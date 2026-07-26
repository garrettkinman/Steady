# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

## Lifetime analysis and arena packing.
##
## Every activation buffer in the graph gets a byte offset into one static
## arena. Because the graph is feed-forward with no dynamic control flow,
## each buffer's live range is a single interval `[firstWrite, lastRead]`,
## and placement reduces to packing intervals into a 1-D address space —
## essentially offline register allocation.
##
## The strategy is greedy-by-size: sort buffers largest first, and give each
## the lowest offset that does not collide with an already-placed buffer
## whose live range overlaps. This is what TFLite Micro's memory planner
## does, and in practice it lands within a few percent of optimal while
## staying trivial to audit.
##
## The planner takes a *set* of graphs rather than one. Several models in a
## single binary (a small always-on detector gating a larger classifier) is
## an ordinary requirement, and they should share one arena sized to the
## worst case rather than each carrying their own.

import std/[algorithm, strformat, tables]
import ./ir

type
  LiveRange* = object
    tensor*: int          ## index into Graph.tensors
    first*, last*: int    ## inclusive op-index interval
    size*: int            ## bytes
    offset*: int          ## assigned; -1 until planned

  Plan* = object
    ranges*: seq[LiveRange]
    offsets*: Table[int, int]   ## tensor index -> byte offset
    arenaSize*: int
    alignment*: int

  ArenaError* = object of CatchableError

const DefaultAlignment* = 8
  ## Wide enough for int32/float32 loads on all the targets in scope, and
  ## for the 8-byte moves an accelerator DMA typically wants.

func alignUp(x, a: int): int = ((x + a - 1) div a) * a

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ALIASING
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

func isPureAlias*(op: Op): bool =
  ## Ops that only reinterpret their input's layout. Their output can share
  ## the input's storage outright, so no buffer and no copy is needed.
  op.kind == okReshape

func canRunInPlace*(op: Op): bool =
  ## Ops whose output element `i` depends only on input element `i`. These
  ## may write into their input buffer, which is frequently the difference
  ## between fitting in RAM and not.
  op.kind == okClamp

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# LIFETIME ANALYSIS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

proc resolveAliases*(g: Graph): Table[int, int] =
  ## Maps each tensor to the tensor whose storage it actually occupies.
  ## Chains (reshape of a reshape) collapse to a single representative.
  result = initTable[int, int]()
  for i in 0 ..< g.tensors.len:
    result[i] = i
  for op in g.ops:
    if op.isPureAlias:
      var root = op.inputs[0]
      while result[root] != root:
        root = result[root]
      # A graph output must keep its own identity only if the source is also
      # an output; otherwise sharing storage is safe and free.
      result[op.outputs[0]] = root

proc computeLiveRanges*(g: Graph, alias: Table[int, int]): seq[LiveRange] =
  ## Live range of every arena-resident buffer, in op-index units.
  var first = initTable[int, int]()
  var last = initTable[int, int]()

  proc touchWrite(t, oi: int) =
    if t notin first: first[t] = oi
    if t notin last or last[t] < oi: last[t] = oi

  proc touchRead(t, oi: int) =
    if t notin first: first[t] = oi
    if t notin last or last[t] < oi: last[t] = oi

  for i in g.inputs:
    let r = alias[i]
    touchWrite(r, -1)

  for oi, op in g.ops:
    if op.isPureAlias: continue     # no storage of its own
    for t in op.inputs:
      if g.tensors[t].kind != tkConst:
        touchRead(alias[t], oi)
    for t in op.outputs:
      touchWrite(alias[t], oi)

  # Graph outputs stay live past the final op so nothing overwrites them.
  for o in g.outputs:
    last[alias[o]] = g.ops.len

  for t, f in first:
    let kind = g.tensors[t].kind
    if kind == tkConst: continue
    result.add LiveRange(tensor: t, first: f, last: last[t],
                         size: g.tensors[t].byteSize, offset: -1)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PACKING
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

func overlaps(a, b: LiveRange): bool =
  a.first <= b.last and b.first <= a.last

proc plan*(graphs: openArray[Graph], alignment = DefaultAlignment): Plan =
  ## Packs all buffers from all graphs into one arena.
  if alignment <= 0 or (alignment and (alignment - 1)) != 0:
    raise newException(ArenaError, &"alignment {alignment} is not a power of two")

  var all: seq[LiveRange]
  var offsetBase = 0
  for g in graphs:
    let alias = resolveAliases(g)
    var rs = computeLiveRanges(g, alias)
    # Shift each graph's op indices into a private band so buffers from
    # different graphs always appear to overlap and never share storage.
    # Models in one binary may be invoked in any order or interleaved.
    for r in rs.mitems:
      r.first += offsetBase
      r.last += offsetBase
    all.add rs
    offsetBase += g.ops.len + 2

  # Largest first; ties broken by longest-lived, then by tensor index so the
  # result is deterministic across runs. Reproducible builds matter when the
  # output is checked in.
  all.sort proc (a, b: LiveRange): int =
    if a.size != b.size: return cmp(b.size, a.size)
    let aLife = a.last - a.first
    let bLife = b.last - b.first
    if aLife != bLife: return cmp(bLife, aLife)
    cmp(a.tensor, b.tensor)

  var placed: seq[LiveRange]
  for r in all.mitems:
    let need = alignUp(r.size, alignment)
    var candidate = 0
    # Walk the gaps between already-placed conflicting buffers, lowest first.
    var conflicts: seq[LiveRange]
    for p in placed:
      if overlaps(r, p): conflicts.add p
    conflicts.sort proc (a, b: LiveRange): int = cmp(a.offset, b.offset)

    for c in conflicts:
      if candidate + need <= c.offset:
        break                          # fits in the gap below this buffer
      candidate = max(candidate, alignUp(c.offset + c.size, alignment))
    r.offset = candidate
    placed.add r
    result.arenaSize = max(result.arenaSize, candidate + need)

  result.ranges = placed
  result.alignment = alignment
  result.offsets = initTable[int, int]()
  for r in placed:
    result.offsets[r.tensor] = r.offset

proc planOne*(g: Graph, alignment = DefaultAlignment): Plan =
  plan([g], alignment)

proc offsetFor*(p: Plan, g: Graph, alias: Table[int, int], tensor: int): int =
  let root = alias[tensor]
  if root notin p.offsets:
    raise newException(ArenaError,
      &"tensor '{g.tensors[tensor].name}' was never assigned an arena offset")
  p.offsets[root]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# REPORTING
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

proc totalConstBytes*(g: Graph): int =
  for t in g.tensors:
    if t.kind == tkConst: result += t.data.len

proc report*(p: Plan, graphs: openArray[Graph]): string =
  ## Build-time resource report. Peak RAM should be a number the user reads
  ## at compile time, not something they discover on the device.
  var naive = 0
  for r in p.ranges: naive += alignUp(r.size, p.alignment)
  var flash = 0
  for g in graphs: flash += g.totalConstBytes

  result = "arena:\n"
  result.add &"  buffers        {p.ranges.len}\n"
  result.add &"  peak RAM       {p.arenaSize} bytes\n"
  result.add &"  without reuse  {naive} bytes\n"
  if naive > 0:
    let saved = 100.0 * (1.0 - float(p.arenaSize) / float(naive))
    result.add &"  saved          {saved:.1f}%\n"
  result.add &"  flash (const)  {flash} bytes\n"
