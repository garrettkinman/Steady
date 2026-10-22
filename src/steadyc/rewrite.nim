# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

## Graph rewrites: chains of ops that a converter writes out step by step,
## found and replaced by the one op they amount to.
##
## The first are relative-position attention's (Shaw et al., Transformer-XL,
## VITS and the TTS models after it). Its logits by relative position, a
## window of `w` either side of the diagonal, are moved onto the diagonal of
## the scores by a skew: pad each row by the length, read the rows back one
## shorter, so that each moves one further right than the last, and crop.
## The weights go the other way. Each is five ops over tensors of `n^2`
## elements and more, for a result whose every element is a read of the
## input at a known place:
##
##   band     out[.., i, j] = x[.., i, j - i + c]   for 0 <= j - i + c < w, else 0
##   unband   out[.., i, k] = p[.., i, i + k - a]   for 0 <= i + k - a < n, else 0
##
## and a band read only by an ADD, as the scores take it, is added there from
## x, its n^2 values never written: outside the window it is 0, whose
## rescaling is 0, so each output is the ADD's to the bit.
##
## Values are moved and never changed, so the rewrite is exact by
## construction; the differential harness checks that it is, the generated
## code taking the rewritten graph and the simulator the one TFLite wrote.

import std/[tables, sets]
import ./ir

proc consumers(g: Graph): Table[int, seq[int]] =
  for oi, op in g.ops:
    for t in op.inputs: result.mgetOrPut(t, @[]).add oi

proc only(g: Graph, uses: Table[int, seq[int]], t: int): int =
  ## The one op reading `t`, if one reads it and it is not an output
  if t in g.outputs or uses.getOrDefault(t).len != 1: -1 else: uses[t][0]

proc lastPad(op: Op): Dim =
  ## What a PAD adds after the last axis, if that is all it pads; -1 if not
  let p = op.padsOf
  for d in 0 ..< p.len - 1:
    if p[d][0] != dim(0) or p[d][1] != dim(0): return dim(-1)
  if p[^1][0] != dim(0): return dim(-1)
  p[^1][1]

proc chain(g: Graph, uses: Table[int, seq[int]], start: int,
           kinds: openArray[OpKind]): seq[int] =
  ## The ops reading `start`, then each the last's output, of these kinds,
  ## each the only reader of what it reads
  var t = start
  for k in kinds:
    let oi = only(g, uses, t)
    if oi < 0 or g.ops[oi].kind != k: return @[]
    result.add oi
    t = g.ops[oi].outputs[0]

proc matchBand(g: Graph, uses: Table[int, seq[int]], pad: int): (bool, seq[int], int) =
  ## PAD [.., n, w] by n after its last axis, RESHAPE to [.., n (n + w)],
  ## SLICE the first n (n + w - 1), RESHAPE to [.., n, n + w - 1], SLICE
  ## [.., n, n] from c along the last
  let p = g.ops[pad]
  let x = g.tensors[p.inputs[0]].dimsOf
  if x.len < 2: return
  let (n, w) = (x[^2], x[^1])
  if not w.isConst or p.lastPad != n: return
  let ops = @[pad] & chain(g, uses, p.outputs[0], [okReshape, okSlice, okReshape, okSlice])
  if ops.len != 5: return
  let lead = x[0 ..< ^2]
  let shapes = [lead & @[n * (n + w)], lead & @[n * (n + w - 1)], lead & @[n, n + w - 1], lead & @[n, n]]
  for k in 1 .. 4:
    if g.tensors[g.ops[ops[k]].outputs[0]].dimsOf != shapes[k - 1]: return
  for b in g.ops[ops[2]].beginOf:
    if b != dim(0): return
  let begin = g.ops[ops[4]].beginOf
  for b in begin[0 ..< ^1]:
    if b != dim(0): return
  if not begin[^1].isConst or begin[^1].value >= w.value: return
  (true, ops, begin[^1].value)

proc matchUnband(g: Graph, uses: Table[int, seq[int]], pad: int): (bool, seq[int], int) =
  ## PAD [.., n, n] by a either side of its last axis, RESHAPE to
  ## [.., n (n + 2a)], PAD by n after it, RESHAPE to [.., n, n + 2a + 1], and
  ## SLICE [.., n, w] from 0
  let p = g.ops[pad]
  let x = g.tensors[p.inputs[0]].dimsOf
  if x.len < 2 or x[^1] != x[^2]: return
  let n = x[^1]
  let pads = p.padsOf
  for d in 0 ..< pads.len - 1:
    if pads[d][0] != dim(0) or pads[d][1] != dim(0): return
  let a = pads[^1][0]
  if not a.isConst or pads[^1][1] != a: return
  let ops = @[pad] & chain(g, uses, p.outputs[0], [okReshape, okPad, okReshape, okSlice])
  if ops.len != 5: return
  let lead = x[0 ..< ^2]
  if g.tensors[g.ops[ops[1]].outputs[0]].dimsOf != lead & @[n * (n + 2 * a.value)]: return
  if g.ops[ops[2]].lastPad != n: return
  if g.tensors[g.ops[ops[3]].outputs[0]].dimsOf != lead & @[n, n + 2 * a.value + 1]: return
  let y = g.tensors[g.ops[ops[4]].outputs[0]].dimsOf
  if y[0 ..< ^1] != lead & @[n] or not y[^1].isConst or y[^1].value > 2 * a.value + 1: return
  for b in g.ops[ops[4]].beginOf:
    if b != dim(0): return
  (true, ops, a.value)

proc addBands(g: Graph): Graph =
  ## An ADD of a band and something else, the band read by nothing more, as
  ## one op reading the band's input
  result = g
  let uses = consumers(g)
  var producer = initTable[int, int]()
  for oi, op in g.ops: producer[op.outputs[0]] = oi
  var gone = initHashSet[int]()
  var fused = initTable[int, Op]()
  for oi, op in g.ops:
    if op.kind != okAdd or g.policy != pkAffineI16x8: continue
    for side in 0 .. 1:
      let t = op.inputs[side]
      if t notin producer or g.ops[producer[t]].kind != okBand or only(g, uses, t) != oi: continue
      if g.tensors[op.inputs[1 - side]].shape != g.tensors[op.outputs[0]].shape: continue
      let band = g.ops[producer[t]]
      var f = op
      f.kind = okAddBand
      f.inputs = @[op.inputs[1 - side], band.inputs[0]]
      f.bandStart = band.bandStart
      f.bandSide = side
      fused[oi] = f
      gone.incl producer[t]
      break
  result.ops = @[]
  for oi, op in g.ops:
    if oi in fused: result.ops.add fused[oi]
    elif oi notin gone: result.ops.add op

proc rewrite*(g: Graph): Graph =
  ## `g` with each chain these rewrites know replaced by its one op. Tensors
  ## left unread stay in the graph, as constants and outputs do; the planner
  ## gives the arena only to what an op writes.
  result = g
  let uses = consumers(g)
  var gone = initHashSet[int]()
  var replaced = initTable[int, Op]()      # first op of a chain -> its one op
  for oi, op in g.ops:
    if op.kind != okPad or oi in gone: continue
    let input = op.inputs[0]
    let (isBand, bandOps, c) = matchBand(g, uses, oi)
    if isBand:
      let last = g.ops[bandOps[^1]]
      replaced[oi] = Op(name: last.name, kind: okBand, inputs: @[input],
                        outputs: last.outputs, bandStart: c)
      for k in bandOps: gone.incl k
      continue
    let (isUnband, unbandOps, a) = matchUnband(g, uses, oi)
    if isUnband:
      let last = g.ops[unbandOps[^1]]
      replaced[oi] = Op(name: last.name, kind: okUnband, inputs: @[input],
                        outputs: last.outputs, bandStart: a)
      for k in unbandOps: gone.incl k
  result.ops = @[]
  for oi, op in g.ops:
    if oi in replaced: result.ops.add replaced[oi]
    elif oi notin gone: result.ops.add op
  result = addBands(result)
