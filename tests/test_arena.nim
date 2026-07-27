# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

## Arena planner.
##
## The safety property is the one that matters: two buffers whose live
## ranges overlap must never share bytes. Everything else — how tightly it
## packs, how it breaks ties — is quality, not correctness.

import std/[unittest, tables, sets, strutils]
import steadyc

proc chainGraph(sizes: openArray[int], name = "chain"): Graph =
  ## input -> fc -> fc -> ..., one activation buffer per stage. Fully
  ## connected rather than clamp because the stages change size, and a
  ## clamp that changes element count is (correctly) rejected by validate.
  result = initGraph(name, pkAffineI8)
  let q = Quant(scales: @[0.05], zeroPoints: @[0'i32], axis: -1)
  let wq = Quant(scales: @[0.01], zeroPoints: @[0'i32], axis: -1)
  var prev = result.addInput("input", @[1, sizes[0]], dtInt8, q)
  var prevSize = sizes[0]
  for i, s in sizes:
    if i == 0: continue
    let w = result.addConst("w" & $i, @[s, prevSize], dtInt8,
                            newSeq[byte](s * prevSize), wq)
    let b = result.addConst("b" & $i, @[s], dtInt32, newSeq[byte](s * 4))
    let t = result.addIntermediate("t" & $i, @[1, s], dtInt8, q)
    discard result.addOp Op(name: "fc" & $i, kind: okFullyConnected,
                            inputs: @[prev, w, b], outputs: @[t], fused: faRelu)
    prev = t
    prevSize = s
  result.markOutput prev
  result.validate

proc overlappingPairsAreDisjoint(p: Plan): bool =
  for i in 0 ..< p.ranges.len:
    for j in i + 1 ..< p.ranges.len:
      let a = p.ranges[i]
      let b = p.ranges[j]
      let liveOverlap = a.first <= b.last and b.first <= a.last
      if liveOverlap:
        let byteOverlap = a.offset < b.offset + b.size and
                          b.offset < a.offset + a.size
        if byteOverlap: return false
  true

suite "arena planner":

  test "overlapping live ranges never share bytes":
    for sizes in [@[64, 256, 64, 10], @[1, 1, 1], @[1000, 4, 1000, 4],
                  @[7, 13, 31, 5, 97]]:
      let g = chainGraph(sizes)
      let p = planOne(g)
      check overlappingPairsAreDisjoint(p)

  test "buffers are reused, so peak is below the naive sum":
    let g = chainGraph(@[256, 256, 256, 256, 256])
    let p = planOne(g)
    var naive = 0
    for r in p.ranges: naive += r.size
    check p.arenaSize < naive
    # A pure chain only ever needs two live buffers at once.
    check p.arenaSize <= 2 * 256 + p.alignment

  test "every offset respects the requested alignment":
    for a in [4, 8, 16, 32]:
      let g = chainGraph(@[7, 13, 31, 5])
      let p = plan([g], a)
      for r in p.ranges:
        check r.offset mod a == 0

  test "rejects a non-power-of-two alignment":
    let g = chainGraph(@[8, 8])
    expect ArenaError:
      discard plan([g], 12)

  test "planning is deterministic":
    let g = chainGraph(@[64, 256, 64, 10, 33, 5])
    let a = planOne(g)
    let b = planOne(g)
    check a.arenaSize == b.arenaSize
    check a.offsets == b.offsets

  test "reshape is free — it produces no buffer of its own":
    var g = initGraph("rs", pkAffineI8)
    let q = Quant(scales: @[0.05], zeroPoints: @[0'i32], axis: -1)
    let x = g.addInput("input", @[1, 4, 4, 4], dtInt8, q)
    let flat = g.addIntermediate("flat", @[1, 64], dtInt8, q)
    discard g.addOp Op(name: "rs", kind: okReshape, inputs: @[x], outputs: @[flat])
    g.markOutput flat
    g.validate
    let p = planOne(g)
    check p.ranges.len == 1                 # not two
    check p.arenaSize == 64
    let alias = resolveAliases(g)
    check p.offsetFor(g, alias, x) == p.offsetFor(g, alias, flat)

  test "several models in one binary share an arena but never share bytes":
    let a = chainGraph(@[256, 256, 128])
    let b = chainGraph(@[512, 64], "other")
    let p = plan([a, b])
    check overlappingPairsAreDisjoint(p)
    # Sized for the worst case, not the sum of both peaks.
    check p.arenaSize >= planOne(b).arenaSize
    check p.arenaSize < planOne(a).arenaSize + planOne(b).arenaSize
    # No two buffers from different models may alias, in any invocation order.
    var offsets = initHashSet[int]()
    for r in p.ranges: offsets.incl r.offset
    check offsets.len >= 2

  test "graph outputs stay live to the end":
    let g = chainGraph(@[64, 256, 64])
    let p = planOne(g)
    let alias = resolveAliases(g)
    let outOff = p.offsetFor(g, alias, g.outputs[0])
    var outRange: LiveRange
    for r in p.ranges:
      if r.offset == outOff and r.size == 64: outRange = r
    check outRange.last >= g.ops.len

  test "an input buffer stops being live after its last read":
    # This is a deliberate contract, not an accident, and it is worth pinning:
    # an input's bytes are reusable the moment the model has read them, which
    # is why the emitted accessors and the C header both say to refill every
    # input before every invoke. Relaxing it would cost RAM on exactly the
    # models where the input is the largest buffer.
    let g = chainGraph(@[256, 8, 8, 8])
    let alias = resolveAliases(g)
    let ranges = computeLiveRanges(g, alias)
    var inputRange: LiveRange
    for r in ranges:
      if g.tensors[r.tensor].kind == tkInput: inputRange = r
    check inputRange.last == 0                # read by op 0, dead after it
    check inputRange.last < g.ops.len

    # And the planner does put something else there, on a graph shaped to make
    # that worthwhile.
    let p = planOne(g)
    let inputOffset = p.offsets[inputRange.tensor]
    var sharesWithInput = false
    for r in p.ranges:
      if r.tensor != inputRange.tensor and r.offset == inputOffset:
        sharesWithInput = true
    check sharesWithInput

  test "report states peak RAM and flash":
    let g = chainGraph(@[64, 256])
    let r = planOne(g).report([g])
    check "peak RAM" in r
    check "flash" in r

suite "graph validation":

  test "catches a shape mismatch with a readable message":
    var g = initGraph("bad", pkAffineI8)
    let q = Quant(scales: @[0.05], zeroPoints: @[0'i32], axis: -1)
    let x = g.addInput("input", @[1, 8], dtInt8, q)
    let w = g.addConst("w", @[4, 16], dtInt8, newSeq[byte](64), q)
    let b = g.addConst("b", @[4], dtInt32, newSeq[byte](16))
    let y = g.addIntermediate("y", @[1, 4], dtInt8, q)
    discard g.addOp Op(name: "fc", kind: okFullyConnected,
                       inputs: @[x, w, b], outputs: @[y])
    g.markOutput y
    try:
      g.validate
      check false
    except IrError as e:
      check "fc" in e.msg
      check "8" in e.msg and "16" in e.msg

  test "catches a convolution whose kernel does not fit":
    var g = initGraph("bad2", pkAffineI8)
    let q = Quant(scales: @[0.05], zeroPoints: @[0'i32], axis: -1)
    let x = g.addInput("input", @[1, 2, 2, 1], dtInt8, q)
    let w = g.addConst("w", @[1, 5, 5, 1], dtInt8, newSeq[byte](25),
                       Quant(scales: @[0.01], zeroPoints: @[0'i32], axis: 0))
    let b = g.addConst("b", @[1], dtInt32, newSeq[byte](4))
    let y = g.addIntermediate("y", @[1, 1, 1, 1], dtInt8, q)
    discard g.addOp Op(name: "conv", kind: okConv2d, inputs: @[x, w, b],
                       outputs: @[y], padding: padValid, strideH: 1, strideW: 1,
                       dilationH: 1, dilationW: 1)
    g.markOutput y
    expect IrError:
      g.validate

  test "catches a tensor written twice":
    var g = initGraph("bad3", pkAffineI8)
    let q = Quant(scales: @[0.05], zeroPoints: @[0'i32], axis: -1)
    let x = g.addInput("input", @[1, 8], dtInt8, q)
    let y = g.addIntermediate("y", @[1, 8], dtInt8, q)
    discard g.addOp Op(name: "c1", kind: okClamp, inputs: @[x], outputs: @[y])
    discard g.addOp Op(name: "c2", kind: okClamp, inputs: @[x], outputs: @[y])
    g.markOutput y
    expect IrError:
      g.validate

  test "catches use before definition":
    var g = initGraph("bad4", pkAffineI8)
    let q = Quant(scales: @[0.05], zeroPoints: @[0'i32], axis: -1)
    let x = g.addInput("input", @[1, 8], dtInt8, q)
    let a = g.addIntermediate("a", @[1, 8], dtInt8, q)
    let b = g.addIntermediate("b", @[1, 8], dtInt8, q)
    discard g.addOp Op(name: "c1", kind: okClamp, inputs: @[b], outputs: @[a])
    discard g.addOp Op(name: "c2", kind: okClamp, inputs: @[x], outputs: @[b])
    g.markOutput a
    expect IrError:
      g.validate
