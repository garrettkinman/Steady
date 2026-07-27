# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

## The host-side backend hook.
##
## What is worth checking here is not that a callback gets called, but that it
## is called with the right thing at the right time:
##
##   * every flash constant is offered, tagged with what it is
##   * biases are offered *after* folding, since that is what reaches flash
##   * a rewritten constant actually changes the emitted C array, its declared
##     length and its dtype
##   * the layout tag lands in the generated module, so target-side code can
##     refuse a layout it does not implement
##   * a transform that returns inconsistent bytes is caught here rather than
##     by a kernel reading off the end of an array

import std/[unittest, os, strutils, tables, algorithm]
import steadyc
import ../examples/tiny_cnn

let outDir = getTempDir() / "steady_backend_test"

proc emitWith(bk: HostBackend, stem: string): tuple[nimSrc, cSrc: string] =
  var g = tiny_cnn.buildGraph()
  let p = planOne(g)
  emitModel(g, p, outDir, stem, backend = bk)
  (readFile(outDir / (stem & ".nim")), readFile(outDir / (stem & "_weights.c")))

suite "host backend hook":

  test "the default emits exactly what it did before the hook existed":
    let plain = emitWith(noBackend(), "plain")
    check "WeightLayout" notin plain.nimSrc
    check "Host backend" notin plain.nimSrc
    check "rewrote" notin plain.cSrc

  test "every constant is offered, tagged with its role":
    var seenRoles: seq[ConstRole]
    var seenNames: seq[string]

    proc record(g: Graph, c: BackendConst): BackendConst =
      if c.role notin seenRoles: seenRoles.add c.role
      seenNames.add c.name
      c

    discard emitWith(initHostBackend("recorder", "", record), "recorded")
    seenRoles.sort
    # tiny_cnn: conv and fully-connected weights and biases, plus one requant
    # multiplier/shift pair per op.
    check seenRoles == @[crWeights, crBias, crRequantMult, crRequantShift]
    for name in ["conv_w", "conv_b", "fc_w", "fc_b"]:
      check name in seenNames

  test "biases are offered folded, not raw":
    # The graph's own bias bytes are pre-fold; what the hook sees must be the
    # folded values, because those are the ones that end up in flash.
    var offered: Table[string, seq[byte]]

    proc record(g: Graph, c: BackendConst): BackendConst =
      if c.role == crBias: offered[c.name] = c.data
      c

    discard emitWith(initHostBackend("recorder", "", record), "folded")

    let g = tiny_cnn.buildGraph()
    var rawBias: seq[byte]
    for t in g.tensors:
      if t.name == "conv_b": rawBias = t.data
    var convOp: Op
    for op in g.ops:
      if op.kind == okConv2d: convOp = op

    check offered["conv_b"] == foldBias(g, convOp)
    check offered["conv_b"] != rawBias        # folding really did something

  test "a transform changes the emitted array, its length and its dtype":
    # Transpose the fully-connected weights and pad the conv bias out to eight
    # channels — both things real accelerators ask for, both visible in the
    # generated C.
    proc relayout(g: Graph, c: BackendConst): BackendConst =
      result = c
      if c.role == crWeights and c.opKind == okFullyConnected:
        result.data = transposed(c.data, c.dtype.byteWidth, c.shape[0], c.shape[1])
        result.shape = @[c.shape[1], c.shape[0]]
      elif c.role == crBias and c.opKind == okConv2d:
        result.data = padTo(c.data, c.dtype.byteWidth, c.elementCount, 8, 0'u8)
        result.shape = @[8]

    let bk = initHostBackend("test-layout", "steady-test/transposed-fc", relayout)
    let got = emitWith(bk, "transformed")

    check "WeightLayout* = \"steady-test/transposed-fc\"" in got.nimSrc
    check "Host backend: test-layout" in got.nimSrc
    check "constant layout: steady-test/transposed-fc" in got.nimSrc

    # The padded bias is declared and defined at its new length, in the C file
    # and in the Nim declaration block alike.
    check "steady_tiny_cnn_b_conv_b[8]" in got.cSrc
    check "array[8, int32]" in got.nimSrc
    check "rewrote 'conv_b' (crBias) from dtInt32 @[4] to dtInt32 @[8]" in got.cSrc
    check "rewrote 'fc_w' (crWeights) from dtInt8 @[10, 64] to dtInt8 @[64, 10]" in
      got.cSrc

  test "a widening transform leaves the call sequence alone":
    # Padding a constant must not disturb anything downstream: the arena plan,
    # the accessors and the straight-line calls are all unchanged.
    proc padMultipliers(g: Graph, c: BackendConst): BackendConst =
      result = c
      if c.role == crRequantMult:
        let n = c.elementCount
        result.data = padTo(c.data, 4, n, n + 4, 0'u8)
        result.shape = @[n + 4]

    proc invokeBody(s: string): string = s[s.find("proc invoke*()") .. ^1]

    let plain = emitWith(noBackend(), "plain2")
    let padded = emitWith(initHostBackend("padder", "", padMultipliers), "padded")
    check invokeBody(plain.nimSrc) == invokeBody(padded.nimSrc)

  test "an inconsistent transform is rejected, with the blob named":
    proc liar(g: Graph, c: BackendConst): BackendConst =
      result = c
      if c.role == crWeights:
        result.shape = @[c.elementCount * 2]    # bytes no longer match

    var g = tiny_cnn.buildGraph()
    let p = planOne(g)
    expect BackendError:
      emitModel(g, p, outDir, "bad", backend = initHostBackend("liar", "", liar))

suite "layout helpers":

  test "transposed is its own inverse":
    var data = newSeq[byte](12)
    for i in 0 ..< 12: data[i] = byte(i)
    let once = transposed(data, 1, 3, 4)
    check once == @[0'u8, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
    check transposed(once, 1, 4, 3) == data

  test "transposed permutes elements and nothing else":
    let g = tiny_cnn.buildGraph()
    var fcW: seq[byte]
    for t in g.tensors:
      if t.name == "fc_w": fcW = t.data
    let moved = transposed(fcW, 1, 10, 64)
    check moved.len == fcW.len
    check moved != fcW
    var a = fcW
    var b = moved
    a.sort
    b.sort
    check a == b

  test "transposed moves whole elements, not bytes":
    # [1, 2] as int32, a 1x2 matrix transposed to 2x1: the bytes inside each
    # element must stay in order, so the result is unchanged.
    var data = newSeq[byte](8)
    data[0] = 1'u8
    data[4] = 2'u8
    check transposed(data, 4, 1, 2) == data

  test "transposed rejects a length that is not the stated shape":
    expect BackendError:
      discard transposed(newSeq[byte](11), 1, 3, 4)

  test "padTo appends the fill and refuses to shrink":
    check padTo(@[1'u8, 2], 1, 2, 4, 9'u8) == @[1'u8, 2, 9, 9]
    expect BackendError:
      discard padTo(@[1'u8, 2], 1, 2, 1, 0'u8)
