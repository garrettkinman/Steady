# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

## A small int8 net whose only purpose is to be awkward.
##
## `tiny_cnn` exercises the straight-line matmul path. This one exercises the
## data-movement and non-linear ops, and deliberately picks quantization
## parameters that force the *general* path through each of them rather than
## the fast one:
##
##   input [1,6,6,2]
##     -> pad 1,1,1,1              -> [1,8,8,2]   explicit spatial padding
##     -> conv 3x3 VALID x4 +relu  -> [1,6,6,4]   branch A, scale 0.02
##     -> depthwise 3x3 SAME       -> [1,6,6,4]   branch B, scale 0.035
##     -> concat A, B on axis 3    -> [1,6,6,8]   A copies, B must rescale
##     -> mean over H, W           -> [1,8]       output scale differs from A
##     -> tanh                     -> [1,8]       256-entry host-built table
##     -> fully connected          -> [1,5]
##     -> softmax                  -> [1,5]       scale 1/256, zero -128
##
## The concatenation is the interesting one: operand A already carries the
## output's quantization and is copied verbatim, operand B does not and goes
## through the rescale path, so one op covers both branches of that decision.

import steadyc

type Lcg = object
  s: uint32

proc next(r: var Lcg): int32 =
  r.s = r.s * 1664525'u32 + 1013904223'u32
  int32((r.s shr 16) and 0xFF'u32) - 128'i32

proc i8Bytes(n: int, r: var Lcg): seq[byte] =
  result = newSeq[byte](n)
  for i in 0 ..< n:
    var v = r.next
    if v < -127: v = -127          # keep weights symmetric; Zw must be 0
    result[i] = cast[byte](int8(v))

proc i32Bytes(n: int, r: var Lcg): seq[int32] =
  result = newSeq[int32](n)
  for i in 0 ..< n:
    result[i] = r.next * 5

proc perChannel(n: int, base, step: float64): Quant =
  var scales: seq[float64]
  for c in 0 ..< n: scales.add base + step * float64(c)
  Quant(scales: scales, zeroPoints: @[0'i32], axis: 0)

proc buildGraph*(): Graph =
  var r = Lcg(s: 987654'u32)
  result = initGraph("branch_net", pkAffineI8)

  let inQ = Quant(scales: @[0.05], zeroPoints: @[-3'i32], axis: -1)
  let x = result.addInput("input", @[1, 6, 6, 2], dtInt8, inQ)

  # --- pad: same quantization as the input, filled with real 0.0 ------------
  let padded = result.addIntermediate("padded", @[1, 8, 8, 2], dtInt8, inQ)
  discard result.addOp Op(
    name: "pad1", kind: okPad,
    inputs: @[x], outputs: @[padded],
    pads: @[[0, 0], [1, 1], [1, 1], [0, 0]], padConst: 0.0)

  # --- branch A: conv 3x3 VALID over the padded input -----------------------
  let convW = result.addConst("conv_w", @[4, 3, 3, 2], dtInt8,
                              i8Bytes(4 * 3 * 3 * 2, r), perChannel(4, 0.004, 0.0009))
  let convB = result.addConst("conv_b", @[4], dtInt32, fromInt32(i32Bytes(4, r)))
  let aQ = Quant(scales: @[0.02], zeroPoints: @[-14'i32], axis: -1)
  let convOut = result.addIntermediate("conv_out", @[1, 6, 6, 4], dtInt8, aQ)
  discard result.addOp Op(
    name: "conv1", kind: okConv2d,
    inputs: @[padded, convW, convB], outputs: @[convOut],
    padding: padValid, strideH: 1, strideW: 1,
    dilationH: 1, dilationW: 1, fused: faRelu)

  # --- branch B: depthwise 3x3 SAME over branch A ---------------------------
  # Per-channel weights on a depthwise filter are laid out [1, kH, kW, outC],
  # so a channel's taps are strided — the bias fold has to know that.
  var dwScales: seq[float64]
  for c in 0 ..< 4: dwScales.add 0.006 + 0.0013 * float64(c)
  let dwW = result.addConst("dw_w", @[1, 3, 3, 4], dtInt8,
                            i8Bytes(1 * 3 * 3 * 4, r),
                            Quant(scales: dwScales, zeroPoints: @[0'i32], axis: 3))
  let dwB = result.addConst("dw_b", @[4], dtInt32, fromInt32(i32Bytes(4, r)))
  let bQ = Quant(scales: @[0.035], zeroPoints: @[5'i32], axis: -1)
  let dwOut = result.addIntermediate("dw_out", @[1, 6, 6, 4], dtInt8, bQ)
  discard result.addOp Op(
    name: "dw1", kind: okDepthwiseConv2d,
    inputs: @[convOut, dwW, dwB], outputs: @[dwOut],
    padding: padSame, strideH: 1, strideW: 1,
    dilationH: 1, dilationW: 1, depthMultiplier: 1, fused: faNone)

  # --- concat on channels: A matches the output quant, B does not -----------
  let cat = result.addIntermediate("cat", @[1, 6, 6, 8], dtInt8, aQ)
  discard result.addOp Op(
    name: "cat1", kind: okConcatenation,
    inputs: @[convOut, dwOut], outputs: @[cat], axis: 3)

  # --- global average, into a different quantization ------------------------
  let mQ = Quant(scales: @[0.03], zeroPoints: @[-8'i32], axis: -1)
  let pooled = result.addIntermediate("pooled", @[1, 8], dtInt8, mQ)
  discard result.addOp Op(
    name: "gap", kind: okMean,
    inputs: @[cat], outputs: @[pooled],
    axes: @[1, 2], keepDims: false)

  # --- tanh from a host-generated table -------------------------------------
  # tanh lands in [-1, 1], so a symmetric scale of 1/128 with zero point 0
  # spends the whole int8 range on the actual output range.
  let tQ = Quant(scales: @[1.0 / 128.0], zeroPoints: @[0'i32], axis: -1)
  let activated = result.addIntermediate("activated", @[1, 8], dtInt8, tQ)
  discard result.addOp Op(
    name: "tanh1", kind: okTanh,
    inputs: @[pooled], outputs: @[activated])

  # --- classifier -----------------------------------------------------------
  let fcW = result.addConst("fc_w", @[5, 8], dtInt8, i8Bytes(5 * 8, r),
                            perChannel(5, 0.005, 0.0008))
  let fcB = result.addConst("fc_b", @[5], dtInt32, fromInt32(i32Bytes(5, r)))
  let lQ = Quant(scales: @[0.11], zeroPoints: @[2'i32], axis: -1)
  let logits = result.addIntermediate("logits", @[1, 5], dtInt8, lQ)
  discard result.addOp Op(
    name: "fc1", kind: okFullyConnected,
    inputs: @[activated, fcW, fcB], outputs: @[logits],
    fused: faNone)

  # --- softmax: TFLite pins int8 output at scale 1/256, zero point -128 -----
  let pQ = Quant(scales: @[1.0 / 256.0], zeroPoints: @[-128'i32], axis: -1)
  let probs = result.addIntermediate("probs", @[1, 5], dtInt8, pQ)
  discard result.addOp Op(
    name: "softmax1", kind: okSoftmax,
    inputs: @[logits], outputs: @[probs])

  result.markOutput probs
  result.validate

when isMainModule:
  import std/[os, strformat]

  let outDir = if paramCount() >= 1: paramStr(1) else: "generated"
  var g = buildGraph()
  let p = planOne(g)

  echo &"model '{g.name}'  policy {g.policy.policyName}  ops {g.ops.len}"
  echo p.report([g])

  emitModel(g, p, outDir, "branch_net")
  emitCApi(g, p, outDir, "branch_net")
  echo &"wrote {outDir}/branch_net.nim, branch_net_weights.c, branch_net_weights.h, " &
       &"branch_net_api.nim, branch_net.h"
