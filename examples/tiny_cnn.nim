# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

## A small int8 CNN, built directly against the IR.
##
## Until the TFLite importer lands this is how graphs are constructed; once
## it does, the importer will populate exactly this structure and everything
## downstream is unchanged. The shape is chosen to exercise the parts that
## are easy to get wrong: SAME padding (which stresses bias folding at the
## edges), per-channel weight quantization, a reshape that should cost
## nothing, and a buffer-reuse opportunity in the arena.
##
##   input [1,8,8,1] -> conv 3x3 SAME x4 +relu -> [1,8,8,4]
##                   -> maxpool 2x2 s2         -> [1,4,4,4]
##                   -> reshape                -> [1,64]
##                   -> fully connected        -> [1,10]

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
    if v < -127: v = -127        # keep symmetric; -128 has no positive twin
    result[i] = cast[byte](int8(v))

proc i32Bytes(n: int, r: var Lcg): seq[int32] =
  result = newSeq[int32](n)
  for i in 0 ..< n:
    result[i] = r.next * 7

proc buildGraph*(): Graph =
  var r = Lcg(s: 12345'u32)
  result = initGraph("tiny_cnn", pkAffineI8)

  let inQ = Quant(scales: @[0.05], zeroPoints: @[-3'i32], axis: -1)
  let x = result.addInput("input", @[1, 8, 8, 1], dtInt8, inQ)

  # --- conv2d, per-channel weights -----------------------------------------
  var convScales: seq[float64]
  for c in 0 ..< 4: convScales.add 0.003 + 0.0011 * float64(c)
  let convWQ = Quant(scales: convScales, zeroPoints: @[0'i32], axis: 0)
  let convW = result.addConst("conv_w", @[4, 3, 3, 1], dtInt8,
                              i8Bytes(4 * 3 * 3 * 1, r), convWQ)
  let convB = result.addConst("conv_b", @[4], dtInt32,
                              fromInt32(i32Bytes(4, r)))
  let convOutQ = Quant(scales: @[0.02], zeroPoints: @[-14'i32], axis: -1)
  let convOut = result.addIntermediate("conv_out", @[1, 8, 8, 4], dtInt8, convOutQ)

  discard result.addOp Op(
    name: "conv1", kind: okConv2d,
    inputs: @[x, convW, convB], outputs: @[convOut],
    padding: padSame, strideH: 1, strideW: 1,
    dilationH: 1, dilationW: 1, fused: faRelu)

  # --- maxpool -------------------------------------------------------------
  let poolOut = result.addIntermediate("pool_out", @[1, 4, 4, 4], dtInt8, convOutQ)
  discard result.addOp Op(
    name: "pool1", kind: okMaxPool2d,
    inputs: @[convOut], outputs: @[poolOut],
    padding: padValid, strideH: 2, strideW: 2,
    kH: 2, kW: 2, fused: faNone)

  # --- reshape (should be free) --------------------------------------------
  let flat = result.addIntermediate("flat", @[1, 64], dtInt8, convOutQ)
  discard result.addOp Op(
    name: "flatten", kind: okReshape,
    inputs: @[poolOut], outputs: @[flat])

  # --- fully connected -----------------------------------------------------
  var fcScales: seq[float64]
  for c in 0 ..< 10: fcScales.add 0.004 + 0.0007 * float64(c)
  let fcWQ = Quant(scales: fcScales, zeroPoints: @[0'i32], axis: 0)
  let fcW = result.addConst("fc_w", @[10, 64], dtInt8, i8Bytes(10 * 64, r), fcWQ)
  let fcB = result.addConst("fc_b", @[10], dtInt32, fromInt32(i32Bytes(10, r)))
  let outQ = Quant(scales: @[0.09], zeroPoints: @[5'i32], axis: -1)
  let logits = result.addIntermediate("logits", @[1, 10], dtInt8, outQ)

  discard result.addOp Op(
    name: "fc1", kind: okFullyConnected,
    inputs: @[flat, fcW, fcB], outputs: @[logits],
    fused: faNone)

  result.markOutput logits
  result.validate

when isMainModule:
  import std/[os, strformat]

  let outDir = if paramCount() >= 1: paramStr(1) else: "generated"
  var g = buildGraph()
  let p = planOne(g)

  echo &"model '{g.name}'  policy {g.policy.policyName}  ops {g.ops.len}"
  echo p.report([g])

  emitModel(g, p, outDir, "tiny_cnn")
  echo &"wrote {outDir}/tiny_cnn.nim, tiny_cnn_weights.c, tiny_cnn_weights.h"
