# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

## A small posit(8,0) net, built directly against the IR.
##
## `tiny_cnn` and `branch_net` are int8; this one exists to run the same
## compiler over a *real-number* policy end to end — no scales, no zero
## points, no requantization, an exact quire where the int8 path has an int32
## accumulator and a Q31 multiplier.
##
##   input [1,6,6,2]
##     -> pad 1,1,1,1              -> [1,8,8,2]   pad value is a real 0.0
##     -> conv 3x3 VALID x4 +relu  -> [1,6,6,4]   clamp in the quire domain
##     -> depthwise 3x3 SAME       -> [1,6,6,4]
##     -> add(conv, depthwise)     -> [1,6,6,4]   exact sum, one rounding
##     -> concat with conv, axis 3 -> [1,6,6,8]
##     -> avgpool 2x2 s2           -> [1,3,3,8]   the quire's only division
##     -> tanh                     -> [1,3,3,8]   table keyed on the encoding
##     -> mean over H, W           -> [1,8]
##     -> fully connected          -> [1,3]
##
## Softmax is absent because it is absent from the policy: it needs a uniform
## integer store domain and a posit has neither. The host rejects it rather
## than the kernel approximating it, which `test_codec` pins.
##
## Magnitudes are chosen deliberately. maxpos is 64 and the format carries
## about five significant bits near 1.0, so weights sit near +/-0.25 and the
## eighteen-tap convolution lands well inside the range. A net that saturated
## everywhere would still be a valid test of the plumbing and a useless test
## of the arithmetic.

import steadyc
import steady/posit8

type Lcg = object
  s: uint32

proc next(r: var Lcg): float64 =
  ## A value in [-1, 1), quantized to the input's own grid so that the
  ## generator is not itself a source of rounding.
  r.s = r.s * 1664525'u32 + 1013904223'u32
  float64(int32((r.s shr 16) and 0xFF'u32) - 128'i32) / 128.0

proc p8Bytes(n: int, r: var Lcg, scale: float64): seq[byte] =
  result = newSeq[byte](n)
  for i in 0 ..< n:
    result[i] = toPosit8(r.next * scale).bits

proc buildGraph*(): Graph =
  var r = Lcg(s: 24680'u32)
  result = initGraph("posit_net", pkRealP8)

  let x = result.addInput("input", @[1, 6, 6, 2], dtPosit8)

  # --- pad: filled with a real 0.0, which for a posit is the zero encoding --
  let padded = result.addIntermediate("padded", @[1, 8, 8, 2], dtPosit8)
  discard result.addOp Op(
    name: "pad1", kind: okPad,
    inputs: @[x], outputs: @[padded],
    pads: @[[0, 0], [1, 1], [1, 1], [0, 0]], padConst: 0.0)

  # --- conv 3x3 VALID, relu ------------------------------------------------
  let convW = result.addConst("conv_w", @[4, 3, 3, 2], dtPosit8,
                              p8Bytes(4 * 3 * 3 * 2, r, 0.25))
  let convB = result.addConst("conv_b", @[4], dtPosit8, p8Bytes(4, r, 0.5))
  let convOut = result.addIntermediate("conv_out", @[1, 6, 6, 4], dtPosit8)
  discard result.addOp Op(
    name: "conv1", kind: okConv2d,
    inputs: @[padded, convW, convB], outputs: @[convOut],
    padding: padValid, strideH: 1, strideW: 1,
    dilationH: 1, dilationW: 1, fused: faRelu)

  # --- depthwise 3x3 SAME --------------------------------------------------
  let dwW = result.addConst("dw_w", @[1, 3, 3, 4], dtPosit8,
                            p8Bytes(1 * 3 * 3 * 4, r, 0.3))
  let dwB = result.addConst("dw_b", @[4], dtPosit8, p8Bytes(4, r, 0.5))
  let dwOut = result.addIntermediate("dw_out", @[1, 6, 6, 4], dtPosit8)
  discard result.addOp Op(
    name: "dw1", kind: okDepthwiseConv2d,
    inputs: @[convOut, dwW, dwB], outputs: @[dwOut],
    padding: padSame, strideH: 1, strideW: 1,
    dilationH: 1, dilationW: 1, depthMultiplier: 1, fused: faNone)

  # --- two-input add: both operands already real, so a plain exact sum -----
  let summed = result.addIntermediate("summed", @[1, 6, 6, 4], dtPosit8)
  discard result.addOp Op(
    name: "add1", kind: okAdd,
    inputs: @[convOut, dwOut], outputs: @[summed], fused: faNone)

  # --- concat on channels --------------------------------------------------
  let cat = result.addIntermediate("cat", @[1, 6, 6, 8], dtPosit8)
  discard result.addOp Op(
    name: "cat1", kind: okConcatenation,
    inputs: @[convOut, summed], outputs: @[cat], axis: 3)

  # --- average pool: the one place this policy rounds twice ----------------
  let pooled = result.addIntermediate("pooled", @[1, 3, 3, 8], dtPosit8)
  discard result.addOp Op(
    name: "pool1", kind: okAvgPool2d,
    inputs: @[cat], outputs: @[pooled],
    padding: padValid, strideH: 2, strideW: 2,
    kH: 2, kW: 2, fused: faNone)

  # --- tanh from a host-generated table ------------------------------------
  let activated = result.addIntermediate("activated", @[1, 3, 3, 8], dtPosit8)
  discard result.addOp Op(
    name: "tanh1", kind: okTanh,
    inputs: @[pooled], outputs: @[activated])

  # --- global average ------------------------------------------------------
  let gap = result.addIntermediate("gap", @[1, 8], dtPosit8)
  discard result.addOp Op(
    name: "gap1", kind: okMean,
    inputs: @[activated], outputs: @[gap],
    axes: @[1, 2], keepDims: false)

  # --- classifier ----------------------------------------------------------
  let fcW = result.addConst("fc_w", @[3, 8], dtPosit8, p8Bytes(3 * 8, r, 0.4))
  let fcB = result.addConst("fc_b", @[3], dtPosit8, p8Bytes(3, r, 0.5))
  let logits = result.addIntermediate("logits", @[1, 3], dtPosit8)
  discard result.addOp Op(
    name: "fc1", kind: okFullyConnected,
    inputs: @[gap, fcW, fcB], outputs: @[logits],
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

  emitModel(g, p, outDir, "posit_net")
  emitCApi(g, p, outDir, "posit_net")
  echo &"wrote {outDir}/posit_net.nim, posit_net_weights.c, posit_net_weights.h, " &
       &"posit_net_api.nim, posit_net.h"
