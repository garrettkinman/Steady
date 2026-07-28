# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

## Host-side codecs, lookup tables, and softmax.
##
## The end-to-end test proves that the generated code and the simulator agree
## about these. It cannot prove that either agrees with the *mathematics*,
## because the table construction is shared between them — the same gap
## `test_quant.nim` fills for requantization. So:
##
##   * tables are checked entry by entry against the true function, evaluated
##     independently here in double precision
##   * softmax is checked against a float64 reference, where the fixed-point
##     pipeline is allowed one LSB of the output grid and no more
##   * the ops that cannot exist under a given policy are checked to fail on
##     the host, with a message, rather than to compile into something wrong

import std/[unittest, math]
import steady
import steadyc

template pa[T](a: var openArray[T]): ptr UncheckedArray[T] =
  cast[ptr UncheckedArray[T]](addr a[0])

let i8Q = Quant(scales: @[0.05], zeroPoints: @[-14'i32], axis: -1)
let unitQ = Quant(scales: @[1.0 / 128.0], zeroPoints: @[0'i32], axis: -1)

suite "storage codecs":

  test "affine decode is scale * (q - zero), keyed on the storage byte":
    check decodeStore(pkAffineI8, i8Q, 0) == 0.05 * 14.0          # byte 0 -> q 0
    check abs(decodeStore(pkAffineI8, i8Q, 128) - 0.05 * -114.0) < 1e-12
    check abs(decodeStore(pkAffineI8, i8Q, 255) - 0.05 * 13.0) < 1e-12

  test "affine encode round-trips every representable value":
    for raw in 0 ..< LutSize:
      check encodeStore(pkAffineI8, i8Q, decodeStore(pkAffineI8, i8Q, raw)) == raw

  test "affine encode saturates rather than wrapping":
    check encodeStore(pkAffineI8, i8Q, 1e6) == int(cast[uint8](127'i8))
    check encodeStore(pkAffineI8, i8Q, -1e6) == int(cast[uint8](-128'i8))

  test "quantizedValue gives the signed value a kernel would see":
    check quantizedValue(pkAffineI8, i8Q, 0.0) == -14'i32          # the zero point
    check quantizedValue(pkAffineI8, unitQ, 1.0) == 127'i32        # saturating
    check quantizedValue(pkAffineI8, unitQ, -1.0) == -128'i32

  test "fp8 decode and encode agree with the runtime's own conversions":
    for raw in 0 ..< LutSize:
      let f = Fp8(uint8(raw)).toFloat32
      if f != f: continue                                          # the one NaN
      check decodeStore(pkRealFp8, Quant(), raw) == float64(f)
      check encodeStore(pkRealFp8, Quant(), float64(f)) == raw

  test "posit decode and encode agree with the runtime's own conversions":
    for raw in 0 ..< LutSize:
      let p = Posit8(uint8(raw))
      if p.isNaR:
        # NaR has no real value; it decodes to NaN and encodes back from one.
        check decodeStore(pkRealP8, Quant(), raw) != decodeStore(pkRealP8, Quant(), raw)
        check encodeStore(pkRealP8, Quant(), NaN) == raw
      else:
        check decodeStore(pkRealP8, Quant(), raw) == p.toFloat64
        check encodeStore(pkRealP8, Quant(), p.toFloat64) == raw

  test "a posit encode saturates rather than wrapping":
    check encodeStore(pkRealP8, Quant(), 1.0e9) == int(Posit8Max.bits)
    check encodeStore(pkRealP8, Quant(), -1.0e9) == int(Posit8Min.bits)

  test "RealF32 has no byte domain, and says so":
    expect CodecError:
      discard decodeStore(pkRealF32, Quant(), 0)
    expect CodecError:
      discard encodeStore(pkRealF32, Quant(), 1.0)

suite "activation tables":

  test "a logistic table matches the true function at every input":
    let table = buildLut(pkAffineI8, i8Q, unitQ, activationFn(okLogistic))
    check table.len == LutSize
    for raw in 0 ..< LutSize:
      let x = decodeStore(pkAffineI8, i8Q, raw)
      let want = 1.0 / (1.0 + exp(-x))
      check int(table[raw]) == encodeStore(pkAffineI8, unitQ, want)

  test "a tanh table is monotone in the value it encodes and saturates":
    let table = buildLut(pkAffineI8, i8Q, unitQ, activationFn(okTanh))
    # Walk quantized inputs in increasing numeric order, not byte order.
    var prev = -129'i32
    for q in -128 .. 127:
      let raw = int(cast[uint8](int8(q)))
      let v = int32(cast[int8](table[raw]))
      check v >= prev
      prev = v
    check int32(cast[int8](table[int(cast[uint8](-128'i8))])) == -128'i32
    check int32(cast[int8](table[int(cast[uint8](127'i8))])) == 127'i32

  test "an fp8 table propagates NaN instead of inventing a value":
    let table = buildLut(pkRealFp8, Quant(), Quant(), activationFn(okTanh))
    check Fp8(table[int(Fp8Nan.bits)]).isNan
    check Fp8(table[int(toFp8(0.5'f32).bits)]).toFloat32 == toFp8(tanh(0.5'f32)).toFloat32

  test "a posit table propagates NaR instead of inventing a value":
    let table = buildLut(pkRealP8, Quant(), Quant(), activationFn(okTanh))
    check Posit8(table[int(Posit8NaR.bits)]).isNaR
    check Posit8(table[int(toPosit8(0.5).bits)]).toFloat64 == toPosit8(tanh(0.5)).toFloat64
    # The table is the function, rounded once: every entry, not a sample.
    for raw in 0 ..< LutSize:
      let p = Posit8(uint8(raw))
      if p.isNaR: continue
      check Posit8(table[raw]).bits == toPosit8(tanh(p.toFloat64)).bits

  test "RealF32 cannot have a table built for it":
    expect CodecError:
      discard buildLut(pkRealF32, Quant(), Quant(), activationFn(okTanh))

  test "only logistic and tanh are table activations":
    expect CodecError:
      discard activationFn(okConv2d)

suite "softmax exp table":

  test "fractional bits leave room to sum one entry per class":
    for classes in [1, 2, 10, 64, 255, 1000, 4096]:
      let bits = expTableBits(classes)
      check float64(classes) * float64(1 shl bits) <= float64(1 shl 30)

  test "entry zero is exactly the unit, so the divisor is never zero":
    for scale in [0.001, 0.05, 0.5, 4.0]:
      let (table, bits) = buildExpLut(scale, 10)
      check table[0] == int32(1 shl bits)
      check table.len == LutSize
      for d in 1 ..< LutSize:
        check table[d] <= table[d - 1]                             # monotone
        check table[d] >= 0

  test "a non-positive input scale is rejected":
    expect CodecError:
      discard buildExpLut(0.0, 10)

suite "softmax against a float64 reference":

  proc reference(logits: seq[int32], inScale: float64, outQ: Quant): seq[int32] =
    ## Plain double-precision softmax, quantized once at the end. Note the
    ## input zero point does not appear: softmax is shift-invariant, and the
    ## kernel's max subtraction is what makes that true of the fixed-point
    ## version too.
    var mx = logits[0]
    for v in logits:
      if v > mx: mx = v
    var e = newSeq[float64](logits.len)
    var total = 0.0
    for i, v in logits:
      e[i] = exp(inScale * float64(v - mx))
      total += e[i]
    result = newSeq[int32](logits.len)
    for i in 0 ..< logits.len:
      result[i] = quantizedValue(pkAffineI8, outQ, e[i] / total)

  proc kernel(logits: seq[int32], inScale: float64, outQ: Quant): seq[int32] =
    let n = logits.len
    var (table, _) = buildExpLut(inScale, n)
    let g = initGraph("t", pkAffineI8)
    let outT = Tensor(name: "probs", shape: @[1, n], dtype: dtInt8,
                      kind: tkOutput, quant: outQ)
    let rp = resolveSoftmaxParams(g, outT)
    var x = newSeq[int8](n)
    var y = newSeq[int8](n)
    for i, v in logits: x[i] = int8(v)
    softmax(AffineI8, pa(y), pa(x), 1, n, pa(table),
            rp.mult[0], rp.shift[0], rp.outZeroPoint, rp.qActMin, rp.qActMax)
    result = newSeq[int32](n)
    for i in 0 ..< n: result[i] = int32(y[i])

  # TFLite pins int8 softmax output to this quantization.
  let probQ = Quant(scales: @[1.0 / 256.0], zeroPoints: @[-128'i32], axis: -1)

  test "within one LSB of the output grid, over random logits":
    var s = 20260726'u32
    for trial in 0 ..< 200:
      let n = 2 + trial mod 12
      var logits = newSeq[int32](n)
      for i in 0 ..< n:
        s = s * 1664525'u32 + 1013904223'u32
        logits[i] = int32((s shr 16) and 0xFF'u32) - 128'i32
      let inScale = 0.02 + 0.01 * float64(trial mod 7)
      let got = kernel(logits, inScale, probQ)
      let want = reference(logits, inScale, probQ)
      for i in 0 ..< n:
        check abs(got[i] - want[i]) <= 1

  test "exactly uniform inputs give an exactly uniform distribution":
    for n in [2, 4, 8, 16]:
      var logits = newSeq[int32](n)
      for i in 0 ..< n: logits[i] = 7'i32
      let got = kernel(logits, 0.05, probQ)
      for i in 1 ..< n:
        check got[i] == got[0]
      check abs(float64(got[0] + 128) / 256.0 - 1.0 / float64(n)) <= 1.0 / 256.0

  test "the dominant class wins and the tail is not negative":
    let got = kernel(@[127'i32, -128'i32, 0'i32], 0.5, probQ)
    check got[0] == 127'i32                     # saturates at probability 1
    check got[1] == -128'i32                    # underflows to probability 0
    for q in got: check q >= -128'i32

  test "a shifted input is the same distribution":
    # Adding a constant to every logit changes nothing, which is the property
    # the max subtraction exists to preserve.
    let a = kernel(@[10'i32, -4'i32, 33'i32, 2'i32], 0.05, probQ)
    let b = kernel(@[40'i32, 26'i32, 63'i32, 32'i32], 0.05, probQ)
    check a == b

suite "ops the host must reject":

  proc oneOpGraph(policy: PolicyKind, kind: OpKind): Graph =
    ## Input -> op -> output, shapes chosen so only the policy is at issue.
    let dt = policy.storeType
    let q =
      if policy.isAffine: Quant(scales: @[0.05], zeroPoints: @[0'i32], axis: -1)
      else: Quant(axis: -1)
    result = initGraph("reject", policy)
    let x = result.addInput("input", @[1, 4], dt, q)
    let y = result.addIntermediate("out", @[1, 4], dt, q)
    discard result.addOp Op(name: "op", kind: kind, inputs: @[x], outputs: @[y])
    result.markOutput y

  test "a table activation needs an 8-bit storage type":
    for kind in [okLogistic, okTanh]:
      var ok8 = oneOpGraph(pkRealFp8, kind)
      ok8.validate                                  # fp8 is fine
      var okP = oneOpGraph(pkRealP8, kind)
      okP.validate                                  # and so is posit8
      var bad = oneOpGraph(pkRealF32, kind)
      expect IrError:
        bad.validate

  test "softmax needs a uniform integer domain, so affine only":
    var good = oneOpGraph(pkAffineI8, okSoftmax)
    good.validate
    for policy in [pkRealF32, pkRealFp8, pkRealP8]:
      var bad = oneOpGraph(policy, okSoftmax)
      expect IrError:
        bad.validate

  test "softmax normalises over the last axis, once per row":
    # A classifier has one row; a detection head has one per output cell, which
    # is what FOMO's [1, 12, 12, 3] output is. Both are legal, and the class
    # count — not the element count — is what the exp table has to cover.
    proc softmaxGraph(inShape, outShape: seq[int]): Graph =
      let q = Quant(scales: @[0.05], zeroPoints: @[0'i32], axis: -1)
      result = initGraph("m", pkAffineI8)
      let x = result.addInput("input", inShape, dtInt8, q)
      let y = result.addIntermediate("out", outShape, dtInt8, q)
      discard result.addOp Op(name: "sm", kind: okSoftmax, inputs: @[x],
                              outputs: @[y])
      result.markOutput y

    var vector = softmaxGraph(@[1, 10], @[1, 10])
    vector.validate
    var rows = softmaxGraph(@[3, 4], @[3, 4])
    rows.validate
    var grid = softmaxGraph(@[1, 12, 12, 3], @[1, 12, 12, 3])
    grid.validate

    var reshaped = softmaxGraph(@[1, 12, 12, 3], @[1, 144, 3])
    expect IrError:
      reshaped.validate                        # must preserve shape, not just size

    var tooMany = softmaxGraph(@[1, MaxSoftmaxClasses + 1],
                               @[1, MaxSoftmaxClasses + 1])
    expect IrError:
      tooMany.validate

  test "pad must not change the quantization, and must be spatial":
    let q = Quant(scales: @[0.05], zeroPoints: @[-3'i32], axis: -1)
    let other = Quant(scales: @[0.07], zeroPoints: @[-3'i32], axis: -1)

    proc padGraph(pads: seq[array[2, int]], outQ: Quant,
                  outShape: seq[int]): Graph =
      result = initGraph("p", pkAffineI8)
      let x = result.addInput("input", @[1, 2, 2, 1], dtInt8, q)
      let y = result.addIntermediate("out", outShape, dtInt8, outQ)
      discard result.addOp Op(name: "pad", kind: okPad, inputs: @[x],
                             outputs: @[y], pads: pads)
      result.markOutput y

    var good = padGraph(@[[0, 0], [1, 1], [1, 1], [0, 0]], q, @[1, 4, 4, 1])
    good.validate

    var requant = padGraph(@[[0, 0], [1, 1], [1, 1], [0, 0]], other, @[1, 4, 4, 1])
    expect IrError:
      requant.validate

    var channels = padGraph(@[[0, 0], [0, 0], [0, 0], [1, 1]], q, @[1, 2, 2, 3])
    expect IrError:
      channels.validate

    var wrongShape = padGraph(@[[0, 0], [1, 1], [1, 1], [0, 0]], q, @[1, 3, 4, 1])
    expect IrError:
      wrongShape.validate

  test "concatenation operands must agree on every other dimension":
    proc catGraph(bShape: seq[int], outShape: seq[int], axis: int): Graph =
      let q = Quant(scales: @[0.05], zeroPoints: @[0'i32], axis: -1)
      result = initGraph("c", pkAffineI8)
      let a = result.addInput("a", @[1, 2, 2, 3], dtInt8, q)
      let b = result.addInput("b", bShape, dtInt8, q)
      let y = result.addIntermediate("out", outShape, dtInt8, q)
      discard result.addOp Op(name: "cat", kind: okConcatenation,
                             inputs: @[a, b], outputs: @[y], axis: axis)
      result.markOutput y

    var good = catGraph(@[1, 2, 2, 5], @[1, 2, 2, 8], 3)
    good.validate

    var negativeAxis = catGraph(@[1, 2, 2, 5], @[1, 2, 2, 8], -1)
    negativeAxis.validate                            # normalized to 3
    check negativeAxis.ops[0].axis == 3

    var mismatched = catGraph(@[1, 3, 2, 5], @[1, 2, 2, 8], 3)
    expect IrError:
      mismatched.validate

    var wrongTotal = catGraph(@[1, 2, 2, 5], @[1, 2, 2, 7], 3)
    expect IrError:
      wrongTotal.validate

  test "mean only reduces the spatial axes, in either order":
    proc meanGraph(axes: seq[int], keepDims: bool, outShape: seq[int]): Graph =
      let q = Quant(scales: @[0.05], zeroPoints: @[0'i32], axis: -1)
      result = initGraph("m", pkAffineI8)
      let x = result.addInput("input", @[1, 4, 4, 3], dtInt8, q)
      let y = result.addIntermediate("out", outShape, dtInt8, q)
      discard result.addOp Op(name: "mean", kind: okMean, inputs: @[x],
                             outputs: @[y], axes: axes, keepDims: keepDims)
      result.markOutput y

    var flat = meanGraph(@[1, 2], false, @[1, 3])
    flat.validate
    var kept = meanGraph(@[2, 1], true, @[1, 1, 1, 3])
    kept.validate                                    # sorted for you
    var negative = meanGraph(@[-3, -2], false, @[1, 3])
    negative.validate

    var channelReduce = meanGraph(@[3], false, @[1, 4, 4])
    expect IrError:
      channelReduce.validate

    var wrongShape = meanGraph(@[1, 2], true, @[1, 3])
    expect IrError:
      wrongShape.validate
