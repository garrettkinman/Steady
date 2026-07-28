# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

## Backend override.
##
## Two fixtures, at the two seams, each deliberately partial.
##
## tests/fixtures/steady_backend.nim implements exactly one thing:
## `fullyConnected` for AffineI8. Everything else must fall through to the
## reference kernels with no forking and no stubs. Build with
## `-d:steadyBackend --path:tests/fixtures`.
##
## tests/fixtures/steady_arith.nim replaces policy *members* instead — `mac`
## and the block widths, for RealP8 only — which is the seam a hardware
## arithmetic unit plugs into. Build with `-d:steadyArith --path:tests/fixtures`.
## Its arithmetic is identical to the default on purpose: what is under test
## is that substituting it changes nothing observable.

import std/[unittest, math]
import steady
when defined(steadyBackend):
  import steady_backend as be
when defined(steadyArith):
  import steady_arith as ar

template pa[T](a: var openArray[T]): ptr UncheckedArray[T] =
  cast[ptr UncheckedArray[T]](addr a[0])

var mult = [1073741824'i32]     # 0.5 in Q31
var shift = [0'i32]

proc affineParams(): AffineParams =
  AffineParams(mult: cast[ptr UncheckedArray[int32]](addr mult[0]),
               shift: cast[ptr UncheckedArray[int32]](addr shift[0]),
               channelStride: 0, outZeroPoint: 0,
               actMin: -128'i32, actMax: 127'i32)

var one = [1073741824'i32]      # 1.0 in Q31 is 2^30 with a shift of 1
var oneShift = [1'i32]

var quarter = [1073741824'i32]  # 0.25: the same mantissa, two shifts down
var quarterShift = [-1'i32]

proc affineIdentity(): AffineParams =
  ## The requantization that changes nothing: input and output scales equal,
  ## zero point zero. Exactly the identity, per tests/test_quant.nim.
  AffineParams(mult: cast[ptr UncheckedArray[int32]](addr one[0]),
               shift: cast[ptr UncheckedArray[int32]](addr oneShift[0]),
               channelStride: 0, outZeroPoint: 0,
               actMin: -128'i32, actMax: 127'i32)

proc affineQuarter(): AffineParams =
  ## Scale by 1/4. `meanSpatial` needs this: the host folds a mean's `1/count`
  ## into the output multiplier rather than dividing in the kernel, so a
  ## four-tap mean is an identity requantization times a quarter.
  AffineParams(mult: cast[ptr UncheckedArray[int32]](addr quarter[0]),
               shift: cast[ptr UncheckedArray[int32]](addr quarterShift[0]),
               channelStride: 0, outZeroPoint: 0,
               actMin: -128'i32, actMax: 127'i32)

suite "kernel correctness (reference path)":

  test "fullyConnected computes W*x + b then requantizes":
    var y = [0'i8, 0]
    var x = [4'i8, 2]
    var w = [6'i8, 3, 1, 1]       # rows: [6,3], [1,1]
    var b = [0'i32, 0]
    fullyConnected(AffineI8, pa(y), pa(x), pa(w), pa(b), affineParams(), 2, 2)
    check y[0] == 15'i8           # (6*4 + 3*2) * 0.5 = 15
    check y[1] == 3'i8            # (1*4 + 1*2) * 0.5 = 3

  test "clamp1d is safe in place":
    var v = [-5'i8, 0, 7, 100]
    clamp1d(AffineI8, pa(v), pa(v), 4, 0'i8, 6'i8)
    check v == [0'i8, 0, 6, 6]

  test "maxPool2d picks the window maximum":
    var y = [0'i8]
    var x = [1'i8, 9, 3, 4]       # 2x2x1
    maxPool2d(AffineI8, pa(y), pa(x), 2, 2, 1, 1, 1, 2, 2, 1, 1, 0, 0,
              -128'i8, 127'i8)
    check y[0] == 9'i8

  test "avgPool2d divides by the valid tap count and rounds away from zero":
    var y = [0'i8]
    var x = [1'i8, 2, 3, 5]       # mean 2.75 -> 3
    avgPool2d(AffineI8, pa(y), pa(x), 2, 2, 1, 1, 1, 2, 2, 1, 1, 0, 0,
              -128'i8, 127'i8)
    check y[0] == 3'i8

  test "conv2d with padding multiplies padded taps by the pad value":
    # 1x1 input, 3x3 SAME kernel: eight taps are padded. With padValue 0 the
    # only contribution is the centre tap.
    var y = [0'i8]
    var x = [10'i8]
    var w: array[9, int8]
    for i in 0 ..< 9: w[i] = 1
    var b = [0'i32]
    conv2d(AffineI8, pa(y), pa(x), pa(w), pa(b), affineParams(),
           1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 0'i8)
    check y[0] == 5'i8            # 10 * 0.5

  test "conv2d pad value actually participates":
    var y = [0'i8]
    var x = [10'i8]
    var w: array[9, int8]
    for i in 0 ..< 9: w[i] = 1
    var b = [0'i32]
    conv2d(AffineI8, pa(y), pa(x), pa(w), pa(b), affineParams(),
           1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 2'i8)
    check y[0] == 13'i8           # (10 + 8*2) * 0.5 = 13

suite "data movement and table kernels":

  test "pad2d fills the border with the pad value and nothing else":
    var y: array[4 * 4 * 1, int8]
    var x = [1'i8, 2, 3, 4]                      # 2x2x1
    pad2d(AffineI8, pa(y), pa(x), 2, 2, 1, 1, 1, 1, 1, -7'i8)
    check y == [-7'i8, -7, -7, -7,
                -7, 1, 2, -7,
                -7, 3, 4, -7,
                -7, -7, -7, -7]

  test "pad2d handles asymmetric padding and multiple channels":
    var y: array[2 * 3 * 2, int8]
    var x = [1'i8, 2]                            # 1x1x2
    pad2d(AffineI8, pa(y), pa(x), 1, 1, 2, 0, 1, 1, 1, 0'i8)
    check y == [0'i8, 0, 1, 2, 0, 0,
                0, 0, 0, 0, 0, 0]

  test "concatSlice interleaves two operands along the channel axis":
    # Two [1,2,2,1] tensors into [1,2,2,2]: outer = 4 pixels, inner 1 -> 2.
    var y: array[8, int8]
    var a = [1'i8, 2, 3, 4]
    var b = [5'i8, 6, 7, 8]
    concatSlice(AffineI8, pa(y), pa(a), 4, 1, 2, 0)
    concatSlice(AffineI8, pa(y), pa(b), 4, 1, 2, 1)
    check y == [1'i8, 5, 2, 6, 3, 7, 4, 8]

  test "concatSlice on the outermost axis is a contiguous append":
    var y: array[6, int8]
    var a = [1'i8, 2, 3, 4]
    var b = [5'i8, 6]
    concatSlice(AffineI8, pa(y), pa(a), 1, 4, 6, 0)
    concatSlice(AffineI8, pa(y), pa(b), 1, 2, 6, 4)
    check y == [1'i8, 2, 3, 4, 5, 6]

  test "concatSliceRescaled maps an operand into the output domain":
    # The operand stage subtracts the zero point and lands the value in the
    # common domain 2^20 times finer; the shared stage takes it out again with
    # a factor of 0.5. Net effect (v - 4) * 0.5, chosen exact so that the two
    # rounding rules cannot muddy what is being checked.
    var y: array[3, int8]
    var a = [8'i8, 12, 4]
    var mult2 = [1073741824'i32]                 # 0.5 in Q31
    var shift2 = [-20'i32]                       # 0.5 * 2^-20, undoing the shift
    let prm = AffineParams(
      mult: cast[ptr UncheckedArray[int32]](addr mult2[0]),
      shift: cast[ptr UncheckedArray[int32]](addr shift2[0]),
      channelStride: 0, outZeroPoint: 0, actMin: -128'i32, actMax: 127'i32)
    concatSliceRescaled(AffineI8, pa(y), pa(a), prm, 1, 3, 3, 0,
                        1073741824'i32, 1'i32, -4'i32)
    check y == [2'i8, 4, 0]

  test "meanSpatial averages each channel independently":
    var y = [0'i8, 0]
    var x = [10'i8, 20, 12, 22, 14, 24, 16, 26]  # 2x2x2, means 13 and 23
    meanSpatial(AffineI8, pa(y), pa(x), 0'i32, affineQuarter(), 2, 2, 2)
    check y == [13'i8, 23]

  test "meanSpatial applies the zero-point correction in accumulator units":
    # Four taps at 10 with Zx = 3: the mean of (x - Zx) is 7.
    var y = [0'i8]
    var x = [10'i8, 10, 10, 10]
    meanSpatial(AffineI8, pa(y), pa(x), -12'i32, affineQuarter(), 2, 2, 1)
    check y == [7'i8]

  test "meanSpatial rounds once, at the requantization":
    # A sum of 50 over 4 taps is 12.5, which the folded multiplier resolves in
    # one step: half away from zero, per RoundingDivideByPOT. Dividing in the
    # kernel first would round the accumulator to 13 and then requantize it,
    # which is where the LSB MobileNetV2's global average pool exposed went.
    var y = [0'i8]
    var x = [20'i8, 20, 5, 5]                    # sum 50
    meanSpatial(AffineI8, pa(y), pa(x), 0'i32, affineQuarter(), 2, 2, 1)
    check y == [13'i8]                           # 12.5 -> 13, not 12

  test "meanSpatial under a real policy divides in the kernel":
    # No multiplier to fold into, so the same kernel source must divide here.
    var y = [0'f32]
    var x = [1'f32, 2, 3, 10]                    # mean 4.0
    meanSpatial(RealF32, pa(y), pa(x), 0'f32, RealF32.noClamp, 2, 2, 1)
    check y[0] == 4'f32

  test "softmax normalises each row independently":
    # A detection head like FOMO emits a grid of per-cell distributions, so one
    # call covers many rows. Two rows here: the second is the first shifted by a
    # constant, which softmax must map to the same distribution.
    var expLut: array[256, int32]
    for d in 0 ..< 256:
      expLut[d] = int32(float64(1 shl 26) * exp(-float64(d) * 0.1))
    var y: array[6, int8]
    var x = [10'i8, 12, 5, 40, 42, 35]
    softmax(AffineI8, pa(y), pa(x), 2, 3, pa(expLut),
            1073741824'i32, -6'i32, -128'i32, -128'i32, 127'i32)
    check y[0] == y[3]
    check y[1] == y[4]
    check y[2] == y[5]
    var total = 0.0
    for i in 0 ..< 3: total += float64(int(y[i]) + 128) / 256.0
    check abs(total - 1.0) < 0.02

  test "lut1d is keyed on the storage byte, two's complement":
    var table: array[256, int8]
    for i in 0 ..< 256: table[i] = int8(i - 128)  # byte b -> b - 128
    var y: array[4, int8]
    var x = [0'i8, 1, 127, -128]
    lut1d(AffineI8, pa(y), pa(x), 4, pa(table))
    check y == [-128'i8, -127, -1, 0]

  test "lut1d is safe in place":
    var table: array[256, int8]
    for i in 0 ..< 256: table[i] = 42'i8
    var v = [1'i8, 2, 3]
    lut1d(AffineI8, pa(v), pa(v), 3, pa(table))
    check v == [42'i8, 42, 42]

  test "lut1d works for fp8 too, which is the point of the canary":
    # A table keyed on fp8 bits: the same kernel, the same host plumbing, a
    # storage type with no scale metadata and no uniform spacing.
    var table: array[256, Fp8]
    for i in 0 ..< 256:
      table[i] = toFp8(Fp8(uint8(i)).toFloat32 * 2'f32)
    var y = [Fp8Zero, Fp8Zero, Fp8Zero]
    var x = [toFp8(1'f32), toFp8(-3'f32), toFp8(0.5'f32)]
    lut1d(RealFp8, pa(y), pa(x), 3, pa(table))
    check y[0].toFloat32 == 2'f32
    check y[1].toFloat32 == -6'f32
    check y[2].toFloat32 == 1'f32

suite "policy independence":

  test "the same kernel runs under all three policies":
    var yf = [0'f32, 0]
    var xf = [4'f32, 2]
    var wf = [6'f32, 3, 1, 1]
    var bf = [0'f32, 0]
    fullyConnected(RealF32, pa(yf), pa(xf), pa(wf), pa(bf), RealF32.noClamp, 2, 2)
    check yf[0] == 30'f32
    check yf[1] == 6'f32

    var y8 = [Fp8Zero, Fp8Zero]
    var x8 = [4'f32.toFp8, 2'f32.toFp8]
    var w8 = [6'f32.toFp8, 3'f32.toFp8, 1'f32.toFp8, 1'f32.toFp8]
    var b8 = [Fp8Zero, Fp8Zero]
    fullyConnected(RealFp8, pa(y8), pa(x8), pa(w8), pa(b8), RealFp8.noClamp, 2, 2)
    check y8[0].toFloat32 == 30'f32
    check y8[1].toFloat32 == 6'f32

  test "fp8 rounds once at the end, not at every accumulation step":
    # Sum of eight 0.1s. 0.1 is not representable in e4m3; accumulating in
    # fp8 would compound the error, accumulating in float32 does not.
    var y = [Fp8Zero]
    var x: array[8, Fp8]
    var w: array[8, Fp8]
    for i in 0 ..< 8:
      x[i] = 0.1'f32.toFp8
      w[i] = 1'f32.toFp8
    var b = [Fp8Zero]
    fullyConnected(RealFp8, pa(y), pa(x), pa(w), pa(b), RealFp8.noClamp, 1, 8)
    let exact = 8'f32 * 0.1'f32.toFp8.toFloat32
    check y[0].bits == exact.toFp8.bits

  test "relu clamp works under fp8 ordering":
    var v = [(-3.0'f32).toFp8, 0.5'f32.toFp8, 9.0'f32.toFp8]
    clamp1d(RealFp8, pa(v), pa(v), 3, 0'f32.toFp8, 6'f32.toFp8)
    check v[0].toFloat32 == 0'f32
    check v[1].toFloat32 == 0.5'f32
    check v[2].toFloat32 == 6'f32

when defined(steadyBackend):
  suite "backend override":

    test "an int8 fullyConnected goes to the backend":
      let before = be.beCalls
      var y = [0'i8, 0]
      var x = [4'i8, 2]
      var w = [6'i8, 3, 1, 1]
      var b = [0'i32, 0]
      fullyConnected(AffineI8, pa(y), pa(x), pa(w), pa(b), affineParams(), 2, 2)
      check be.beCalls == before + 1
      check y[0] == 15'i8          # and it still produces the right answer

    test "the same op under a different policy falls back":
      let before = be.beCalls
      var y = [0'f32, 0]
      var x = [4'f32, 2]
      var w = [6'f32, 3, 1, 1]
      var b = [0'f32, 0]
      fullyConnected(RealF32, pa(y), pa(x), pa(w), pa(b), RealF32.noClamp, 2, 2)
      check be.beCalls == before
      check y[0] == 30'f32

    test "a policy the backend has never heard of falls back":
      # The override is resolved per policy, so a policy added after the
      # backend was written needs nothing from it and gets nothing.
      let before = be.beCalls
      var y = [Posit8Zero, Posit8Zero]
      var x = [toPosit8(0.5), toPosit8(0.25)]
      var w = [toPosit8(1.5), toPosit8(0.5), toPosit8(1.0), toPosit8(1.0)]
      var b = [Posit8Zero, Posit8Zero]
      fullyConnected(RealP8, pa(y), pa(x), pa(w), pa(b), RealP8.noClamp, 2, 2)
      check be.beCalls == before
      check y[0].toFloat64 == 0.875

    test "ops the backend does not implement fall back":
      let before = be.beCalls
      var v = [-5'i8, 0, 7]
      clamp1d(AffineI8, pa(v), pa(v), 3, 0'i8, 6'i8)
      check be.beCalls == before
      check v == [0'i8, 0, 6]

when defined(steadyArith):
  suite "arithmetic override":

    test "mac for RealP8 goes to the arithmetic backend":
      let before = ar.arithCalls
      var y = [Posit8Zero, Posit8Zero]
      var x = [toPosit8(0.5), toPosit8(0.25)]
      var w = [toPosit8(1.5), toPosit8(0.5), toPosit8(1.0), toPosit8(1.0)]
      var b = [Posit8Zero, Posit8Zero]
      fullyConnected(RealP8, pa(y), pa(x), pa(w), pa(b), RealP8.noClamp, 2, 2)
      check ar.arithCalls == before + 4        # one per tap, not one per op
      check y[0].toFloat64 == 0.875            # and the answer is unchanged

    test "one overridden member does not drag in the others":
      # `finish` was not overridden, so the clamp and the rounding are still
      # the defaults even on the op whose `mac` was replaced.
      let prm = RealParams[Quire](actMin: 0'i64, actMax: 6 * QuireOne)
      check finish(RealP8, Quire(9 * QuireOne), prm, 0).bits == toPosit8(6.0).bits

    test "a policy the arithmetic backend does not cover falls through":
      let before = ar.arithCalls
      var y = [0'i8, 0]
      var x = [4'i8, 2]
      var w = [6'i8, 3, 1, 1]
      var b = [0'i32, 0]
      fullyConnected(AffineI8, pa(y), pa(x), pa(w), pa(b), affineParams(), 2, 2)
      check ar.arithCalls == before
      check y[0] == 15'i8

    test "kernels that do not multiply never reach it":
      let before = ar.arithCalls
      var v = [toPosit8(-3.0), toPosit8(0.5), toPosit8(9.0)]
      clamp1d(RealP8, pa(v), pa(v), 3, toPosit8(0.0), toPosit8(6.0))
      check ar.arithCalls == before
      check v[0].toFloat64 == 0.0

    test "the block width is 1 here, and the results are the same anyway":
      # The fixture sets every block width to 1 for RealP8, so these kernels
      # take their unblocked path. Each accumulator is supposed to see the
      # same taps in the same order whatever the blocking; this is a nine-row
      # matmul, wide enough that a width of 4 would have blocked twice and
      # left a remainder.
      var y: array[9, Posit8]
      var x: array[4, Posit8]
      var w: array[36, Posit8]
      var b: array[9, Posit8]
      for i in 0 ..< 4: x[i] = toPosit8(0.25 + 0.125 * float64(i))
      for i in 0 ..< 36: w[i] = toPosit8(0.5 - 0.03125 * float64(i mod 8))
      for i in 0 ..< 9: b[i] = toPosit8(0.125)
      fullyConnected(RealP8, pa(y), pa(x), pa(w), pa(b), RealP8.noClamp, 9, 4)
      # Computed against the exact real-valued sum, rounded once — the same
      # oracle the end-to-end posit test uses.
      for o in 0 ..< 9:
        var acc = 0.0
        for i in 0 ..< 4:
          acc += w[o * 4 + i].toFloat64 * x[i].toFloat64
        check y[o].bits == toPosit8(acc + b[o].toFloat64).bits
