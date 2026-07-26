# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

## Fixed-point requantization, pinned to TFLite/gemmlowp semantics.
##
## The end-to-end test shares this arithmetic between the runtime and the
## simulator, so it cannot catch an error here — that is what these vectors
## are for. Every value below is derived from the published gemmlowp
## definitions rather than from this implementation.

import std/[unittest, math]
import steady/policy
import steadyc/[ir, quant]

suite "SaturatingRoundingDoublingHighMul":

  test "multiplying by the Q31 representation of 0.5 halves the input":
    const half = 1073741824'i32          # 2^30
    check satRoundDoublingHighMul(100'i32, half) == 50'i32
    check satRoundDoublingHighMul(24'i32, half) == 12'i32
    check satRoundDoublingHighMul(-100'i32, half) == -50'i32

  test "rounds to nearest, ties toward positive infinity":
    # Note this is NOT the same tie rule as RoundingDivideByPOT below, which
    # ties away from zero. gemmlowp's asymmetric nudge (1 - 2^30 on the
    # negative side) combined with C's truncating division produces half-up.
    # Matching TFLite means reproducing the asymmetry, not "fixing" it.
    const half = 1073741824'i32
    check satRoundDoublingHighMul(1'i32, half) == 1'i32      # 0.5 -> 1
    check satRoundDoublingHighMul(-1'i32, half) == 0'i32     # -0.5 -> 0
    check satRoundDoublingHighMul(3'i32, half) == 2'i32      # 1.5 -> 2
    check satRoundDoublingHighMul(-3'i32, half) == -1'i32    # -1.5 -> -1

  test "the two rounding helpers genuinely differ on negative ties":
    const half = 1073741824'i32
    check satRoundDoublingHighMul(-3'i32, half) == -1'i32    # half-up
    check roundingDivideByPOT(-3'i32, 1'i32) == -2'i32       # half-away-from-zero

  test "identity multiplier is very nearly the identity":
    for x in [0'i32, 1, -1, 12345, -12345, 1000000, -1000000]:
      check abs(satRoundDoublingHighMul(x, high(int32)) - x) <= 1

  test "saturates on the single overflow case":
    check satRoundDoublingHighMul(low(int32), low(int32)) == high(int32)

  test "zero annihilates":
    check satRoundDoublingHighMul(0'i32, high(int32)) == 0'i32
    check satRoundDoublingHighMul(high(int32), 0'i32) == 0'i32

suite "RoundingDivideByPOT":

  test "rounds half away from zero":
    check roundingDivideByPOT(7'i32, 2'i32) == 2'i32     # 1.75
    check roundingDivideByPOT(6'i32, 2'i32) == 2'i32     # 1.5 ties up
    check roundingDivideByPOT(5'i32, 2'i32) == 1'i32     # 1.25
    check roundingDivideByPOT(-7'i32, 2'i32) == -2'i32
    check roundingDivideByPOT(-6'i32, 2'i32) == -2'i32   # -1.5 ties down
    check roundingDivideByPOT(-5'i32, 2'i32) == -1'i32

  test "exponent zero is the identity":
    for x in [0'i32, 1, -1, 12345, -12345, high(int32), low(int32)]:
      check roundingDivideByPOT(x, 0'i32) == x

  test "is not a bare arithmetic shift":
    # An arithmetic shift floors; this rounds. The two agree on exact
    # values and on ties that happen to floor the same way, so the
    # distinction has to be probed where rounding actually moves the result.
    check roundingDivideByPOT(7'i32, 2'i32) == 2'i32
    check (7'i32 shr 2'i32) == 1'i32
    check roundingDivideByPOT(-5'i32, 2'i32) == -1'i32
    check (-5'i32 shr 2'i32) == -2'i32

suite "QuantizeMultiplier":

  test "known encodings":
    check quantizeMultiplier(0.5) == (1073741824'i32, 0'i32)
    check quantizeMultiplier(1.0) == (1073741824'i32, 1'i32)
    check quantizeMultiplier(0.25) == (1073741824'i32, -1'i32)
    check quantizeMultiplier(0.0) == (0'i32, 0'i32)

  test "normalised mantissa always lands in [2^30, 2^31)":
    var m = 1e-8
    while m < 1e6:
      let (q, _) = quantizeMultiplier(m)
      check q >= 1073741824'i32
      check q <= high(int32)
      m *= 1.7

  test "underflow and saturation are clamped, not wrapped":
    check quantizeMultiplier(1e-30)[0] == 0'i32
    let (qBig, sBig) = quantizeMultiplier(1e12)
    check sBig == 30'i32
    check qBig == high(int32)

  test "rejects a negative effective scale":
    expect QuantError:
      discard quantizeMultiplier(-0.5)

suite "MultiplyByQuantizedMultiplier":

  test "approximates the real multiply to within one LSB":
    for scale in [0.001, 0.0125, 0.5, 0.999, 1.0, 3.7, 250.0]:
      let (m, s) = quantizeMultiplier(scale)
      for x in [0'i32, 1, -1, 37, -37, 1000, -1000, 32767, -32768]:
        let got = multiplyByQuantizedMultiplier(x, m, s)
        let want = round(float64(x) * scale)
        check abs(float64(got) - want) <= 1.0

  test "a scale of exactly one is exactly the identity":
    let (m, s) = quantizeMultiplier(1.0)
    for x in [0'i32, 1, -1, 12345, -12345, 100000, -100000]:
      check multiplyByQuantizedMultiplier(x, m, s) == x

suite "activation ranges":

  test "relu clamps at the output zero point, not at zero":
    # This is the classic quantized-ReLU mistake: the representable value of
    # real 0.0 is the zero point, not the integer 0.
    let q = Quant(scales: @[0.05], zeroPoints: @[-14'i32], axis: -1)
    let (lo, hi) = affineActRange(faRelu, q)
    check lo == -14'i32
    check hi == 127'i32

  test "relu6 clamps the top at the quantized value of 6":
    let q = Quant(scales: @[0.05], zeroPoints: @[-128'i32], axis: -1)
    let (lo, hi) = affineActRange(faRelu6, q)
    check lo == -128'i32
    check hi == int32(round(6.0 / 0.05)) - 128'i32

  test "no fused activation spans the full int8 range":
    let q = Quant(scales: @[0.05], zeroPoints: @[3'i32], axis: -1)
    check affineActRange(faNone, q) == (-128'i32, 127'i32)

  test "real policies get plain bounds with no quantization involved":
    check realActRange(faRelu) == (0'f32, Inf.float32)
    check realActRange(faRelu6) == (0'f32, 6'f32)
    check realActRange(faReluN1To1) == (-1'f32, 1'f32)
