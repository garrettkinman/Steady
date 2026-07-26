# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

## Backend override.
##
## The fixture in tests/fixtures/steady_backend.nim implements exactly one
## thing: `fullyConnected` for AffineI8. Everything else must fall through
## to the reference kernels with no forking and no stubs. Build this file
## with `-d:steadyBackend --path:tests/fixtures`.

import std/unittest
import steady
when defined(steadyBackend):
  import steady_backend as be

template pa[T](a: var openArray[T]): ptr UncheckedArray[T] =
  cast[ptr UncheckedArray[T]](addr a[0])

var mult = [1073741824'i32]     # 0.5 in Q31
var shift = [0'i32]

proc affineParams(): AffineParams =
  AffineParams(mult: cast[ptr UncheckedArray[int32]](addr mult[0]),
               shift: cast[ptr UncheckedArray[int32]](addr shift[0]),
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

    test "ops the backend does not implement fall back":
      let before = be.beCalls
      var v = [-5'i8, 0, 7]
      clamp1d(AffineI8, pa(v), pa(v), 3, 0'i8, 6'i8)
      check be.beCalls == before
      check v == [0'i8, 0, 6]
