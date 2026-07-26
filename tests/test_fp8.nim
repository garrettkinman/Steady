# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import std/[unittest, math]
import steady/fp8

suite "fp8 e4m3 decode":

  test "special values":
    check Fp8(0x00'u8).toFloat32 == 0.0'f32
    check Fp8(0x80'u8).toFloat32 == 0.0'f32
    check signbit(Fp8(0x80'u8).toFloat32)
    check Fp8(0x38'u8).toFloat32 == 1.0'f32          # e4=7, m3=0
    check Fp8(0xB8'u8).toFloat32 == -1.0'f32
    check Fp8(0x7E'u8).toFloat32 == 448.0'f32        # max finite
    check Fp8(0xFE'u8).toFloat32 == -448.0'f32
    check Fp8(0x7F'u8).isNan
    check Fp8(0xFF'u8).isNan

  test "subnormals":
    # step is 2^-9; encodings 0x01..0x07 are k * 2^-9
    for k in 1 .. 7:
      check Fp8(uint8(k)).toFloat32 == float32(k) * pow(2.0'f32, -9.0'f32)
    # smallest normal is 2^-6, immediately above the largest subnormal
    check Fp8(0x08'u8).toFloat32 == pow(2.0'f32, -6.0'f32)

  test "decode is total and finite except for the two NaNs":
    for b in 0 .. 255:
      let v = Fp8(uint8(b))
      if v.isNan:
        check isNaN(v.toFloat32)
      else:
        check classify(v.toFloat32) in {fcNormal, fcSubnormal, fcZero, fcNegZero}

suite "fp8 e4m3 encode":

  test "round-trips every finite encoding exactly":
    for b in 0 .. 255:
      let v = Fp8(uint8(b))
      if v.isNan: continue
      let back = v.toFloat32.toFp8
      check back.bits == v.bits

  test "saturates rather than overflowing to NaN":
    check 1.0e30'f32.toFp8.bits == Fp8Max.bits
    check (-1.0e30'f32).toFp8.bits == Fp8Min.bits
    check 449.0'f32.toFp8.bits == Fp8Max.bits
    check Inf.float32.toFp8.bits == Fp8Max.bits
    check NegInf.float32.toFp8.bits == Fp8Min.bits
    check NaN.float32.toFp8.isNan

  test "ties round to even":
    # 1.0 and the next fp8 up (1.125) — midpoint 1.0625 ties to even (1.0).
    check 1.0625'f32.toFp8.bits == Fp8(0x38'u8).bits      # m3=0, even
    # midpoint between 1.125 (m3=1) and 1.25 (m3=2) is 1.1875 -> ties to 1.25
    check 1.1875'f32.toFp8.bits == Fp8(0x3A'u8).bits      # m3=2, even
    # just above a midpoint always rounds up
    check 1.07'f32.toFp8.bits == Fp8(0x39'u8).bits        # 1.125

  test "underflow rounds toward zero at the bottom":
    check 1.0e-10'f32.toFp8.bits == 0x00'u8
    # half of the smallest subnormal ties to even -> zero
    check (pow(2.0'f32, -10.0'f32)).toFp8.bits == 0x00'u8
    # just above that midpoint rounds up to the smallest subnormal
    check (pow(2.0'f32, -10.0'f32) * 1.01'f32).toFp8.bits == 0x01'u8

  test "nearest-representable for arbitrary values":
    # brute-force reference: scan all encodings for the true nearest
    for i in 0 .. 2000:
      let x = (float32(i) - 1000.0'f32) * 0.37'f32
      let got = x.toFp8
      var bestErr = Inf.float32
      for b in 0 .. 255:
        let cand = Fp8(uint8(b))
        if cand.isNan: continue
        let err = abs(cand.toFloat32 - x)
        if err < bestErr: bestErr = err
      check abs(got.toFloat32 - x) == bestErr

suite "fp8 ordering":

  test "total order over finite values matches float order":
    for a in 0 .. 255:
      for b in 0 .. 255:
        let x = Fp8(uint8(a))
        let y = Fp8(uint8(b))
        if x.isNan or y.isNan: continue
        check (x < y) == (x.toFloat32 < y.toFloat32)

  test "signed zeros compare equal":
    check Fp8(0x00'u8) == Fp8(0x80'u8)
    check not (Fp8(0x00'u8) < Fp8(0x80'u8))
    check not (Fp8(0x80'u8) < Fp8(0x00'u8))
