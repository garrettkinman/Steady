# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

## OCP FP8 E4M3 — 1 sign / 4 exponent (bias 7) / 3 mantissa.
##
## This type exists primarily as the *canary* for the numeric policy
## abstraction: it is a non-native storage type with software arithmetic, a
## widening conversion into a different accumulator type, and no scale
## metadata. Every code path a posit policy will need is exercised here.
## If a kernel compiles and runs correctly under `RealFp8`, adding posits
## later is a matter of supplying arithmetic, not of reshaping kernels.
##
## Deviations from IEEE binary8 are per the OCP FP8 spec:
##   * no infinities — the encoding space is reclaimed for finite values
##   * S.1111.111 is the only NaN
##   * max finite magnitude is 448.0
##   * subnormals are supported (step 2^-9, min subnormal 2^-9)
##
## Conversion out of float32 saturates rather than overflowing to NaN, which
## is the behaviour inference workloads want.

type Fp8* = distinct uint8

const
  Fp8Nan* = Fp8(0x7F'u8)
  Fp8Max* = Fp8(0x7E'u8)      ## +448.0
  Fp8Min* = Fp8(0xFE'u8)      ## -448.0
  Fp8Zero* = Fp8(0x00'u8)

func bits*(x: Fp8): uint8 {.inline.} = uint8(x)

func isNan*(x: Fp8): bool {.inline.} =
  (uint8(x) and 0x7F'u8) == 0x7F'u8

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DECODE  (fp8 -> float32)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Exact: every one of the 256 encodings maps to a float32 with no rounding.

func toFloat32*(x: Fp8): float32 =
  let b = uint8(x)
  let sign = uint32(b and 0x80'u8) shl 24
  let e4 = int((b shr 3) and 0x0F'u8)
  let m3 = uint32(b and 0x07'u8)

  if e4 == 0:
    if m3 == 0:
      return cast[float32](sign)              # +/- 0
    # Subnormal: value = m3 * 2^-9. Renormalise into a float32 normal.
    var m = m3
    var e = -6                                # exponent of the leading bit slot
    while (m and 0x04'u32) == 0:              # shift until bit 2 is the leading 1
      m = m shl 1
      dec e
    let mant = (m and 0x03'u32) shl 21        # drop implicit bit, align to 23
    let bits = sign or (uint32(e - 1 + 127) shl 23) or mant
    return cast[float32](bits)

  if e4 == 0x0F and m3 == 0x07:
    return cast[float32](0x7FC0_0000'u32)     # NaN

  let bits = sign or (uint32(e4 - 7 + 127) shl 23) or (m3 shl 20)
  cast[float32](bits)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ENCODE  (float32 -> fp8, round-to-nearest-even, saturating)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

func rneShift(v: uint32, shift: int): uint32 {.inline.} =
  ## Shift `v` right by `shift`, rounding to nearest with ties to even.
  if shift <= 0:
    return v shl (-shift)
  if shift > 31:
    return 0
  let low = v and ((1'u32 shl shift) - 1)
  let half = 1'u32 shl (shift - 1)
  result = v shr shift
  if low > half or (low == half and (result and 1'u32) == 1'u32):
    inc result

func toFp8*(x: float32): Fp8 =
  let xb = cast[uint32](x)
  let sign = uint8((xb shr 24) and 0x80'u32)
  let absb = xb and 0x7FFF_FFFF'u32

  if absb >= 0x7F80_0000'u32:                 # NaN or Inf
    if absb > 0x7F80_0000'u32:
      return Fp8Nan
    return Fp8(sign or 0x7E'u8)               # Inf saturates to +/-448

  if absb == 0:
    return Fp8(sign)

  let e32 = int((absb shr 23) and 0xFF'u32)
  let m32 = absb and 0x007F_FFFF'u32
  let e = e32 - 127                           # true exponent of 1.m32

  if e < -6:
    # Subnormal target. Value in units of 2^-9 is (1.m32 * 2^e) * 2^9,
    # i.e. (2^23 + m32) >> (14 - e), rounded to nearest even.
    let k = rneShift(0x0080_0000'u32 or m32, 14 - e)
    # k == 8 carries cleanly into the smallest normal (e4 = 1, m3 = 0).
    return Fp8(sign or uint8(k and 0x0F'u32))

  var e4 = e + 7
  var m3 = rneShift(m32, 20)
  if m3 == 8:                                 # mantissa carry
    m3 = 0
    inc e4

  if e4 > 15 or (e4 == 15 and m3 >= 7):       # beyond +/-448 -> saturate
    return Fp8(sign or 0x7E'u8)

  Fp8(sign or uint8((uint32(e4) shl 3) or m3))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ORDERING
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Sign-magnitude, so ordering needs a fold — unlike posits, which are
# monotonic under two's-complement comparison and get this for free.

func `<`*(a, b: Fp8): bool {.inline.} =
  let x = uint8(a)
  let y = uint8(b)
  let sx = (x and 0x80'u8) != 0
  let sy = (y and 0x80'u8) != 0
  if sx != sy:
    # Distinct signs; -0 < +0 must still compare false.
    return sx and ((x or y) and 0x7F'u8) != 0
  if sx: (y and 0x7F'u8) < (x and 0x7F'u8)
  else:  (x and 0x7F'u8) < (y and 0x7F'u8)

func `<=`*(a, b: Fp8): bool {.inline.} =
  not (b < a)

func `==`*(a, b: Fp8): bool {.inline.} =
  if a.isNan or b.isNan: false
  elif ((uint8(a) or uint8(b)) and 0x7F'u8) == 0: true   # +0 == -0
  else: uint8(a) == uint8(b)

func `$`*(x: Fp8): string =
  $x.toFloat32
