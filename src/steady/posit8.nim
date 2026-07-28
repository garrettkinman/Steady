# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

## Posit(8,0) — 1 sign / run-length regime / no exponent field / fraction.
##
## The format `RealFp8` was a stand-in for. Two properties make it a better
## fit for this compiler than fp8 rather than merely a different one, and
## both are consequences of `es = 0` specifically:
##
## **Every representable value is an exact multiple of 2^-6.** The fraction
## field loses one bit for each bit the regime gains, so the ulp of the
## densest binade and the ulp of the sparsest coincide at the bottom: 2^-6.
## The largest magnitude is 2^6. So a posit is an integer in units of 2^-6
## — `units` below — and the whole format is exactly representable in an
## int16 with room to spare.
##
## **Therefore a product of two posits is an exact multiple of 2^-12**, and
## a sum of products is too. That is what makes the quire below a *real*
## quire rather than a float32 approximation of one: `Quire` is an int64 in
## units of 2^-12, `mac` is one 16x16 multiply accumulated into it, and a
## convolution's entire reduction is exact. Rounding happens once, in
## `finish`, which is what the policy contract promised and what a float
## accumulator cannot actually deliver.
##
## The headroom is not tight. A product is at most 2^24 quire units, so
## 2^38 multiply-accumulates fit before an int64 overflows — six orders of
## magnitude more taps than the largest layer here has.
##
## Rounding follows the posit standard rather than IEEE habits:
##   * round to nearest, ties to even *in the encoding*
##   * a magnitude above maxpos saturates to maxpos — there is no infinity
##   * a nonzero magnitude below minpos rounds to minpos, never to zero
##
## The last rule is the one that surprises people. It is deliberate: a posit
## carries no subnormals and no gradual underflow, so flushing to zero would
## turn a small weight into an absent one.
##
## Layout, for the seven bits below the sign:
##
##   regime  a run of m identical bits, then the opposite bit as terminator
##           run of ones  -> k = m - 1
##           run of zeros -> k = -m
##   frac    whatever is left, 7 - (m + 1) bits, possibly none
##   value   2^k * (1 + frac / 2^fracBits)
##
## and a negative value is the two's complement of the whole byte, which is
## why `<` is a plain signed comparison and needs no fold — unlike fp8.

type
  Posit8* = distinct uint8

  Quire* = int64
    ## Exact accumulator, in units of 2^-12.
    ##
    ## Not a distinct type. It is arithmetic in the same sense `AffineI8`'s
    ## int32 accumulator is, the policy templates are the only things that
    ## put values into it, and a distinct int64 would buy unit safety at the
    ## cost of borrowing every operator the clamp in `finish` needs.

const
  Posit8Zero* = Posit8(0x00'u8)
  Posit8NaR* = Posit8(0x80'u8)   ## Not a Real: the one non-numeric encoding
  Posit8Max* = Posit8(0x7F'u8)   ## +64  (maxpos)
  Posit8Min* = Posit8(0x81'u8)   ## -64  (-maxpos, and the lowest real value)
  Posit8MinPos* = Posit8(0x01'u8) ## +1/64 (minpos)

  UnitScale* = 6
    ## `units` are 2^-UnitScale. A property of es = 0 and nbits = 8; see the
    ## module doc.
  QuireScale* = 2 * UnitScale
    ## Quire units are 2^-QuireScale, because a quire holds *products*.
  QuireOne* = 1'i64 shl QuireScale
    ## The quire representation of 1.0. Handy for building clamp bounds.

func bits*(x: Posit8): uint8 {.inline.} = uint8(x)

func isNaR*(x: Posit8): bool {.inline.} = uint8(x) == 0x80'u8

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DECODE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

func decodeUnits(b: uint8): int32 =
  ## The exact value of encoding `b`, in units of 2^-6. NaR and zero both
  ## decode to 0 here; callers that care distinguish them by the encoding.
  ##
  ## Runs at compile time to build `UnitsTable`, so this is host-side code in
  ## practice and clarity beats cleverness.
  if b == 0x00'u8 or b == 0x80'u8:
    return 0'i32

  let neg = (b and 0x80'u8) != 0
  # Magnitude pattern: two's complement of the byte when negative.
  let mag = if neg: uint8((0'i32 - int32(b)) and 0xFF'i32) else: b
  let payload = uint32(mag and 0x7F'u8)      # seven bits, MSB first

  # Regime: count the leading run in bits 6..0.
  let lead = (payload shr 6) and 1'u32
  var m = 0
  var i = 6
  while i >= 0 and ((payload shr uint32(i)) and 1'u32) == lead:
    inc m
    dec i
  let k = if lead == 1'u32: m - 1 else: -m

  # Fraction: whatever survives the run and its terminator.
  let fracBits = max(0, 6 - m)
  let frac = payload and ((1'u32 shl fracBits) - 1'u32)

  # value * 2^6 = 2^(k+6) + frac * 2^(k+6-fracBits), and both exponents are
  # non-negative for every k this format can produce — that is the 2^-6 grid
  # claim, and `test_posit8` checks it encoding by encoding.
  let shiftFrac = k + UnitScale - fracBits
  let v = int32(1'i32 shl (k + UnitScale)) + int32(frac shl uint32(shiftFrac))
  if neg: -v else: v

const UnitsTable: array[256, int16] = block:
  ## Decode as a table, built by the Nim compiler on the build machine.
  ##
  ## 512 bytes of `.rodata` buys a one-load decode in the innermost loop
  ## there is, against a regime scan of up to seven iterations. This is the
  ## only table the runtime carries that is not model-specific, and it is
  ## dropped by the linker when no posit model is built into the image.
  var t: array[256, int16]
  for b in 0 .. 255:
    t[b] = int16(decodeUnits(uint8(b)))
  t

func units*(x: Posit8): int32 {.inline.} =
  ## The value of `x` in units of 2^-6, exactly. The hot path: `mac` is this
  ## twice and one multiply.
  int32(UnitsTable[int(uint8(x))])

func toFloat64*(x: Posit8): float64 {.inline.} =
  ## Exact for every encoding: `units` is an integer and 64 is a power of two.
  ##
  ## NaR maps to a quiet NaN, spelled as a bit pattern rather than imported
  ## from `std/math` — nothing else in this module needs libm and the target
  ## build should not acquire a dependency for one constant.
  if x.isNaR: return cast[float64](0x7FF8_0000_0000_0000'u64)
  float64(units(x)) / 64.0

func toFloat32*(x: Posit8): float32 {.inline.} =
  if x.isNaR: return cast[float32](0x7FC0_0000'u32)
  float32(units(x)) / 64.0'f32

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ENCODE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

func fracBitsFor(k: int): int {.inline.} =
  ## Bits left for the fraction at scale `k`: the regime costs `m + 1` of the
  ## seven, where `m` is `k + 1` going up and `-k` going down.
  if k >= 0: max(0, 5 - k) else: max(0, 6 + k)

func assemble(k: int, frac: uint32): Posit8 =
  ## Pack a *positive* value. The run is written first and may fill the field
  ## entirely, in which case there is no terminator and no fraction — which is
  ## exactly how maxpos and minpos are encoded.
  ##
  ## `left` ends up equal to `fracBitsFor(k)` by construction; the width is
  ## tracked here rather than passed in so that a fraction can never be
  ## shifted into the regime.
  var v = 0'u32
  var left = 7
  if k >= 0:
    for _ in 0 ..< min(k + 1, left):
      v = (v shl 1) or 1'u32
      dec left
    if left > 0:
      v = v shl 1                      # terminating zero
      dec left
  else:
    for _ in 0 ..< min(-k, left):
      v = v shl 1                      # the run itself is zeros
      dec left
    if left > 0:
      v = (v shl 1) or 1'u32           # terminating one
      dec left
  Posit8(uint8((v shl uint32(left)) or (frac and ((1'u32 shl left) - 1'u32))))

func tiesUp(k: int, frac: int64): bool {.inline.} =
  ## Which way an exact midpoint goes: to the neighbour whose *encoding* is
  ## even.
  ##
  ## Not the same as "the fraction's low bit is set", though it reduces to
  ## that whenever the fraction field is non-empty. Where the field is empty
  ## the tie is between two adjacent regimes and the encoding's low bit is the
  ## last regime bit instead — at the bottom of the range that inverts the
  ## answer, so minpos and 2*minpos tie upward, not down.
  (uint8(assemble(k, uint32(frac))) and 1'u8) == 1'u8

func negated(x: Posit8): Posit8 {.inline.} =
  Posit8(uint8((0'i32 - int32(uint8(x))) and 0xFF'i32))

func positFromQuire*(q: Quire): Posit8 =
  ## Round an exact quire to the nearest posit: ties to even, saturating at
  ## maxpos, and never flushing a nonzero to zero.
  ##
  ## This is the only rounding a convolution, a fully-connected layer, an add
  ## or a concatenation performs — `mac` and `addBias` are exact, so the whole
  ## reduction rounds here, once.
  if q == 0: return Posit8Zero
  let neg = q < 0
  # `low(int64)` has no positive counterpart; it is far above maxpos anyway.
  let m = if q == low(int64): high(int64) else: abs(q)

  const MaxQuire = 1'i64 shl (UnitScale + QuireScale)   ## maxpos, 2^18
  const MinQuire = 1'i64 shl (QuireScale - UnitScale)   ## minpos, 2^6
  if m >= MaxQuire:
    return if neg: Posit8Min else: Posit8Max
  if m < MinQuire:
    return if neg: negated(Posit8MinPos) else: Posit8MinPos

  # 2^e <= m < 2^(e+1), so the value's binade is k = e - QuireScale.
  var e = 0
  var t = m shr 1
  while t != 0:
    inc e
    t = t shr 1
  var k = e - QuireScale
  let fracBits = fracBitsFor(k)

  # frac = round((m - 2^e) * 2^fracBits / 2^e), ties to even.
  let rem = m - (1'i64 shl e)
  let num = rem shl fracBits
  var frac = num shr e
  let r = num - (frac shl e)
  let half = 1'i64 shl (e - 1)
  if r > half or (r == half and tiesUp(k, frac)):
    inc frac

  if frac == (1'i64 shl fracBits):     # carried into the next binade
    frac = 0
    inc k
    if k > UnitScale:
      return if neg: Posit8Min else: Posit8Max

  result = assemble(k, uint32(frac))
  if neg: result = negated(result)

func toPosit8*(x: float64): Posit8 =
  ## Host-side encode, from a real value rather than from a quire.
  ##
  ## Deliberately *not* implemented via `positFromQuire`: the compiler encodes
  ## every weight with this and the runtime rounds every activation with that,
  ## so keeping the two derivations independent turns the end-to-end test into
  ## a cross-check of both. They agree on all 256 encodings and on a large
  ## random sample; `test_posit8` is where that is pinned.
  if x != x: return Posit8NaR
  if x == 0.0: return Posit8Zero

  let neg = x < 0.0
  var a = abs(x)
  if a >= 64.0:
    return if neg: Posit8Min else: Posit8Max
  if a < 1.0 / 64.0:
    return if neg: negated(Posit8MinPos) else: Posit8MinPos

  # k = floor(log2 a), by halving rather than by `log2` — no libm rounding
  # gets to influence which binade a boundary value lands in.
  var k = 0
  while a >= 2.0:
    a = a / 2.0
    inc k
  while a < 1.0:
    a = a * 2.0
    dec k
  let fracBits = fracBitsFor(k)

  let scaled = (a - 1.0) * float64(1 shl fracBits)
  var frac = int64(scaled)                     # a is in [1, 2), so >= 0
  let r = scaled - float64(frac)
  if r > 0.5 or (r == 0.5 and tiesUp(k, frac)):
    inc frac

  if frac == (1'i64 shl fracBits):
    frac = 0
    inc k
    if k > UnitScale:
      return if neg: Posit8Min else: Posit8Max

  result = assemble(k, uint32(frac))
  if neg: result = negated(result)

func toPosit8*(x: float32): Posit8 {.inline.} = toPosit8(float64(x))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ORDERING
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Two's complement, so the encodings are monotonic in the value they stand
# for and comparison is one signed compare. This is the property fp8 needs a
# sign-magnitude fold for, and it is free here.
#
# NaR sorts below every real value, being 0x80. That is a consequence of the
# encoding rather than a decision, and it is the harmless direction: a NaR in
# a max-pool window is ignored rather than swallowing the window.

func `<`*(a, b: Posit8): bool {.inline.} =
  cast[int8](a) < cast[int8](b)

func `<=`*(a, b: Posit8): bool {.inline.} =
  cast[int8](a) <= cast[int8](b)

func `==`*(a, b: Posit8): bool {.inline.} =
  uint8(a) == uint8(b)

func `$`*(x: Posit8): string =
  if x.isNaR: "NaR" else: $x.toFloat64
