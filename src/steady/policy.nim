# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

## Numeric policies.
##
## A policy is an empty tag type plus a set of overloaded templates. It
## abstracts over *what happens between accumulate and store*, which is the
## only place integer-affine quantization and real-number formats genuinely
## differ once the host compiler has done its job.
##
## The contract, for a policy `P`:
##
##   Store(P)      typedesc   stored element type
##   Accum(P)      typedesc   accumulator type
##   Bias(P)       typedesc   bias element type (differs from Store for affine!)
##   Params(P)     typedesc   per-op finish metadata
##
##   zeroAccum(P)                   -> Accum(P)
##   mac(P, acc: var Accum, a, b: Store)
##   addBias(P, acc: var Accum, b: Bias)
##   finish(P, acc: Accum, prm: Params, ch: int) -> Store
##
## `mac` is a *primitive*, not sugar for `acc += a * b`. A posit policy
## accumulates into a quire, which supports fused multiply-accumulate and
## nothing else — there is no `*` on a quire. Writing kernels in terms of
## `mac` is what keeps that door open, and it is the single most expensive
## thing to retrofit later.
##
## Two further contract requirements, both relied on by kernels:
##
##   * `<` on `Store(P)` must agree with the ordering of the real values it
##     encodes. True for int8 under a positive affine scale, for fp8 via an
##     explicit fold, and free for posits (monotonic under two's-complement).
##   * `Bias(P)` values are in *accumulator* units. For affine that means the
##     host has already folded zero-point correction and bias rescaling into
##     an int32 constant; the kernel never sees a zero point.
##
## Templates are used rather than concepts deliberately: they inline
## unconditionally (so the abstraction is genuinely free), and a missing
## member produces "undeclared identifier: mac" rather than a concept-match
## essay.

import ./fp8

export fp8

type
  AffineI8* = object
    ## TFLite-style affine quantization: real = scale * (q - zeroPoint).
    ## int8 storage, int32 accumulation, Q31 multiplier + shift on the way out.

  RealF32* = object
    ## Plain float32. Trivial, but it exercises the empty-metadata path.

  RealFp8* = object
    ## OCP FP8 E4M3 storage, float32 accumulation. The stand-in for posits.

  AffineParams* = object
    ## Per-op requantization metadata. `mult` and `shift` point at const
    ## (flash-resident) arrays emitted by the host compiler.
    ##
    ## `channelStride` is 0 for per-tensor quantization and 1 for
    ## per-channel, so the same indexing expression serves both without a
    ## branch and without duplicating a per-tensor scale N times in flash.
    mult*: ptr UncheckedArray[int32]
    shift*: ptr UncheckedArray[int32]
    channelStride*: int
    outZeroPoint*: int32
    actMin*, actMax*: int32     ## fused activation clamp, quantized domain

  RealParams*[A] = object
    ## Accumulator-domain clamp for fused activations.
    ##
    ## Note this is deliberately *not* an empty type. Real formats need no
    ## scale metadata, but they are not metadata-free in general: a posit
    ## policy will likely want a per-tensor power-of-two rescale here to
    ## centre a tensor's distribution on the high-accuracy band around 1.0.
    actMin*, actMax*: A

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ASSOCIATED TYPES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template Store*(_: typedesc[AffineI8]): typedesc = int8
template Accum*(_: typedesc[AffineI8]): typedesc = int32
template Bias*(_: typedesc[AffineI8]): typedesc = int32
template Params*(_: typedesc[AffineI8]): typedesc = AffineParams

template Store*(_: typedesc[RealF32]): typedesc = float32
template Accum*(_: typedesc[RealF32]): typedesc = float32
template Bias*(_: typedesc[RealF32]): typedesc = float32
template Params*(_: typedesc[RealF32]): typedesc = RealParams[float32]

template Store*(_: typedesc[RealFp8]): typedesc = Fp8
template Accum*(_: typedesc[RealFp8]): typedesc = float32
template Bias*(_: typedesc[RealFp8]): typedesc = Fp8
template Params*(_: typedesc[RealFp8]): typedesc = RealParams[float32]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# GEMMLOWP FIXED-POINT REQUANTIZATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# These must match TFLite bit-for-bit. Deviating by one LSB here is wrong
# across an entire network and is invisible without a reference harness.

func satRoundDoublingHighMul*(a, b: int32): int32 {.inline.} =
  ## gemmlowp SaturatingRoundingDoublingHighMul: the high 32 bits of
  ## 2*a*b, rounded, saturating on the single overflow case.
  ##
  ## Ties round toward positive infinity, *not* away from zero: the nudge is
  ## asymmetric (`1 - 2^30` on the negative side) and the division truncates.
  ## `roundingDivideByPOT` below uses the opposite tie rule. The asymmetry is
  ## surprising but it is what TFLite does, so reproducing it is the point.
  if a == low(int32) and b == low(int32):
    return high(int32)
  let ab = int64(a) * int64(b)
  let nudge: int64 = if ab >= 0: (1'i64 shl 30) else: (1'i64 - (1'i64 shl 30))
  int32((ab + nudge) div (1'i64 shl 31))

func roundingDivideByPOT*(x: int32, exponent: int32): int32 {.inline.} =
  ## gemmlowp RoundingDivideByPOT: x / 2^exponent, rounding half away from
  ## zero. Note this is *not* an arithmetic shift — the tie handling differs.
  ##
  ## `exponent` reaches 31, so the mask is built in 64 bits and narrowed:
  ## `1'i32 shl 31` does not fit an int32. gemmlowp writes `1ll << exponent`
  ## for the same reason. This is not a theoretical case — `quantizeMultiplier`
  ## clamps its shift at -31, and a convolution channel whose weights are all
  ## but zero produces exactly that, which is how a real MobileNet found it.
  if exponent == 0:
    return x
  let mask = int32((1'i64 shl exponent) - 1'i64)
  let remainder = x and mask
  let threshold = (mask shr 1) + (if x < 0: 1'i32 else: 0'i32)
  result = (x shr exponent)
  if remainder > threshold:
    inc result

func multiplyByQuantizedMultiplier*(x, quantizedMultiplier, shift: int32): int32 {.inline.} =
  ## TFLite's MultiplyByQuantizedMultiplier. `shift` is positive for a left
  ## shift applied before the multiply, negative for a right shift after.
  let leftShift = if shift > 0: shift else: 0'i32
  let rightShift = if shift > 0: 0'i32 else: -shift
  roundingDivideByPOT(
    satRoundDoublingHighMul(x * (1'i32 shl leftShift), quantizedMultiplier),
    rightShift)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# AffineI8
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template zeroAccum*(_: typedesc[AffineI8]): int32 = 0'i32

template mac*(_: typedesc[AffineI8], acc: var int32, a, b: int8) =
  acc = acc + int32(a) * int32(b)

template addBias*(_: typedesc[AffineI8], acc: var int32, b: int32) =
  acc = acc + b

func finish*(_: typedesc[AffineI8], acc: int32, prm: AffineParams, ch: int): int8 {.inline.} =
  let i = ch * prm.channelStride
  var v = multiplyByQuantizedMultiplier(acc, prm.mult[i], prm.shift[i]) + prm.outZeroPoint
  if v < prm.actMin: v = prm.actMin
  if v > prm.actMax: v = prm.actMax
  int8(v)

template lowestStore*(_: typedesc[AffineI8]): int8 = low(int8)

template accumulate*(_: typedesc[AffineI8], acc: var int32, v: int8) =
  ## Widen and add, with no multiply. Used by reductions such as avgpool.
  acc = acc + int32(v)

func divAccum*(_: typedesc[AffineI8], acc: int32, n: int): int32 {.inline.} =
  ## Rounds half away from zero, matching TFLite's avgpool.
  let d = int32(n)
  if acc > 0: (acc + d div 2) div d
  else: (acc - d div 2) div d

template meanScale*(_: typedesc[AffineI8], acc: int32, count: int): int32 =
  ## Brings an accumulated sum into the units `finish` expects for a *mean*.
  ##
  ## For affine this is the identity: the host folds `1/count` into the output
  ## multiplier, so the whole reduction rounds exactly once, in `finish`. That
  ## is both more accurate than dividing first — an integer divide would round
  ## to the input's grid before requantizing, costing up to half an LSB — and
  ## cheaper, since the kernel then contains no division at all.
  ##
  ## Distinct from `divAccum`, which average-pooling needs to keep doing:
  ## pooling preserves quantization, so its division is the whole operation and
  ## TFLite performs it as an integer divide that we match bit for bit.
  acc

func storeOf*(_: typedesc[AffineI8], acc: int32): int8 {.inline.} =
  ## Narrow without requantizing — for ops whose input and output share a
  ## quantization (pooling, reductions). Saturates rather than wrapping.
  if acc < -128'i32: -128'i8
  elif acc > 127'i32: 127'i8
  else: int8(acc)

template lutIndex*(_: typedesc[AffineI8], v: int8): int =
  ## Raw storage byte of a value, which is what a host-generated lookup table
  ## is keyed on. Two's-complement, so -128 lands at 128 and the table the
  ## host writes has to agree — that agreement is what the end-to-end test
  ## checks.
  ##
  ## Deliberately absent for `RealF32`: a 32-bit store has no enumerable
  ## domain, so `lut1d` simply does not instantiate for it and the host
  ## rejects LUT ops under that policy rather than reaching for libm.
  int(cast[uint8](v))

func addRescaled*(_: typedesc[AffineI8], acc: var int32, v: int8,
                  mult, shift, offset: int32) {.inline.} =
  ## Rescale one operand of a two-input elementwise op into the common
  ## domain and accumulate. `offset` is TFLite's input_offset, i.e. the
  ## negated zero point. The fixed left shift of 20 matches TFLite's ADD
  ## reference and exists to preserve precision before the Q31 multiply.
  const AddLeftShift = 20'i32
  let shifted = (int32(v) + offset) * (1'i32 shl AddLeftShift)
  acc = acc + multiplyByQuantizedMultiplier(shifted, mult, shift)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# RealF32
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template zeroAccum*(_: typedesc[RealF32]): float32 = 0'f32

template mac*(_: typedesc[RealF32], acc: var float32, a, b: float32) =
  acc = acc + a * b

template addBias*(_: typedesc[RealF32], acc: var float32, b: float32) =
  acc = acc + b

func finish*(_: typedesc[RealF32], acc: float32, prm: RealParams[float32],
             ch: int): float32 {.inline.} =
  result = acc
  if result < prm.actMin: result = prm.actMin
  if result > prm.actMax: result = prm.actMax

template lowestStore*(_: typedesc[RealF32]): float32 = -Inf.float32

template accumulate*(_: typedesc[RealF32], acc: var float32, v: float32) =
  acc = acc + v

func divAccum*(_: typedesc[RealF32], acc: float32, n: int): float32 {.inline.} =
  acc / float32(n)

template meanScale*(_: typedesc[RealF32], acc: float32, count: int): float32 =
  ## No multiplier to fold into, so a real policy divides here. Same kernel
  ## source either way — this is the policy absorbing the difference, which is
  ## what it is for.
  acc / float32(count)

func storeOf*(_: typedesc[RealF32], acc: float32): float32 {.inline.} = acc

func addRescaled*(_: typedesc[RealF32], acc: var float32, v: float32,
                  mult, shift, offset: int32) {.inline.} =
  ## Rescale parameters are meaningless for a real format and are ignored;
  ## the host emits zeros for them. This is the abstraction earning its keep
  ## — same kernel source, no branch, no cost.
  acc = acc + v

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# RealFp8
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Widening happens inside `mac`, exactly where a posit quire would do its
# exact fused accumulate. Rounding to storage precision happens once, in
# `finish` — which is the whole reason the accumulator is a separate type.

template zeroAccum*(_: typedesc[RealFp8]): float32 = 0'f32

template mac*(_: typedesc[RealFp8], acc: var float32, a, b: Fp8) =
  acc = acc + a.toFloat32 * b.toFloat32

template addBias*(_: typedesc[RealFp8], acc: var float32, b: Fp8) =
  acc = acc + b.toFloat32

func finish*(_: typedesc[RealFp8], acc: float32, prm: RealParams[float32],
             ch: int): Fp8 {.inline.} =
  var v = acc
  if v < prm.actMin: v = prm.actMin
  if v > prm.actMax: v = prm.actMax
  v.toFp8

template lowestStore*(_: typedesc[RealFp8]): Fp8 = Fp8Min

template accumulate*(_: typedesc[RealFp8], acc: var float32, v: Fp8) =
  acc = acc + v.toFloat32

func divAccum*(_: typedesc[RealFp8], acc: float32, n: int): float32 {.inline.} =
  acc / float32(n)

template meanScale*(_: typedesc[RealFp8], acc: float32, count: int): float32 =
  acc / float32(count)

func storeOf*(_: typedesc[RealFp8], acc: float32): Fp8 {.inline.} = acc.toFp8

func addRescaled*(_: typedesc[RealFp8], acc: var float32, v: Fp8,
                  mult, shift, offset: int32) {.inline.} =
  acc = acc + v.toFloat32

template lutIndex*(_: typedesc[RealFp8], v: Fp8): int =
  ## The whole point of the canary: a posit policy's table lookup is this
  ## same expression, and it needs no arithmetic on the target at all.
  int(v.bits)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CONVENIENCE CONSTRUCTORS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

func noClamp*(_: typedesc[RealF32]): RealParams[float32] {.inline.} =
  RealParams[float32](actMin: -Inf.float32, actMax: Inf.float32)

func noClamp*(_: typedesc[RealFp8]): RealParams[float32] {.inline.} =
  RealParams[float32](actMin: -Inf.float32, actMax: Inf.float32)

func reluClamp*(_: typedesc[RealF32]): RealParams[float32] {.inline.} =
  RealParams[float32](actMin: 0'f32, actMax: Inf.float32)

func reluClamp*(_: typedesc[RealFp8]): RealParams[float32] {.inline.} =
  RealParams[float32](actMin: 0'f32, actMax: Inf.float32)
