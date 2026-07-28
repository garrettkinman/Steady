# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

## The numeric policy contract.
##
## A policy is an empty tag type plus a set of overloaded templates. It
## abstracts over *what happens between accumulate and store*, which is the
## only place integer-affine quantization and real-number formats genuinely
## differ once the host compiler has done its job.
##
## This module is the contract; `policy.nim` is the default implementation of
## it. They are separate modules for one reason: **an arithmetic backend has
## to be able to name the policy tags without importing the implementation it
## replaces.** A backend that overrides `mac` for `RealP8` imports this file
## and nothing else, `policy.nim` imports the backend, and the cycle that
## would otherwise exist does not.
##
## That split is also the layering, stated once:
##
##   contract.nim      tags, associated types, params, block widths
##     steady_arith    (optional) hardware arithmetic — sits *below* the policy
##   policy.nim        default members
##   kernels/arith.nim per-member dispatch
##   kernels/reference sequenced loops
##     steady_backend  (optional) whole-kernel override — sits *above* them
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
## thing to retrofit later. It is also what lets a hardware multiply-accumulate
## be substituted for the software one without a kernel changing.
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
##
## The associated *types* are deliberately not overridable. A backend may
## replace the arithmetic over an accumulator; changing what the accumulator
## **is** — a hardware quire register rather than an int64, say — changes what
## the host emits as well, so it is a new policy rather than an override. Same
## rule as fp8: E5M2 hardware does not accelerate `RealFp8`, it is a different
## format and gets its own policy and its own codec.

import ./fp8, ./posit8

export fp8, posit8

type
  AffineI8* = object
    ## TFLite-style affine quantization: real = scale * (q - zeroPoint).
    ## int8 storage, int32 accumulation, Q31 multiplier + shift on the way out.

  RealF32* = object
    ## Plain float32. Trivial, but it exercises the empty-metadata path.

  RealFp8* = object
    ## OCP FP8 E4M3 storage, float32 accumulation. The stand-in for posits.

  RealP8* = object
    ## Posit(8,0) storage, exact int64 quire accumulation.
    ##
    ## The format the `mac`-as-a-primitive contract above was written for. A
    ## quire is not an accumulator that happens to be wider: it is a fixed-point
    ## register in which the entire reduction is *exact*, so a convolution
    ## rounds once, at `finish`, and never in between. `RealFp8` approximates
    ## that with a float32 accumulator, which rounds at every step and merely
    ## rounds finely; this one does not round at all.
    ##
    ## What makes it affordable is `es = 0`: every posit8 value is an integer
    ## multiple of 2^-6, so a product is a multiple of 2^-12 and the quire is a
    ## plain int64 counting those. `mac` is two table loads and one 16x16
    ## multiply — no floating point anywhere on the target, which is the point
    ## on a part with no FPU.

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

template Store*(_: typedesc[RealP8]): typedesc = Posit8
template Accum*(_: typedesc[RealP8]): typedesc = Quire
template Bias*(_: typedesc[RealP8]): typedesc = Posit8
template Params*(_: typedesc[RealP8]): typedesc = RealParams[Quire]
  ## Clamp bounds live in the *quire* domain, not in storage. The host knows
  ## a fused activation's bounds as real numbers and 0, 1 and 6 are all exact
  ## multiples of 2^-12, so converting them costs nothing and the clamp then
  ## happens before the single rounding rather than after it.

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BLOCK WIDTHS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## How many outputs the matmul-family kernels compute at once, and therefore
## how many accumulators are in flight.
##
## These are a property of the **policy**, not of the kernel file, because
## what they are trading against is the target's register file and how many
## accumulators of `Accum(P)` fit in it. Four is right for a general-purpose
## core holding int32 or float32 accumulators, and it is a measurement rather
## than a preference: two was about 10% worse and eight was worse than two,
## because with eight accumulators plus eight weights and eight activations
## live at once the compiler spilled seven of the eight to the stack and the
## reuse did not pay for the traffic.
##
## They are three numbers rather than one because the kernels want them for
## different reasons, and because a machine with a vector unit will not want
## the same number in all three places.
##
## A part with hardware arithmetic will often want something else entirely. A
## posit unit with a *single* quire register wants 1 — blocking by four would
## spill the quire to memory every iteration and lose far more than the load
## reuse gains. So these dispatch through `kernels/arith.nim` like any other
## policy member, and an arithmetic backend overrides them by defining
## `OcBlock(P)` and friends.

template OcBlock*(_: typedesc[AffineI8]): int = 4
template DwBlock*(_: typedesc[AffineI8]): int = 4
template FcBlock*(_: typedesc[AffineI8]): int = 4

template OcBlock*(_: typedesc[RealF32]): int = 4
template DwBlock*(_: typedesc[RealF32]): int = 4
template FcBlock*(_: typedesc[RealF32]): int = 4

template OcBlock*(_: typedesc[RealFp8]): int = 4
template DwBlock*(_: typedesc[RealFp8]): int = 4
template FcBlock*(_: typedesc[RealFp8]): int = 4

template OcBlock*(_: typedesc[RealP8]): int = 4
template DwBlock*(_: typedesc[RealP8]): int = 4
template FcBlock*(_: typedesc[RealP8]): int = 4

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# GEMMLOWP FIXED-POINT REQUANTIZATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# These must match TFLite bit-for-bit. Deviating by one LSB here is wrong
# across an entire network and is invisible without a reference harness.
#
# They live in the contract rather than beside `AffineI8`'s members because a
# backend implementing `finish` for that policy needs them: the point of
# overriding `finish` is usually to use a hardware multiplier for the Q31 step
# while keeping TFLite's exact rounding, and that is only possible if the
# reference arithmetic is reachable.

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

func noClamp*(_: typedesc[RealP8]): RealParams[Quire] {.inline.} =
  ## The quire's own limits rather than an infinity — a fixed-point
  ## accumulator has no such value, and saturating at the ends of the type is
  ## the same clamp in practice.
  RealParams[Quire](actMin: low(Quire), actMax: high(Quire))

func reluClamp*(_: typedesc[RealP8]): RealParams[Quire] {.inline.} =
  RealParams[Quire](actMin: 0'i64, actMax: high(Quire))
