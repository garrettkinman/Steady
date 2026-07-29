# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

## The numeric policy contract.
##
## A policy is an empty tag type plus a set of overloaded templates. It
## abstracts over *what happens between accumulate and store*, which is the
## only place two numeric formats genuinely differ once the host compiler has
## done its job.
##
## **One policy ships.** `AffineI8` is the format every interchange file can
## carry and every model in the test suite uses, and a contract with one
## implementor is a fair thing to be suspicious of — so the reason it is here
## is worth stating rather than assuming.
##
## It is not to make adding formats easy. It is that `mac` below is a
## *primitive*, and that is the seam hardware arithmetic plugs into: a
## CMSIS-NN backend substituting SMLAD, a vector unit widening the accumulate,
## or a format whose accumulator supports fused multiply-accumulate and
## nothing else. Kernels written in terms of `acc += a * b` close that door,
## and reopening it means rewriting every kernel rather than adding a file.
## `tests/fixtures/steady_arith.nim` substitutes the arithmetic under this
## policy and requires bit-identical results, which is what keeps the claim
## that the abstraction is free from being an assertion.
##
## Three further policies once lived here — float32, OCP fp8 E4M3, and
## posit(8,0) over an exact int64 quire — and were removed rather than
## deprecated. Two reasons, both worth knowing before adding a fourth. No
## interchange format can carry them, so none could be driven from a `.tflite`
## and they were reachable only through the library API. And the posit one was
## not the format its name claimed: the 2022 standard fixes es = 2 and a 16n
## quire, so a conforming posit8 needs a 128-bit accumulator, while the cheap
## `mac` that made the implementation interesting — two table loads and one
## 16x16 multiply, no floating point — depended entirely on es = 0 making
## every value a multiple of 2^-6. The result was real; it was a result about
## posit(8,0), not about posits.
##
## This module is the contract; `policy.nim` is the default implementation of
## it. They are separate modules for one reason: **an arithmetic backend has
## to be able to name the policy tags without importing the implementation it
## replaces.** A backend that overrides `mac` for `AffineI8` imports this file
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
## `mac` is a *primitive*, not sugar for `acc += a * b`, and that is the whole
## point of the file. Some accumulators support fused multiply-accumulate and
## nothing else — there is no `*` on a fixed-point accumulator that a hardware
## unit owns — and some cores have one instruction that does the multiply and
## the add together. Writing kernels in terms of `mac` is what keeps both
## doors open, and it is the single most expensive thing to retrofit later.
##
## Two further contract requirements, both relied on by kernels:
##
##   * `<` on `Store(P)` must agree with the ordering of the real values it
##     encodes, so that max-pooling can compare stored values directly. True
##     for int8 under a positive affine scale.
##   * `Bias(P)` values are in *accumulator* units. For affine that means the
##     host has already folded zero-point correction and bias rescaling into
##     an int32 constant; the kernel never sees a zero point.
##
## Templates are used rather than concepts deliberately: they inline
## unconditionally (so the abstraction is genuinely free), and a missing
## member produces "undeclared identifier: mac" rather than a concept-match
## essay.
##
## **Known limitation.** The associated *types* are not overridable, only the
## arithmetic over them. A backend can replace what `mac` does to an `int32`;
## it cannot say that the accumulator is now an architectural register of some
## other width, because that changes what the host emits as well. So a part
## with a hardware accumulator is a new policy rather than a backend — which
## is exactly the case this contract was written to serve, and the one place
## it does not reach. Making `Accum(P)` opaque and overridable is the change
## that would fix it, and it is not a large one; it has not been made because
## there is no hardware here to test it against.

type
  AffineI8* = object
    ## TFLite-style affine quantization: real = scale * (q - zeroPoint).
    ## int8 storage, int32 accumulation, Q31 multiplier + shift on the way out.

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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ASSOCIATED TYPES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template Store*(_: typedesc[AffineI8]): typedesc = int8
template Accum*(_: typedesc[AffineI8]): typedesc = int32
template Bias*(_: typedesc[AffineI8]): typedesc = int32
template Params*(_: typedesc[AffineI8]): typedesc = AffineParams

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BLOCK WIDTHS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## How many outputs the matmul-family kernels compute at once, and therefore
## how many accumulators are in flight.
##
## These are a property of the **policy**, not of the kernel file, because
## what they are trading against is the target's register file and how many
## accumulators of `Accum(P)` fit in it. Four is right for a general-purpose
## core holding int32 accumulators, and it is a measurement rather
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
## unit with a *single* accumulator register wants 1 — blocking by four would
## spill it to memory every iteration and lose far more than the load reuse
## gains — and a vector unit wants considerably more than four. So these
## dispatch through `kernels/arith.nim` like any other policy member, and an
## arithmetic backend overrides them by defining `OcBlock(P)` and friends.

template OcBlock*(_: typedesc[AffineI8]): int = 4
template DwBlock*(_: typedesc[AffineI8]): int = 4
template FcBlock*(_: typedesc[AffineI8]): int = 4

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
