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
## **Two policies ship.** `AffineI8` is the format every interchange file can
## carry and every model in the test suite uses. `AffineI16x8` is TFLite's
## other integer format, int16 activations with int8 weights, which a model
## producing audio needs: at 8 bits its output is 8-bit sound. A contract with
## one implementor was a fair thing to be suspicious of, and the reason it is
## here is worth stating rather than assuming.
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
##   Weight(P)     typedesc   weight element type (int8 for both policies)
##   Accum(P)      typedesc   accumulator type
##   Bias(P)       typedesc   bias element type (differs from Store for affine!)
##   Params(P)     typedesc   per-op finish metadata
##
##   zeroAccum(P)                   -> Accum(P)
##   mac(P, acc: var Accum, w: Weight, x: Store)
##   addBias(P, acc: var Accum, b: Bias)
##   finish(P, acc: Accum, prm: Params, ch: int) -> Store
##   finishRescaled(P, acc: Accum, prm: Params) -> Store
##
## and, for a policy that runs matmuls between two activations,
##
##   ProductAccum(P)  typedesc   accumulator of Store x Store products
##   zeroProduct(P)                 -> ProductAccum(P)
##   macStores(P, acc: var ProductAccum, a, b: Store)
##   finishProduct(P, acc: ProductAccum, prm: Params) -> Store
##
## `finish` ends a matmul-family op and `finishRescaled` an elementwise one,
## whose operands `addRescaled` has brought to a common domain. They are the
## same arithmetic for `AffineI8` and not for `AffineI16x8`, where TFLite
## requantizes a convolution through a 16-bit multiplier and an addition
## through the Q31 one.
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

# Inner-loop arithmetic, without Nim's checks on a bare-metal target, as the
# kernels are
when hostOS == "any": {.push checks: off.}

type
  AffineI8* = object
    ## TFLite-style affine quantization: real = scale * (q - zeroPoint).
    ## int8 storage, int32 accumulation, Q31 multiplier + shift on the way out.

  AffineI16x8* = object
    ## TFLite's 16x8 format: int16 activations with zero point 0, int8
    ## weights, an int32 accumulator whose range the host checks layer by layer
    ## (TFLite's is int64), and the 16-bit multiplier TFLite uses for it.

  AffineParams* = object
    ## Per-op requantization metadata. `mult` and `shift` point at const
    ## (flash-resident) arrays emitted by the host compiler.
    ##
    ## `channelStride` is 0 for per-tensor quantization and 1 for
    ## per-channel, so the same indexing expression serves both without a
    ## branch and without duplicating a per-tensor scale N times in flash.
    mult*: ptr UncheckedArray[int32]
    shift*: ptr UncheckedArray[int32]
    ##
    ## For `AffineI16x8`'s matmul family `mult` is TFLite's 16-bit reduced
    ## multiplier, worked out on the host; see `reducedMultiplier`.
    channelStride*: int
    outZeroPoint*: int32
    actMin*, actMax*: int32     ## fused activation clamp, quantized domain

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ASSOCIATED TYPES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template Store*(_: typedesc[AffineI8]): typedesc = int8
template Weight*(_: typedesc[AffineI8]): typedesc = int8
template Accum*(_: typedesc[AffineI8]): typedesc = int32
template Bias*(_: typedesc[AffineI8]): typedesc = int32
template Params*(_: typedesc[AffineI8]): typedesc = AffineParams

template Store*(_: typedesc[AffineI16x8]): typedesc = int16
template Weight*(_: typedesc[AffineI16x8]): typedesc = int8
template Accum*(_: typedesc[AffineI16x8]): typedesc = int32
template Bias*(_: typedesc[AffineI16x8]): typedesc = int32
template Params*(_: typedesc[AffineI16x8]): typedesc = AffineParams

## `ProductAccum(P)` sums products of two *stored* values, as a matmul between
## two activations does (attention's Q times K, and its weights times V),
## where `Accum(P)` sums a weight times a stored value. Two int16 values
## multiply to 2^30, so a 16-bit store needs an int64 sum, as TFLite's has;
## and unlike a weight's, its range cannot be checked on the host, as both
## operands are only known at run time.
template ProductAccum*(_: typedesc[AffineI16x8]): typedesc = int64

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
## That measurement is a Cortex-M4's, with its flash accelerators on, and it
## does not generalise — which is the case for these being dispatched rather
## than fixed, and is now a result rather than an argument. Measured on the
## four convolutional models in the benchmark suite, `OcBlock = 8` against the
## default 4:
##
##   ESP32-C3 (RV32IMC, 32 registers)         1.01x - 1.13x *faster*
##   STM32L475, flash accelerators on          0.93x - 0.99x, i.e. slower
##   STM32L475, accelerators off               1.05x - 1.10x faster
##
## Three answers from two parts. RISC-V has 32 general-purpose registers where
## ARM32 has about 13, so the spill that made eight a loss on the M4 does not
## happen there. And the same M4 flips sign when its flash accelerators are
## switched off: with nothing in front of flash the kernel is memory-bound, so
## a wider block pays for its spills by reading the activation patch fewer
## times. The right width is a property of the core *and* of what sits between
## it and its weights, and there is no single value that is correct.
##
## Four stays the default because it is right for the Cortex-M4 in the
## configuration a product would ship. A target that wants another value
## defines `OcBlock(P)` in a `steady_arith` module; the benchmark harness
## deliberately does not, so that its published tables stay comparable across
## boards.
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

template OcBlock*(_: typedesc[AffineI16x8]): int = 4
template DwBlock*(_: typedesc[AffineI16x8]): int = 4
template FcBlock*(_: typedesc[AffineI16x8]): int = 4

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
# 16x8 REQUANTIZATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TFLite requantizes a 16x8 convolution through its int64 overload of
# MultiplyByQuantizedMultiplier: the Q31 multiplier rounded to 16 bits, then
# one shift that rounds a tie up. Not the gemmlowp path above, and it must
# match just as exactly. The host reduces each multiplier once, so the kernel
# only multiplies and shifts.

func reducedMultiplier*(quantizedMultiplier: int32): int32 =
  ## TFLite's 16-bit form of a Q31 multiplier.
  if quantizedMultiplier < 0x7FFF0000'i32:
    (quantizedMultiplier + (1'i32 shl 15)) shr 16
  else:
    0x7FFF'i32

func multiplyByReducedMultiplier*(x, reduced, shift: int32): int32 {.inline.} =
  ## `x * reduced * 2^(shift - 15)`, a tie rounded up. TFLite takes an int64
  ## `x`; an int32 holds every sum the host has allowed. `shift` is below 8,
  ## so the total shift is at least 8, and the product fits an int64.
  let total = 15'i32 - shift
  int32((int64(x) * int64(reduced) + (1'i64 shl (total - 1))) shr total)

func multiplyByReducedMultiplier64*(x: int64, reduced, shift: int32): int32 {.inline.} =
  ## The same for an int64 `x`, a sum of products of two stored values.
  ## TFLite narrows the result to an int32 by its low 32 bits, without
  ## saturating, so this does too.
  let total = 15'i64 - int64(shift)
  cast[int32](uint32((x * int64(reduced) + (1'i64 shl (total - 1))) shr total and 0xFFFF_FFFF'i64))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# INT16 TANH
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TFLite's int16 tanh, which is all integer: tanh(x) = 2 sigmoid(2x) - 1, from
# a table of sigmoid(i / 24) in 0.16 fixed point, interpolated linearly. No
# formula found regenerates the table, so it is TFLite's own.

const sigmoidTableU16: array[256, uint16] = [
  32768'u16, 33451, 34133, 34813, 35493, 36169, 36843, 37513, 38180, 38841, 39498, 40149,
  40794, 41432, 42064, 42688, 43304, 43912, 44511, 45102, 45683, 46255, 46817, 47369,
  47911, 48443, 48964, 49475, 49975, 50464, 50942, 51409, 51865, 52311, 52745, 53169,
  53581, 53983, 54374, 54755, 55125, 55485, 55834, 56174, 56503, 56823, 57133, 57433,
  57724, 58007, 58280, 58544, 58800, 59048, 59288, 59519, 59743, 59959, 60168, 60370,
  60565, 60753, 60935, 61110, 61279, 61441, 61599, 61750, 61896, 62036, 62172, 62302,
  62428, 62549, 62666, 62778, 62886, 62990, 63090, 63186, 63279, 63368, 63454, 63536,
  63615, 63691, 63765, 63835, 63903, 63968, 64030, 64090, 64148, 64204, 64257, 64308,
  64357, 64405, 64450, 64494, 64536, 64576, 64614, 64652, 64687, 64721, 64754, 64786,
  64816, 64845, 64873, 64900, 64926, 64950, 64974, 64997, 65019, 65039, 65060, 65079,
  65097, 65115, 65132, 65149, 65164, 65179, 65194, 65208, 65221, 65234, 65246, 65258,
  65269, 65280, 65291, 65301, 65310, 65319, 65328, 65337, 65345, 65352, 65360, 65367,
  65374, 65381, 65387, 65393, 65399, 65404, 65410, 65415, 65420, 65425, 65429, 65433,
  65438, 65442, 65445, 65449, 65453, 65456, 65459, 65462, 65465, 65468, 65471, 65474,
  65476, 65479, 65481, 65483, 65485, 65488, 65489, 65491, 65493, 65495, 65497, 65498,
  65500, 65501, 65503, 65504, 65505, 65507, 65508, 65509, 65510, 65511, 65512, 65513,
  65514, 65515, 65516, 65517, 65517, 65518, 65519, 65520, 65520, 65521, 65522, 65522,
  65523, 65523, 65524, 65524, 65525, 65525, 65526, 65526, 65526, 65527, 65527, 65528,
  65528, 65528, 65529, 65529, 65529, 65529, 65530, 65530, 65530, 65530, 65531, 65531,
  65531, 65531, 65531, 65532, 65532, 65532, 65532, 65532, 65532, 65533, 65533, 65533,
  65533, 65533, 65533, 65533, 65533, 65534, 65534, 65534, 65534, 65534, 65534, 65534,
  65534, 65534, 65534, 65535
]
  ## `sigmoid_table_uint16` from TensorFlow Lite's
  ## tensorflow/lite/kernels/internal/common.h, copyright The TensorFlow
  ## Authors, under the Apache License, Version 2.0.

func tanhInt16*(x: int16, inputMultiplier, inputLeftShift: int32): int16 =
  ## TFLite's reference int16 tanh, for an output scale of 2^-15. The host
  ## works out `inputMultiplier` and `inputLeftShift` from the input scale, as
  ## TFLite's TanhPrepare does; a multiplier of 0 means a power-of-two scale.
  var mult = inputMultiplier
  var leftShift = inputLeftShift
  if mult == 0:
    mult = 3'i32 shl leftShift
    leftShift = 0
  let round = if leftShift > 0: 1'i32 shl (leftShift - 1) else: 0'i32
  let input = (int32(x) * mult + round) shr leftShift
  let magnitude = uint32(abs(input))
  let hi = magnitude shr 8
  var r: int32
  if hi >= 255:
    r = 0xFFFF'i32 shl 8
  else:
    let a = uint32(sigmoidTableU16[hi])
    let b = uint32(sigmoidTableU16[hi + 1])
    r = int32((a shl 8) + (magnitude and 0xFF) * (b - a))
  r = if input >= 0: r - (1'i32 shl 23) + (1'i32 shl 7)
      else: -r + (1'i32 shl 23) + (1'i32 shl 7) - 1
  int16(r shr 8)

func logisticInt16*(x: int16, inputMultiplier, inputLeftShift: int32): int16 =
  ## TFLite's reference int16 logistic, for an output scale of 2^-15: the
  ## same table as `tanhInt16`, read at a quarter of the slope
  var mult = inputMultiplier
  var leftShift = inputLeftShift
  if mult == 0:
    mult = 3'i32 shl leftShift
    leftShift = 0
  let round = if leftShift > 0: 1'i32 shl (leftShift - 1) else: 0'i32
  let input = (int32(x) * mult + round) shr leftShift
  let magnitude = uint32(abs(input))
  let hi = magnitude shr 9
  var r: uint32
  if hi >= 255:
    r = 0x7FFF'u32 shl 10
  else:
    let a = uint32(sigmoidTableU16[hi])
    let b = uint32(sigmoidTableU16[hi + 1])
    r = (a shl 9) + (magnitude and 0x1FF) * (b - a)
  r = if input >= 0: r + (1'u32 shl 9) else: (1'u32 shl 25) - r + (1'u32 shl 9) - 1
  cast[int16](uint16((r shr 10) and 0xFFFF))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# INT16 TABLES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TFLite's other way of evaluating a function of a 16-bit value: 513 entries,
# the top 9 bits of the input choosing two and the low 7 interpolating between
# them. The host builds the table as TFLite does, from the function itself;
# see `steadyc/codec.nim`.

const Lut16Size* = 513

func lutLookup16*(value: int16, lut: ptr UncheckedArray[int16]): int16 {.inline.} =
  ## TFLite's LUTLookup for int16
  let index = 256 + (int(value) shr 7)
  let offset = int32(value) and 0x7f
  let base = int32(lut[index])
  # TFLite holds the slope in an int16 and the result is narrowed the same
  # way, by its low bits; a steep table, 1/sqrt near 0, does wrap
  let slope = int32(cast[int16](uint16((int32(lut[index + 1]) - base) and 0xFFFF)))
  cast[int16](uint16((base + ((slope * offset + 64) shr 7)) and 0xFFFF))

when hostOS == "any": {.pop.}
