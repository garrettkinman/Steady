# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

## Default implementations of the numeric policy contract.
##
## The contract itself — the policy tags, the associated types, the params,
## the block widths — lives in `contract.nim`. This module supplies the
## members: what `mac` does, what `finish` does, and so on, for each of the
## four policies that ship.
##
## The split exists so that an arithmetic backend can override those members.
## A backend has to name `RealP8` to overload on it, and the module it is
## overriding cannot import the module overriding it, so the tags sit one
## level below in `contract.nim` and both import that. Kernels do not call
## into here directly; they go through `kernels/arith.nim`, which selects
## between a backend's member and the default below at compile time. See that
## module for how to supply one.
##
## Every default here stays compiled and reachable whatever a backend does,
## which is what makes a backend differential-testable rather than merely a
## substitute.

import ./contract

export contract

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
# RealP8
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Every operation below except the two divisions is exact. `mac` is an exact
# product of exact decodes into a register that cannot lose a bit until it
# overflows, which takes 2^38 taps; `addBias` and `accumulate` are exact
# rescalings by a power of two. So the reduction a convolution performs is
# the real-valued reduction, and `finish` rounds it once.
#
# NaR does not survive arithmetic: `units` decodes it as zero, so a NaR input
# contributes nothing to a quire rather than poisoning it. Carrying NaR
# through would mean a flag bit in every accumulator, which is a real cost
# paid on every model for a value no kernel here can produce — there is no
# division and no square root in this operator set. NaR *is* preserved
# through `lut1d`, where it costs nothing, because the host tabulates it.

template zeroAccum*(_: typedesc[RealP8]): Quire = 0'i64

template mac*(_: typedesc[RealP8], acc: var Quire, a, b: Posit8) =
  ## Exact. `units` is the value in 2^-6, so the product is in 2^-12 — the
  ## quire's own units, with no shift and no rounding.
  acc = acc + int64(units(a)) * int64(units(b))

template addBias*(_: typedesc[RealP8], acc: var Quire, b: Posit8) =
  acc = acc + (int64(units(b)) shl UnitScale)

func finish*(_: typedesc[RealP8], acc: Quire, prm: RealParams[Quire],
             ch: int): Posit8 {.inline.} =
  var v = acc
  if v < prm.actMin: v = prm.actMin
  if v > prm.actMax: v = prm.actMax
  positFromQuire(v)

template lowestStore*(_: typedesc[RealP8]): Posit8 = Posit8Min
  ## -maxpos, not NaR. NaR does sort below it, so a NaR in a max-pool window
  ## loses rather than winning — see the note above.

template accumulate*(_: typedesc[RealP8], acc: var Quire, v: Posit8) =
  acc = acc + (int64(units(v)) shl UnitScale)

func divRoundEven(a: Quire, n: int): Quire {.inline.} =
  ## `a / n` on the quire grid, ties to even. The only inexact step in this
  ## policy, and it exists because a quire is fixed point: a mean has to land
  ## back on the 2^-12 grid before it can be rounded to a posit.
  ##
  ## That is a double rounding, and it is worth being precise about how small
  ## it is. The quire grid is 2^-12; the coarsest posit spacing this can
  ## precede is 2^-6, the finest 2^-6 as well — the whole format sits on a
  ## 2^-6 grid — so the intermediate rounding is at most half of 1/64 of a
  ## posit ulp. It can only change an answer when the exact quotient lands
  ## within that of a posit midpoint, and never by more than one ulp.
  let d = Quire(n)
  let neg = a < 0
  let m = if neg: -a else: a
  var q = m div d
  let r = (m - q * d) * 2
  if r > d or (r == d and (q and 1) == 1): inc q
  if neg: -q else: q

func divAccum*(_: typedesc[RealP8], acc: Quire, n: int): Quire {.inline.} =
  divRoundEven(acc, n)

template meanScale*(_: typedesc[RealP8], acc: Quire, count: int): Quire =
  ## A real format has no output multiplier to fold `1/count` into, so the
  ## division happens here — same as `RealF32` and `RealFp8`, on the quire.
  divRoundEven(acc, count)

func storeOf*(_: typedesc[RealP8], acc: Quire): Posit8 {.inline.} =
  positFromQuire(acc)

func addRescaled*(_: typedesc[RealP8], acc: var Quire, v: Posit8,
                  mult, shift, offset: int32) {.inline.} =
  ## Rescale parameters are meaningless for a real format; the host emits
  ## zeros. The sum itself is exact, so a two-input add rounds exactly once.
  acc = acc + (int64(units(v)) shl UnitScale)

template lutIndex*(_: typedesc[RealP8], v: Posit8): int =
  ## What the fp8 canary was rehearsing: a table keyed on the raw encoding,
  ## no arithmetic on the target at all.
  int(v.bits)

