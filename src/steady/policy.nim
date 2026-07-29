# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

## Default implementations of the numeric policy contract.
##
## The contract itself — the policy tags, the associated types, the params,
## the block widths — lives in `contract.nim`. This module supplies the
## members: what `mac` does, what `finish` does, and so on.
##
## The split exists so that an arithmetic backend can override those members.
## A backend has to name `AffineI8` to overload on it, and the module it is
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
  ## A policy whose store is wider than a byte has no enumerable domain and
  ## simply does not define this, so `lut1d` fails to instantiate for it and
  ## the host rejects LUT ops rather than the kernels reaching for libm.
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
