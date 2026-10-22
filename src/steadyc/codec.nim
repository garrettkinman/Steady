# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

## Host-side numeric codecs and lookup-table generation.
##
## The host and the target need *different* things from a numeric format. The
## target needs `mac`, `finish`, `zero` — arithmetic, and as little of it as
## possible. The host needs a codec:
##
##     decode: raw storage bits -> float64
##     encode: float64          -> raw storage bits
##
## and nothing else. That asymmetry is worth keeping in mind if a format is
## ever added: the host side may use libm and float64 freely, and the target
## side stays a handful of templates that use neither.
##
## What the codec buys is this module's real product: **non-linear activations
## become tables**. For a storage type with 256 values the host can evaluate
## the true function at every representable input and quantize the result, so
## the device does a load instead of arithmetic. No libm, no software float,
## no fixed-point series expansion, no accuracy argument — the table *is* the
## function, rounded once.
##
## This works for any 8-bit storage type and for no others, which is why
## `hasLutDomain` gates it and `validate` rejects LUT ops for a policy whose
## store is wider, rather than the kernels quietly reaching for `expf`.

import std/[math, strformat]
import ./ir

type CodecError* = object of CatchableError

const LutSize* = 256
  ## One entry per raw storage byte. The index is the byte pattern, not the
  ## numeric value — see `lutIndex` in the runtime's policy module, which must
  ## agree with `decodeStore` below.

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CODECS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

proc decodeStore*(p: PolicyKind, q: Quant, raw: int): float64 =
  ## The real number that storage byte `raw` stands for.
  case p
  of pkAffineI8:
    if q.scales.len == 0:
      raise newException(CodecError, "affine decode needs a quantization scale")
    let v = int32(cast[int8](uint8(raw)))
    let zero = if q.zeroPoints.len == 0: 0'i32 else: q.zeroPoints[0]
    q.scales[0] * float64(v - zero)
  of pkAffineI16x8:
    raise newException(CodecError, "a 16-bit store has no byte to decode")

proc encodeStore*(p: PolicyKind, q: Quant, v: float64): int =
  ## The storage byte that best represents `v`. Saturating, never wrapping —
  ## a clipped activation is a small error, a wrapped one is a wrong answer.
  case p
  of pkAffineI8:
    if q.scales.len == 0:
      raise newException(CodecError, "affine encode needs a quantization scale")
    if v != v:
      raise newException(CodecError, "cannot encode NaN in an affine format")
    let zero = if q.zeroPoints.len == 0: 0'i32 else: q.zeroPoints[0]
    var qv = int64(round(v / q.scales[0])) + int64(zero)
    if qv < -128: qv = -128
    if qv > 127: qv = 127
    int(cast[uint8](int8(qv)))
  of pkAffineI16x8:
    raise newException(CodecError, "a 16-bit store has no byte to encode")

proc quantizedValue*(p: PolicyKind, q: Quant, v: float64): int32 =
  ## The *numeric* quantized value of `v`, as opposed to its storage byte.
  ## For affine int8 that is the signed value a kernel would see, which is
  ## what pad values and clamp bounds are expressed in.
  case p
  of pkAffineI8: int32(cast[int8](uint8(encodeStore(p, q, v))))
  of pkAffineI16x8:
    let zero = if q.zeroPoints.len == 0: 0'i64 else: int64(q.zeroPoints[0])
    int32(clamp(int64(round(v / q.scales[0])) + zero, -32768'i64, 32767'i64))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ACTIVATION TABLES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

proc activationFn*(kind: OpKind): proc (x: float64): float64 {.nimcall.} =
  ## The mathematical function, in double precision, with no quantization
  ## anywhere near it. Evaluated on the build machine exactly 256 times.
  case kind
  of okLogistic: (proc (x: float64): float64 = 1.0 / (1.0 + exp(-x)))
  of okTanh:     (proc (x: float64): float64 = tanh(x))
  else: raise newException(CodecError, &"op kind {kind} is not a table activation")

proc buildLut*(p: PolicyKind, inQ, outQ: Quant,
               f: proc (x: float64): float64 {.nimcall.}): seq[byte] =
  ## A 256-entry table, indexed by input storage byte, holding output storage
  ## bytes. Both quantizations are baked in, so the table is specific to this
  ## op in this graph — which is the entire point of an ahead-of-time
  ## compiler.
  if not p.hasLutDomain:
    raise newException(CodecError,
      &"policy {p.policyName} stores {p.storeType}, which has no 256-value " &
      "domain to tabulate")
  result = newSeq[byte](LutSize)
  for raw in 0 ..< LutSize:
    let x = decodeStore(p, inQ, raw)
    # An fp8 NaN input has no sensible function value; propagate it rather
    # than folding it to a finite number the model would then trust.
    result[raw] =
      if x != x: byte(encodeStore(p, outQ, NaN))
      else: byte(encodeStore(p, outQ, f(x)))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SOFTMAX EXP TABLE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

proc expTableBits*(classes: int): int =
  ## Fractional bits for the exp table, chosen so that summing one entry per
  ## class cannot overflow an int32: `classes * 2^bits <= 2^30`.
  ##
  ## Deriving this from the class count rather than fixing it means a
  ## ten-class model gets a table 2^12 times finer than a worst-case bound
  ## would allow, for free, at compile time.
  var headroom = 0
  while (1 shl headroom) < classes: inc headroom
  result = 30 - headroom
  if result > 26: result = 26          # beyond this the extra bits buy nothing
  if result < 8:
    raise newException(CodecError,
      &"{classes} classes leaves only {result} fractional bits for the exp table")

proc buildExpLut*(inScale: float64, classes: int): tuple[table: seq[int32], bits: int] =
  ## `table[d] = exp(-d * inScale) * 2^bits`, where `d` is the quantized
  ## difference `max - x` and therefore an index in `0 ..< 256`.
  ##
  ## Entry 0 is exactly `2^bits`, so the runtime's total is always at least
  ## that — the normalising divide cannot see a zero denominator, whatever the
  ## input. Far entries round down to zero, which is the correct answer: a
  ## class that many LSBs below the maximum has a probability far under one
  ## part in 256.
  if inScale <= 0.0:
    raise newException(CodecError, &"softmax input scale must be positive, got {inScale}")
  let bits = expTableBits(classes)
  var t = newSeq[int32](LutSize)
  let unit = float64(1 shl bits)
  for d in 0 ..< LutSize:
    t[d] = int32(round(exp(-float64(d) * inScale) * unit))
  (t, bits)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# INT16 TABLES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# A 16-bit store has too many values to tabulate each, so TFLite tabulates 513
# and interpolates (`lutLookup16` in the contract). Its tables are made from
# the function, not shipped, so Steady makes them the same way: TFLite's
# LUTPopulateInt16, step for step, in the precision TFLite uses for that op,
# float32 for most and float64 where it keeps that for compatibility.

proc lutPopulateInt16*[F: float32 | float64](inScale: F, inZero: int32,
                                              outScale: F, outZero: int32,
                                              f: proc (x: F): F): seq[int16] =
  let inputMin = inScale * F(-32768'i32 - inZero)
  let inputMax = inScale * F(32767'i32 - inZero)
  let outputMin = outScale * F(-32768'i32 - outZero)
  let outputMax = outScale * F(32767'i32 - outZero)
  const steps = 512
  let step = (inputMax - inputMin) / F(steps)
  let halfStep = step / F(2)
  let scaling = F(65536) / (outputMax - outputMin)
  result = newSeq[int16](steps + 1)
  for i in 0 ..< steps:
    let v = f(inputMin + F(i) * step)
    let midpoint = f(inputMin + F(i) * step + halfStep)
    let next = f(inputMin + F(i + 1) * step)
    let sample = round(v * scaling)
    let interpolated = round((next * scaling + round(v * scaling)) / F(2))
    let bias = round((interpolated - round(midpoint * scaling)) / F(2))
    result[i] = int16(min(max(sample - bias, F(-32768)), F(32767)))
  result[steps] = int16(min(max(round(f(inputMax) * scaling), F(-32768)), F(32767)))

proc rsqrtLut16*(inScale, outScale: float32): seq[int16] =
  ## TFLite's table for int16 RSQRT, in float32. At 0 and below it gives the
  ## store's largest value, as TFLite's does.
  lutPopulateInt16[float32](inScale, 0, outScale, 0, proc (x: float32): float32 =
    if x <= 0'f32: 32767'f32 * outScale else: 1'f32 / sqrt(x))

proc softmaxLuts16*(): tuple[exp, reciprocal: seq[int16]] =
  ## TFLite's two tables for int16 softmax, in float64: exp over [-10, 0] and
  ## 1 / (1 + x) over [0, 1], both onto [-1, 1]. They depend on nothing in the
  ## model, so every softmax shares them.
  const range = 65535.0
  (lutPopulateInt16[float64](10.0 / range, 32767, 2.0 / range, 0,
                             proc (x: float64): float64 = exp(x)),
   lutPopulateInt16[float64](1.0 / range, -32768, 2.0 / range, 0,
                             proc (x: float64): float64 = 1.0 / (1.0 + x)))
