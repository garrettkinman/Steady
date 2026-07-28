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
## and nothing else. That asymmetry is why a new format does not have to
## arrive all at once: a SoftPosit binding satisfies this module on a real OS
## with a real libm, and the target side stays a handful of templates.
##
## What the codec buys is this module's real product: **non-linear activations
## become tables**. For a storage type with 256 values the host can evaluate
## the true function at every representable input and quantize the result, so
## the device does a load instead of arithmetic. No libm, no software float,
## no fixed-point series expansion, no accuracy argument — the table *is* the
## function, rounded once.
##
## This works for any 8-bit storage type and for no others, which is why
## `hasLutDomain` gates it and `validate` rejects LUT ops under `RealF32`
## rather than the kernels quietly reaching for `expf`.

import std/[math, strformat]
import ./ir
import ../steady/[fp8, posit8]

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
  of pkRealFp8:
    float64(Fp8(uint8(raw)).toFloat32)
  of pkRealP8:
    Posit8(uint8(raw)).toFloat64
  of pkRealF32:
    raise newException(CodecError,
      "RealF32 has no enumerable storage domain; there is nothing to decode " &
      "byte-wise and no lookup table to build")

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
  of pkRealFp8:
    if v != v: return int(Fp8Nan.bits)
    int(toFp8(float32(v)).bits)
  of pkRealP8:
    # NaN maps to NaR, which is how a table activation keeps an
    # unrepresentable input unrepresentable instead of inventing a value.
    if v != v: return int(Posit8NaR.bits)
    int(toPosit8(v).bits)
  of pkRealF32:
    raise newException(CodecError,
      "RealF32 values are emitted as literals; there is no byte encoding step")

proc quantizedValue*(p: PolicyKind, q: Quant, v: float64): int32 =
  ## The *numeric* quantized value of `v`, as opposed to its storage byte.
  ## For affine int8 that is the signed value a kernel would see, which is
  ## what pad values and clamp bounds are expressed in.
  case p
  of pkAffineI8: int32(cast[int8](uint8(encodeStore(p, q, v))))
  else: raise newException(CodecError,
    &"quantizedValue is meaningful only for an integer-affine policy, not {p}")

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
