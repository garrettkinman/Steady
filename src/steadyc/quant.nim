# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

## Host-side quantization resolution.
##
## Everything here happens at build time. Its job is to absorb the
## difference between affine-integer and real-number formats so that the
## kernels do not have to: it computes Q31 requantization multipliers, folds
## input zero-point correction into the int32 bias, and reduces fused
## activations to plain clamp bounds.
##
## The folding identity, for symmetric weights (Zw = 0, which TFLite
## mandates for int8 conv and fully-connected weights):
##
##     acc_true = sum_k w_k * (x_k - Zx)
##              = sum_k w_k * x_k  -  Zx * sum_k w_k
##                ^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^^
##                what the kernel     constant per output channel,
##                computes            folded into the bias here
##
## which is why the kernel never sees a zero point.

import std/[math, strformat]
import ./ir

type
  ResolvedParams* = object
    ## What the emitter needs to materialise a `Params(P)` value.
    case affine*: bool
    of true:
      mult*, shift*: seq[int32]     ## one entry (per-tensor) or one per channel
      outZeroPoint*: int32
      qActMin*, qActMax*: int32
    of false:
      fActMin*, fActMax*: float32

  AddRescale* = object
    aMult*, aShift*, bMult*, bShift*: int32
    aOffset*, bOffset*: int32       ## TFLite input_offset, i.e. negated zero point

  QuantError* = object of CatchableError

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TFLITE MULTIPLIER ENCODING
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

proc quantizeMultiplier*(m: float64): tuple[mult: int32, shift: int32] =
  ## TFLite QuantizeMultiplier: express `m` as `mult * 2^(shift-31)` with
  ## `mult` in [2^30, 2^31). Must match TFLite exactly — this value is
  ## baked into flash and any drift here is a systematic error.
  if m == 0.0:
    return (0'i32, 0'i32)
  if m < 0.0:
    raise newException(QuantError, &"negative effective scale {m}")

  let (frac, e) = frexp(m)          # m == frac * 2^e, frac in [0.5, 1)
  var shift = int32(e)
  var qFixed = int64(round(frac * float64(1'i64 shl 31)))
  doAssert qFixed <= (1'i64 shl 31)

  if qFixed == (1'i64 shl 31):      # frac rounded up to 1.0
    qFixed = qFixed div 2
    inc shift

  if shift < -31:                   # underflows to zero
    return (0'i32, 0'i32)
  if shift > 30:                    # saturates
    return (high(int32), 30'i32)
  (int32(qFixed), shift)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TYPED VIEWS OVER RAW TENSOR DATA
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

proc asInt8*(t: Tensor): seq[int8] =
  if t.data.len != t.numElements:
    raise newException(QuantError,
      &"tensor '{t.name}': {t.data.len} bytes for {t.numElements} int8 elements")
  result = newSeq[int8](t.numElements)
  for i in 0 ..< t.numElements: result[i] = cast[int8](t.data[i])

proc asInt32*(t: Tensor): seq[int32] =
  if t.data.len != t.numElements * 4:
    raise newException(QuantError,
      &"tensor '{t.name}': {t.data.len} bytes for {t.numElements} int32 elements")
  result = newSeq[int32](t.numElements)
  for i in 0 ..< t.numElements:
    var v: int32
    copyMem(addr v, unsafeAddr t.data[i * 4], 4)
    result[i] = v

proc fromInt32*(xs: seq[int32]): seq[byte] =
  result = newSeq[byte](xs.len * 4)
  for i, v in xs:
    var tmp = v
    copyMem(addr result[i * 4], addr tmp, 4)

proc scaleAt*(q: Quant, ch: int): float64 =
  if q.scales.len == 0:
    raise newException(QuantError, "missing quantization scale")
  if q.scales.len == 1: q.scales[0] else: q.scales[ch]

proc zeroAt*(q: Quant, ch: int): int32 =
  if q.zeroPoints.len == 0: 0'i32
  elif q.zeroPoints.len == 1: q.zeroPoints[0]
  else: q.zeroPoints[ch]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FUSED ACTIVATION -> CLAMP BOUNDS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

proc quantizeValue(v: float64, scale: float64, zero: int32): int32 =
  let q = int32(round(v / scale)) + zero
  max(-128'i32, min(127'i32, q))

proc affineActRange*(fused: FusedAct, outQ: Quant): (int32, int32) =
  let s = outQ.scaleAt(0)
  let z = outQ.zeroAt(0)
  case fused
  of faNone:       (-128'i32, 127'i32)
  of faRelu:       (max(-128'i32, z), 127'i32)
  of faRelu6:      (max(-128'i32, z), quantizeValue(6.0, s, z))
  of faReluN1To1:  (quantizeValue(-1.0, s, z), quantizeValue(1.0, s, z))

proc realActRange*(fused: FusedAct): (float32, float32) =
  case fused
  of faNone:       (-Inf.float32, Inf.float32)
  of faRelu:       (0'f32, Inf.float32)
  of faRelu6:      (0'f32, 6'f32)
  of faReluN1To1:  (-1'f32, 1'f32)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PARAMETER RESOLUTION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

proc resolveMatmulParams*(g: Graph, inT, wT, outT: Tensor,
                          fused: FusedAct, outChannels: int): ResolvedParams =
  ## Requantization for the matmul family: conv, depthwise, fully-connected.
  if not g.policy.isAffine:
    let (lo, hi) = realActRange(fused)
    return ResolvedParams(affine: false, fActMin: lo, fActMax: hi)

  let inScale = inT.quant.scaleAt(0)
  let outScale = outT.quant.scaleAt(0)
  let perChannel = wT.quant.isPerChannel
  let n = if perChannel: outChannels else: 1

  result = ResolvedParams(affine: true)
  result.mult = newSeq[int32](n)
  result.shift = newSeq[int32](n)
  for c in 0 ..< n:
    let effective = (inScale * wT.quant.scaleAt(c)) / outScale
    let (m, s) = quantizeMultiplier(effective)
    result.mult[c] = m
    result.shift[c] = s
  result.outZeroPoint = outT.quant.zeroAt(0)
  let (lo, hi) = affineActRange(fused, outT.quant)
  result.qActMin = lo
  result.qActMax = hi

proc resolveAddParams*(g: Graph, aT, bT, outT: Tensor,
                       fused: FusedAct): (ResolvedParams, AddRescale) =
  ## Elementwise add genuinely needs runtime zero points: the two operands
  ## carry different scales and there is no per-channel constant to fold
  ## them into. TFLite pre-shifts both operands left by 20 to preserve
  ## precision, so each input multiplier is taken relative to a common
  ## domain 2^20 times finer than the larger input scale.
  if not g.policy.isAffine:
    let (lo, hi) = realActRange(fused)
    return (ResolvedParams(affine: false, fActMin: lo, fActMax: hi),
            AddRescale())

  const LeftShift = 20
  let aScale = aT.quant.scaleAt(0)
  let bScale = bT.quant.scaleAt(0)
  let outScale = outT.quant.scaleAt(0)
  let twiceMax = 2.0 * max(aScale, bScale)
  let realA = aScale / twiceMax
  let realB = bScale / twiceMax
  let realOut = twiceMax / (float64(1 shl LeftShift) * outScale)

  let (am, ash) = quantizeMultiplier(realA)
  let (bm, bsh) = quantizeMultiplier(realB)
  let (om, osh) = quantizeMultiplier(realOut)

  var params = ResolvedParams(affine: true)
  params.mult = @[om]
  params.shift = @[osh]
  params.outZeroPoint = outT.quant.zeroAt(0)
  let (lo, hi) = affineActRange(fused, outT.quant)
  params.qActMin = lo
  params.qActMax = hi

  (params, AddRescale(aMult: am, aShift: ash, bMult: bm, bShift: bsh,
                      aOffset: -aT.quant.zeroAt(0),
                      bOffset: -bT.quant.zeroAt(0)))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BIAS FOLDING
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

proc foldBias*(g: Graph, op: Op): seq[byte] =
  ## Returns replacement bias data with the input zero-point correction
  ## folded in. A no-op for real-number policies, which have no zero points.
  if not g.policy.isAffine:
    return g.tensors[op.inputs[2]].data

  let inT = g.tensors[op.inputs[0]]
  let wT = g.tensors[op.inputs[1]]
  let bT = g.tensors[op.inputs[2]]
  let zx = inT.quant.zeroAt(0)
  let w = wT.asInt8
  var bias = bT.asInt32

  case op.kind
  of okFullyConnected:
    let outDim = wT.shape[0]
    let inDim = wT.shape[1]
    for o in 0 ..< outDim:
      var s = 0'i32
      for i in 0 ..< inDim:
        s += int32(w[o * inDim + i])
      bias[o] -= zx * s

  of okConv2d:
    let outC = wT.shape[0]
    let taps = wT.shape[1] * wT.shape[2] * wT.shape[3]
    for o in 0 ..< outC:
      var s = 0'i32
      for k in 0 ..< taps:
        s += int32(w[o * taps + k])
      bias[o] -= zx * s

  of okDepthwiseConv2d:
    # Filter layout is [1, kH, kW, outC]; a channel's taps are strided.
    let outC = wT.shape[3]
    let window = wT.shape[1] * wT.shape[2]
    for o in 0 ..< outC:
      var s = 0'i32
      for k in 0 ..< window:
        s += int32(w[k * outC + o])
      bias[o] -= zx * s

  else:
    raise newException(QuantError, &"foldBias: op kind {op.kind} has no bias")

  fromInt32(bias)

proc padValueFor*(g: Graph, inT: Tensor): int32 =
  ## The value padded taps are multiplied against. For affine this is the
  ## input zero point, which is exactly what keeps the folded bias valid at
  ## the edges; for real formats it is plain zero.
  if g.policy.isAffine: inT.quant.zeroAt(0) else: 0'i32
