# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

## Portable reference kernels.
##
## Every kernel here is:
##   * generic over a numeric policy, never over a shape — shapes are ordinary
##     runtime ints, so there is one instantiation per dtype rather than one
##     per layer. Shape *checking* is the host compiler's job, where errors
##     are readable.
##   * destination-passing — no value returns, no temporaries. This is what
##     makes arena allocation possible at all.
##   * free of zero-point semantics. The host folds input zero-point
##     correction into the int32 bias (legal because TFLite mandates
##     symmetric int8 weights, Zw = 0), so kernels see a plain product-sum.
##
## Padding deserves a note. Folding `-Zx * sum(w)` into the bias assumes
## *every* tap contributes, which stops being true at the edges if padded
## taps are skipped. Rather than special-case boundary pixels or push a zero
## point into the kernel, padded taps are multiplied against an explicit
## `padValue: Store` supplied by the host — Zx for affine, 0 for real
## formats. The folding stays valid everywhere, the kernel stays
## policy-agnostic, and the inner loop stays branch-free, which on a
## cacheless part is often faster than the bounds test it replaces.

import ../policy

export policy

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BLOCKING
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The matmul-family kernels compute several outputs at once. Two reasons, and
# the second is the one that shows up on a machine with no cache to speak of:
#
#   * **Loads.** A convolution's input patch is read once per output channel by
#     the obvious loop nest. Reading it once per *block* of output channels
#     divides activation loads by the block size, and reduction-heavy layers on
#     small parts are load-bound long before they are multiply-bound.
#   * **Dependency chains.** One accumulator per output means one `mac` cannot
#     start until the last one retires. Several independent accumulators in
#     flight let a pipelined core overlap them, which is most of the difference
#     between a 3-multiply-deep loop and a saturated multiplier.
#
# What is *not* done here is splitting one output's reduction into partial sums
# and adding them at the end. That is the other standard way to break a
# dependency chain, and it re-associates the addition — exact for a wrapping
# integer accumulator, not exact for float32 or fp8, and not exact for a posit
# quire's rounding either. Every transform below leaves each individual
# accumulator seeing exactly the taps it saw before, in exactly the order it saw
# them, so bit-exactness against TFLite holds by construction rather than by
# retesting. The differential harness confirms it on the six real models it can
# compare, agreement statistics unchanged to the digit.
#
# Blocks are `static` template parameters rather than runtime values, so each
# instantiation has constant loop bounds and the accumulator array can live in
# registers. A runtime block count would put it back in memory and cost more
# than the reuse gains.

const
  OcBlock = 4
    ## Output channels per convolution block.
  DwBlock = 4
    ## Channels per depthwise block.
  FcBlock = 4
    ## Rows of a fully-connected weight matrix per block, sharing one pass over
    ## the input vector.
    ##
    ## All three are four because four measured fastest, not because four is a
    ## natural number: two was about 10% worse and eight was worse than two.
    ## Eight is worth understanding, because it is the width a vector unit
    ## would want — with eight accumulators plus eight weights and eight
    ## activations live at once, the compiler spilled seven of the eight
    ## accumulators to the stack, and the reuse did not pay for the traffic.
    ## So the right width is a property of the target's register file, and
    ## these are the first constants to re-tune for a different one. They are
    ## three constants rather than one because the kernels want them for
    ## different reasons, and a machine with SIMD will not want the same number
    ## in all three places.

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FULLY CONNECTED
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

proc fullyConnected*[P, S, B, PT](
    _: typedesc[P],
    y: ptr UncheckedArray[S],
    x: ptr UncheckedArray[S],
    w: ptr UncheckedArray[S],
    bias: ptr UncheckedArray[B],
    prm: PT,
    outDim, inDim: int) =
  ## y[outDim] = finish(W[outDim, inDim] * x[inDim] + bias)
  ## Weights are row-major over [outDim, inDim].
  ##
  ## Rows are computed `FcBlock` at a time. The input vector is read once for
  ## the whole block instead of once per row, which is the entire trick: a
  ## fully-connected layer reads its weights exactly once no matter what, so
  ## activation loads are the only thing left to save.
  static:
    assert S is Store(P), "fullyConnected: storage type mismatch"
    assert B is Bias(P), "fullyConnected: bias type mismatch"
    assert PT is Params(P), "fullyConnected: params type mismatch"

  template rows(Blk: static int, o: int) =
    var acc: array[Blk, Accum(P)]
    for k in 0 ..< Blk: acc[k] = zeroAccum(P)
    for i in 0 ..< inDim:
      let v = x[i]
      for k in 0 ..< Blk:
        mac(P, acc[k], w[(o + k) * inDim + i], v)
    for k in 0 ..< Blk:
      addBias(P, acc[k], bias[o + k])
      y[o + k] = finish(P, acc[k], prm, o + k)

  var o = 0
  while o + FcBlock <= outDim:
    rows(FcBlock, o)
    o += FcBlock
  while o < outDim:
    rows(1, o)
    inc o

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CONV2D  (NHWC activations, OHWI filters, batch 1)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

proc conv2d*[P, S, B, PT](
    _: typedesc[P],
    y: ptr UncheckedArray[S],
    x: ptr UncheckedArray[S],
    w: ptr UncheckedArray[S],
    bias: ptr UncheckedArray[B],
    prm: PT,
    inH, inW, inC: int,
    outH, outW, outC: int,
    kH, kW: int,
    strideH, strideW: int,
    padTop, padLeft: int,
    dilationH, dilationW: int,
    padValue: S) =
  ## Output channels are computed `OcBlock` at a time, so one pass over an
  ## input patch feeds four filters rather than being repeated for each.
  ##
  ## A stride-1-or-more 1x1 convolution with no padding gets its own path.
  ## That is not a micro-optimization on this class of model: in a
  ## MobileNet-class network the overwhelming majority of multiplies are in
  ## pointwise convolutions, and for them the "patch" is a single contiguous
  ## run of `inC` values. The general path's per-tap work — deriving `iy` and
  ## `ix`, testing them against the input bounds, re-deriving a weight base —
  ## is then pure overhead wrapped around a reduction that may be only eight
  ## elements long. The fast path deletes all of it and leaves one loop.
  ##
  ## The 1x1 path needs no bounds test because it cannot go out of bounds: with
  ## no padding the host's own shape rule gives `(outH - 1) * strideH <= inH - 1`,
  ## so every sampled position is inside by construction. Shapes are validated
  ## on the host precisely so kernels can rely on them.
  static:
    assert S is Store(P), "conv2d: storage type mismatch"
    assert B is Bias(P), "conv2d: bias type mismatch"
    assert PT is Params(P), "conv2d: params type mismatch"

  let filterSize = kH * kW * inC

  template filters(Blk: static int, oc: int, pointwise: static bool) =
    ## `Blk` filters over one output pixel. Each accumulator sees the same taps
    ## in the same order as the unblocked kernel: ky, then kx, then ic.
    var acc: array[Blk, Accum(P)]
    for k in 0 ..< Blk: acc[k] = zeroAccum(P)

    when pointwise:
      let xBase = (inYOrigin * inW + inXOrigin) * inC
      for ic in 0 ..< inC:
        let v = x[xBase + ic]
        for k in 0 ..< Blk:
          mac(P, acc[k], w[(oc + k) * filterSize + ic], v)
    else:
      for ky in 0 ..< kH:
        let iy = inYOrigin + ky * dilationH
        let rowInside = iy >= 0 and iy < inH
        for kx in 0 ..< kW:
          let ix = inXOrigin + kx * dilationW
          let wBase = (ky * kW + kx) * inC
          if rowInside and ix >= 0 and ix < inW:
            let xBase = (iy * inW + ix) * inC
            for ic in 0 ..< inC:
              let v = x[xBase + ic]
              for k in 0 ..< Blk:
                mac(P, acc[k], w[(oc + k) * filterSize + wBase + ic], v)
          else:
            for ic in 0 ..< inC:
              for k in 0 ..< Blk:
                mac(P, acc[k], w[(oc + k) * filterSize + wBase + ic], padValue)

    for k in 0 ..< Blk:
      addBias(P, acc[k], bias[oc + k])
      y[yBase + oc + k] = finish(P, acc[k], prm, oc + k)

  let pointwise = kH == 1 and kW == 1 and padTop == 0 and padLeft == 0

  for oy in 0 ..< outH:
    let inYOrigin = oy * strideH - padTop
    for ox in 0 ..< outW:
      let inXOrigin = ox * strideW - padLeft
      let yBase = (oy * outW + ox) * outC
      var oc = 0
      if pointwise:
        while oc + OcBlock <= outC:
          filters(OcBlock, oc, true)
          oc += OcBlock
        while oc < outC:
          filters(1, oc, true)
          inc oc
      else:
        while oc + OcBlock <= outC:
          filters(OcBlock, oc, false)
          oc += OcBlock
        while oc < outC:
          filters(1, oc, false)
          inc oc

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DEPTHWISE CONV2D  (NHWC activations, 1HWC filters)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

proc depthwiseConv2d*[P, S, B, PT](
    _: typedesc[P],
    y: ptr UncheckedArray[S],
    x: ptr UncheckedArray[S],
    w: ptr UncheckedArray[S],
    bias: ptr UncheckedArray[B],
    prm: PT,
    inH, inW, inC: int,
    outH, outW: int,
    kH, kW: int,
    depthMultiplier: int,
    strideH, strideW: int,
    padTop, padLeft: int,
    dilationH, dilationW: int,
    padValue: S) =
  ## Output channel count is inC * depthMultiplier. Filter layout is
  ## [1, kH, kW, outC], matching TFLite.
  ##
  ## Channels are the blocked axis, and for this operator that is a statement
  ## about memory rather than about arithmetic. A depthwise convolution has no
  ## channel reduction: every output is nine multiplies over nine values that
  ## are `inC` elements apart in NHWC. Walking the kernel window per channel —
  ## the obvious nest — therefore touches nine cache lines, uses one byte of
  ## each, and does it again for the next channel.
  ##
  ## Blocking the channel axis turns that inside out. For a block of
  ## `DwBlock` channels the nine activation loads become nine *contiguous*
  ## runs, and the nine weight loads likewise, because the filter's channel
  ## axis is its innermost one. Same multiplies, a `DwBlock`-th of the touched
  ## lines, and the inner loop is a shape a vector unit could take over
  ## unchanged.
  ##
  ## The block requires `oc` and its input channel to advance together, which
  ## holds exactly when `depthMultiplier` is 1 — the only value real converters
  ## emit. A larger multiplier falls back to one channel at a time, where
  ## `oc div depthMultiplier` is the right input channel and the arithmetic
  ## costs nothing because it happens once per block rather than once per tap.
  static:
    assert S is Store(P), "depthwiseConv2d: storage type mismatch"
    assert B is Bias(P), "depthwiseConv2d: bias type mismatch"
    assert PT is Params(P), "depthwiseConv2d: params type mismatch"

  let outC = inC * depthMultiplier

  template channels(Blk: static int, oc: int) =
    ## `Blk` channels over one output pixel, each accumulator seeing its taps
    ## in the original ky-then-kx order.
    var acc: array[Blk, Accum(P)]
    for k in 0 ..< Blk: acc[k] = zeroAccum(P)
    let ic0 = oc div depthMultiplier      # == oc whenever Blk > 1

    for ky in 0 ..< kH:
      let iy = inYOrigin + ky * dilationH
      let rowInside = iy >= 0 and iy < inH
      for kx in 0 ..< kW:
        let ix = inXOrigin + kx * dilationW
        let wBase = (ky * kW + kx) * outC + oc
        if rowInside and ix >= 0 and ix < inW:
          let xBase = (iy * inW + ix) * inC + ic0
          for k in 0 ..< Blk:
            mac(P, acc[k], w[wBase + k], x[xBase + k])
        else:
          for k in 0 ..< Blk:
            mac(P, acc[k], w[wBase + k], padValue)

    for k in 0 ..< Blk:
      addBias(P, acc[k], bias[oc + k])
      y[yBase + oc + k] = finish(P, acc[k], prm, oc + k)

  let blocked = depthMultiplier == 1

  for oy in 0 ..< outH:
    let inYOrigin = oy * strideH - padTop
    for ox in 0 ..< outW:
      let inXOrigin = ox * strideW - padLeft
      let yBase = (oy * outW + ox) * outC
      var oc = 0
      if blocked:
        while oc + DwBlock <= outC:
          channels(DwBlock, oc)
          oc += DwBlock
      while oc < outC:
        channels(1, oc)
        inc oc

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# POOLING
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Pooling preserves quantization, so there is no requantize step — only the
# fused activation clamp, expressed directly in Store units. Padded taps are
# genuinely skipped here: there is no bias folding to invalidate.

proc maxPool2d*[P, S](
    _: typedesc[P],
    y: ptr UncheckedArray[S],
    x: ptr UncheckedArray[S],
    inH, inW, channels: int,
    outH, outW: int,
    kH, kW: int,
    strideH, strideW: int,
    padTop, padLeft: int,
    actMin, actMax: S) =
  static:
    assert S is Store(P), "maxPool2d: storage type mismatch"

  for oy in 0 ..< outH:
    let inYOrigin = oy * strideH - padTop
    let yStart = max(0, -inYOrigin)
    let yEnd = min(kH, inH - inYOrigin)
    for ox in 0 ..< outW:
      let inXOrigin = ox * strideW - padLeft
      let xStart = max(0, -inXOrigin)
      let xEnd = min(kW, inW - inXOrigin)
      for c in 0 ..< channels:
        var best = lowestStore(P)
        for ky in yStart ..< yEnd:
          for kx in xStart ..< xEnd:
            let v = x[((inYOrigin + ky) * inW + (inXOrigin + kx)) * channels + c]
            if best < v: best = v
        if best < actMin: best = actMin
        if actMax < best: best = actMax
        y[(oy * outW + ox) * channels + c] = best

proc avgPool2d*[P, S](
    _: typedesc[P],
    y: ptr UncheckedArray[S],
    x: ptr UncheckedArray[S],
    inH, inW, channels: int,
    outH, outW: int,
    kH, kW: int,
    strideH, strideW: int,
    padTop, padLeft: int,
    actMin, actMax: S) =
  ## Divides by the count of *valid* taps, not the window area — this is
  ## what TFLite does, and getting it wrong shows up only at the borders.
  static:
    assert S is Store(P), "avgPool2d: storage type mismatch"

  for oy in 0 ..< outH:
    let inYOrigin = oy * strideH - padTop
    let yStart = max(0, -inYOrigin)
    let yEnd = min(kH, inH - inYOrigin)
    for ox in 0 ..< outW:
      let inXOrigin = ox * strideW - padLeft
      let xStart = max(0, -inXOrigin)
      let xEnd = min(kW, inW - inXOrigin)
      let count = (yEnd - yStart) * (xEnd - xStart)
      for c in 0 ..< channels:
        var acc = zeroAccum(P)
        for ky in yStart ..< yEnd:
          for kx in xStart ..< xEnd:
            accumulate(P, acc, x[((inYOrigin + ky) * inW + (inXOrigin + kx)) * channels + c])
        var v = storeOf(P, divAccum(P, acc, count))
        if v < actMin: v = actMin
        if actMax < v: v = actMax
        y[(oy * outW + ox) * channels + c] = v

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ELEMENTWISE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

proc clamp1d*[P, S](
    _: typedesc[P],
    y: ptr UncheckedArray[S],
    x: ptr UncheckedArray[S],
    n: int,
    lo, hi: S) =
  ## Standalone activation (ReLU, ReLU6, hard clamp). `lo`/`hi` are in Store
  ## units and computed by the host, so this needs no quantization knowledge
  ## at all — it relies only on `<` over Store agreeing with real ordering.
  ## Safe to run in-place (y may alias x).
  static:
    assert S is Store(P), "clamp1d: storage type mismatch"
  for i in 0 ..< n:
    var v = x[i]
    if v < lo: v = lo
    if hi < v: v = hi
    y[i] = v

proc addElementwise*[P, S, PT](
    _: typedesc[P],
    y: ptr UncheckedArray[S],
    a: ptr UncheckedArray[S],
    b: ptr UncheckedArray[S],
    prm: PT,
    n: int,
    aMult, aShift, bMult, bShift: int32,
    aZero, bZero: int32) =
  ## Two-input add. Unlike matmul-family ops this genuinely needs zero points
  ## at runtime: the two inputs carry different scales, so each must be
  ## rescaled into a common domain before summing, and there is no per-channel
  ## constant to fold them into.
  ##
  ## For real policies the rescale parameters are ignored entirely (the
  ## host emits zeros) and this reduces to a plain elementwise sum.
  static:
    assert S is Store(P), "addElementwise: storage type mismatch"
    assert PT is Params(P), "addElementwise: params type mismatch"
  for i in 0 ..< n:
    var acc = zeroAccum(P)
    addRescaled(P, acc, a[i], aMult, aShift, aZero)
    addRescaled(P, acc, b[i], bMult, bShift, bZero)
    y[i] = finish(P, acc, prm, 0)

proc copy1d*[P, S](
    _: typedesc[P],
    y: ptr UncheckedArray[S],
    x: ptr UncheckedArray[S],
    n: int) =
  static:
    assert S is Store(P), "copy1d: storage type mismatch"
  for i in 0 ..< n:
    y[i] = x[i]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PAD  (NHWC, spatial only, batch 1)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

proc pad2d*[P, S](
    _: typedesc[P],
    y: ptr UncheckedArray[S],
    x: ptr UncheckedArray[S],
    inH, inW, channels: int,
    padTop, padBottom, padLeft, padRight: int,
    padValue: S) =
  ## Writes the output in address order: whole padded rows, then per row a
  ## left border, the input run, and a right border. The output is written
  ## exactly once and read never, so there is no interior bounds test.
  ##
  ## `padValue` is in Store units and comes from the host, which quantizes the
  ## op's real pad constant — for affine int8 that is the zero point when the
  ## constant is 0.0, which is what TFLite's PAD means and a place it is easy
  ## to accidentally write an integer zero instead.
  static:
    assert S is Store(P), "pad2d: storage type mismatch"

  let outW = padLeft + inW + padRight
  let rowLen = outW * channels
  let runLen = inW * channels
  var o = 0
  for _ in 0 ..< padTop * rowLen:
    y[o] = padValue
    inc o
  for iy in 0 ..< inH:
    for _ in 0 ..< padLeft * channels:
      y[o] = padValue
      inc o
    let xBase = iy * runLen
    for i in 0 ..< runLen:
      y[o] = x[xBase + i]
      inc o
    for _ in 0 ..< padRight * channels:
      y[o] = padValue
      inc o
  for _ in 0 ..< padBottom * rowLen:
    y[o] = padValue
    inc o

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CONCATENATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# There is no whole-op concatenation kernel. Concatenation is a set of writes
# into disjoint slices of one output buffer, and the host knows every slice
# offset at compile time, so it emits one call per operand and no operand ever
# learns that the others exist.
#
# `outer` is the product of the dimensions before the concat axis — identical
# for every operand, since those dimensions must agree. `innerIn` is the
# product from the axis onward for this operand, `innerOut` the same for the
# output. Both degenerate to a single flat copy when the axis is the outermost
# non-trivial one.

proc concatSlice*[P, S](
    _: typedesc[P],
    y: ptr UncheckedArray[S],
    x: ptr UncheckedArray[S],
    outer, innerIn, innerOut, dstOffset: int) =
  ## Verbatim copy, emitted when the operand already carries the output's
  ## quantization — the overwhelmingly common case, and exact.
  static:
    assert S is Store(P), "concatSlice: storage type mismatch"
  for o in 0 ..< outer:
    let src = o * innerIn
    let dst = o * innerOut + dstOffset
    for i in 0 ..< innerIn:
      y[dst + i] = x[src + i]

proc concatSliceRescaled*[P, S, PT](
    _: typedesc[P],
    y: ptr UncheckedArray[S],
    x: ptr UncheckedArray[S],
    prm: PT,
    outer, innerIn, innerOut, dstOffset: int,
    mult, shift, offset: int32) =
  ## Same copy, for an operand whose quantization differs from the output's.
  ## This is deliberately the *add* kernel's rescale path with one operand
  ## instead of two: same primitive, same two-stage multiplier, already
  ## covered by the same tests. For real policies the rescale parameters are
  ## ignored and this reduces to the copy above.
  static:
    assert S is Store(P), "concatSliceRescaled: storage type mismatch"
    assert PT is Params(P), "concatSliceRescaled: params type mismatch"
  for o in 0 ..< outer:
    let src = o * innerIn
    let dst = o * innerOut + dstOffset
    for i in 0 ..< innerIn:
      var acc = zeroAccum(P)
      addRescaled(P, acc, x[src + i], mult, shift, offset)
      y[dst + i] = finish(P, acc, prm, 0)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MEAN  (reduction over H and W — global average)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

proc meanSpatial*[P, S, B, PT](
    _: typedesc[P],
    y: ptr UncheckedArray[S],
    x: ptr UncheckedArray[S],
    corr: B,
    prm: PT,
    inH, inW, channels: int) =
  ## y[c] = finish(mean over all pixels of x[.., c])
  ##
  ## `corr` carries the zero-point correction in accumulator units, exactly as
  ## a folded bias does: for affine int8 the host passes `-count * Zx`, so
  ## `divAccum` sees `sum(x - Zx)` and the kernel again never learns what a
  ## zero point is. For real policies it is zero and the whole op is a plain
  ## average.
  ##
  ## `meanScale` is where the division goes, and what it does depends on the
  ## policy: for affine it is the identity, because the host has folded
  ## `1/count` into the output multiplier so that the reduction rounds exactly
  ## once; for a real format, which has no multiplier to fold into, it divides.
  ## One kernel, a mean under every policy, and no rounding of the accumulator
  ## before `finish` sees it — which is worth a full LSB on a global average
  ## pool, as MobileNetV2 demonstrated.
  ##
  ## Channels are the outer loop because the alternative needs one accumulator
  ## per channel, and a variable-length accumulator array is precisely the
  ## dynamic allocation this runtime does not do.
  static:
    assert S is Store(P), "meanSpatial: storage type mismatch"
    assert B is Bias(P), "meanSpatial: correction type mismatch"
    assert PT is Params(P), "meanSpatial: params type mismatch"

  let count = inH * inW
  for c in 0 ..< channels:
    var acc = zeroAccum(P)
    for i in 0 ..< count:
      accumulate(P, acc, x[i * channels + c])
    addBias(P, acc, corr)
    y[c] = finish(P, meanScale(P, acc, count), prm, 0)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# LOOKUP-TABLE ACTIVATIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

proc lut1d*[P, S](
    _: typedesc[P],
    y: ptr UncheckedArray[S],
    x: ptr UncheckedArray[S],
    n: int,
    table: ptr UncheckedArray[S]) =
  ## Every non-linear activation on an 8-bit storage type, for the price of
  ## one load: the host evaluated the true function at each of the 256
  ## representable inputs and quantized the result. Exact by construction,
  ## constant-time, and it needs no arithmetic — no libm, no software float, no
  ## fixed-point series expansion.
  ##
  ## Safe to run in place. Only instantiates for policies that define
  ## `lutIndex`, which is the compile-time expression of "this storage type
  ## has 256 values".
  static:
    assert S is Store(P), "lut1d: storage type mismatch"
  for i in 0 ..< n:
    y[i] = table[lutIndex(P, x[i])]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SOFTMAX
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

proc softmax*[P, S](
    _: typedesc[P],
    y: ptr UncheckedArray[S],
    x: ptr UncheckedArray[S],
    rows, classes: int,
    expLut: ptr UncheckedArray[int32],
    outMult, outShift, outZeroPoint, actMin, actMax: int32) =
  ## Softmax over the last axis, from a host-generated exp table.
  ##
  ## `rows` is the product of every dimension before the last, so one call
  ## covers both a classifier's single vector and a detector's grid of
  ## per-cell distributions — FOMO's output is a 12x12 grid of them. Each row
  ## normalises independently; the table is shared, since it depends only on
  ## the input scale.
  ##
  ## The table is keyed on the *difference* `max - x[i]`, which for a uniform
  ## integer store is itself an index in `0 ..< 256`. That is what makes the
  ## table possible: subtracting the max is required for numerical range, and
  ## only a uniform integer domain keeps the subtracted value enumerable. A
  ## non-uniform format (fp8, posits) has no such index, so this op is affine
  ## only and the host rejects it elsewhere rather than the kernel branching.
  ##
  ## Entry `d` holds `exp(-d * inScale) * 2^expBits`, with `expBits` chosen by
  ## the host from the class count so the sum cannot overflow an int32. The
  ## normalising divide is done once per element in Q15; the 64-bit numerator
  ## is the only wide arithmetic on the target, and softmax runs once per
  ## inference over a handful of classes.
  ##
  ## Safe to run in place: the max and the total are complete before anything
  ## is written, and the final pass reads `x[i]` only to write `y[i]`.
  static:
    assert S is Store(P), "softmax: storage type mismatch"
    assert P is AffineI8,
      "softmax needs a uniform integer store domain; the host rejects this " &
      "op for real-number policies rather than approximating it here"

  for r in 0 ..< rows:
    let base = r * classes

    var mx = int(x[base])
    for i in 1 ..< classes:
      let v = int(x[base + i])
      if v > mx: mx = v

    var total = 0'i32
    for i in 0 ..< classes:
      total += expLut[mx - int(x[base + i])]

    for i in 0 ..< classes:
      let e = expLut[mx - int(x[base + i])]
      let p = int32((int64(e) shl 15) div int64(total))
      var q = multiplyByQuantizedMultiplier(p, outMult, outShift) + outZeroPoint
      if q < actMin: q = actMin
      if q > actMax: q = actMax
      y[base + i] = S(q)
