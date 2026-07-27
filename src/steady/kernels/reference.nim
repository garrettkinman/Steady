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
  static:
    assert S is Store(P), "fullyConnected: storage type mismatch"
    assert B is Bias(P), "fullyConnected: bias type mismatch"
    assert PT is Params(P), "fullyConnected: params type mismatch"

  for o in 0 ..< outDim:
    var acc = zeroAccum(P)
    let base = o * inDim
    for i in 0 ..< inDim:
      mac(P, acc, w[base + i], x[i])
    addBias(P, acc, bias[o])
    y[o] = finish(P, acc, prm, o)

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
  static:
    assert S is Store(P), "conv2d: storage type mismatch"
    assert B is Bias(P), "conv2d: bias type mismatch"
    assert PT is Params(P), "conv2d: params type mismatch"

  for oy in 0 ..< outH:
    let inYOrigin = oy * strideH - padTop
    for ox in 0 ..< outW:
      let inXOrigin = ox * strideW - padLeft
      for oc in 0 ..< outC:
        var acc = zeroAccum(P)
        let wOC = oc * kH * kW * inC
        for ky in 0 ..< kH:
          let iy = inYOrigin + ky * dilationH
          let rowInside = iy >= 0 and iy < inH
          for kx in 0 ..< kW:
            let ix = inXOrigin + kx * dilationW
            let inside = rowInside and ix >= 0 and ix < inW
            let wBase = wOC + (ky * kW + kx) * inC
            if inside:
              let xBase = (iy * inW + ix) * inC
              for ic in 0 ..< inC:
                mac(P, acc, w[wBase + ic], x[xBase + ic])
            else:
              for ic in 0 ..< inC:
                mac(P, acc, w[wBase + ic], padValue)
        addBias(P, acc, bias[oc])
        y[(oy * outW + ox) * outC + oc] = finish(P, acc, prm, oc)

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
  static:
    assert S is Store(P), "depthwiseConv2d: storage type mismatch"
    assert B is Bias(P), "depthwiseConv2d: bias type mismatch"
    assert PT is Params(P), "depthwiseConv2d: params type mismatch"

  let outC = inC * depthMultiplier
  for oy in 0 ..< outH:
    let inYOrigin = oy * strideH - padTop
    for ox in 0 ..< outW:
      let inXOrigin = ox * strideW - padLeft
      for ic in 0 ..< inC:
        for m in 0 ..< depthMultiplier:
          let oc = ic * depthMultiplier + m
          var acc = zeroAccum(P)
          for ky in 0 ..< kH:
            let iy = inYOrigin + ky * dilationH
            let rowInside = iy >= 0 and iy < inH
            for kx in 0 ..< kW:
              let ix = inXOrigin + kx * dilationW
              let wIdx = (ky * kW + kx) * outC + oc
              if rowInside and ix >= 0 and ix < inW:
                mac(P, acc, w[wIdx], x[(iy * inW + ix) * inC + ic])
              else:
                mac(P, acc, w[wIdx], padValue)
          addBias(P, acc, bias[oc])
          y[(oy * outW + ox) * outC + oc] = finish(P, acc, prm, oc)

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
