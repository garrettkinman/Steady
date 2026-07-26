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
