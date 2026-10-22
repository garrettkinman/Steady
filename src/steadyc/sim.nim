# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

## Host-side graph simulator — the verification reference.
##
## This deliberately evaluates the graph in the **unfolded** form: it
## subtracts the input zero point explicitly at every tap and uses the
## original, unmodified bias. The generated code does the opposite, relying
## on the compiler having folded `-Zx * sum(w)` into the bias ahead of time.
##
## Requiring the two to agree bit-for-bit is what proves the folding
## transformation correct, including at padded edges where it is easiest to
## get wrong. An independent implementation of the same optimisation would
## prove nothing; an implementation that skips the optimisation entirely
## proves quite a lot.
##
## Requantization arithmetic *is* shared with the runtime, so this harness
## does not validate it. That is covered separately by unit tests pinned to
## TFLite's published values.

import std/[strformat, tables, math]
import ./ir, ./quant, ./codec
import ../steady/policy

type
  SimError* = object of CatchableError
  Buffers* = Table[int, seq[int32]]
    ## Tensor index -> values, held widened to int32 regardless of dtype.

proc getQ(t: Tensor): (float64, int32) = (t.quant.scaleAt(0), t.quant.zeroAt(0))

proc actClamp(v: int32, fused: FusedAct, outT: Tensor): int32 =
  let (lo, hi) = affineActRange(fused, outT)
  max(lo, min(hi, v))

proc multiplyByQuantizedMultiplier64(x: int64, quantizedMultiplier, shift: int32): int32 =
  ## TFLite's int64 MultiplyByQuantizedMultiplier, which its 16x8 kernels
  ## requantize through, written out here from the full Q31 multiplier. The
  ## runtime instead takes a multiplier the host reduced, and an int32 sum.
  let reduced = if quantizedMultiplier < 0x7FFF0000'i32:
                  (int64(quantizedMultiplier) + (1'i64 shl 15)) shr 16
                else: 0x7FFF'i64
  let total = 15'i64 - int64(shift)
  int32((x * reduced + (1'i64 shl (total - 1))) shr total)

proc requant64(acc: int64, inScale, wScale, outScale: float64,
               fused: FusedAct, outT: Tensor): int32 =
  let (m, s) = quantizeMultiplier((inScale * wScale) / outScale)
  actClamp(multiplyByQuantizedMultiplier64(acc, m, s), fused, outT)

proc requant(acc: int32, inScale, wScale, outScale: float64,
             outZero: int32, fused: FusedAct, outT: Tensor): int32 =
  let (m, s) = quantizeMultiplier((inScale * wScale) / outScale)
  actClamp(multiplyByQuantizedMultiplier(acc, m, s) + outZero, fused, outT)

proc operand(g: Graph, bufs: Buffers, t: int): seq[int32] =
  ## Values of an op's input. Constants are read from the tensor itself;
  ## everything else was produced by an earlier op.
  let tt = g.tensors[t]
  if tt.kind != tkConst:
    if t notin bufs:
      raise newException(SimError, &"tensor '{tt.name}' has not been produced")
    return bufs[t]
  case tt.dtype
  of dtInt8:
    let v = tt.asInt8
    result = newSeq[int32](v.len)
    for i, e in v: result[i] = int32(e)
  of dtInt32:
    result = tt.asInt32
  of dtInt16:
    result = newSeq[int32](tt.numElements)
    for i in 0 ..< tt.numElements:
      var v: int16
      copyMem(addr v, unsafeAddr tt.data[i * 2], 2)
      result[i] = int32(v)
  else:
    raise newException(SimError,
      &"constant '{tt.name}' of dtype {tt.dtype} is not simulatable")

proc unravel(i: int, shape: seq[int]): seq[int] =
  ## The position of element `i` of a row-major tensor
  result = newSeq[int](shape.len)
  var r = i
  for d in countdown(shape.high, 0):
    result[d] = r mod shape[d]
    r = r div shape[d]

proc ravel(pos, shape: seq[int]): int =
  ## Element of `shape` at `pos`, aligned from the right and taking each
  ## dimension of 1 as broadcast: NumPy's rule, worked out here position by
  ## position where the kernels walk strides
  let skip = pos.len - shape.len
  for d in 0 ..< shape.len:
    result = result * shape[d] + (if shape[d] == 1: 0 else: pos[skip + d])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

proc simulate*(g: Graph, inputs: Table[int, seq[int32]]): Buffers =
  ## Runs the graph over int8-affine semantics. `inputs` maps graph input
  ## tensor indices to quantized values.
  result = initTable[int, seq[int32]]()
  for k, v in inputs: result[k] = v

  for t in g.inputs:
    if t notin result:
      raise newException(SimError, &"no data supplied for input '{g.tensors[t].name}'")

  for op in g.ops:
    let outT = g.tensors[op.outputs[0]]
    let (outScale, outZero) = getQ(outT)

    case op.kind
    of okFullyConnected:
      let xT = g.tensors[op.inputs[0]]
      let wT = g.tensors[op.inputs[1]]
      let bT = g.tensors[op.inputs[2]]
      let (inScale, inZero) = getQ(xT)
      let x = result[op.inputs[0]]
      let w = wT.asInt8
      let bias = bT.asInt32
      let outDim = wT.shape[0]
      let inDim = wT.shape[1]
      var y = newSeq[int32](outDim)
      for o in 0 ..< outDim:
        var acc = 0'i32
        for i in 0 ..< inDim:
          acc += int32(w[o * inDim + i]) * (x[i] - inZero)   # unfolded
        acc += bias[o]
        y[o] = requant(acc, inScale, wT.quant.scaleAt(o), outScale,
                       outZero, op.fused, outT)
      result[op.outputs[0]] = y

    of okConv2d:
      let xT = g.tensors[op.inputs[0]]
      let wT = g.tensors[op.inputs[1]]
      let (inScale, inZero) = getQ(xT)
      let x = result[op.inputs[0]]
      let w = wT.asInt8
      let bias = g.tensors[op.inputs[2]].asInt32
      let inH = xT.shape[1]
      let inW = xT.shape[2]
      let inC = xT.shape[3]
      let outH = outT.shape[1]
      let outW = outT.shape[2]
      let outC = outT.shape[3]
      var y = newSeq[int32](outT.numElements)
      for oy in 0 ..< outH:
        for ox in 0 ..< outW:
          for oc in 0 ..< outC:
            # int64, as TFLite's 16x8 kernel sums; int8 sums fit int32 anyway
            var acc = 0'i64
            for ky in 0 ..< op.kH:
              let iy = oy * op.strideH - op.padTop + ky * op.dilationH
              for kx in 0 ..< op.kW:
                let ix = ox * op.strideW - op.padLeft + kx * op.dilationW
                if iy < 0 or iy >= inH or ix < 0 or ix >= inW:
                  continue                       # padded taps contribute zero
                for ic in 0 ..< inC:
                  let wv = int64(w[((oc * op.kH + ky) * op.kW + kx) * inC + ic])
                  acc += wv * int64(x[(iy * inW + ix) * inC + ic] - inZero)
            acc += int64(bias[oc])
            y[(oy * outW + ox) * outC + oc] =
              if g.policy == pkAffineI16x8:
                requant64(acc, inScale, wT.quant.scaleAt(oc), outScale, op.fused, outT)
              else:
                requant(int32(acc), inScale, wT.quant.scaleAt(oc), outScale,
                        outZero, op.fused, outT)
      result[op.outputs[0]] = y

    of okDepthwiseConv2d:
      let xT = g.tensors[op.inputs[0]]
      let wT = g.tensors[op.inputs[1]]
      let (inScale, inZero) = getQ(xT)
      let x = result[op.inputs[0]]
      let w = wT.asInt8
      let bias = g.tensors[op.inputs[2]].asInt32
      let inH = xT.shape[1]
      let inW = xT.shape[2]
      let inC = xT.shape[3]
      let outH = outT.shape[1]
      let outW = outT.shape[2]
      let outC = outT.shape[3]
      var y = newSeq[int32](outT.numElements)
      for oy in 0 ..< outH:
        for ox in 0 ..< outW:
          for ic in 0 ..< inC:
            for m in 0 ..< op.depthMultiplier:
              let oc = ic * op.depthMultiplier + m
              var acc = 0'i32
              for ky in 0 ..< op.kH:
                let iy = oy * op.strideH - op.padTop + ky * op.dilationH
                for kx in 0 ..< op.kW:
                  let ix = ox * op.strideW - op.padLeft + kx * op.dilationW
                  if iy < 0 or iy >= inH or ix < 0 or ix >= inW:
                    continue
                  let wv = int32(w[(ky * op.kW + kx) * outC + oc])
                  acc += wv * (x[(iy * inW + ix) * inC + ic] - inZero)
              acc += bias[oc]
              y[(oy * outW + ox) * outC + oc] =
                requant(acc, inScale, wT.quant.scaleAt(oc), outScale,
                        outZero, op.fused, outT)
      result[op.outputs[0]] = y

    of okMaxPool2d, okAvgPool2d:
      let xT = g.tensors[op.inputs[0]]
      let x = result[op.inputs[0]]
      let inH = xT.shape[1]
      let inW = xT.shape[2]
      let ch = xT.shape[3]
      let outH = outT.shape[1]
      let outW = outT.shape[2]
      let (lo, hi) = affineActRange(op.fused, outT.quant)
      var y = newSeq[int32](outT.numElements)
      for oy in 0 ..< outH:
        for ox in 0 ..< outW:
          for c in 0 ..< ch:
            var acc = if op.kind == okMaxPool2d: low(int32) else: 0'i32
            var count = 0
            for ky in 0 ..< op.kH:
              let iy = oy * op.strideH - op.padTop + ky
              if iy < 0 or iy >= inH: continue
              for kx in 0 ..< op.kW:
                let ix = ox * op.strideW - op.padLeft + kx
                if ix < 0 or ix >= inW: continue
                let v = x[(iy * inW + ix) * ch + c]
                inc count
                if op.kind == okMaxPool2d: acc = max(acc, v)
                else: acc += v
            var v =
              if op.kind == okMaxPool2d: acc
              else: AffineI8.divAccum(acc, count)
            y[(oy * outW + ox) * ch + c] = max(lo, min(hi, v))
      result[op.outputs[0]] = y

    of okAdd, okSub:
      let aT = g.tensors[op.inputs[0]]
      let bT = g.tensors[op.inputs[1]]
      let a = operand(g, result, op.inputs[0])
      let b = operand(g, result, op.inputs[1])
      let (_, rescale) = resolveAddParams(g, aT, bT, outT, op.fused)
      let leftShift = addLeftShift(g.policy)
      let (om, osh) = quantizeMultiplier(
        (2.0 * max(aT.quant.scaleAt(0), bT.quant.scaleAt(0))) /
        (float64(1 shl leftShift) * outScale))
      var y = newSeq[int32](outT.numElements)
      let same = aT.shape == bT.shape
      for i in 0 ..< outT.numElements:
        let p = unravel(i, outT.shape)
        let (ia, ib) = if same: (i, i) else: (ravel(p, aT.shape), ravel(p, bT.shape))
        let av = multiplyByQuantizedMultiplier(
          (a[ia] + rescale.aOffset) * (1'i32 shl leftShift), rescale.aMult, rescale.aShift)
        let bv = multiplyByQuantizedMultiplier(
          (b[ib] + rescale.bOffset) * (1'i32 shl leftShift), rescale.bMult, rescale.bShift)
        let sum = if op.kind == okAdd: av + bv else: av - bv
        y[i] = actClamp(multiplyByQuantizedMultiplier(sum, om, osh) + outZero,
                        op.fused, outT)
      result[op.outputs[0]] = y

    of okMul:
      let aT = g.tensors[op.inputs[0]]
      let bT = g.tensors[op.inputs[1]]
      let a = operand(g, result, op.inputs[0])
      let b = operand(g, result, op.inputs[1])
      let (m, sh) = quantizeMultiplier(float64(float32(aT.quant.scaleAt(0)) *
        float32(bT.quant.scaleAt(0)) / float32(outScale)))
      var y = newSeq[int32](outT.numElements)
      for i in 0 ..< y.len:
        let p = unravel(i, outT.shape)
        y[i] = actClamp(multiplyByQuantizedMultiplier(
          a[ravel(p, aT.shape)] * b[ravel(p, bT.shape)], m, sh) + outZero, op.fused, outT)
      result[op.outputs[0]] = y

    of okBatchMatmul:
      # Element by element from its position, in int64 as TFLite sums, and
      # requantized from the full Q31 multiplier
      let aT = g.tensors[op.inputs[0]]
      let bT = g.tensors[op.inputs[1]]
      let a = operand(g, result, op.inputs[0])
      let b = operand(g, result, op.inputs[1])
      let (m, sh) = quantizeMultiplier(float64(float32(aT.quant.scaleAt(0)) *
        float32(bT.quant.scaleAt(0))) / outScale)
      let depth = aT.shape[^1]
      var y = newSeq[int32](outT.numElements)
      for e in 0 ..< y.len:
        var p = unravel(e, outT.shape)
        let (i, j) = (p[^2], p[^1])
        var acc = 0'i64
        for k in 0 ..< depth:
          p[^2] = i
          p[^1] = k
          let av = a[ravel(p, aT.shape)]
          if op.adjY: (p[^2], p[^1]) = (j, k) else: (p[^2], p[^1]) = (k, j)
          acc += int64(av) * int64(b[ravel(p, bT.shape)])
        y[e] = actClamp(multiplyByQuantizedMultiplier64(acc, m, sh), faNone, outT)
      result[op.outputs[0]] = y

    of okTranspose:
      let xT = g.tensors[op.inputs[0]]
      let x = operand(g, result, op.inputs[0])
      var y = newSeq[int32](outT.numElements)
      for i in 0 ..< y.len:
        let p = unravel(i, outT.shape)
        var q = newSeq[int](p.len)
        for d, src in op.perm: q[src] = p[d]
        y[i] = x[ravel(q, xT.shape)]
      result[op.outputs[0]] = y

    of okGather:
      let table = operand(g, result, op.inputs[0])
      let ids = operand(g, result, op.inputs[1])
      let inner = g.tensors[op.inputs[0]].shape[1]
      var y = newSeq[int32]()
      for id in ids:
        y.add table[int(id) * inner ..< (int(id) + 1) * inner]
      result[op.outputs[0]] = y

    of okRequantize:
      let xT = g.tensors[op.inputs[0]]
      let x = operand(g, result, op.inputs[0])
      let (m, sh) = quantizeMultiplier(xT.quant.scaleAt(0) / outScale)
      var y = newSeq[int32](x.len)
      for i in 0 ..< x.len:
        y[i] = actClamp(multiplyByQuantizedMultiplier(x[i] - xT.quant.zeroAt(0), m, sh) +
                        outZero, faNone, outT)
      result[op.outputs[0]] = y

    of okAddBand:
      # The band's value at each position, then the ADD as above
      let aT = g.tensors[op.inputs[0]]
      let xT = g.tensors[op.inputs[1]]
      let a = operand(g, result, op.inputs[0])
      let x = operand(g, result, op.inputs[1])
      let (first, second) = if op.bandSide == 0: (xT, aT) else: (aT, xT)
      let (_, r) = resolveAddParams(g, first, second, outT, op.fused)
      let leftShift = addLeftShift(g.policy)
      let (om, osh) = quantizeMultiplier(
        (2.0 * max(aT.quant.scaleAt(0), xT.quant.scaleAt(0))) / (float64(1 shl leftShift) * outScale))
      var y = newSeq[int32](outT.numElements)
      for e in 0 ..< y.len:
        var p = unravel(e, outT.shape)
        let k = p[^1] - p[^2] + op.bandStart
        p[^1] = k
        let band = if k >= 0 and k < xT.shape[^1]: x[ravel(p, xT.shape)] else: xT.quant.zeroAt(0)
        let (u, v) = if op.bandSide == 0: (band, a[e]) else: (a[e], band)
        let su = multiplyByQuantizedMultiplier((u + r.aOffset) * (1'i32 shl leftShift), r.aMult, r.aShift)
        let sv = multiplyByQuantizedMultiplier((v + r.bOffset) * (1'i32 shl leftShift), r.bMult, r.bShift)
        y[e] = actClamp(multiplyByQuantizedMultiplier(su + sv, om, osh) + outZero, op.fused, outT)
      result[op.outputs[0]] = y

    of okBand, okUnband:
      # By position, from what each element of the five ops' result is
      let xT = g.tensors[op.inputs[0]]
      let x = operand(g, result, op.inputs[0])
      let n = xT.shape[^2]
      var y = newSeq[int32](outT.numElements)
      for e in 0 ..< y.len:
        var p = unravel(e, outT.shape)
        let k = if op.kind == okBand: p[^1] - p[^2] + op.bandStart else: p[^2] + p[^1] - op.bandStart
        let inside = k >= 0 and k < (if op.kind == okBand: xT.shape[^1] else: n)
        p[^1] = k
        y[e] = if inside: x[ravel(p, xT.shape)] else: xT.quant.zeroAt(0)
      result[op.outputs[0]] = y

    of okRsqrt:
      # TFLite's table, as the runtime's: it is shared, as the int16 tanh's is
      let x = operand(g, result, op.inputs[0])
      var lut = rsqrtLut16(float32(g.tensors[op.inputs[0]].quant.scaleAt(0)), float32(outScale))
      var y = newSeq[int32](x.len)
      for i in 0 ..< x.len:
        y[i] = int32(lutLookup16(int16(x[i]), cast[ptr UncheckedArray[int16]](addr lut[0])))
      result[op.outputs[0]] = y

    of okClamp:
      let x = result[op.inputs[0]]
      let (lo, hi) = affineActRange(op.fused, outT.quant)
      var y = newSeq[int32](x.len)
      for i in 0 ..< x.len: y[i] = max(lo, min(hi, x[i]))
      result[op.outputs[0]] = y

    of okReshape:
      result[op.outputs[0]] = result[op.inputs[0]]

    of okPad:
      let xT = g.tensors[op.inputs[0]]
      let x = operand(g, result, op.inputs[0])
      if g.policy == pkAffineI16x8:
        # Scattered from each input position, of any dimensions
        var y = newSeq[int32](outT.numElements)
        for i in 0 ..< y.len: y[i] = quantizedValue(g.policy, xT.quant, op.padConst)
        for i in 0 ..< x.len:
          var p = unravel(i, xT.shape)
          for d in 0 ..< p.len: p[d] += op.pads[d][0]
          y[ravel(p, outT.shape)] = x[i]
        result[op.outputs[0]] = y
        continue
      let inH = xT.shape[1]
      let inW = xT.shape[2]
      let ch = xT.shape[3]
      let padTop = op.pads[1][0]
      let padLeft = op.pads[2][0]
      let outH = outT.shape[1]
      let outW = outT.shape[2]
      let fill = quantizedValue(pkAffineI8, xT.quant, op.padConst)
      var y = newSeq[int32](outT.numElements)
      for i in 0 ..< y.len: y[i] = fill
      # Scatter, rather than the kernel's gather — same result if and only if
      # the offsets agree, which is the thing worth checking.
      for iy in 0 ..< inH:
        for ix in 0 ..< inW:
          for c in 0 ..< ch:
            y[((iy + padTop) * outW + (ix + padLeft)) * ch + c] =
              x[(iy * inW + ix) * ch + c]
      result[op.outputs[0]] = y

    of okConcatenation:
      var ins: seq[Tensor]
      for t in op.inputs: ins.add g.tensors[t]
      let ax = op.axis
      var outer = 1
      for d in 0 ..< ax: outer *= outT.shape[d]
      var innerOut = 1
      for d in ax ..< outT.shape.len: innerOut *= outT.shape[d]
      let (rp, rescales) = resolveConcatParams(g, ins, outT, op.fused)
      var y = newSeq[int32](outT.numElements)
      var dstOff = 0
      for k, ti in op.inputs:
        let t = g.tensors[ti]
        let x = operand(g, result, ti)
        var innerIn = 1
        for d in ax ..< t.shape.len: innerIn *= t.shape[d]
        let copyVerbatim = sameQuant(t.quant, outT.quant)
        for o in 0 ..< outer:
          for i in 0 ..< innerIn:
            let v = x[o * innerIn + i]
            y[o * innerOut + dstOff + i] =
              if copyVerbatim: v
              else:
                # The rescale arithmetic is shared with the runtime, as
                # requantization is; what this proves is the slice geometry.
                let r = rescales[k]
                let stage1 = multiplyByQuantizedMultiplier(
                  (v + r.offset) * (1'i32 shl RescaleLeftShift), r.mult, r.shift)
                actClamp(multiplyByQuantizedMultiplier(
                  stage1, rp.mult[0], rp.shift[0]) + outZero, op.fused, outT)
        dstOff += innerIn
      result[op.outputs[0]] = y

    of okMean:
      let xT = g.tensors[op.inputs[0]]
      let (inScale, inZero) = getQ(xT)
      let x = operand(g, result, op.inputs[0])
      if g.policy == pkAffineI16x8:
        # Each row summed, and TFLite's integer division in the multiplier
        let n = xT.shape[^1]
        let rp = resolveMeanLastParams(xT, outT, n)
        var y = newSeq[int32](x.len div n)
        for r in 0 ..< y.len:
          var acc = 0'i32
          for v in x[r * n ..< (r + 1) * n]: acc += v - inZero
          y[r] = actClamp(multiplyByQuantizedMultiplier(acc, rp.mult[0], rp.shift[0]) + outZero,
                          faNone, outT)
        result[op.outputs[0]] = y
        continue
      let ch = xT.shape[3]
      let count = xT.shape[1] * xT.shape[2]
      var y = newSeq[int32](ch)
      for c in 0 ..< ch:
        var acc = 0'i32
        for i in 0 ..< count:
          acc += x[i * ch + c] - inZero          # unfolded, tap by tap
        # `1/count` rides in the effective scale, exactly as the compiler folds
        # it; passing it as the "weight scale" is the same arithmetic the
        # matmul family does.
        y[c] = requant(acc, inScale, 1.0 / float64(count), outScale, outZero,
                       op.fused, outT)
      result[op.outputs[0]] = y

    of okLogistic, okTanh:
      # Evaluated from the true function at the tensor's own quantization,
      # indexed by *value*. The runtime indexes a table by storage *byte*;
      # requiring the two to agree is what checks the table's key encoding.
      let xT = g.tensors[op.inputs[0]]
      let x = operand(g, result, op.inputs[0])
      if g.policy == pkAffineI16x8:
        # TFLite's own integer tanh, the only one for 16 bits; its arithmetic
        # is shared with the runtime, as requantization is.
        let (mult, leftShift) = resolveTanhInt16Params(xT, op.kind == okLogistic)
        var y = newSeq[int32](x.len)
        for i in 0 ..< x.len:
          y[i] = int32(if op.kind == okLogistic: logisticInt16(int16(x[i]), mult, leftShift)
                       else: tanhInt16(int16(x[i]), mult, leftShift))
        result[op.outputs[0]] = y
        continue
      let (inScale, inZero) = getQ(xT)
      let f = activationFn(op.kind)
      var y = newSeq[int32](x.len)
      for i in 0 ..< x.len:
        y[i] = quantizedValue(pkAffineI8, outT.quant,
                              f(inScale * float64(x[i] - inZero)))
      result[op.outputs[0]] = y

    of okSoftmax:
      # The fixed-point normalisation is shared with the runtime, in the same
      # way requantization is; `tests/test_lut.nim` pins it against a float64
      # reference independently.
      let xT = g.tensors[op.inputs[0]]
      let x = operand(g, result, op.inputs[0])
      let classes = xT.shape[^1]
      if g.policy == pkAffineI16x8:
        # TFLite's SoftmaxInt16, as written there, from the tables it makes
        var (expLut, recipLut) = softmaxLuts16()
        let ep = cast[ptr UncheckedArray[int16]](addr expLut[0])
        let rcp = cast[ptr UncheckedArray[int16]](addr recipLut[0])
        let (m, sh) = resolveSoftmax16Params(xT)
        var y = newSeq[int32](x.len)
        for r in 0 ..< x.len div classes:
          let row = x[r * classes ..< (r + 1) * classes]
          let mx = max(row)
          var exps = newSeq[int32](classes)
          var sum = 0'i32
          for c, v in row:
            let d = clamp(multiplyByQuantizedMultiplier(v - mx, m, sh) + 32767, -32768'i32, 32767'i32)
            exps[c] = int32(lutLookup16(int16(d), ep))
            sum += exps[c]
          var headroom = 0'i32         # leading zeros, counted
          while (uint32(sum) shl headroom) < 0x8000_0000'u32: inc headroom
          let shifted = int32(((int64(sum) shl (headroom - 1)) + (1 shl 13)) shr 14)
          let recip = int64(lutLookup16(int16(clamp(shifted - 98304'i32, -32768'i32, 32767'i32)), rcp))
          let rs = 31 - headroom
          for c in 0 ..< classes:
            y[r * classes + c] = clamp(int32((int64(exps[c]) * recip + (1'i64 shl (rs - 1))) shr rs),
                                       0'i32, 32767'i32)
        result[op.outputs[0]] = y
        continue
      let rows = x.len div classes
      let (tbl, _) = buildExpLut(xT.quant.scaleAt(0), classes)
      let rp = resolveSoftmaxParams(g, outT)
      var y = newSeq[int32](x.len)
      for r in 0 ..< rows:
        let base = r * classes
        var mx = x[base]
        for i in 0 ..< classes:
          if x[base + i] > mx: mx = x[base + i]
        var total = 0'i32
        for i in 0 ..< classes: total += tbl[mx - x[base + i]]
        for i in 0 ..< classes:
          let p = int32((int64(tbl[mx - x[base + i]]) shl SoftmaxProbBits) div
                        int64(total))
          y[base + i] = actClamp(
            multiplyByQuantizedMultiplier(p, rp.mult[0], rp.shift[0]) + outZero,
            faNone, outT)
      result[op.outputs[0]] = y

    of okLeakyRelu:
      # TFLite's QuantizeLeakyRelu, element by element
      let x = operand(g, result, op.inputs[0])
      let lp = resolveLeakyReluParams(g.tensors[op.inputs[0]], outT, op.alpha)
      var y = newSeq[int32](x.len)
      for i in 0 ..< x.len:
        let v = x[i] - lp.inZero
        let r = if v >= 0: multiplyByQuantizedMultiplier(v, lp.identityMult, lp.identityShift)
                else: multiplyByQuantizedMultiplier(v, lp.alphaMult, lp.alphaShift)
        y[i] = actClamp(r + lp.outZero, faNone, outT)
      result[op.outputs[0]] = y

    of okSlice:
      # Indexed afresh from the four begins, where the kernel walks rows
      let xT = g.tensors[op.inputs[0]]
      let x = operand(g, result, op.inputs[0])
      let ys = outT.shape
      let xs = xT.shape
      var y = newSeq[int32](outT.numElements)
      if xs.len != 4 or op.begin[0] != 0:
        for i in 0 ..< y.len:
          var p = unravel(i, ys)
          for d in 0 ..< p.len: p[d] += op.begin[d]
          y[i] = x[ravel(p, xs)]
        result[op.outputs[0]] = y
        continue
      var o = 0
      for h in 0 ..< ys[1]:
        for w in 0 ..< ys[2]:
          for c in 0 ..< ys[3]:
            y[o] = x[((h + op.begin[1]) * xs[2] + w + op.begin[2]) * xs[3] + c + op.begin[3]]
            inc o
      result[op.outputs[0]] = y
