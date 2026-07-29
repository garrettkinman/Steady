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
  let (lo, hi) = affineActRange(fused, outT.quant)
  max(lo, min(hi, v))

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
  else:
    raise newException(SimError,
      &"constant '{tt.name}' of dtype {tt.dtype} is not simulatable")

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
            var acc = 0'i32
            for ky in 0 ..< op.kH:
              let iy = oy * op.strideH - op.padTop + ky * op.dilationH
              for kx in 0 ..< op.kW:
                let ix = ox * op.strideW - op.padLeft + kx * op.dilationW
                if iy < 0 or iy >= inH or ix < 0 or ix >= inW:
                  continue                       # padded taps contribute zero
                for ic in 0 ..< inC:
                  let wv = int32(w[((oc * op.kH + ky) * op.kW + kx) * inC + ic])
                  acc += wv * (x[(iy * inW + ix) * inC + ic] - inZero)
            acc += bias[oc]
            y[(oy * outW + ox) * outC + oc] =
              requant(acc, inScale, wT.quant.scaleAt(oc), outScale,
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

    of okAdd:
      let aT = g.tensors[op.inputs[0]]
      let bT = g.tensors[op.inputs[1]]
      let a = result[op.inputs[0]]
      let b = result[op.inputs[1]]
      let (_, rescale) = resolveAddParams(g, aT, bT, outT, op.fused)
      let (om, osh) = quantizeMultiplier(
        (2.0 * max(aT.quant.scaleAt(0), bT.quant.scaleAt(0))) /
        (float64(1 shl 20) * outScale))
      var y = newSeq[int32](outT.numElements)
      for i in 0 ..< outT.numElements:
        let av = multiplyByQuantizedMultiplier(
          (a[i] + rescale.aOffset) * (1'i32 shl 20), rescale.aMult, rescale.aShift)
        let bv = multiplyByQuantizedMultiplier(
          (b[i] + rescale.bOffset) * (1'i32 shl 20), rescale.bMult, rescale.bShift)
        y[i] = actClamp(multiplyByQuantizedMultiplier(av + bv, om, osh) + outZero,
                        op.fused, outT)
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
      let (inScale, inZero) = getQ(xT)
      let x = operand(g, result, op.inputs[0])
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
