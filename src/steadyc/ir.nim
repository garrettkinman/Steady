# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

## Host-side graph IR.
##
## This module runs on the build machine only. It may use `seq`, `string`
## and exceptions freely — none of it reaches the target, which sees only
## the straight-line code emitted from a planned graph.
##
## The IR is deliberately frontend-neutral. A TFLite importer populates it,
## and an ONNX importer could later populate the same structure; everything
## downstream (validation, folding, planning, emission) is written against
## the IR rather than against any file format.

import std/[math, strformat, strutils, tables, sequtils]
import ./symbolic
export symbolic

type
  DType* = enum
    dtInt8, dtInt16, dtInt32, dtFloat32

  PolicyKind* = enum
    ## Chosen by what a graph's activations are stored as. See
    ## `steady/contract.nim` for what the abstraction is for, what was removed
    ## from it, and why the second policy is there.
    pkAffineI8
    pkAffineI16x8   ## int16 activations, int8 weights

  TensorKind* = enum
    tkInput        ## supplied by the caller each invocation
    tkConst        ## weights and biases; lives in flash
    tkIntermediate ## arena-resident activation
    tkOutput       ## arena-resident, but live until the end of invoke

  Padding* = enum
    padValid, padSame

  FusedAct* = enum
    faNone, faRelu, faRelu6, faReluN1To1

  OpKind* = enum
    okFullyConnected, okConv2d, okDepthwiseConv2d,
    okMaxPool2d, okAvgPool2d, okAdd, okClamp, okReshape,
    okPad, okConcatenation, okMean,
    okLogistic, okTanh, okSoftmax,
    okLeakyRelu, okSlice,
    okSub, okMul, okRsqrt, okBatchMatmul, okTranspose, okGather, okRequantize,
    okBand, okUnband, okAddBand   ## From `steadyc/rewrite`, not from a file

  Quant* = object
    ## Per-tensor when `scales` has one entry, per-channel when it has one
    ## per slice along `axis`. Real-number policies leave this empty.
    scales*: seq[float64]
    zeroPoints*: seq[int32]
    axis*: int                 ## -1 for per-tensor

  Tensor* = object
    name*: string
    shape*: seq[int]           ## At the bound, for a model that takes a length
    dims*: seq[Dim]            ## The shape in terms of the length; empty when it is fixed
    dtype*: DType
    kind*: TensorKind
    quant*: Quant
    data*: seq[byte]           ## populated for tkConst only

  Op* = object
    name*: string
    kind*: OpKind
    inputs*: seq[int]
    outputs*: seq[int]
    padding*: Padding
    strideH*, strideW*: int
    dilationH*, dilationW*: int
    kH*, kW*: int
    depthMultiplier*: int
    fused*: FusedAct
    # okPad: `pads[d]` is (before, after) for dimension `d`, exactly the
    # layout TFLite's PAD paddings tensor uses. `padConst` is the *real*
    # value to pad with (TFLite PAD is 0.0, PADV2 supplies its own); the host
    # quantizes it, so a zero point never has to be spelled out by hand.
    pads*: seq[array[2, int]]
    padConst*: float64
    axis*: int                 ## okConcatenation
    alpha*: float64            ## okLeakyRelu: the slope below zero
    begin*: seq[int]           ## okSlice: the first index in each dimension
    potScaleInt16*: bool       ## okAdd: TFLite's flag for its power-of-two int16 path
    axes*: seq[int]            ## okMean, normalized by validate()
    keepDims*: bool            ## okMean
    perm*: seq[int]            ## okTranspose: output dimension d is input dimension perm[d]
    padDims*: seq[array[2, Dim]]  ## okPad: `pads` in terms of the length, when they depend on it
    beginDims*: seq[Dim]       ## okSlice: `begin` in terms of the length, when it depends on it
    bandStart*: int            ## okBand: c in x[i, j - i + c]; okUnband: a in p[i, i + k - a]
    bandSide*: int             ## okAddBand: which of the ADD's operands the band was, 0 or 1
    adjX*, adjY*: bool         ## okBatchMatmul: an operand's last two dimensions swapped
    # Filled in by validate(); explicit padding derived from `padding`.
    padTop*, padLeft*: int

  Graph* = object
    name*: string
    bound*: int                ## The longest length it takes at run time; 0 for a fixed model
    policy*: PolicyKind
    tensors*: seq[Tensor]
    ops*: seq[Op]
    inputs*, outputs*: seq[int]

  IrError* = object of CatchableError

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BASICS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

func byteWidth*(d: DType): int =
  case d
  of dtInt8: 1
  of dtInt16: 2
  of dtInt32, dtFloat32: 4

func nimTypeName*(d: DType): string =
  case d
  of dtInt8: "int8"
  of dtInt16: "int16"
  of dtInt32: "int32"
  of dtFloat32: "float32"

func cTypeName*(d: DType): string =
  ## Fixed-width types throughout — `int` is not 32 bits everywhere these
  ## models are going to run.
  case d
  of dtInt8: "int8_t"
  of dtInt16: "int16_t"
  of dtInt32: "int32_t"
  of dtFloat32: "float"

func policyName*(p: PolicyKind): string =
  case p
  of pkAffineI8: "AffineI8"
  of pkAffineI16x8: "AffineI16x8"

func storeType*(p: PolicyKind): DType =
  case p
  of pkAffineI8: dtInt8
  of pkAffineI16x8: dtInt16

func weightType*(p: PolicyKind): DType =
  ## int8 for both: 16x8 is int8 weights under int16 activations.
  dtInt8

func biasType*(p: PolicyKind): DType =
  ## Accumulator units, zero point pre-folded. TFLite's 16x8 biases are int64;
  ## the importer narrows them, refusing one that does not fit.
  dtInt32

func hasLutDomain*(p: PolicyKind): bool =
  ## True when the storage type's whole value domain is small enough for the
  ## host to enumerate — that is, 8 bits, 256 entries. Non-linear activations
  ## are lookup tables when this holds and a host-side error when it does not;
  ## see `steadyc/codec.nim`.
  p.storeType.byteWidth == 1

const MaxSoftmaxClasses* = 4096
  ## The softmax exp table is summed in an int32, and the host picks the
  ## table's fractional bits from the class count to guarantee that fits
  ## (see `buildExpLut`). Beyond this the table would be too coarse to be
  ## worth having.

func dimsOf*(t: Tensor): seq[Dim] =
  ## The shape in terms of the length, which for a fixed one is its numbers
  if t.dims.len == t.shape.len: t.dims else: t.shape.mapIt(dim(it))

func elements*(t: Tensor): Dim = product(t.dimsOf)
  ## How many, in terms of the length

func padsOf*(op: Op): seq[array[2, Dim]] =
  if op.padDims.len == op.pads.len: op.padDims
  else: op.pads.mapIt([dim(it[0]), dim(it[1])])

func beginOf*(op: Op): seq[Dim] =
  if op.beginDims.len == op.begin.len: op.beginDims else: op.begin.mapIt(dim(it))

func atLength*(g: Graph, n: int): Graph =
  ## The graph for one length: every shape and every attribute that depends
  ## on the length worked out for `n`, which is how it is validated and
  ## simulated
  result = g
  for t in result.tensors.mitems:
    if t.dims.len == t.shape.len: t.shape = t.dims.mapIt(it.at(n))
  for op in result.ops.mitems:
    if op.padDims.len == op.pads.len and op.pads.len > 0:
      op.pads = op.padDims.mapIt([it[0].at(n), it[1].at(n)])
    if op.beginDims.len == op.begin.len and op.begin.len > 0:
      op.begin = op.beginDims.mapIt(it.at(n))

func numElements*(t: Tensor): int =
  result = 1
  for d in t.shape: result *= d

func byteSize*(t: Tensor): int =
  t.numElements * t.dtype.byteWidth

func isPerChannel*(q: Quant): bool = q.axis >= 0 and q.scales.len > 1

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# COST MODEL
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# A single number per op, computed on the host, so the benchmark can report
# MAC/s rather than only cycles. Cycles say which op is slow; MAC/s says
# whether that is because it does more work or because the kernel is worse,
# and only the second is actionable.

func macCount*(g: Graph, op: Op): int =
  ## Multiply-accumulates one execution of `op` performs, counted the way the
  ## kernel actually performs them rather than the way the mathematics
  ## minimally requires.
  ##
  ## Padded taps are included, because they are multiplied against `padValue`
  ## rather than skipped — that is what keeps the folded bias valid at the
  ## edges. Counting the ideal number instead would credit a SAME-padded 3x3
  ## layer with work it does not do and quietly overstate its MAC/s.
  ##
  ## Zero for ops that move, compare or tabulate rather than multiply; their
  ## cost is in loads and stores, and `numElements` of the output is the
  ## honest measure there.
  let y = g.tensors[op.outputs[0]]
  case op.kind
  of okConv2d:
    let inC = g.tensors[op.inputs[0]].shape[3]
    y.numElements * op.kH * op.kW * inC
  of okDepthwiseConv2d:
    y.numElements * op.kH * op.kW
  of okFullyConnected:
    y.numElements * g.tensors[op.inputs[1]].shape[1]
  of okBatchMatmul:
    let a = g.tensors[op.inputs[0]]
    y.numElements * a.shape[if op.adjX: ^2 else: ^1]
  else:
    0

func totalMacs*(g: Graph): int =
  for op in g.ops: result += g.macCount(op)

func macDims*(g: Graph, op: Op): Dim =
  ## `macCount` in terms of the length
  let y = g.tensors[op.outputs[0]]
  case op.kind
  of okConv2d:
    y.elements * (op.kH * op.kW * g.tensors[op.inputs[0]].shape[3])
  of okDepthwiseConv2d:
    y.elements * (op.kH * op.kW)
  of okFullyConnected:
    y.elements * g.tensors[op.inputs[1]].shape[1]
  of okBatchMatmul:
    let a = g.tensors[op.inputs[0]].dimsOf
    y.elements * a[if op.adjX: ^2 else: ^1]
  else:
    dim(0)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BUILDER
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

proc initGraph*(name: string, policy: PolicyKind): Graph =
  Graph(name: name, policy: policy)

proc addTensor*(g: var Graph, t: Tensor): int =
  ## Returns the tensor's index, which is how ops refer to it.
  for i, existing in g.tensors:
    if existing.name == t.name:
      raise newException(IrError, &"duplicate tensor name '{t.name}'")
  g.tensors.add t
  g.tensors.high

proc addInput*(g: var Graph, name: string, shape: seq[int], dtype: DType,
               quant = Quant(axis: -1)): int =
  result = g.addTensor Tensor(name: name, shape: shape, dtype: dtype,
                              kind: tkInput, quant: quant)
  g.inputs.add result

proc addConst*(g: var Graph, name: string, shape: seq[int], dtype: DType,
               data: seq[byte], quant = Quant(axis: -1)): int =
  g.addTensor Tensor(name: name, shape: shape, dtype: dtype, kind: tkConst,
                     data: data, quant: quant)

proc addIntermediate*(g: var Graph, name: string, shape: seq[int], dtype: DType,
                      quant = Quant(axis: -1)): int =
  g.addTensor Tensor(name: name, shape: shape, dtype: dtype,
                     kind: tkIntermediate, quant: quant)

proc markOutput*(g: var Graph, idx: int) =
  g.tensors[idx].kind = tkOutput
  g.outputs.add idx

proc addOp*(g: var Graph, op: Op): int =
  g.ops.add op
  g.ops.high

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SHAPE INFERENCE AND VALIDATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This is where shape errors are supposed to surface. Kernels take runtime
# ints precisely so that this check can live here, on the host, where an
# error message can name the op and print both shapes — rather than in a
# generic instantiation failure.

func outputDim(inDim, k, stride, dilation: int, pad: Padding): int =
  let effK = dilation * (k - 1) + 1
  case pad
  of padValid: (inDim - effK + stride) div stride
  of padSame: (inDim + stride - 1) div stride

func padBefore(inDim, outDim, k, stride, dilation: int): int =
  let effK = dilation * (k - 1) + 1
  let needed = max(0, (outDim - 1) * stride + effK - inDim)
  needed div 2

proc check(cond: bool, opName: string, msg: string) =
  if not cond:
    raise newException(IrError, &"op '{opName}': {msg}")

func zeroAt0(q: Quant): int32 =
  if q.zeroPoints.len == 0: 0'i32 else: q.zeroPoints[0]

func sameQuant*(a, b: Quant): bool =
  ## Structural equality. Ops that move values without touching them require
  ## it; anything looser would silently reinterpret the data. The emitter uses
  ## it too, to decide when an op reduces to a plain copy.
  a.scales == b.scales and a.zeroPoints == b.zeroPoints

const MaxRank* = 4
  ## The kernels for tensors other than images take up to four dimensions,
  ## a shorter shape padded with leading 1s.

func pad4*(shape: seq[int], fill = 1): array[4, int] =
  ## `shape` with leading 1s to four dimensions, or with `fill`s for what
  ## is not a shape
  result = [fill, fill, fill, fill]
  for i, d in shape: result[4 - shape.len + i] = d

func strides4*(shape: seq[int]): array[4, int] =
  ## Row-major strides of `shape`, padded to four dimensions
  let d = pad4(shape)
  result[3] = 1
  for i in countdown(2, 0): result[i] = result[i + 1] * d[i + 1]

func broadcastShape*(a, b: seq[int]): seq[int] =
  ## NumPy's rule: dimensions aligned from the right, each equal or 1. Empty
  ## when they are not broadcastable.
  let n = max(a.len, b.len)
  for i in 0 ..< n:
    let da = if i < n - a.len: 1 else: a[i - (n - a.len)]
    let db = if i < n - b.len: 1 else: b[i - (n - b.len)]
    if da != db and da != 1 and db != 1: return @[]
    result.add max(da, db)

func broadcastStrides*(operand, output: seq[int]): array[4, int] =
  ## The strides to read `operand` at each of `output`'s positions, 0 along a
  ## dimension it is broadcast in
  let o = pad4(output)
  let d = pad4(operand)
  let s = strides4(operand)
  for i in 0 ..< 4:
    result[i] = if d[i] == 1 and o[i] != 1: 0 else: s[i]

func pad4*(shape: seq[Dim], fill = dim(1)): array[4, Dim] =
  ## The same for sizes in terms of the length
  result = [fill, fill, fill, fill]
  for i, d in shape: result[4 - shape.len + i] = d

func strides4*(shape: seq[Dim]): array[4, Dim] =
  let d = pad4(shape)
  result[3] = dim(1)
  for i in countdown(2, 0): result[i] = result[i + 1] * d[i + 1]

func broadcastStrides*(operand, output: seq[Dim]): array[4, Dim] =
  ## Broadcast along a dimension that is 1 in the operand, whatever the length
  let o = pad4(output)
  let d = pad4(operand)
  let s = strides4(operand)
  for i in 0 ..< 4:
    result[i] = if d[i] == dim(1) and o[i] != dim(1): dim(0) else: s[i]

proc validate*(g: var Graph) =
  ## Verifies shapes and dtypes, derives explicit padding, and confirms
  ## every tensor is produced before it is consumed.
  var produced = initTable[int, int]()   # tensor index -> producing op index
  for i, t in g.tensors:
    if t.kind in {tkInput, tkConst}:
      produced[i] = -1

  let storeDt = g.policy.storeType
  let biasDt = g.policy.biasType

  for oi in 0 ..< g.ops.len:
    template op: untyped = g.ops[oi]
    let nm = op.name

    for t in op.inputs:
      check t in produced, nm,
        &"input tensor '{g.tensors[t].name}' is consumed before it is produced"

    # The ops TFLite's reference kernels have been matched on for 16x8. The
    # others would compile, perhaps rightly; unchecked, they are refused.
    check g.policy != pkAffineI16x8 or
          op.kind in {okConv2d, okAdd, okReshape, okTanh, okLeakyRelu, okSlice,
                      okSub, okMul, okRsqrt, okBatchMatmul, okTranspose, okGather,
                      okRequantize, okPad, okMean, okSoftmax, okConcatenation,
                      okLogistic, okBand, okUnband, okAddBand}, nm,
      &"{op.kind} is not supported under {g.policy.policyName} yet"
    # And the ones only matched for 16x8 so far, the other way round
    check g.policy == pkAffineI16x8 or
          op.kind notin {okSub, okMul, okRsqrt, okBatchMatmul, okTranspose,
                         okGather, okRequantize, okBand, okUnband, okAddBand}, nm,
      &"{op.kind} is not supported under {g.policy.policyName} yet"

    case op.kind
    of okFullyConnected:
      check op.inputs.len == 3, nm, "expects [input, weights, bias]"
      let x = g.tensors[op.inputs[0]]
      let w = g.tensors[op.inputs[1]]
      let b = g.tensors[op.inputs[2]]
      let y = g.tensors[op.outputs[0]]
      check w.shape.len == 2, nm, &"weights must be rank 2, got {w.shape}"
      let inDim = w.shape[1]
      let outDim = w.shape[0]
      check x.numElements == inDim, nm,
        &"input has {x.numElements} elements but weights expect {inDim}"
      check b.numElements == outDim, nm,
        &"bias has {b.numElements} elements but output dim is {outDim}"
      check y.numElements == outDim, nm,
        &"output has {y.numElements} elements, expected {outDim}"
      check x.dtype == storeDt and y.dtype == storeDt, nm,
        "input/output dtype must match the graph policy storage type"
      check w.dtype == g.policy.weightType, nm,
        &"weights must be {g.policy.weightType}"
      check b.dtype == biasDt, nm, &"bias dtype must be {biasDt}"

    of okConv2d, okDepthwiseConv2d:
      check op.inputs.len == 3, nm, "expects [input, filter, bias]"
      let x = g.tensors[op.inputs[0]]
      let w = g.tensors[op.inputs[1]]
      check w.dtype == g.policy.weightType, nm,
        &"filter must be {g.policy.weightType}"
      check g.tensors[op.inputs[2]].dtype == biasDt, nm, &"bias dtype must be {biasDt}"
      check x.shape.len == 4, nm, &"input must be NHWC rank 4, got {x.shape}"
      check x.shape[0] == 1, nm, "only batch size 1 is supported"
      check w.shape.len == 4, nm, &"filter must be rank 4, got {w.shape}"
      check op.strideH > 0 and op.strideW > 0, nm, "stride must be positive"
      check op.dilationH > 0 and op.dilationW > 0, nm, "dilation must be positive"

      let inH = x.shape[1]
      let inW = x.shape[2]
      let inC = x.shape[3]
      op.kH = w.shape[1]
      op.kW = w.shape[2]

      let outH = outputDim(inH, op.kH, op.strideH, op.dilationH, op.padding)
      let outW = outputDim(inW, op.kW, op.strideW, op.dilationW, op.padding)
      check outH > 0 and outW > 0, nm,
        &"kernel {op.kH}x{op.kW} with stride {op.strideH}x{op.strideW} " &
        &"does not fit input {inH}x{inW}"
      op.padTop = padBefore(inH, outH, op.kH, op.strideH, op.dilationH)
      op.padLeft = padBefore(inW, outW, op.kW, op.strideW, op.dilationW)

      let outC =
        if op.kind == okConv2d:
          check w.shape[3] == inC, nm,
            &"filter input channels {w.shape[3]} != input channels {inC}"
          w.shape[0]
        else:
          check op.depthMultiplier > 0, nm, "depthMultiplier must be positive"
          check w.shape[3] == inC * op.depthMultiplier, nm,
            &"depthwise filter channels {w.shape[3]} != " &
            &"inC*depthMultiplier ({inC}*{op.depthMultiplier})"
          inC * op.depthMultiplier

      let expected = @[1, outH, outW, outC]
      let y = g.tensors[op.outputs[0]]
      check y.shape == expected, nm,
        &"output shape {y.shape} does not match inferred {expected}"
      check g.tensors[op.inputs[2]].numElements == outC, nm,
        &"bias length {g.tensors[op.inputs[2]].numElements} != output channels {outC}"

    of okMaxPool2d, okAvgPool2d:
      let x = g.tensors[op.inputs[0]]
      check x.shape.len == 4, nm, &"input must be NHWC rank 4, got {x.shape}"
      check op.kH > 0 and op.kW > 0, nm, "pool window must be positive"
      let outH = outputDim(x.shape[1], op.kH, op.strideH, 1, op.padding)
      let outW = outputDim(x.shape[2], op.kW, op.strideW, 1, op.padding)
      check outH > 0 and outW > 0, nm, "pool window does not fit input"
      op.padTop = padBefore(x.shape[1], outH, op.kH, op.strideH, 1)
      op.padLeft = padBefore(x.shape[2], outW, op.kW, op.strideW, 1)
      let expected = @[1, outH, outW, x.shape[3]]
      check g.tensors[op.outputs[0]].shape == expected, nm,
        &"output shape {g.tensors[op.outputs[0]].shape} != inferred {expected}"

    of okAdd, okSub, okMul:
      check op.inputs.len == 2, nm, "expects two inputs"
      let a = g.tensors[op.inputs[0]]
      let b = g.tensors[op.inputs[1]]
      let y = g.tensors[op.outputs[0]]
      if g.policy == pkAffineI16x8:
        # Broadcast as NumPy does, within the four dimensions the kernels take
        check y.shape.len <= MaxRank, nm, &"output {y.shape} has more than {MaxRank} dimensions"
        check broadcastShape(a.shape, b.shape) == y.shape or
              (a.shape == b.shape and a.numElements == y.numElements), nm,
          &"operands {a.shape} and {b.shape} do not broadcast to the output's {y.shape}"
      else:
        check op.kind == okAdd, nm, &"{op.kind} is not supported under {g.policy.policyName} yet"
        check a.numElements == b.numElements, nm,
          &"operand element counts differ ({a.numElements} vs {b.numElements}); " &
          "broadcasting is not supported"
        check y.numElements == a.numElements, nm,
          "output element count must match operands"
      if op.kind == okMul:
        for t in [a, b, y]:
          check t.quant.zeroPoints.len == 0 or t.quant.zeroPoints[0] == 0, nm,
            "int16 MUL needs zero points of 0"
      if g.policy == pkAffineI16x8 and op.kind != okMul:
        # TFLite takes another path when all three scales are powers of two
        # and this flag allows it; nothing here implements that one.
        var pot = op.potScaleInt16
        for t in [a, b, g.tensors[op.outputs[0]]]:
          let l = log2(t.quant.scales[0])
          pot = pot and abs(l - round(l)) < 1e-3
        check not pot, nm,
          "int16 ADD with power-of-two scales takes TFLite's special path, " &
          "which is not implemented"

    of okClamp:
      check g.tensors[op.outputs[0]].numElements ==
            g.tensors[op.inputs[0]].numElements, nm,
        "clamp must preserve element count"

    of okReshape:
      check g.tensors[op.outputs[0]].numElements ==
            g.tensors[op.inputs[0]].numElements, nm,
        &"reshape changes element count " &
        &"({g.tensors[op.inputs[0]].numElements} -> " &
        &"{g.tensors[op.outputs[0]].numElements})"

    of okPad:
      check op.inputs.len == 1, nm, "expects one input"
      let x = g.tensors[op.inputs[0]]
      let y = g.tensors[op.outputs[0]]
      if g.policy == pkAffineI16x8:
        # Any dimension, of a tensor of up to four
        check x.shape.len <= MaxRank and op.pads.len == x.shape.len, nm,
          &"pads needs one (before, after) pair for each of {x.shape}'s dimensions"
        for d in 0 ..< x.shape.len:
          check op.pads[d][0] >= 0 and op.pads[d][1] >= 0, nm,
            &"negative padding {op.pads[d]} on dimension {d}"
          check y.shape[d] == x.shape[d] + op.pads[d][0] + op.pads[d][1], nm,
            &"output shape {y.shape} does not match {x.shape} padded by {op.pads}"
        check sameQuant(x.quant, y.quant), nm,
          "pad copies its input through unchanged, so input and output " &
          "quantization must be identical"
      else:
        check x.shape.len == 4, nm, &"input must be NHWC rank 4, got {x.shape}"
        check x.shape[0] == 1, nm, "only batch size 1 is supported"
        check op.pads.len == 4, nm,
          &"pads needs one (before, after) pair per dimension; got {op.pads.len} " &
          "for a rank-4 input"
        for d in 0 ..< 4:
          check op.pads[d][0] >= 0 and op.pads[d][1] >= 0, nm,
            &"negative padding {op.pads[d]} on dimension {d}"
        check op.pads[0] == [0, 0] and op.pads[3] == [0, 0], nm,
          "only spatial padding is supported; batch and channel pads must be zero"
        let expected = @[x.shape[0],
                         x.shape[1] + op.pads[1][0] + op.pads[1][1],
                         x.shape[2] + op.pads[2][0] + op.pads[2][1],
                         x.shape[3]]
        check y.shape == expected, nm,
          &"output shape {y.shape} does not match inferred {expected}"
        check x.dtype == storeDt and y.dtype == storeDt, nm,
          "input/output dtype must match the graph policy storage type"
        # Interior values are copied through bit-for-bit, so a differing
        # quantization would silently reinterpret them.
        check sameQuant(x.quant, y.quant), nm,
          "pad copies its input through unchanged, so input and output " &
          "quantization must be identical"
        check op.fused == faNone, nm,
          "pad has no fused activation; add an explicit Clamp op instead"

    of okConcatenation:
      check op.inputs.len >= 1, nm, "expects at least one input"
      let y = g.tensors[op.outputs[0]]
      let rank = y.shape.len
      var ax = op.axis
      if ax < 0: ax += rank
      check ax >= 0 and ax < rank, nm,
        &"axis {op.axis} is out of range for a rank-{rank} output"
      op.axis = ax
      var total = 0
      for k, ti in op.inputs:
        let t = g.tensors[ti]
        check t.shape.len == rank, nm,
          &"operand {k} '{t.name}' has rank {t.shape.len}, output has rank {rank}"
        check t.dtype == storeDt, nm,
          &"operand {k} '{t.name}' dtype must match the graph policy storage type"
        for d in 0 ..< rank:
          if d != ax:
            check t.shape[d] == y.shape[d], nm,
              &"operand {k} '{t.name}' dimension {d} is {t.shape[d]}, " &
              &"output has {y.shape[d]}"
        total += t.shape[ax]
      check total == y.shape[ax], nm,
        &"operand extents along axis {ax} sum to {total}, output has {y.shape[ax]}"
      # An operand whose quantization already matches the output is copied
      # verbatim by the emitter, which would silently skip a fused clamp.
      check op.fused == faNone, nm,
        "concatenation has no fused activation; add an explicit Clamp op instead"
      if g.policy == pkAffineI16x8:
        # TFLite's int16 concatenation rescales in float; the converter puts a
        # QUANTIZE in front of an operand that needs it, so here they agree
        for k, ti in op.inputs:
          check sameQuant(g.tensors[ti].quant, y.quant), nm,
            &"operand {k} '{g.tensors[ti].name}' is quantized otherwise than the output"

    of okMean:
      check op.inputs.len == 1, nm, "expects one input"
      let x = g.tensors[op.inputs[0]]
      let y = g.tensors[op.outputs[0]]
      if g.policy == pkAffineI16x8:
        # Over the last axis, as LayerNorm takes it
        var axes = op.axes
        for i in 0 ..< axes.len:
          if axes[i] < 0: axes[i] += x.shape.len
        check axes == @[x.shape.len - 1], nm,
          &"under {g.policy.policyName} only a mean over the last axis is supported, got {op.axes}"
        op.axes = axes
        check y.numElements * x.shape[^1] == x.numElements, nm,
          &"output {y.shape} is not {x.shape} reduced over its last axis"
        for t in [x, y]:
          check t.quant.zeroPoints.len == 0 or t.quant.zeroPoints[0] == 0, nm,
            "int16 MEAN needs zero points of 0"
      else:
        check x.shape.len == 4, nm, &"input must be NHWC rank 4, got {x.shape}"
        check x.shape[0] == 1, nm, "only batch size 1 is supported"
        var axes = op.axes
        for i in 0 ..< axes.len:
          if axes[i] < 0: axes[i] += 4
        if axes.len == 2 and axes[0] > axes[1]:
          swap axes[0], axes[1]
        check axes == @[1, 2], nm,
          &"only reduction over the spatial axes [1, 2] — global average — is " &
          &"supported, got {op.axes}"
        op.axes = axes
        let expected =
          if op.keepDims: @[1, 1, 1, x.shape[3]] else: @[1, x.shape[3]]
        check y.shape == expected, nm,
          &"output shape {y.shape} does not match inferred {expected} " &
          &"(keepDims = {op.keepDims})"
        check x.dtype == storeDt and y.dtype == storeDt, nm,
          "input/output dtype must match the graph policy storage type"

    of okLogistic, okTanh:
      check op.inputs.len == 1, nm, "expects one input"
      if g.policy == pkAffineI16x8:
        # TFLite's int16 tanh and logistic, not a table of every input
        let xq = g.tensors[op.inputs[0]].quant
        let yq = g.tensors[op.outputs[0]].quant
        check xq.zeroPoints.len == 0 or xq.zeroPoints[0] == 0, nm,
          &"int16 {op.kind} needs input zero point 0"
        check yq.scales.len == 1 and abs(log2(yq.scales[0]) + 15.0) < 1e-3, nm,
          &"int16 {op.kind} needs output scale 2^-15"
      else:
        check g.policy.hasLutDomain, nm,
          &"{op.kind} is evaluated from a host-generated lookup table, which " &
          &"needs an 8-bit storage type; policy {g.policy.policyName} stores " &
          &"{storeDt}"
      check g.tensors[op.outputs[0]].numElements ==
            g.tensors[op.inputs[0]].numElements, nm,
        "must preserve element count"
      check g.tensors[op.inputs[0]].dtype == storeDt and
            g.tensors[op.outputs[0]].dtype == storeDt, nm,
        "input/output dtype must match the graph policy storage type"
      check op.fused == faNone, nm,
        "the lookup table already is the activation; a fused clamp on top of " &
        "it would be silently ignored"

    of okSoftmax:
      check op.inputs.len == 1, nm, "expects one input"
      let x = g.tensors[op.inputs[0]]
      let y = g.tensors[op.outputs[0]]
      check y.numElements == x.numElements, nm, "must preserve element count"
      check x.shape.len >= 1, nm, "input must have at least one dimension"
      check x.shape == y.shape, nm,
        &"softmax must preserve shape, got {x.shape} -> {y.shape}"
      # Softmax normalises over the last axis, once per row. A classifier has
      # one row; a detection head like FOMO has one per output cell.
      let classes = x.shape[^1]
      check classes >= 1, nm, "needs at least one class"
      check classes <= MaxSoftmaxClasses, nm,
        &"{classes} classes exceeds the {MaxSoftmaxClasses} the exp table can " &
        "cover without overflowing its int32 sum"
      check x.dtype == storeDt and y.dtype == storeDt, nm,
        "input/output dtype must match the graph policy storage type"
      check op.fused == faNone, nm, "softmax has no fused activation"
      if g.policy == pkAffineI16x8:
        # TFLite's int16 softmax gives [0, 1] at 2^-15, whatever the file says
        check x.quant.zeroAt0 == 0 and y.quant.zeroAt0 == 0, nm,
          "int16 softmax needs zero points of 0"
        check abs(log2(y.quant.scales[0]) + 15.0) < 1e-3, nm,
          "int16 softmax needs an output scale of 2^-15"

    of okLeakyRelu:
      check op.inputs.len == 1, nm, "expects one input"
      let x = g.tensors[op.inputs[0]]
      let y = g.tensors[op.outputs[0]]
      check y.numElements == x.numElements, nm, "must preserve element count"
      check x.dtype == storeDt and y.dtype == storeDt, nm,
        "input/output dtype must match the graph policy storage type"
      check op.alpha >= 0.0, nm,
        &"alpha {op.alpha} is negative; TFLite's quantizer takes no negative scale"

    of okSlice:
      check op.inputs.len == 1, nm, "expects one input"
      let x = g.tensors[op.inputs[0]]
      let y = g.tensors[op.outputs[0]]
      let rank = x.shape.len
      check rank == y.shape.len and (rank == 4 or
            (g.policy == pkAffineI16x8 and rank <= MaxRank)), nm,
        &"only NHWC rank 4 is sliced, or up to {MaxRank} dimensions under " &
        &"AffineI16x8, got {x.shape} -> {y.shape}"
      check op.begin.len == rank, nm,
        &"begin {op.begin} must give one index per dimension"
      for d in 0 ..< rank:
        check op.begin[d] >= 0 and op.begin[d] + y.shape[d] <= x.shape[d], nm,
          &"slice of {y.shape} from {op.begin} falls outside {x.shape}"
      check sameQuant(x.quant, y.quant), nm,
        "slice copies its input through unchanged, so input and output " &
        "quantization must be identical"

    of okRsqrt:
      check op.inputs.len == 1, nm, "expects one input"
      let x = g.tensors[op.inputs[0]]
      let y = g.tensors[op.outputs[0]]
      check x.numElements == y.numElements, nm, "must preserve element count"
      check x.quant.zeroAt0 == 0 and y.quant.zeroAt0 == 0, nm,
        "int16 RSQRT needs zero points of 0"

    of okBatchMatmul:
      check op.inputs.len == 2, nm, "expects two inputs"
      let a = g.tensors[op.inputs[0]]
      let b = g.tensors[op.inputs[1]]
      let y = g.tensors[op.outputs[0]]
      check not op.adjX, nm, "a transposed left operand is not supported"
      check a.shape.len >= 2 and a.shape.len == b.shape.len and a.shape.len <= MaxRank, nm,
        &"operands {a.shape} and {b.shape} must have the same rank, 2 to {MaxRank}"
      let depth = a.shape[^1]
      let (bDepth, cols) = if op.adjY: (b.shape[^1], b.shape[^2]) else: (b.shape[^2], b.shape[^1])
      check depth == bDepth, nm, &"{a.shape} cannot multiply {b.shape} (adjY = {op.adjY})"
      check y.shape == a.shape[0 ..< ^2] & @[a.shape[^2], cols], nm,
        &"output {y.shape} is not {a.shape} times {b.shape}"
      check a.shape[0 ..< ^2] == b.shape[0 ..< ^2], nm,
        "the operands' batch dimensions must agree; broadcasting them is not supported"
      for t in [a, b, y]:
        check t.quant.zeroAt0 == 0, nm, "int16 BATCH_MATMUL needs zero points of 0"

    of okTranspose:
      check op.inputs.len == 1, nm, "expects one input"
      let x = g.tensors[op.inputs[0]]
      let y = g.tensors[op.outputs[0]]
      check x.shape.len <= MaxRank and op.perm.len == x.shape.len, nm,
        &"perm {op.perm} must give one dimension for each of {x.shape}'s"
      for d, p in op.perm:
        check p >= 0 and p < x.shape.len and y.shape[d] == x.shape[p], nm,
          &"output {y.shape} is not {x.shape} permuted by {op.perm}"
      check sameQuant(x.quant, y.quant), nm, "transpose moves values without changing them"

    of okGather:
      check op.inputs.len == 2, nm, "expects a table and indices"
      let table = g.tensors[op.inputs[0]]
      let ids = g.tensors[op.inputs[1]]
      let y = g.tensors[op.outputs[0]]
      check table.kind == tkConst and table.shape.len == 2, nm,
        "only rows of a constant table of rank 2 are gathered"
      check ids.dtype == dtInt32, nm, "indices must be int32"
      check y.numElements == ids.numElements * table.shape[1], nm,
        &"output {y.shape} is not a row of {table.shape} for each of {ids.shape}"
      check sameQuant(table.quant, y.quant), nm, "gather copies rows without changing them"

    of okBand, okUnband:
      check op.inputs.len == 1, nm, "expects one input"
      let x = g.tensors[op.inputs[0]].shape
      let y = g.tensors[op.outputs[0]].shape
      let n = x[^2]
      check x.len == y.len and x[0 ..< ^1] == y[0 ..< ^1], nm, &"{x} and {y} differ in more than their last dimension"
      if op.kind == okBand:
        check y[^1] == n and op.bandStart >= 0 and op.bandStart < x[^1], nm,
          &"a band of {x} is [.., n, n], from inside the window"
      else:
        check x[^1] == n and y[^1] <= 2 * op.bandStart + 1, nm,
          &"the band of {x} is narrower than {y}"
      check sameQuant(g.tensors[op.inputs[0]].quant, g.tensors[op.outputs[0]].quant), nm,
        "a band moves values without changing them"

    of okAddBand:
      check op.inputs.len == 2, nm, "expects the other operand and the band's input"
      let a = g.tensors[op.inputs[0]].shape
      let x = g.tensors[op.inputs[1]].shape
      check a == g.tensors[op.outputs[0]].shape and a.len == x.len and a[0 ..< ^1] == x[0 ..< ^1] and
            a[^1] == a[^2] and op.bandStart < x[^1], nm,
        &"{a} cannot take the band of {x}"

    of okRequantize:
      check op.inputs.len == 1, nm, "expects one input"
      check g.tensors[op.inputs[0]].numElements == g.tensors[op.outputs[0]].numElements, nm,
        "must preserve element count"

    for t in op.outputs:
      check t notin produced, nm,
        &"tensor '{g.tensors[t].name}' is written by more than one op"
      produced[t] = oi

  for t in g.outputs:
    if t notin produced:
      raise newException(IrError,
        &"graph output '{g.tensors[t].name}' is never produced")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# REPORTING
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

proc summary*(g: Graph): string =
  ## One line per op, with shapes and the attributes that change the answer.
  ## For `steadyc --dump`: reading an imported graph against the model it came
  ## from is the cheapest way to catch a frontend that misread a field.
  result = &"graph '{g.name}'  policy {g.policy.policyName}  " &
           &"{g.ops.len} ops  {g.tensors.len} tensors\n"
  for i, t in g.inputs:
    result.add &"  input{i}   '{g.tensors[t].name}' {g.tensors[t].shape} " &
               &"{g.tensors[t].dtype}\n"
  for i, t in g.outputs:
    result.add &"  output{i}  '{g.tensors[t].name}' {g.tensors[t].shape} " &
               &"{g.tensors[t].dtype}\n"
  for oi, op in g.ops:
    var attrs: seq[string]
    case op.kind
    of okConv2d, okDepthwiseConv2d:
      attrs.add &"k{op.kH}x{op.kW}"
      attrs.add &"s{op.strideH}x{op.strideW}"
      if op.dilationH != 1 or op.dilationW != 1:
        attrs.add &"d{op.dilationH}x{op.dilationW}"
      attrs.add (if op.padding == padSame: "SAME" else: "VALID")
      if op.kind == okDepthwiseConv2d and op.depthMultiplier != 1:
        attrs.add &"dm{op.depthMultiplier}"
      if g.tensors[op.inputs[1]].quant.isPerChannel:
        attrs.add &"per-channel({g.tensors[op.inputs[1]].quant.scales.len})"
    of okMaxPool2d, okAvgPool2d:
      attrs.add &"k{op.kH}x{op.kW}"
      attrs.add &"s{op.strideH}x{op.strideW}"
      attrs.add (if op.padding == padSame: "SAME" else: "VALID")
    of okPad:
      attrs.add $op.pads
    of okConcatenation:
      attrs.add &"axis {op.axis}"
    of okMean:
      attrs.add &"axes {op.axes}"
      if op.keepDims: attrs.add "keepDims"
    of okLeakyRelu:
      attrs.add &"alpha {op.alpha}"
    of okSlice:
      attrs.add &"begin {op.begin}"
    of okTranspose:
      attrs.add &"perm {op.perm}"
    of okBatchMatmul:
      if op.adjY: attrs.add "adjY"
    else: discard
    if op.fused != faNone:
      attrs.add $op.fused
    var shapes: seq[string]
    for t in op.inputs:
      let tt = g.tensors[t]
      shapes.add $tt.shape & (if tt.kind == tkConst: "c" else: "")
    result.add &"  {oi:3d} {op.kind:<20} " & shapes.join(" ") &
               &" -> {g.tensors[op.outputs[0]].shape}"
    if attrs.len > 0:
      result.add "  " & attrs.join(" ")
    result.add "\n"
