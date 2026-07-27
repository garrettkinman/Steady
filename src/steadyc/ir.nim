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

import std/[strformat, strutils, tables]

type
  DType* = enum
    dtInt8, dtInt32, dtFloat32, dtFp8

  PolicyKind* = enum
    pkAffineI8, pkRealF32, pkRealFp8

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
    okLogistic, okTanh, okSoftmax

  Quant* = object
    ## Per-tensor when `scales` has one entry, per-channel when it has one
    ## per slice along `axis`. Real-number policies leave this empty.
    scales*: seq[float64]
    zeroPoints*: seq[int32]
    axis*: int                 ## -1 for per-tensor

  Tensor* = object
    name*: string
    shape*: seq[int]
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
    axes*: seq[int]            ## okMean, normalized by validate()
    keepDims*: bool            ## okMean
    # Filled in by validate(); explicit padding derived from `padding`.
    padTop*, padLeft*: int

  Graph* = object
    name*: string
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
  of dtInt8, dtFp8: 1
  of dtInt32, dtFloat32: 4

func nimTypeName*(d: DType): string =
  case d
  of dtInt8: "int8"
  of dtInt32: "int32"
  of dtFloat32: "float32"
  of dtFp8: "Fp8"

func cTypeName*(d: DType): string =
  ## Fixed-width types throughout — `int` is not 32 bits everywhere these
  ## models are going to run.
  case d
  of dtInt8: "int8_t"
  of dtInt32: "int32_t"
  of dtFloat32: "float"
  of dtFp8: "uint8_t"

func policyName*(p: PolicyKind): string =
  case p
  of pkAffineI8: "AffineI8"
  of pkRealF32: "RealF32"
  of pkRealFp8: "RealFp8"

func storeType*(p: PolicyKind): DType =
  case p
  of pkAffineI8: dtInt8
  of pkRealF32: dtFloat32
  of pkRealFp8: dtFp8

func biasType*(p: PolicyKind): DType =
  case p
  of pkAffineI8: dtInt32       ## accumulator units, zero point pre-folded
  of pkRealF32: dtFloat32
  of pkRealFp8: dtFp8

func isAffine*(p: PolicyKind): bool = p == pkAffineI8

func hasLutDomain*(p: PolicyKind): bool =
  ## True when the storage type's whole value domain is small enough for the
  ## host to enumerate — that is, 8 bits, 256 entries. Non-linear activations
  ## are lookup tables for exactly these policies and are a host-side error
  ## for the others; see `steadyc/codec.nim`.
  p.storeType.byteWidth == 1

const MaxSoftmaxClasses* = 4096
  ## The softmax exp table is summed in an int32, and the host picks the
  ## table's fractional bits from the class count to guarantee that fits
  ## (see `buildExpLut`). Beyond this the table would be too coarse to be
  ## worth having.

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
# MAC/s rather than only milliseconds. Milliseconds say which op is slow;
# MAC/s says whether that is because it does more work or because the kernel
# is worse, and only the second is actionable.

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
  else:
    0

func totalMacs*(g: Graph): int =
  for op in g.ops: result += g.macCount(op)

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

func sameQuant*(a, b: Quant): bool =
  ## Structural equality. Ops that move values without touching them require
  ## it; anything looser would silently reinterpret the data. The emitter uses
  ## it too, to decide when an op reduces to a plain copy.
  a.scales == b.scales and a.zeroPoints == b.zeroPoints

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
      check x.dtype == storeDt and w.dtype == storeDt and y.dtype == storeDt, nm,
        "input/weights/output dtype must match the graph policy storage type"
      check b.dtype == biasDt, nm, &"bias dtype must be {biasDt}"

    of okConv2d, okDepthwiseConv2d:
      check op.inputs.len == 3, nm, "expects [input, filter, bias]"
      let x = g.tensors[op.inputs[0]]
      let w = g.tensors[op.inputs[1]]
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

    of okAdd:
      check op.inputs.len == 2, nm, "expects two inputs"
      let a = g.tensors[op.inputs[0]]
      let b = g.tensors[op.inputs[1]]
      check a.numElements == b.numElements, nm,
        &"operand element counts differ ({a.numElements} vs {b.numElements}); " &
        "broadcasting is not supported"
      check g.tensors[op.outputs[0]].numElements == a.numElements, nm,
        "output element count must match operands"

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

    of okMean:
      check op.inputs.len == 1, nm, "expects one input"
      let x = g.tensors[op.inputs[0]]
      let y = g.tensors[op.outputs[0]]
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
      check g.policy.isAffine, nm,
        "softmax needs a uniform integer store domain, so that a " &
        "max-subtracted difference is itself a table index; only the affine " &
        "int8 policy qualifies"
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
