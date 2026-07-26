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

import std/[strformat, tables]

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
    okMaxPool2d, okAvgPool2d, okAdd, okClamp, okReshape

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

func numElements*(t: Tensor): int =
  result = 1
  for d in t.shape: result *= d

func byteSize*(t: Tensor): int =
  t.numElements * t.dtype.byteWidth

func isPerChannel*(q: Quant): bool = q.axis >= 0 and q.scales.len > 1

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

    for t in op.outputs:
      check t notin produced, nm,
        &"tensor '{g.tensors[t].name}' is written by more than one op"
      produced[t] = oi

  for t in g.outputs:
    if t notin produced:
      raise newException(IrError,
        &"graph output '{g.tensors[t].name}' is never produced")
