# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

## TFLite importer.
##
## Populates the IR from a `.tflite` file. Everything downstream — validation,
## bias folding, planning, emission — was written against the IR and does not
## change, which is what "frontend-neutral" was for.
##
## The field indices below are the schema, spelled out. They are named
## constants rather than magic numbers so they can be read against
## `schema.fbs` line by line, because a wrong index does not fail loudly: it
## silently reads a neighbouring field, and a stride that arrives as a dilation
## produces a model that runs and is wrong. The two defences against that are
## this table being auditable, and the differential harness in
## `tests/models/`, which compares against TFLite itself on real files.
##
## Two deliberate refusals:
##
##   * **Only operator codes this importer is sure of are mapped.** An unknown
##     code is an error naming it, never a guess. Mapping an op we cannot test
##     would be the same silent-wrongness failure as a bad field index.
##   * **uint8 models are rejected**, with an explanation. The TF1-era "quant"
##     models use asymmetric weights (`Zw != 0`), which breaks the bias-folding
##     identity the kernels depend on: it leaves a `Zw * sum(x)` term that is
##     data-dependent and cannot be folded on the host. Supporting them means a
##     runtime row-sum pass inside every matmul kernel. Convert to int8 instead.

import std/[strformat, tables, sets, math, os]
import ./ir, ./flatbuffers
from ./emit import cIdent, nimIdentKey

# A malformed file raises `FbError` from the reader; callers want to catch that
# alongside `TfliteError` without importing the reader themselves.
export flatbuffers.FbError

type TfliteError* = object of CatchableError

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SCHEMA
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

const
  # table Model
  mdlVersion = 0
  mdlOperatorCodes = 1
  mdlSubgraphs = 2
  mdlBuffers = 4

  # table OperatorCode
  ocDeprecatedBuiltinCode = 0
  ocCustomCode = 1
  ocBuiltinCode = 3

  # table SubGraph
  sgTensors = 0
  sgInputs = 1
  sgOutputs = 2
  sgOperators = 3
  sgName = 4

  # table Tensor
  tsShape = 0
  tsType = 1
  tsBuffer = 2
  tsName = 3
  tsQuantization = 4

  # table QuantizationParameters
  qpScale = 2
  qpZeroPoint = 3
  qpQuantizedDimension = 6

  # table Operator
  opOpcodeIndex = 0
  opInputs = 1
  opOutputs = 2
  opBuiltinOptions = 4

  # table Buffer
  bufData = 0

  # table Conv2DOptions
  convPadding = 0
  convStrideW = 1
  convStrideH = 2
  convFusedActivation = 3
  convDilationW = 4
  convDilationH = 5

  # table DepthwiseConv2DOptions
  dwPadding = 0
  dwStrideW = 1
  dwStrideH = 2
  dwDepthMultiplier = 3
  dwFusedActivation = 4
  dwDilationW = 5
  dwDilationH = 6

  # table Pool2DOptions
  poolPadding = 0
  poolStrideW = 1
  poolStrideH = 2
  poolFilterW = 3
  poolFilterH = 4
  poolFusedActivation = 5

  # table FullyConnectedOptions
  fcFusedActivation = 0
  fcWeightsFormat = 1

  # table AddOptions
  addFusedActivation = 0

  # table ConcatenationOptions
  catAxis = 0
  catFusedActivation = 1

  # table SoftmaxOptions
  softmaxBeta = 0

  # table ReducerOptions
  reducerKeepDims = 0

  # enum Padding
  padSameCode = 0
  padValidCode = 1

  # enum ActivationFunctionType
  actNone = 0
  actRelu = 1
  actReluN1To1 = 2
  actRelu6 = 3

  # enum TensorType
  ttFloat32 = 0
  ttInt32 = 2
  ttUInt8 = 3
  ttInt64 = 4
  ttInt16 = 7
  ttInt8 = 9

  # enum BuiltinOperator — only the codes this importer maps.
  bopAdd = 0
  bopAveragePool2d = 1
  bopConcatenation = 2
  bopConv2d = 3
  bopDepthwiseConv2d = 4
  bopDequantize = 6
  bopFullyConnected = 9
  bopLogistic = 14
  bopMaxPool2d = 17
  bopRelu = 19
  bopReluN1To1 = 20
  bopRelu6 = 21
  bopReshape = 22
  bopSoftmax = 25
  bopTanh = 28
  bopPad = 34
  bopMean = 40
  bopQuantize = 114

func builtinName(code: int): string =
  ## For error messages only. Codes this importer does not map are reported by
  ## number, which is more useful than a name that might be wrong.
  case code
  of bopAdd: "ADD"
  of bopAveragePool2d: "AVERAGE_POOL_2D"
  of bopConcatenation: "CONCATENATION"
  of bopConv2d: "CONV_2D"
  of bopDepthwiseConv2d: "DEPTHWISE_CONV_2D"
  of bopDequantize: "DEQUANTIZE"
  of bopFullyConnected: "FULLY_CONNECTED"
  of bopLogistic: "LOGISTIC"
  of bopMaxPool2d: "MAX_POOL_2D"
  of bopRelu: "RELU"
  of bopReluN1To1: "RELU_N1_TO_1"
  of bopRelu6: "RELU6"
  of bopReshape: "RESHAPE"
  of bopSoftmax: "SOFTMAX"
  of bopTanh: "TANH"
  of bopPad: "PAD"
  of bopMean: "MEAN"
  of bopQuantize: "QUANTIZE"
  else: &"builtin #{code}"

func dtypeOf(tt: int, where: string): DType =
  case tt
  of ttInt8: dtInt8
  of ttInt32: dtInt32
  of ttFloat32: dtFloat32
  of ttUInt8:
    raise newException(TfliteError,
      &"{where} is uint8. TF1-era 'quant' models use asymmetric weights, " &
      "which the bias-folding identity this compiler is built on cannot " &
      "absorb — it leaves a data-dependent term that has to be computed on " &
      "the device. Re-convert the model with TFLite's int8 post-training " &
      "quantization.")
  of ttInt16:
    raise newException(TfliteError,
      &"{where} is int16; only int8 and float32 graphs are supported")
  of ttInt64:
    raise newException(TfliteError, &"{where} is int64, which no kernel consumes")
  else:
    raise newException(TfliteError, &"{where} has unsupported tensor type {tt}")

func fusedOf(code: int, opName: string): FusedAct =
  case code
  of actNone: faNone
  of actRelu: faRelu
  of actRelu6: faRelu6
  of actReluN1To1: faReluN1To1
  else:
    raise newException(TfliteError,
      &"op '{opName}': fused activation {code} is not supported; only NONE, " &
      "RELU, RELU6 and RELU_N1_TO_1 reduce to a clamp")

func paddingOf(code: int, opName: string): Padding =
  case code
  of padSameCode: padSame
  of padValidCode: padValid
  else:
    raise newException(TfliteError, &"op '{opName}': unknown padding mode {code}")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# RAW VIEW
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

type
  RawTensor = object
    shape: seq[int]
    typeCode: int
    bufferIdx: int
    name: string
    scales: seq[float64]
    zeroPoints: seq[int32]
    quantDim: int

  RawOp = object
    code: int
    inputs: seq[int]
    outputs: seq[int]
    options: FbTable
    hasOptions: bool

  Reader = object
    buf: FbBuffer
    buffers: FbVector
    tensors: seq[RawTensor]
    ops: seq[RawOp]
    graphInputs: seq[int]
    graphOutputs: seq[int]
    subgraphName: string

proc readQuant(t: FbTable): tuple[scales: seq[float64], zeros: seq[int32], dim: int] =
  if not t.has(tsQuantization):
    return (@[], @[], -1)
  let q = t.fieldTable(tsQuantization)
  let scales = q.fieldVector(qpScale).toFloat64Seq
  var zeros: seq[int32]
  let zv = q.fieldVector(qpZeroPoint)
  for i in 0 ..< zv.len:
    let z = zv.i64At(i)
    if z < low(int32) or z > high(int32):
      raise newException(TfliteError, &"zero point {z} does not fit in an int32")
    zeros.add int32(z)
  # `quantized_dimension` is meaningless for a per-tensor scale, and the IR
  # spells that case as axis -1 so `isPerChannel` needs no special case.
  let dim = if scales.len > 1: int(q.fieldI32(qpQuantizedDimension)) else: -1
  (scales, zeros, dim)

proc readModel(bytes: sink seq[byte], path: string): Reader =
  result.buf = newFbBuffer(bytes)
  let ident = result.buf.fileIdentifier
  if ident != "TFL3":
    raise newException(TfliteError,
      &"'{path}' has file identifier '{ident}', expected 'TFL3'; this does " &
      "not look like a TFLite flatbuffer")
  let model = result.buf.root
  let version = model.fieldU32(mdlVersion)
  if version != 3:
    raise newException(TfliteError,
      &"schema version {version} is not supported; this importer reads version 3")

  var codes: seq[int]
  let ocs = model.fieldVector(mdlOperatorCodes)
  for i in 0 ..< ocs.len:
    let oc = ocs.tableAt(i)
    # `builtin_code` was widened to int32 once codes passed 127; the deprecated
    # byte field is still what older writers filled in.
    let wide = int(oc.fieldI32(ocBuiltinCode))
    let narrow = int(oc.fieldI8(ocDeprecatedBuiltinCode))
    var code = if wide != 0: wide else: narrow
    if code == 32:                      # CUSTOM
      let custom = oc.fieldString(ocCustomCode)
      raise newException(TfliteError,
        &"the model uses custom operator '{custom}', which has no portable " &
        "definition to compile")
    codes.add code

  result.buffers = model.fieldVector(mdlBuffers)

  let sgs = model.fieldVector(mdlSubgraphs)
  if sgs.len != 1:
    raise newException(TfliteError,
      &"the model has {sgs.len} subgraphs; only single-subgraph models are " &
      "supported (control flow is out of scope)")
  let sg = sgs.tableAt(0)
  result.subgraphName = sg.fieldString(sgName)

  let ts = sg.fieldVector(sgTensors)
  for i in 0 ..< ts.len:
    let t = ts.tableAt(i)
    var raw = RawTensor(
      shape: t.fieldVector(tsShape).toIntSeq,
      typeCode: int(t.fieldI8(tsType)),
      bufferIdx: int(t.fieldU32(tsBuffer)),
      name: t.fieldString(tsName))
    let q = readQuant(t)
    raw.scales = q.scales
    raw.zeroPoints = q.zeros
    raw.quantDim = q.dim
    if raw.name.len == 0:
      raw.name = &"tensor{i}"
    result.tensors.add raw

  result.graphInputs = sg.fieldVector(sgInputs).toIntSeq
  result.graphOutputs = sg.fieldVector(sgOutputs).toIntSeq

  let os = sg.fieldVector(sgOperators)
  for i in 0 ..< os.len:
    let o = os.tableAt(i)
    let ci = int(o.fieldU32(opOpcodeIndex))
    if ci < 0 or ci >= codes.len:
      raise newException(TfliteError, &"operator {i} has opcode index {ci}, out of range")
    var raw = RawOp(code: codes[ci],
                    inputs: o.fieldVector(opInputs).toIntSeq,
                    outputs: o.fieldVector(opOutputs).toIntSeq,
                    hasOptions: o.has(opBuiltinOptions))
    if raw.hasOptions:
      raw.options = o.fieldTable(opBuiltinOptions)
    result.ops.add raw

proc optI(op: RawOp, field: int, default: int32): int32 =
  ## Builtin options are absent whenever every field is at its default, so
  ## "no options table" and "all defaults" have to mean the same thing.
  if op.hasOptions: op.options.fieldI32(field, default) else: default

proc optByte(op: RawOp, field: int, default: int): int =
  ## Enum fields in the schema are `: byte`. Reading one as an int32 silently
  ## picks up the next field's low byte, which is exactly the class of mistake
  ## the named-constant table above is meant to make findable: RELU6 arrives as
  ## 515 rather than 3.
  if op.hasOptions: int(op.options.fieldI8(field, int8(default))) else: default

proc optBool(op: RawOp, field: int, default: bool): bool =
  if op.hasOptions: op.options.fieldBool(field, default) else: default

proc optF32(op: RawOp, field: int, default: float32): float32 =
  if op.hasOptions: op.options.fieldF32(field, default) else: default

proc bufferBytes(r: Reader, idx: int): seq[byte] =
  if idx <= 0 or idx >= r.buffers.len:
    return @[]                          # buffer 0 is the shared empty buffer
  let b = r.buffers.tableAt(idx)
  b.fieldVector(bufData).copyBytes

proc hasData(r: Reader, idx: int): bool =
  if idx <= 0 or idx >= r.buffers.len: return false
  r.buffers.tableAt(idx).fieldVector(bufData).len > 0

proc constInts(r: Reader, ti: int, what: string): seq[int] =
  ## An int32 parameter tensor — a reshape's shape, a pad's paddings, a mean's
  ## axes. These are graph *constants* in TFLite and op *fields* in the IR,
  ## which is the whole reason the IR needs no runtime shape machinery.
  let t = r.tensors[ti]
  if t.typeCode != ttInt32:
    raise newException(TfliteError,
      &"{what}: '{t.name}' must be an int32 constant, got type {t.typeCode}")
  let data = r.bufferBytes(t.bufferIdx)
  if data.len == 0:
    raise newException(TfliteError,
      &"{what}: '{t.name}' has no constant data, so it is computed at runtime; " &
      "this compiler resolves shapes on the host")
  result = newSeq[int](data.len div 4)
  for i in 0 ..< result.len:
    var v: int32
    copyMem(addr v, unsafeAddr data[i * 4], 4)
    result[i] = int(v)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BOUNDARY QUANTIZE / DEQUANTIZE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TFLite's "full integer with float I/O" conversion wraps the graph in a
# QUANTIZE and a DEQUANTIZE. Implementing them would put two storage types in
# one graph, which is exactly what the one-policy-per-graph design exists to
# avoid; and it would make the device do work the caller can do for free.
#
# So they are stripped, and the quantized tensor becomes the model's own input
# or output. Its scale and zero point are already published in the generated C
# header, so a caller holding real values converts in one line. A QUANTIZE
# anywhere other than the boundary is a genuine mid-graph type change and is
# an error.

proc stripBoundary(r: var Reader): seq[bool] =
  result = newSeq[bool](r.ops.len)

  var consumers = initTable[int, seq[int]]()
  var producer = initTable[int, int]()
  for oi, op in r.ops:
    for t in op.inputs:
      if t >= 0: consumers.mgetOrPut(t, @[]).add oi
    for t in op.outputs:
      producer[t] = oi

  for k, gi in r.graphInputs:
    if gi notin consumers or consumers[gi].len != 1: continue
    let oi = consumers[gi][0]
    if r.ops[oi].code != bopQuantize: continue
    if gi in producer: continue                  # not actually a boundary
    result[oi] = true
    r.graphInputs[k] = r.ops[oi].outputs[0]

  for k, go in r.graphOutputs:
    if go notin producer: continue
    let oi = producer[go]
    if r.ops[oi].code != bopDequantize: continue
    let src = r.ops[oi].inputs[0]
    if src notin producer: continue               # would leave nothing behind
    if go in consumers: continue                  # feeds something else too
    result[oi] = true
    r.graphOutputs[k] = src

  for oi, op in r.ops:
    if result[oi]: continue
    if op.code in [bopQuantize, bopDequantize]:
      raise newException(TfliteError,
        &"operator {oi} is {builtinName(op.code)} in the middle of the graph, " &
        "not at its boundary; a storage-type change mid-graph is not " &
        "something one numeric policy can express")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# IMPORT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

proc choosePolicy(r: Reader, keep: seq[bool]): PolicyKind =
  ## Decided by what the activations are stored as. Weights and activations
  ## always agree in a converted model; biases are int32 and say nothing.
  var sawInt8 = false
  var sawFloat = false
  for oi, op in r.ops:
    if not keep[oi]: continue
    for t in op.inputs & op.outputs:
      if t < 0: continue
      case r.tensors[t].typeCode
      of ttInt8: sawInt8 = true
      of ttFloat32: sawFloat = true
      of ttInt32: discard                         # bias or parameter
      else:
        discard dtypeOf(r.tensors[t].typeCode, &"tensor '{r.tensors[t].name}'")
  if sawInt8 and sawFloat:
    raise newException(TfliteError,
      "the graph mixes int8 and float32 activations; one graph is compiled " &
      "under one numeric policy")
  if sawInt8: return pkAffineI8
  if sawFloat: return pkRealF32
  raise newException(TfliteError, "the graph has no activations to compile")

proc importTflite*(bytes: sink seq[byte], name = "", path = "<memory>"): Graph =
  ## Reads a TFLite flatbuffer into a validated `Graph`.
  var r = readModel(bytes, path)
  let skipped = stripBoundary(r)
  var keep = newSeq[bool](r.ops.len)
  for i in 0 ..< r.ops.len: keep[i] = not skipped[i]

  let policy = choosePolicy(r, keep)
  let modelName = if name.len > 0: name
                  elif r.subgraphName.len > 0: r.subgraphName
                  else: "model"
  var g = initGraph(modelName, policy)

  # ---- which tensors are parameters rather than operands -------------------
  # A reshape's shape, a pad's paddings and a mean's axes are constants in the
  # file and *fields* in the IR. Keeping them out of the graph is what lets the
  # arena hold activations only.
  var isParam = initHashSet[int]()
  for oi, op in r.ops:
    if not keep[oi]: continue
    if op.code in [bopReshape, bopPad, bopMean] and
       op.inputs.len > 1 and op.inputs[1] >= 0:
      isParam.incl op.inputs[1]

  var needed = initHashSet[int]()
  for oi, op in r.ops:
    if not keep[oi]: continue
    for t in op.inputs:
      if t >= 0 and t notin isParam: needed.incl t
    for t in op.outputs:
      if t >= 0: needed.incl t
  for t in r.graphInputs: needed.incl t
  for t in r.graphOutputs: needed.incl t

  # ---- tensors ------------------------------------------------------------
  var idx = initTable[int, int]()        # tflite tensor index -> IR index
  var usedNames = initHashSet[string]()
  let inputSet = toHashSet(r.graphInputs)

  for ti in 0 ..< r.tensors.len:
    if ti notin needed: continue
    let t = r.tensors[ti]
    var shape = t.shape
    if shape.len == 0:
      raise newException(TfliteError,
        &"tensor '{t.name}' has no shape; dynamic shapes are out of scope")
    # A dynamic batch is written as 0 or -1 depending on the converter. Every
    # kernel here is batch-1 anyway, and `validate` is what enforces that.
    if shape[0] <= 0: shape[0] = 1
    for d in 1 ..< shape.len:
      if shape[d] <= 0:
        raise newException(TfliteError,
          &"tensor '{t.name}' has dynamic shape {t.shape}; shapes have to be " &
          "resolved for the arena to be planned at compile time")

    # Symbols in the emitted code derive from these names, so a collision
    # *after* sanitizing is a real hazard, and there are three ways to get one:
    # punctuation that maps to the same identifier ("conv/1" and "conv_1"),
    # truncation of two very long names that share a prefix, and Nim's own
    # style-insensitive comparison, which reads `block_1` and `block1` as one
    # name. `nimIdentKey` is the strictest of the three, so it decides.
    var uname = t.name
    if nimIdentKey(uname) in usedNames:
      uname = &"{t.name}_t{ti}"
    usedNames.incl nimIdentKey(uname)

    let dtype = dtypeOf(t.typeCode, &"tensor '{t.name}'")
    let quant = Quant(scales: t.scales, zeroPoints: t.zeroPoints, axis: t.quantDim)

    if r.hasData(t.bufferIdx):
      idx[ti] = g.addConst(uname, shape, dtype, r.bufferBytes(t.bufferIdx), quant)
    elif ti in inputSet:
      idx[ti] = g.addInput(uname, shape, dtype, quant)
    else:
      idx[ti] = g.addIntermediate(uname, shape, dtype, quant)

  proc ir(ti: int): int =
    if ti notin idx:
      raise newException(TfliteError,
        &"tensor {ti} is referenced by an operator but was not imported")
    idx[ti]

  proc zeroBias(opName: string, n: int): int =
    ## TFLite allows a bias-free convolution or matmul; the IR does not, because
    ## the bias is where zero-point correction gets folded. A zero bias costs
    ## `n` words of flash and keeps one code path instead of two.
    let dt = policy.biasType
    g.addConst(&"{opName}_zero_bias", @[n], dt, newSeq[byte](n * dt.byteWidth))

  # ---- operators ----------------------------------------------------------
  for oi, op in r.ops:
    if not keep[oi]: continue
    let nm = if op.outputs.len > 0 and r.tensors[op.outputs[0]].name.len > 0:
               r.tensors[op.outputs[0]].name
             else: &"op{oi}"
    if op.outputs.len != 1:
      raise newException(TfliteError,
        &"op '{nm}' has {op.outputs.len} outputs; every operator this importer " &
        "maps has exactly one")
    if op.inputs.len == 0:
      raise newException(TfliteError, &"op '{nm}' has no inputs")

    var built = Op(name: nm, outputs: @[ir(op.outputs[0])])

    case op.code
    of bopConv2d, bopDepthwiseConv2d:
      let isDw = op.code == bopDepthwiseConv2d
      built.kind = if isDw: okDepthwiseConv2d else: okConv2d
      built.padding = paddingOf(
        op.optByte(if isDw: dwPadding else: convPadding, padSameCode), nm)
      built.strideW = int(op.optI(if isDw: dwStrideW else: convStrideW, 1))
      built.strideH = int(op.optI(if isDw: dwStrideH else: convStrideH, 1))
      built.dilationW = int(op.optI(if isDw: dwDilationW else: convDilationW, 1))
      built.dilationH = int(op.optI(if isDw: dwDilationH else: convDilationH, 1))
      built.fused = fusedOf(op.optByte(
        if isDw: dwFusedActivation else: convFusedActivation, actNone), nm)
      if isDw:
        built.depthMultiplier = int(op.optI(dwDepthMultiplier, 1))
      if op.inputs.len < 2:
        raise newException(TfliteError, &"op '{nm}' has no filter")
      let w = ir(op.inputs[1])
      let outC = if isDw: g.tensors[w].shape[3] else: g.tensors[w].shape[0]
      let b = if op.inputs.len > 2 and op.inputs[2] >= 0: ir(op.inputs[2])
              else: zeroBias(nm, outC)
      built.inputs = @[ir(op.inputs[0]), w, b]

    of bopFullyConnected:
      built.kind = okFullyConnected
      if op.optByte(fcWeightsFormat, 0) != 0:
        raise newException(TfliteError,
          &"op '{nm}' uses a shuffled weight format; only the default layout " &
          "is supported")
      built.fused = fusedOf(op.optByte(fcFusedActivation, actNone), nm)
      if op.inputs.len < 2:
        raise newException(TfliteError, &"op '{nm}' has no weights")
      let w = ir(op.inputs[1])
      let outC = g.tensors[w].shape[0]
      let b = if op.inputs.len > 2 and op.inputs[2] >= 0: ir(op.inputs[2])
              else: zeroBias(nm, outC)
      built.inputs = @[ir(op.inputs[0]), w, b]

    of bopAveragePool2d, bopMaxPool2d:
      built.kind = if op.code == bopMaxPool2d: okMaxPool2d else: okAvgPool2d
      built.padding = paddingOf(op.optByte(poolPadding, padSameCode), nm)
      built.strideW = int(op.optI(poolStrideW, 1))
      built.strideH = int(op.optI(poolStrideH, 1))
      built.kW = int(op.optI(poolFilterW, 1))
      built.kH = int(op.optI(poolFilterH, 1))
      built.fused = fusedOf(op.optByte(poolFusedActivation, actNone), nm)
      built.inputs = @[ir(op.inputs[0])]

    of bopAdd:
      built.kind = okAdd
      built.fused = fusedOf(op.optByte(addFusedActivation, actNone), nm)
      if op.inputs.len < 2:
        raise newException(TfliteError, &"op '{nm}': ADD needs two operands")
      built.inputs = @[ir(op.inputs[0]), ir(op.inputs[1])]

    of bopConcatenation:
      built.kind = okConcatenation
      built.axis = int(op.optI(catAxis, 0))
      if fusedOf(op.optByte(catFusedActivation, actNone), nm) != faNone:
        raise newException(TfliteError,
          &"op '{nm}': a fused activation on a concatenation would be skipped " &
          "for operands that are copied verbatim; split it into a Clamp")
      for t in op.inputs:
        built.inputs.add ir(t)

    of bopReshape:
      built.kind = okReshape
      built.inputs = @[ir(op.inputs[0])]

    of bopPad:
      built.kind = okPad
      built.inputs = @[ir(op.inputs[0])]
      if op.inputs.len < 2:
        raise newException(TfliteError, &"op '{nm}': PAD needs a paddings tensor")
      let flat = r.constInts(op.inputs[1], &"op '{nm}' paddings")
      if flat.len mod 2 != 0:
        raise newException(TfliteError,
          &"op '{nm}': paddings has {flat.len} entries, which is not whole pairs")
      for i in 0 ..< flat.len div 2:
        built.pads.add [flat[i * 2], flat[i * 2 + 1]]
      built.padConst = 0.0            # TFLite PAD is defined to pad with zero

    of bopMean:
      built.kind = okMean
      built.inputs = @[ir(op.inputs[0])]
      if op.inputs.len < 2:
        raise newException(TfliteError, &"op '{nm}': MEAN needs an axes tensor")
      built.axes = r.constInts(op.inputs[1], &"op '{nm}' axes")
      built.keepDims = op.optBool(reducerKeepDims, false)

    of bopLogistic, bopTanh:
      built.kind = if op.code == bopTanh: okTanh else: okLogistic
      built.inputs = @[ir(op.inputs[0])]

    of bopSoftmax:
      built.kind = okSoftmax
      built.inputs = @[ir(op.inputs[0])]
      let beta = op.optF32(softmaxBeta, 1'f32)
      if abs(beta - 1'f32) > 1e-6'f32:
        raise newException(TfliteError,
          &"op '{nm}': softmax beta is {beta}; only 1.0 is supported. Beta " &
          "would fold into the exp table, but no converter emits it for int8 " &
          "and an untested path is worse than an error.")

    of bopRelu, bopRelu6, bopReluN1To1:
      built.kind = okClamp
      built.fused =
        case op.code
        of bopRelu: faRelu
        of bopRelu6: faRelu6
        else: faReluN1To1
      built.inputs = @[ir(op.inputs[0])]

    else:
      raise newException(TfliteError,
        &"operator {oi} is {builtinName(op.code)}, which this importer does " &
        "not map. Unmapped codes are refused rather than guessed at.")

    discard g.addOp built

  for t in r.graphOutputs:
    g.markOutput ir(t)

  g.validate
  g

proc importTfliteFile*(path: string, name = ""): Graph =
  ## Reads and imports a `.tflite` file from disk.
  if not fileExists(path):
    raise newException(TfliteError, &"no such file: {path}")
  let content = readFile(path)
  var bytes = newSeq[byte](content.len)
  if content.len > 0:
    copyMem(addr bytes[0], unsafeAddr content[0], content.len)
  let stem = if name.len > 0: name else: path.splitFile.name
  importTflite(bytes, stem, path)
