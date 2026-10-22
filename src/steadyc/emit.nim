# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

## Code emission.
##
## Produces three files per model:
##
##   <name>_weights.h   extern declarations
##   <name>_weights.c   const arrays — weights, folded biases, requant tables
##   <name>.nim         arena, accessors, and a straight-line `invoke`
##
## Weights are emitted as C rather than as Nim `const` deliberately. A Nim
## `const` cannot have its address taken, and a module-level `let` lands in
## `.data` — that is, it gets copied into RAM at startup, which on a part
## with 64 KB of SRAM and 512 KB of flash is precisely backwards. Emitting
## `const` C arrays puts them in `.rodata`, keeps them in flash, gives
## explicit control over alignment and section placement, and keeps the Nim
## compiler from having to parse megabytes of array literals.
##
## `invoke` contains no dispatch, no op table, and no interpreter — just the
## calls, in order, with every buffer address resolved to a constant offset.

import std/[strformat, strutils, tables, times, os, math]
import ./ir, ./quant, ./arena, ./codec, ./backend

const ToolVersion* = "0.1.0"

type EmitError* = object of CatchableError

const MaxIdentLen* = 56
  ## Emitted identifiers are truncated to this. Real TFLite tensor names run to
  ## several hundred characters — a fused MobileNetV2 depthwise carries the
  ## whole batch-norm folding history in its name — and while a long identifier
  ## is legal, it makes generated code unreadable. Truncation can collide, so
  ## the importer checks for that and disambiguates; see `nimIdentKey`.

func cIdent*(s: string): string =
  ## An identifier from an arbitrary tensor or model name, valid in **both** C
  ## and Nim. Nim is the stricter of the two: it rejects a double underscore
  ## and a trailing underscore, both of which fall straight out of mapping
  ## punctuation to `_` — `conv/BiasAdd;conv/Conv2D:0` has runs of separators
  ## and ends in one.
  ##
  ## Exported because the importer needs the *same* mapping to notice names
  ## that are distinct in a model and identical as symbols.
  for c in s:
    if c in {'a'..'z', 'A'..'Z', '0'..'9'}:
      result.add c
    elif result.len > 0 and result[^1] != '_':
      result.add '_'                  # collapse runs, skip leading separators
  while result.len > 0 and result[^1] == '_':
    result.setLen result.len - 1
  if result.len > MaxIdentLen:
    result.setLen MaxIdentLen
    while result.len > 0 and result[^1] == '_':
      result.setLen result.len - 1
  if result.len == 0:
    result = "t"
  elif result[0] in {'0'..'9'}:
    result = "t" & result

func nimIdentKey*(s: string): string =
  ## The form Nim compares identifiers by: first character as written, the rest
  ## case-folded with underscores ignored. `block_1_depthwise` and
  ## `block1Depthwise` are one identifier to Nim and two strings to everyone
  ## else, so uniqueness has to be decided on this key or the generated module
  ## can redeclare a symbol it believes is new.
  let id = cIdent(s)
  if id.len == 0: return id
  result.add id[0]
  for i in 1 ..< id.len:
    if id[i] == '_': continue
    result.add (if id[i] in {'A'..'Z'}: char(ord(id[i]) + 32) else: id[i])

func cSym(g: Graph, prefix, name: string): string =
  &"steady_{cIdent(g.name)}_{prefix}{cIdent(name)}"

func reindent(code: string, by: int): string =
  ## Shifts an already-indented block of generated code further right, keeping
  ## exactly one trailing newline. `strutils.indent` pads the empty line a
  ## trailing "\n" produces and then drops the newline itself, which puts the
  ## next line of output in the wrong place.
  let pad = spaces(by)
  for line in code.strip(leading = false, chars = {'\n'}).splitLines:
    result.add pad & line & "\n"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# C LITERAL FORMATTING
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

proc f32Literal(x: float32): string =
  ## Nine significant digits round-trips float32 exactly, so the constant in
  ## flash is bit-identical to the one the host validated against.
  if x != x: return "(0.0f/0.0f)"
  if x == Inf.float32: return "(1.0f/0.0f)"
  if x == NegInf.float32: return "(-1.0f/0.0f)"
  result = formatFloat(float(x), ffScientific, 9) & "f"

proc emitArrayBody(s: var string, dtype: DType, count: int, data: seq[byte]) =
  const PerLine = 12
  case dtype
  of dtInt8:
    for i in 0 ..< count:
      if i mod PerLine == 0: s.add "\n  "
      s.add $cast[int8](data[i]) & ","
  of dtInt16:
    for i in 0 ..< count:
      if i mod 10 == 0: s.add "\n  "
      var v: int16
      copyMem(addr v, unsafeAddr data[i * 2], 2)
      s.add $v & ","
  of dtInt32:
    for i in 0 ..< count:
      if i mod 8 == 0: s.add "\n  "
      var v: int32
      copyMem(addr v, unsafeAddr data[i * 4], 4)
      s.add $v & ","
  of dtFloat32:
    for i in 0 ..< count:
      if i mod 6 == 0: s.add "\n  "
      var v: float32
      copyMem(addr v, unsafeAddr data[i * 4], 4)
      s.add f32Literal(v) & ","
  s.add "\n"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# WEIGHTS FILES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

type ConstBlob = object
  sym: string
  dtype: DType
  count: int
  body: string
  note: string      ## set when a host backend rewrote the constant

proc emitWeightsFiles(g: Graph, blobs: seq[ConstBlob], stem: string,
                      hdr, src: var string) =
  let guard = "STEADY_" & cIdent(stem).toUpperAscii & "_WEIGHTS_H"
  hdr = &"""
/* Generated by steadyc {ToolVersion}. Do not edit. */
#ifndef {guard}
#define {guard}

#include <stdint.h>

#ifdef __cplusplus
extern "C" {{
#endif

"""
  for b in blobs:
    hdr.add &"extern const {b.dtype.cTypeName} {b.sym}[{b.count}];\n"
  hdr.add """
#ifdef __cplusplus
}
#endif
#endif
"""

  src = &"""
/* Generated by steadyc {ToolVersion}. Do not edit. */
#include "{stem}_weights.h"

/* Alignment keeps word loads legal on strict-alignment cores and keeps DMA
   engines happy. Override STEADY_SECTION to relocate weights into external
   flash. */
#if defined(__GNUC__) || defined(__clang__)
#  define STEADY_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER)
#  define STEADY_ALIGN(n) __declspec(align(n))
#else
#  define STEADY_ALIGN(n)
#endif

#ifndef STEADY_SECTION
#  define STEADY_SECTION
#endif

"""
  for b in blobs:
    if b.note.len > 0:
      src.add &"/* {b.note} */\n"
    src.add &"STEADY_SECTION const {b.dtype.cTypeName} {b.sym}[{b.count}] STEADY_ALIGN(8) = {{"
    src.add b.body
    src.add "};\n\n"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PARAMS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

proc paramsExpr(g: Graph, rp: ResolvedParams, multSym, shiftSym: string,
                perChannel: bool): string =
  let stride = if perChannel: 1 else: 0
  &"AffineParams(mult: cast[ptr UncheckedArray[int32]](addr {multSym}), " &
  &"shift: cast[ptr UncheckedArray[int32]](addr {shiftSym}), " &
  &"channelStride: {stride}, outZeroPoint: {rp.outZeroPoint}'i32, " &
  &"actMin: {rp.qActMin}'i32, actMax: {rp.qActMax}'i32)"

func lit4(a: array[4, int] | array[4, Dim]): string =
  &"[{a[0]}, {a[1]}, {a[2]}, {a[3]}]"

proc storeLiteral(p: PolicyKind, v: int32): string =
  ## A Store-typed literal for clamp bounds and pad values.
  case p
  of pkAffineI8: &"{v}'i8"
  of pkAffineI16x8: &"{v}'i16"

proc storeLiteralF(p: PolicyKind, q: Quant, v: float64): string =
  ## A Store-typed literal for a *real* value the graph specified — the pad
  ## constant. That means quantizing it here, on the host, which is how PAD's
  ## "fill with 0.0" becomes "fill with the zero point" without the kernel or
  ## the caller ever spelling that out.
  case p
  of pkAffineI8: &"{quantizedValue(p, q, v)}'i8"
  of pkAffineI16x8: &"{quantizedValue(p, q, v)}'i16"

proc biasLiteral(p: PolicyKind, v: int32): string =
  ## A Bias-typed literal, in accumulator units: it carries a folded
  ## zero-point correction rather than the model's own bias value.
  case p
  of pkAffineI8, pkAffineI16x8: &"{v}'i32"

proc clampLiteral(p: PolicyKind, fused: FusedAct, outQ: Quant): (string, string) =
  case p
  of pkAffineI8:
    let (lo, hi) = affineActRange(fused, outQ)
    (&"{lo}'i8", &"{hi}'i8")
  of pkAffineI16x8:
    let (lo, hi) = affineActRange(fused, outQ, -32768'i32, 32767'i32)
    (&"{lo}'i16", &"{hi}'i16")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MAIN EMITTER
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

proc emitModel*(g: Graph, p: Plan, outDir, stem: string, sourceHash = "",
                backend = noBackend()) =
  ## Writes <stem>.nim, <stem>_weights.c and <stem>_weights.h into `outDir`.
  ##
  ## Every constant is offered to `backend` on its way out, which is where a
  ## vendor kernel's preferred weight layout gets applied — at build time, once,
  ## rather than on the device every invocation. See `steadyc/backend.nim`.
  createDir(outDir)
  if g.bound > 0:
    # The arena is planned at the bound, which holds only if no buffer is
    # larger at a shorter length
    for t in g.tensors:
      if t.dims.len == 0: continue
      let most = t.elements.at(g.bound)
      for n in 1 ..< g.bound:
        if t.elements.at(n) > most:
          raise newException(EmitError,
            &"tensor '{t.name}' is larger at a length of {n} than at the bound, {g.bound}")
  let alias = resolveAliases(g)
  let policy = g.policy.policyName
  let storeDt = g.policy.storeType
  let storeNim = storeDt.nimTypeName

  var blobs: seq[ConstBlob]
  var nimSrc = ""

  # ---- header -------------------------------------------------------------
  nimSrc.add &"""
# Generated by steadyc {ToolVersion} on {now().format("yyyy-MM-dd HH:mm:ss")}.
# Model: {g.name}   policy: {policy}
"""
  if sourceHash.len > 0:
    nimSrc.add &"# Source model hash: {sourceHash}\n"
  if backend.isActive:
    nimSrc.add &"# Host backend: {backend.name}"
    if backend.layoutTag.len > 0:
      nimSrc.add &"   constant layout: {backend.layoutTag}"
    nimSrc.add "\n"
  nimSrc.add &"""#
# DO NOT EDIT. Regenerate with steadyc.
#
# Peak arena: {p.arenaSize} bytes    Flash constants: {g.totalConstBytes} bytes

import steady/kernels/dispatch

{{.compile: "{stem}_weights.c".}}

const
  ArenaSize* = {p.arenaSize}
  ModelName* = "{g.name}"
  ToolVersion* = "{ToolVersion}"
  TotalMacs* = {g.totalMacs}
    ## Multiply-accumulates per invocation, counted on the host including
    ## padded taps. A compile-time constant, so a benchmark can report MAC/s
    ## and a target-side budget check costs nothing.
"""
  if g.bound > 0:
    nimSrc.add &"  MaxLength* = {g.bound}\n"
    nimSrc.add "    ## The longest length `invoke` takes. The arena and every size below\n"
    nimSrc.add "    ## are for it; `TotalMacs` too, and `macs` gives them for another.\n"
  for i, t in g.inputs:
    nimSrc.add &"  Input{i}Elems* = {g.tensors[t].numElements}\n"
  for i, t in g.outputs:
    nimSrc.add &"  Output{i}Elems* = {g.tensors[t].numElements}\n"
  # The real value an input or output stands for is scale * (q - zero point),
  # as the C header says too; a program moving values between models needs it
  for (port, ts) in [("Input", g.inputs), ("Output", g.outputs)]:
    for i, t in ts:
      let q = g.tensors[t].quant
      if q.scales.len == 1:
        nimSrc.add &"  {port}{i}Scale* = {q.scales[0]:.17e}\n" &
                   &"  {port}{i}ZeroPoint* = {q.zeroAt(0)}'i32\n"
  if sourceHash.len > 0:
    nimSrc.add &"  SourceHash* = \"{sourceHash}\"\n"
  if backend.layoutTag.len > 0:
    # So that target-side code can refuse to build against a constant layout
    # it does not implement: `static: assert model.WeightLayout == "..."`.
    nimSrc.add &"  WeightLayout* = \"{backend.layoutTag}\"\n"

  # ---- constants ----------------------------------------------------------
  var declBlock = ""

  # Code is accumulated one op at a time rather than into a single string, so
  # that the same text can be assembled twice: into `invoke`, and — only under
  # `-d:steadyProfile` — into a per-op entry point the on-target harness calls
  # individually. Attribution is otherwise guesswork, and a profiler that
  # cannot name the operator cannot tell "this layer does more work" from
  # "this kernel is worse".
  var opChunks: seq[string]
  var opCalls = ""

  # Extern declarations are emitted straight into the generated C rather than
  # pulled in via the header. The header exists for people integrating from
  # C; making Nim depend on it would mean depending on an include path that
  # varies with how the project is built.
  var externBlock = ""

  proc declRawConst(sym: string, dtype: DType, count: int) =
    externBlock.add &"extern const {dtype.cTypeName} {sym}[{count}];\n"
    declBlock.add &"  {sym} {{.importc, nodecl.}}: array[{count}, " &
                  &"{dtype.nimTypeName}]\n"

  proc emitConst(sym: string, c0: BackendConst): string =
    ## The single path every flash constant takes: past the host backend,
    ## into the C file, and into the Nim declaration block. Length and dtype
    ## are re-derived from whatever the backend returned, so a transform is
    ## free to change either.
    let c = apply(backend, g, c0)
    let count = c.elementCount
    var body = ""
    emitArrayBody(body, c.dtype, count, c.data)
    var note = ""
    if changed(c0, c):
      let who = if backend.name.len > 0: backend.name else: "host backend"
      note = &"{who}: rewrote '{c0.name}' ({c0.role}) " &
             &"from {c0.dtype} {c0.shape} to {c.dtype} {c.shape}"
    blobs.add ConstBlob(sym: sym, dtype: c.dtype, count: count, body: body,
                        note: note)
    declRawConst(sym, c.dtype, count)
    sym

  proc tensorConst(role: ConstRole, oi: int, kind: OpKind, t: Tensor,
                   data: seq[byte]): BackendConst =
    BackendConst(role: role, opIndex: oi, opKind: kind, name: t.name,
                 dtype: t.dtype, shape: t.shape, data: data)

  proc tableConst(role: ConstRole, oi: int, kind: OpKind, name: string,
                  dtype: DType, xs: seq[int32]): BackendConst =
    BackendConst(role: role, opIndex: oi, opKind: kind, name: name,
                 dtype: dtype, shape: @[xs.len], data: fromInt32(xs))

  # Weight and bias tensors, with zero-point correction folded into biases.
  # The backend sees the *folded* bias, because that is what actually reaches
  # flash — a backend rearranging biases must rearrange the real ones.
  var constSym = initTable[int, string]()
  for oi, op in g.ops:
    if op.kind in {okFullyConnected, okConv2d, okDepthwiseConv2d}:
      let wIdx = op.inputs[1]
      let bIdx = op.inputs[2]
      if wIdx notin constSym:
        constSym[wIdx] = emitConst(
          cSym(g, "w_", g.tensors[wIdx].name),
          tensorConst(crWeights, oi, op.kind, g.tensors[wIdx],
                      g.tensors[wIdx].data))
      if bIdx notin constSym:
        constSym[bIdx] = emitConst(
          cSym(g, "b_", g.tensors[bIdx].name),
          tensorConst(crBias, oi, op.kind, g.tensors[bIdx], foldBias(g, op)))

  # ---- op calls -----------------------------------------------------------
  # The op currently being emitted, so that a constant pulled in on demand can
  # still tell the backend which op reads it.
  var curOp = -1
  var curKind = okReshape
  var softmaxSyms: array[2, string]   # int16 softmax's tables, once per model

  proc bufExpr(t: int): string =
    let tt = g.tensors[t]
    let off = p.offsetFor(g, alias, t)
    &"A({off}, {tt.dtype.nimTypeName})"

  proc castConst(sym: string, dtype: DType): string =
    &"cast[ptr UncheckedArray[{dtype.nimTypeName}]](addr {sym})"

  proc ensureConst(t: int): string =
    ## Weights and biases are registered above, because their bytes are
    ## rewritten by bias folding. Any other constant an op reads — a
    ## concatenation operand, say — is emitted verbatim, on demand.
    if t notin constSym:
      let tt = g.tensors[t]
      if tt.data.len != tt.byteSize:
        raise newException(EmitError,
          &"const tensor '{tt.name}' holds {tt.data.len} bytes, expected " &
          &"{tt.byteSize} for {tt.shape} of {tt.dtype}")
      constSym[t] = emitConst(
        cSym(g, "c_", tt.name),
        tensorConst(crOperand, curOp, curKind, tt, tt.data))
    constSym[t]

  proc constExpr(t: int): string =
    castConst(ensureConst(t), g.tensors[t].dtype)

  proc D(t: int): seq[Dim] = g.tensors[t].dimsOf
    ## A tensor's shape, in terms of the length if the model takes one

  proc E(t: int): Dim = g.tensors[t].elements

  proc paramsOf(rp: ResolvedParams, oi: int, op: Op): string =
    ## A per-tensor op's requantization, its multiplier and shift in flash
    let multSym = emitConst(cSym(g, &"m{oi}_", op.name),
      tableConst(crRequantMult, oi, op.kind, op.name, dtInt32, rp.mult))
    let shiftSym = emitConst(cSym(g, &"s{oi}_", op.name),
      tableConst(crRequantShift, oi, op.kind, op.name, dtInt32, rp.shift))
    paramsExpr(g, rp, multSym, shiftSym, false)

  proc srcExpr(t: int): string =
    ## Where an op reads an operand from: flash if it is constant, the arena
    ## otherwise. Only ops whose operands are genuinely interchangeable use
    ## this — the matmul family keeps its explicit split, since it treats
    ## activations and weights differently anyway.
    if g.tensors[t].kind == tkConst: constExpr(t) else: bufExpr(t)

  for oi, op in g.ops:
    curOp = oi
    curKind = op.kind
    let mark = opCalls.len       # where this op's code starts

    if op.isPureAlias:
      opCalls.add &"  # op{oi} '{op.name}': reshape — aliased, no code emitted\n"
      opChunks.add opCalls[mark .. ^1]
      continue

    opCalls.add &"  # op{oi} '{op.name}' ({op.kind})\n"
    let y = op.outputs[0]

    case op.kind
    of okFullyConnected, okConv2d, okDepthwiseConv2d:
      let x = op.inputs[0]
      let w = op.inputs[1]
      let b = op.inputs[2]
      let wT = g.tensors[w]
      let outC =
        if op.kind == okFullyConnected: wT.shape[0]
        elif op.kind == okConv2d: wT.shape[0]
        else: wT.shape[3]
      if g.policy == pkAffineI16x8:
        checkAccumulator(g, op)
      let rp = resolveMatmulParams(g, g.tensors[x], wT, g.tensors[y],
                                   op.fused, outC)
      let multSym = emitConst(cSym(g, &"m{oi}_", op.name),
        tableConst(crRequantMult, oi, op.kind, op.name, dtInt32, rp.mult))
      let shiftSym = emitConst(cSym(g, &"s{oi}_", op.name),
        tableConst(crRequantShift, oi, op.kind, op.name, dtInt32, rp.shift))
      let prm = paramsExpr(g, rp, multSym, shiftSym, wT.quant.isPerChannel)
      let pv = storeLiteral(g.policy, padValueFor(g, g.tensors[x]))

      case op.kind
      of okFullyConnected:
        opCalls.add &"  fullyConnected({policy}, {bufExpr(y)}, {bufExpr(x)}, " &
                    &"{constExpr(w)}, {constExpr(b)},\n    {prm}, " &
                    &"{wT.shape[0]}, {wT.shape[1]})\n"
      of okConv2d:
        let xs = D(x)
        let ys = D(y)
        opCalls.add &"  conv2d({policy}, {bufExpr(y)}, {bufExpr(x)}, " &
                    &"{constExpr(w)}, {constExpr(b)},\n    {prm},\n" &
                    &"    {xs[1]}, {xs[2]}, {xs[3]}, {ys[1]}, {ys[2]}, {ys[3]}, " &
                    &"{op.kH}, {op.kW}, {op.strideH}, {op.strideW}, " &
                    &"{op.padTop}, {op.padLeft}, {op.dilationH}, {op.dilationW}, {pv})\n"
      of okDepthwiseConv2d:
        let xs = D(x)
        let ys = D(y)
        opCalls.add &"  depthwiseConv2d({policy}, {bufExpr(y)}, {bufExpr(x)}, " &
                    &"{constExpr(w)}, {constExpr(b)},\n    {prm},\n" &
                    &"    {xs[1]}, {xs[2]}, {xs[3]}, {ys[1]}, {ys[2]}, " &
                    &"{op.kH}, {op.kW}, {op.depthMultiplier}, " &
                    &"{op.strideH}, {op.strideW}, {op.padTop}, {op.padLeft}, " &
                    &"{op.dilationH}, {op.dilationW}, {pv})\n"
      else: discard

    of okMaxPool2d, okAvgPool2d:
      let x = op.inputs[0]
      let xs = D(x)
      let ys = D(y)
      let (lo, hi) = clampLiteral(g.policy, op.fused, g.tensors[y].quant)
      let fn = if op.kind == okMaxPool2d: "maxPool2d" else: "avgPool2d"
      opCalls.add &"  {fn}({policy}, {bufExpr(y)}, {bufExpr(x)},\n" &
                  &"    {xs[1]}, {xs[2]}, {xs[3]}, {ys[1]}, {ys[2]}, " &
                  &"{op.kH}, {op.kW}, {op.strideH}, {op.strideW}, " &
                  &"{op.padTop}, {op.padLeft}, {lo}, {hi})\n"

    of okAdd, okSub:
      let a = op.inputs[0]
      let b = op.inputs[1]
      let (rp, rescale) = resolveAddParams(g, g.tensors[a], g.tensors[b],
                                           g.tensors[y], op.fused)
      let prm = paramsOf(rp, oi, op)
      let ys = D(y)
      if op.kind == okAdd and D(a) == ys and D(b) == ys:
        opCalls.add &"  addElementwise({policy}, {bufExpr(y)}, {srcExpr(a)}, " &
                    &"{srcExpr(b)},\n    {prm}, {E(y)},\n"
      else:
        # Broadcast, or a subtraction, which has only the general kernel
        let fn = if op.kind == okAdd: "add4d" else: "sub4d"
        opCalls.add &"  {fn}({policy}, {bufExpr(y)}, {srcExpr(a)}, {srcExpr(b)},\n" &
                    &"    {prm},\n    {lit4(pad4(ys))}, " &
                    &"{lit4(broadcastStrides(D(a), ys))}, " &
                    &"{lit4(broadcastStrides(D(b), ys))},\n"
      opCalls.add &"    {rescale.aMult}'i32, {rescale.aShift}'i32, " &
                  &"{rescale.bMult}'i32, {rescale.bShift}'i32, " &
                  &"{rescale.aOffset}'i32, {rescale.bOffset}'i32)\n"

    of okMul:
      let a = op.inputs[0]
      let b = op.inputs[1]
      let ys = D(y)
      let prm = paramsOf(resolveMulParams(g.tensors[a], g.tensors[b], g.tensors[y], op.fused), oi, op)
      opCalls.add &"  mul4d({policy}, {bufExpr(y)}, {srcExpr(a)}, {srcExpr(b)},\n" &
                  &"    {prm},\n    {lit4(pad4(ys))}, " &
                  &"{lit4(broadcastStrides(D(a), ys))}, " &
                  &"{lit4(broadcastStrides(D(b), ys))})\n"

    of okBatchMatmul:
      let a = op.inputs[0]
      let b = op.inputs[1]
      let xs = D(a)
      let ys = D(y)
      let prm = paramsOf(resolveBatchMatmulParams(g.tensors[a], g.tensors[b], g.tensors[y]), oi, op)
      let (rows, depth, cols) = (xs[^2], xs[^1], ys[^1])
      let batches = E(y) div (rows * cols)
      opCalls.add &"  batchMatmul({policy}, {bufExpr(y)}, {srcExpr(a)}, {srcExpr(b)},\n" &
                  &"    {prm},\n    {batches}, {rows}, {depth}, {cols}, " &
                  &"{rows * depth}, {depth * cols}, {op.adjY})\n"

    of okTranspose:
      let x = op.inputs[0]
      let xs = D(x)
      let ys = D(y)
      let inStrides = strides4(xs)
      var strides: array[4, Dim]
      for d in 0 ..< 4:
        # Output dimension d, padded to four, reads the input's perm[d]
        let k = d - (4 - ys.len)
        strides[d] = if k < 0: dim(0) else: inStrides[op.perm[k] + 4 - xs.len]
      opCalls.add &"  copy4d({policy}, {bufExpr(y)}, {srcExpr(x)}, " &
                  &"{lit4(pad4(ys))}, {lit4(strides)}, 0)\n"

    of okGather:
      let table = g.tensors[op.inputs[0]]
      let ids = op.inputs[1]
      opCalls.add &"  gather({policy}, {bufExpr(y)}, {constExpr(op.inputs[0])}, " &
                  &"{bufExpr(ids)}, {E(ids)}, {table.shape[1]})\n"

    of okRequantize:
      let x = op.inputs[0]
      let rp = resolveRequantizeParams(g.tensors[x], g.tensors[y])
      opCalls.add &"  requantize({policy}, {bufExpr(y)}, {srcExpr(x)}, " &
                  &"{E(y)}, {rp.mult[0]}'i32, {rp.shift[0]}'i32, " &
                  &"{g.tensors[x].quant.zeroAt(0)}'i32, {rp.outZeroPoint}'i32)\n"

    of okAddBand:
      # The ADD's rescaling, its operands in the order the file had them
      let (a, x) = (op.inputs[0], op.inputs[1])
      let (first, second) = if op.bandSide == 0: (x, a) else: (a, x)
      let (rp, r) = resolveAddParams(g, g.tensors[first], g.tensors[second], g.tensors[y], op.fused)
      let prm = paramsOf(rp, oi, op)
      let (am, ash, ao, bm, bsh, bo) =
        if op.bandSide == 0: (r.bMult, r.bShift, r.bOffset, r.aMult, r.aShift, r.aOffset)
        else: (r.aMult, r.aShift, r.aOffset, r.bMult, r.bShift, r.bOffset)
      let xs = D(x)
      opCalls.add &"  addBand({policy}, {bufExpr(y)}, {srcExpr(a)}, {srcExpr(x)},\n    {prm},\n" &
                  &"    {E(x) div (xs[^2] * xs[^1])}, {xs[^2]}, {xs[^1]}, {op.bandStart},\n" &
                  &"    {am}'i32, {ash}'i32, {bm}'i32, {bsh}'i32, {ao}'i32, {bo}'i32)\n"

    of okBand, okUnband:
      let x = op.inputs[0]
      let xs = D(x)
      let n = xs[^2]
      let w = if op.kind == okBand: xs[^1] else: D(y)[^1]
      let fn = if op.kind == okBand: "band" else: "unband"
      opCalls.add &"  {fn}({policy}, {bufExpr(y)}, {srcExpr(x)}, {E(x) div (n * xs[^1])}, " &
                  &"{n}, {w}, {op.bandStart}, {storeLiteral(g.policy, g.tensors[x].quant.zeroAt(0))})\n"

    of okRsqrt:
      let x = op.inputs[0]
      let lut = rsqrtLut16(float32(g.tensors[x].quant.scaleAt(0)),
                           float32(g.tensors[y].quant.scaleAt(0)))
      let sym = emitConst(cSym(g, &"lut{oi}_", op.name),
        BackendConst(role: crLut, opIndex: oi, opKind: op.kind, name: op.name,
                     dtype: dtInt16, shape: @[lut.len], data: fromInt16(lut)))
      opCalls.add &"  lut16({policy}, {bufExpr(y)}, {srcExpr(x)}, " &
                  &"{E(y)}, {castConst(sym, dtInt16)})\n"

    of okClamp:
      let x = op.inputs[0]
      let (lo, hi) = clampLiteral(g.policy, op.fused, g.tensors[y].quant)
      opCalls.add &"  clamp1d({policy}, {bufExpr(y)}, {bufExpr(x)}, " &
                  &"{E(y)}, {lo}, {hi})\n"

    of okPad:
      let x = op.inputs[0]
      let xs = D(x)
      let pv = storeLiteralF(g.policy, g.tensors[x].quant, op.padConst)
      if g.policy == pkAffineI16x8:
        var before = newSeq[Dim]()
        for p in op.padsOf: before.add p[0]
        opCalls.add &"  pad4d({policy}, {bufExpr(y)}, {srcExpr(x)}, " &
                    &"{lit4(pad4(D(y)))}, {lit4(pad4(before, dim(0)))}, " &
                    &"{lit4(pad4(xs))}, {pv})\n"
        opChunks.add opCalls[mark .. ^1]
        continue
      opCalls.add &"  pad2d({policy}, {bufExpr(y)}, {srcExpr(x)},\n" &
                  &"    {xs[1]}, {xs[2]}, {xs[3]}, " &
                  &"{op.pads[1][0]}, {op.pads[1][1]}, " &
                  &"{op.pads[2][0]}, {op.pads[2][1]}, {pv})\n"

    of okConcatenation:
      let ys = D(y)
      let ax = op.axis
      var outer = dim(1)
      for d in 0 ..< ax: outer = outer * ys[d]
      var innerOut = dim(1)
      for d in ax ..< ys.len: innerOut = innerOut * ys[d]

      var ins: seq[Tensor]
      for t in op.inputs: ins.add g.tensors[t]

      # An operand already carrying the output's quantization is copied
      # verbatim; only the others pay for arithmetic. For a real-number policy
      # there is no quantization to differ, so this is always a copy.
      var anyRescaled = false
      for t in ins:
        if not sameQuant(t.quant, g.tensors[y].quant): anyRescaled = true

      var prm = ""
      var rescales: seq[OperandRescale]
      if anyRescaled:
        let (rp, rs) = resolveConcatParams(g, ins, g.tensors[y], op.fused)
        rescales = rs
        let multSym = emitConst(cSym(g, &"m{oi}_", op.name),
          tableConst(crRequantMult, oi, op.kind, op.name, dtInt32, rp.mult))
        let shiftSym = emitConst(cSym(g, &"s{oi}_", op.name),
          tableConst(crRequantShift, oi, op.kind, op.name, dtInt32, rp.shift))
        prm = paramsExpr(g, rp, multSym, shiftSym, false)

      var dstOff = dim(0)
      for k, t in op.inputs:
        let ts = D(t)
        var innerIn = dim(1)
        for d in ax ..< ts.len: innerIn = innerIn * ts[d]
        if sameQuant(g.tensors[t].quant, g.tensors[y].quant):
          opCalls.add &"  concatSlice({policy}, {bufExpr(y)}, {srcExpr(t)}, " &
                      &"{outer}, {innerIn}, {innerOut}, {dstOff})\n"
        else:
          let r = rescales[k]
          opCalls.add &"  concatSliceRescaled({policy}, {bufExpr(y)}, " &
                      &"{srcExpr(t)},\n    {prm},\n" &
                      &"    {outer}, {innerIn}, {innerOut}, {dstOff}, " &
                      &"{r.mult}'i32, {r.shift}'i32, {r.offset}'i32)\n"
        dstOff = dstOff + innerIn

    of okMean:
      let x = op.inputs[0]
      let xs = D(x)
      if g.policy == pkAffineI16x8:
        let prm = paramsOf(resolveMeanLastParams(g.tensors[x], g.tensors[y], xs[^1].value), oi, op)
        opCalls.add &"  meanLastAxis({policy}, {bufExpr(y)}, {srcExpr(x)}, {prm}, " &
                    &"{E(y)}, {xs[^1]})\n"
        opChunks.add opCalls[mark .. ^1]
        continue
      let count = (xs[1] * xs[2]).value
      let (rp, corr) = resolveMeanParams(g, g.tensors[x], g.tensors[y], count,
                                         op.fused)
      let multSym = emitConst(cSym(g, &"m{oi}_", op.name),
        tableConst(crRequantMult, oi, op.kind, op.name, dtInt32, rp.mult))
      let shiftSym = emitConst(cSym(g, &"s{oi}_", op.name),
        tableConst(crRequantShift, oi, op.kind, op.name, dtInt32, rp.shift))
      let prm = paramsExpr(g, rp, multSym, shiftSym, false)
      opCalls.add &"  meanSpatial({policy}, {bufExpr(y)}, {srcExpr(x)}, " &
                  &"{biasLiteral(g.policy, corr)},\n    {prm},\n" &
                  &"    {xs[1]}, {xs[2]}, {xs[3]})\n"

    of okLogistic, okTanh:
      let x = op.inputs[0]
      if g.policy == pkAffineI16x8:
        # No table of every input for 16 bits: TFLite's integer tanh and
        # logistic, whose input scaling is worked out here.
        let (mult, leftShift) = resolveTanhInt16Params(g.tensors[x], op.kind == okLogistic)
        let fn = if op.kind == okLogistic: "logistic16" else: "tanh16"
        opCalls.add &"  {fn}({policy}, {bufExpr(y)}, {srcExpr(x)}, " &
                    &"{E(y)}, {mult}'i32, {leftShift}'i32)\n"
        opChunks.add opCalls[mark .. ^1]
        continue
      # The activation is evaluated on the host at all 256 representable
      # inputs and quantized into flash; the target does a load.
      let lut = buildLut(g.policy, g.tensors[x].quant, g.tensors[y].quant,
                         activationFn(op.kind))
      let sym = emitConst(cSym(g, &"lut{oi}_", op.name),
        BackendConst(role: crLut, opIndex: oi, opKind: op.kind, name: op.name,
                     dtype: storeDt, shape: @[LutSize], data: lut))
      opCalls.add &"  lut1d({policy}, {bufExpr(y)}, {srcExpr(x)}, " &
                  &"{E(y)}, {castConst(sym, storeDt)})\n"

    of okSoftmax:
      let x = op.inputs[0]
      let classes = D(x)[^1]
      let rows = E(x) div classes
      if g.policy == pkAffineI16x8:
        # TFLite's two tables, which depend on nothing in the model, are shared
        if softmaxSyms[0].len == 0:
          let (e, r) = softmaxLuts16()
          softmaxSyms[0] = emitConst(cSym(g, "softmax_exp", ""),
            BackendConst(role: crLut, opIndex: oi, opKind: op.kind, name: "softmax_exp",
                         dtype: dtInt16, shape: @[e.len], data: fromInt16(e)))
          softmaxSyms[1] = emitConst(cSym(g, "softmax_reciprocal", ""),
            BackendConst(role: crLut, opIndex: oi, opKind: op.kind, name: "softmax_reciprocal",
                         dtype: dtInt16, shape: @[r.len], data: fromInt16(r)))
        let (mult, shift) = resolveSoftmax16Params(g.tensors[x])
        opCalls.add &"  softmax16({policy}, {bufExpr(y)}, {srcExpr(x)}, {rows}, {classes}, " &
                    &"{castConst(softmaxSyms[0], dtInt16)}, {castConst(softmaxSyms[1], dtInt16)}, " &
                    &"{mult}'i32, {shift}'i32)\n"
        opChunks.add opCalls[mark .. ^1]
        continue
      let (tbl, bits) = buildExpLut(g.tensors[x].quant.scaleAt(0), classes.value)
      let sym = emitConst(cSym(g, &"exp{oi}_", op.name),
        tableConst(crExpTable, oi, op.kind, op.name, dtInt32, tbl))
      let rp = resolveSoftmaxParams(g, g.tensors[y])
      opCalls.add &"  # exp table: exp(-d * inScale) in Q{bits}, " &
                  &"{tbl.len} entries; {rows} row(s) of {classes}\n"
      opCalls.add &"  softmax({policy}, {bufExpr(y)}, {srcExpr(x)}, " &
                  &"{rows}, {classes}, {castConst(sym, dtInt32)},\n" &
                  &"    {rp.mult[0]}'i32, {rp.shift[0]}'i32, " &
                  &"{rp.outZeroPoint}'i32, {rp.qActMin}'i32, {rp.qActMax}'i32)\n"

    of okLeakyRelu:
      let x = op.inputs[0]
      let lp = resolveLeakyReluParams(g.tensors[x], g.tensors[y], op.alpha)
      opCalls.add &"  leakyRelu1d({policy}, {bufExpr(y)}, {srcExpr(x)}, " &
                  &"{E(y)},\n    {lp.inZero}'i32, {lp.outZero}'i32, " &
                  &"{lp.identityMult}'i32, {lp.identityShift}'i32, " &
                  &"{lp.alphaMult}'i32, {lp.alphaShift}'i32)\n"

    of okSlice:
      let x = op.inputs[0]
      let xs = D(x)
      let ys = D(y)
      if xs.len != 4 or op.begin[0] != 0:
        # Not a crop of an image: walked along the input's strides
        let st = strides4(xs)
        var first = dim(0)
        for d, b in op.beginOf: first = first + b * st[d + 4 - xs.len]
        opCalls.add &"  copy4d({policy}, {bufExpr(y)}, {srcExpr(x)}, " &
                    &"{lit4(pad4(ys))}, {lit4(st)}, {first})\n"
        opChunks.add opCalls[mark .. ^1]
        continue
      opCalls.add &"  slice2d({policy}, {bufExpr(y)}, {srcExpr(x)}, " &
                  &"{xs[2]}, {xs[3]}, {ys[1]}, {ys[2]}, {ys[3]}, " &
                  &"{op.begin[1]}, {op.begin[2]}, {op.begin[3]})\n"

    of okReshape:
      discard

    opChunks.add opCalls[mark .. ^1]

  # ---- assemble -----------------------------------------------------------
  if declBlock.len > 0:
    nimSrc.add "\n# Flash-resident constants, defined in the companion C file.\n"
    nimSrc.add "{.emit: \"\"\"/*TYPESECTION*/\n#include <stdint.h>\n"
    nimSrc.add externBlock
    nimSrc.add "\"\"\".}\n\nlet\n"
    nimSrc.add declBlock

  nimSrc.add &"""

var arena {{.align(8).}}: array[ArenaSize, byte]

template A(off: int, T: typedesc): ptr UncheckedArray[T] =
  cast[ptr UncheckedArray[T]](addr arena[off])

"""
  if g.bound > 0:
    var total = dim(0)
    for op in g.ops: total = total + g.macDims(op)
    nimSrc.add &"func macs*(n: int): int =\n" &
               &"  ## Multiply-accumulates for a length of `n`, counted as `TotalMacs` is\n" &
               &"  {total}\n\n"

  for i, t in g.inputs:
    let tt = g.tensors[t]
    nimSrc.add &"proc input{i}*(): ptr UncheckedArray[{tt.dtype.nimTypeName}] " &
               &"{{.inline.}} =\n" &
               &"  ## '{tt.name}', shape {tt.shape}, {tt.numElements} elements\n" &
               &"  ##\n" &
               &"  ## Scratch space, not storage: the arena reuses these bytes " &
               &"once the\n" &
               &"  ## model has read them, so refill every input before every " &
               &"invoke.\n" &
               &"  {bufExpr(t)}\n\n"
  for i, t in g.outputs:
    let tt = g.tensors[t]
    nimSrc.add &"proc output{i}*(): ptr UncheckedArray[{tt.dtype.nimTypeName}] " &
               &"{{.inline.}} =\n" &
               &"  ## '{tt.name}', shape {tt.shape}, {tt.numElements} elements\n" &
               &"  ##\n" &
               &"  ## Valid until the next invoke.\n" &
               &"  {bufExpr(t)}\n\n"

  if g.bound > 0:
    nimSrc.add "proc invoke*(n: int) =\n"
    nimSrc.add "  ## Straight-line execution, for a length `n` from 1 to `MaxLength`. The\n"
    nimSrc.add "  ## inputs hold their first `n` rows, and so do the outputs after.\n"
    nimSrc.add "  doAssert n >= 1 and n <= MaxLength, \"the length is from 1 to MaxLength\"\n"
  else:
    nimSrc.add "proc invoke*() =\n"
    nimSrc.add "  ## Straight-line execution. No interpreter, no dispatch table.\n"
  if opCalls.len == 0:
    nimSrc.add "  discard\n"
  else:
    nimSrc.add opCalls

  # ---- profiling entry points ---------------------------------------------
  # Off unless asked for, and additive when it is on: `invoke` above is
  # byte-for-byte the same code either way, so a profiled build measures the
  # kernels the unprofiled one runs. The cost of a hook inside `invoke` would
  # be small but it would be measured, and on this scale it is the measurement
  # that has to be trustworthy.
  if g.ops.len > 0:
    nimSrc.add &"""
when defined(steadyProfile):
  ## Per-operator entry points, for the on-target benchmark harness. One `case`
  ## over a compile-time constant, so each branch is still the same
  ## straight-line code `invoke` contains — no dispatch reaches a kernel.
  ##
  ## Calling an op out of sequence computes nonsense, deliberately: the arena
  ## packs buffers, so op N's inputs generally no longer exist by the time op
  ## N+1 has run. Integer kernels are branch-light and data-independent, so
  ## nonsense costs exactly what real data costs, which is what makes timing
  ## one op at a time legitimate. Anything that reads a *result* must call
  ## `invoke`.

  type OpProfile* = object
    ## `cstring` rather than `string`: these are plain pointers into flash,
    ## with no runtime behind them, so the table costs a freestanding build
    ## nothing and an on-target profiler can print op names without linking
    ## Nim's string machinery. Use `$name` on a host.
    name*: cstring
    kind*: cstring
    macs*: int          ## host-counted, padded taps included
    outElems*: int      ## output tensor size, the measure for data movement

  const
    OpCount* = {g.ops.len}
    OpProfiles*: array[OpCount, OpProfile] = [
"""
    for oi, op in g.ops:
      let outElems = g.tensors[op.outputs[0]].numElements
      let macs = if op.isPureAlias: 0 else: g.macCount(op)
      nimSrc.add &"      OpProfile(name: \"{op.name}\", kind: \"{op.kind}\", " &
                 &"macs: {macs}, outElems: {outElems}),\n"
    nimSrc.add "    ]\n\n  proc invokeOp*(" & (if g.bound > 0: "i, n: int" else: "i: int") &
               ") =\n    case i\n"
    for oi, chunk in opChunks:
      nimSrc.add &"    of {oi}:\n"
      nimSrc.add reindent(chunk, 4)
      # A branch holding only a comment — an aliased reshape — is not a
      # statement list Nim will accept.
      var hasCode = false
      for line in chunk.splitLines:
        let s = line.strip
        if s.len > 0 and not s.startsWith("#"): hasCode = true
      if not hasCode:
        nimSrc.add "      discard\n"
    nimSrc.add "    else: discard\n"

  var hdr, src: string
  emitWeightsFiles(g, blobs, stem, hdr, src)

  writeFile(outDir / (stem & "_weights.h"), hdr)
  writeFile(outDir / (stem & "_weights.c"), src)
  writeFile(outDir / (stem & ".nim"), nimSrc)
