# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

## Host-side backend hook: build-time constant layout.
##
## The runtime's dispatch layer lets a backend replace a *kernel*. That is
## only half of what a real accelerator needs. CMSIS-NN wants weights in a
## particular order; an NPU wants them pre-tiled; a SIMD kernel wants pairs
## interleaved so it can issue one wide load. None of that belongs on the
## device — the arrangement is fixed the moment the model is, so it is a
## compile-time transformation of bytes that are already being written into
## flash.
##
## This module is that transformation. A backend supplies one function:
##
##     transform: (Graph, BackendConst) -> BackendConst
##
## called once for every constant the emitter is about to write, with enough
## context to know what it is looking at — the role, the producing op, the
## dtype and shape. It may return different bytes, a different shape, even a
## different dtype. The emitter re-derives the array length from what comes
## back, so a transform that changes the element count is legal.
##
## Which makes the pairing a real hazard: a reordered weight array is wrong
## unless the target-side kernel reads it in the new order. Nothing in the
## build can check that for you, so a backend declares a `layoutTag`, the
## emitter writes it into the generated module as `WeightLayout`, and the
## target side can refuse to compile against a layout it does not implement:
##
##     static: assert model.WeightLayout == "cmsis-nn/int8-hwc"
##
## The identity backend is the default and costs nothing: an unset `transform`
## is not called at all.

import std/[strformat]
import ./ir

type
  ConstRole* = enum
    ## What a constant *is*. A backend usually cares about exactly one of
    ## these and should pass the rest through untouched.
    crWeights          ## conv / depthwise / fully-connected filter
    crBias             ## bias, with zero-point correction already folded in
    crRequantMult      ## Q31 requantization multipliers
    crRequantShift     ## matching shifts
    crLut              ## 256-entry activation table
    crExpTable         ## softmax exp table
    crOperand          ## a constant read directly by an op

  BackendConst* = object
    ## One constant, on its way to flash.
    role*: ConstRole
    opIndex*: int              ## index into Graph.ops, -1 if not op-specific
    opKind*: OpKind
    name*: string              ## tensor name, or the op name for a table
    dtype*: DType
    shape*: seq[int]
    data*: seq[byte]           ## must stay consistent with dtype and shape

  HostBackend* = object
    ## `name` and `layoutTag` are documentation that ends up in the generated
    ## output; `transform` is the whole mechanism.
    name*: string
    layoutTag*: string
    transform*: proc (g: Graph, c: BackendConst): BackendConst {.closure.}

  BackendError* = object of CatchableError

func noBackend*(): HostBackend =
  ## The default: no transform, no tag, no code emitted differently.
  HostBackend(name: "", layoutTag: "")

func initHostBackend*(name, layoutTag: string,
                      transform: proc (g: Graph, c: BackendConst): BackendConst
                        {.closure.}): HostBackend =
  HostBackend(name: name, layoutTag: layoutTag, transform: transform)

func isActive*(bk: HostBackend): bool = bk.transform != nil

func elementCount*(c: BackendConst): int =
  result = 1
  for d in c.shape: result *= d

proc apply*(bk: HostBackend, g: Graph, c: BackendConst): BackendConst =
  ## Runs the hook and checks the result is self-consistent, because a
  ## mismatch here would otherwise surface as a C array with the wrong length
  ## and a kernel reading past the end of it.
  if not bk.isActive:
    return c
  result = bk.transform(g, c)
  let want = result.elementCount * result.dtype.byteWidth
  if result.data.len != want:
    raise newException(BackendError,
      &"backend '{bk.name}' returned {result.data.len} bytes for " &
      &"'{result.name}' ({result.role}), but shape {result.shape} of " &
      &"{result.dtype} needs {want}")

func changed*(before, after: BackendConst): bool =
  ## Whether the hook actually did anything — used to annotate the output.
  before.dtype != after.dtype or before.shape != after.shape or
    before.data != after.data

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# LAYOUT HELPERS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Byte-level, dtype-agnostic, so they work for int8 weights and int32 tables
# alike. Reordering is all any of these transformations are.

proc transposed*(data: seq[byte], elemSize, rows, cols: int): seq[byte] =
  ## Row-major [rows, cols] to row-major [cols, rows]. The fully-connected
  ## case: a kernel that walks one output channel's weights contiguously wants
  ## the opposite layout from one that walks all channels for a single input.
  if data.len != rows * cols * elemSize:
    raise newException(BackendError,
      &"transposed: {data.len} bytes is not {rows}x{cols} of {elemSize}")
  result = newSeq[byte](data.len)
  for r in 0 ..< rows:
    for c in 0 ..< cols:
      let src = (r * cols + c) * elemSize
      let dst = (c * rows + r) * elemSize
      for b in 0 ..< elemSize:
        result[dst + b] = data[src + b]

proc padTo*(data: seq[byte], elemSize, count, target: int, fill: byte): seq[byte] =
  ## Extends a constant to `target` elements. Accelerators that process a
  ## fixed number of channels per pass want the tail padded rather than
  ## special-cased, and flash is the cheap place to pay for that.
  if target < count:
    raise newException(BackendError,
      &"padTo: target {target} is smaller than the existing {count}")
  result = newSeq[byte](target * elemSize)
  for i in 0 ..< count * elemSize: result[i] = data[i]
  for i in count * elemSize ..< result.len: result[i] = fill
