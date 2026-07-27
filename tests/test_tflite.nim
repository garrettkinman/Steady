# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

## The FlatBuffers reader and the TFLite importer.
##
## Two halves, tested differently. The reader is checked against a buffer
## assembled byte by byte here, because its whole job is offset arithmetic and
## the only way to be sure of that is to know exactly what the bytes are. The
## importer is checked against real files when they are present — see
## `tests/models/fetch.sh` — and against malformed input always, since refusing
## a bad file cleanly is the part that has to work even with no fixtures around.

import std/[unittest, os, strutils]
import steadyc
# The reader is an implementation detail of the importer, so it is imported
# directly rather than widening what `steadyc` re-exports.
import steadyc/flatbuffers

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

proc le32(v: int32): seq[byte] =
  result = newSeq[byte](4)
  copyMem(addr result[0], unsafeAddr v, 4)

proc le16(v: uint16): seq[byte] =
  result = newSeq[byte](2)
  copyMem(addr result[0], unsafeAddr v, 2)

proc handBuilt(): seq[byte] =
  ## A minimal flatbuffer holding one table with two int32 fields:
  ##
  ##   0   u32 16      root table offset
  ##   4   "TFL3"      file identifier
  ##   8   u16 8       vtable length
  ##   10  u16 12      table length
  ##   12  u16 4       field 0 lives at table + 4
  ##   14  u16 8       field 1 lives at table + 8
  ##   16  i32 8       table: signed offset back to the vtable (16 - 8)
  ##   20  i32 42      field 0
  ##   24  i32 -7      field 1
  result.add le32(16'i32)
  for c in "TFL3": result.add byte(c)
  result.add le16(8'u16)
  result.add le16(12'u16)
  result.add le16(4'u16)
  result.add le16(8'u16)
  result.add le32(8'i32)
  result.add le32(42'i32)
  result.add le32(-7'i32)

suite "flatbuffers reader":

  test "reads fields from a buffer whose every byte is known":
    let buf = newFbBuffer(handBuilt())
    check buf.fileIdentifier == "TFL3"
    let t = buf.root
    check t.fieldI32(0) == 42'i32
    check t.fieldI32(1) == -7'i32
    check t.has(0)
    check t.has(1)

  test "a field past the end of the vtable is absent, not an error":
    # This is how FlatBuffers encodes defaults, and how a file written against
    # an older schema stays readable.
    let buf = newFbBuffer(handBuilt())
    let t = buf.root
    check not t.has(2)
    check t.fieldI32(2, 99'i32) == 99'i32
    check t.fieldU32(7, 5'u32) == 5'u32
    check t.fieldVector(3).len == 0        # absent vectors read as empty

  test "a truncated buffer is rejected rather than read off the end":
    let full = handBuilt()
    for cut in [0, 4, 8, 12, 16, 20, 24]:
      let buf = newFbBuffer(full[0 ..< cut])
      var raised = false
      try:
        let t = buf.root
        discard t.fieldI32(0)
        discard t.fieldI32(1)
      except FbError:
        raised = true
      check raised

  test "a root offset pointing outside the buffer is rejected":
    var bytes = handBuilt()
    let bad = le32(int32(bytes.len + 64))
    for i in 0 ..< 4: bytes[i] = bad[i]
    expect FbError:
      discard newFbBuffer(bytes).root

suite "importer rejections":

  test "a file that is not a flatbuffer at all":
    var junk = newSeq[byte](64)
    for i in 0 ..< junk.len: junk[i] = byte(i)
    expect CatchableError:                  # FbError or TfliteError, both fine
      discard importTflite(junk, "junk")

  test "a flatbuffer without the TFLite identifier":
    var bytes = handBuilt()
    for i, c in "XXXX": bytes[4 + i] = byte(c)
    expect TfliteError:
      discard importTflite(bytes, "wrong")

  test "a missing file says so":
    expect TfliteError:
      discard importTfliteFile("no/such/model.tflite")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Real files, when they have been fetched. What is checked here is the shape of
# the result — that the importer produced a graph the rest of the compiler
# accepts, with the operators and quantization the file describes. Whether the
# *numbers* match TFLite is tests/models/check.sh, which needs an interpreter.

const ModelDir = currentSourcePath().parentDir / "models"

proc fixture(name: string): string = ModelDir / (name & ".tflite")

suite "importing real models":

  if not fileExists(fixture("kws")):
    echo "  (skipped: run tests/models/fetch.sh to enable)"
  else:
    test "a depthwise-separable CNN imports, validates and plans":
      let g = importTfliteFile(fixture("kws"))
      check g.policy == pkAffineI8
      check g.ops.len == 13
      check g.inputs.len == 1
      check g.outputs.len == 1
      check g.tensors[g.inputs[0]].shape == @[1, 49, 10, 1]
      check g.tensors[g.outputs[0]].shape == @[1, 12]
      var kinds: seq[OpKind]
      for op in g.ops:
        if op.kind notin kinds: kinds.add op.kind
      check okConv2d in kinds
      check okDepthwiseConv2d in kinds
      check okSoftmax in kinds
      let p = planOne(g)
      check p.arenaSize > 0
      # Reuse is the point of the planner; on a 13-layer model it should be
      # saving most of the naive footprint.
      var naive = 0
      for r in p.ranges: naive += r.size
      check p.arenaSize * 2 < naive

    test "every operator's weights keep the file's quantization axis":
      let g = importTfliteFile(fixture("kws"))
      for op in g.ops:
        if op.kind == okDepthwiseConv2d:
          # TFLite lays depthwise filters out as [1, kH, kW, outC], so the
          # per-channel axis is 3 and not 0. Getting this wrong silently
          # transposes the requantization.
          let w = g.tensors[op.inputs[1]]
          check w.shape[0] == 1
          if w.quant.isPerChannel:
            check w.quant.axis == 3
            check w.quant.scales.len == w.shape[3]
        elif op.kind == okConv2d:
          let w = g.tensors[op.inputs[1]]
          if w.quant.isPerChannel:
            check w.quant.axis == 0
            check w.quant.scales.len == w.shape[0]

  if not fileExists(fixture("resnet8")):
    discard
  else:
    test "residual adds survive with their operands' own quantization":
      let g = importTfliteFile(fixture("resnet8"))
      var adds = 0
      for op in g.ops:
        if op.kind == okAdd:
          inc adds
          let a = g.tensors[op.inputs[0]]
          let b = g.tensors[op.inputs[1]]
          check a.numElements == b.numElements
          check a.quant.scales.len == 1
          check b.quant.scales.len == 1
      check adds == 3

  if not fileExists(fixture("vww")):
    discard
  else:
    test "a float I/O boundary is stripped, leaving an int8 model":
      # The file's own input is float32 followed by a QUANTIZE. What the
      # compiler exposes is the quantized tensor, so the device does no
      # conversion and the caller gets the scale in the generated header.
      let g = importTfliteFile(fixture("vww"))
      check g.tensors[g.inputs[0]].dtype == dtInt8
      check g.tensors[g.outputs[0]].dtype == dtInt8
      check g.tensors[g.inputs[0]].quant.scales.len == 1
      check g.tensors[g.inputs[0]].name.contains("int8")
      # The first surviving op consumes the graph input directly: nothing was
      # left behind to convert it.
      check g.ops[0].kind == okConv2d
      check g.ops[0].inputs[0] == g.inputs[0]
      # 27 conv/depthwise layers plus pool, reshape, matmul and softmax — and
      # no leftover boundary op, which would have failed validation anyway
      # since there is no op kind for one.
      check g.ops.len == 31

  if not fileExists(fixture("person_detect")):
    discard
  else:
    test "a 2019-vintage model imports despite its stale bias metadata":
      # Modern TFLite refuses this file: its converter wrote a
      # quantized_dimension of 3 on rank-1 bias tensors. Bias scales are
      # recomputed here from input and weight scales, so the stale field is
      # simply never read.
      let g = importTfliteFile(fixture("person_detect"))
      check g.ops.len == 31
      check g.tensors[g.inputs[0]].shape == @[1, 96, 96, 1]
      discard planOne(g)
