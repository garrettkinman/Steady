# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

## Minimal read-only FlatBuffers access.
##
## Reading FlatBuffers is small enough to own outright: a table is a position
## plus a vtable of field offsets, a vector is a length followed by elements,
## and every reference is a relative offset. There is no allocation, no schema
## compiler, and no generated code — the importer names its own field indices,
## which keeps them auditable against `schema.fbs` instead of buried in a
## megabyte of machine-written accessors.
##
## Owning it also means the bounds checks are ours. A `.tflite` file is
## untrusted input: it arrives from a converter, a download, or a customer,
## and a malformed one must produce an error naming the offset, not a
## segfault. Every offset computed here is checked against the buffer length.
## This is host-side code, so the checks cost nothing that matters.
##
## Absent fields are the normal case in FlatBuffers, not an error — a writer
## omits anything equal to its default. So scalar readers take a default and
## `has` exists for the cases where present-but-default and absent differ.

import std/strformat

type
  FbError* = object of CatchableError

  FbBuffer* = ref object
    bytes*: seq[byte]

  FbTable* = object
    buf*: FbBuffer
    pos*: int             ## first byte of the table
    vt*: int              ## first byte of its vtable
    vtSize*: int          ## vtable length in bytes

  FbVector* = object
    buf*: FbBuffer
    pos*: int             ## first byte of element 0
    len*: int             ## element count

proc newFbBuffer*(bytes: sink seq[byte]): FbBuffer =
  FbBuffer(bytes: bytes)

proc need(buf: FbBuffer, at, n: int, what: string) =
  ## A proc rather than a template: `&` expands before a template's parameters
  ## are substituted, so the interpolation would not see them.
  if at < 0 or n < 0 or at + n > buf.bytes.len:
    raise newException(FbError,
      &"{what}: {n} bytes at offset {at} is outside a {buf.bytes.len}-byte buffer")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SCALARS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FlatBuffers is little-endian on the wire regardless of host, but every target
# this compiler runs on is little-endian too, so a copy is enough.

proc readAt[T](buf: FbBuffer, at: int, what: string): T =
  buf.need(at, sizeof(T), what)
  copyMem(addr result, unsafeAddr buf.bytes[at], sizeof(T))

proc u8*(buf: FbBuffer, at: int): uint8 = readAt[uint8](buf, at, "u8")
proc i8*(buf: FbBuffer, at: int): int8 = readAt[int8](buf, at, "i8")
proc u16*(buf: FbBuffer, at: int): uint16 = readAt[uint16](buf, at, "u16")
proc i32*(buf: FbBuffer, at: int): int32 = readAt[int32](buf, at, "i32")
proc u32*(buf: FbBuffer, at: int): uint32 = readAt[uint32](buf, at, "u32")
proc i64*(buf: FbBuffer, at: int): int64 = readAt[int64](buf, at, "i64")
proc f32*(buf: FbBuffer, at: int): float32 = readAt[float32](buf, at, "f32")

proc offsetAt(buf: FbBuffer, at: int): int =
  ## Follows a uoffset: a forward relative offset stored at `at`.
  let rel = int(buf.u32(at))
  result = at + rel
  if result <= at or result >= buf.bytes.len:
    raise newException(FbError,
      &"offset at {at} points to {result}, outside a {buf.bytes.len}-byte buffer")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TABLES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

proc initFbTable*(buf: FbBuffer, pos: int): FbTable =
  ## A table stores a *signed* offset back to its vtable, which several tables
  ## may share.
  let soffset = int(buf.i32(pos))
  let vt = pos - soffset
  if vt < 0 or vt + 4 > buf.bytes.len:
    raise newException(FbError, &"table at {pos} has a vtable at {vt}, out of range")
  let vtSize = int(buf.u16(vt))
  if vtSize < 4 or vt + vtSize > buf.bytes.len:
    raise newException(FbError, &"vtable at {vt} claims {vtSize} bytes, out of range")
  FbTable(buf: buf, pos: pos, vt: vt, vtSize: vtSize)

proc root*(buf: FbBuffer): FbTable =
  ## The root table, whose uoffset sits at the very start of the file.
  if buf.bytes.len < 8:
    raise newException(FbError, &"a {buf.bytes.len}-byte buffer is too short to hold a root table")
  initFbTable(buf, buf.offsetAt(0))

proc fileIdentifier*(buf: FbBuffer): string =
  ## The optional four-byte identifier at offset 4 — "TFL3" for TFLite.
  if buf.bytes.len < 8: return ""
  result = newString(4)
  for i in 0 ..< 4: result[i] = char(buf.bytes[4 + i])

proc fieldPos(t: FbTable, field: int): int =
  ## 0 when the field is absent, which is how FlatBuffers encodes "default".
  let vo = 4 + field * 2
  if vo + 2 > t.vtSize: return 0
  let rel = int(t.buf.u16(t.vt + vo))
  if rel == 0: 0 else: t.pos + rel

proc has*(t: FbTable, field: int): bool = t.fieldPos(field) != 0

proc fieldU8*(t: FbTable, field: int, default = 0'u8): uint8 =
  let p = t.fieldPos(field)
  if p == 0: default else: t.buf.u8(p)

proc fieldI8*(t: FbTable, field: int, default = 0'i8): int8 =
  let p = t.fieldPos(field)
  if p == 0: default else: t.buf.i8(p)

proc fieldBool*(t: FbTable, field: int, default = false): bool =
  let p = t.fieldPos(field)
  if p == 0: default else: t.buf.u8(p) != 0

proc fieldI32*(t: FbTable, field: int, default = 0'i32): int32 =
  let p = t.fieldPos(field)
  if p == 0: default else: t.buf.i32(p)

proc fieldU32*(t: FbTable, field: int, default = 0'u32): uint32 =
  let p = t.fieldPos(field)
  if p == 0: default else: t.buf.u32(p)

proc fieldF32*(t: FbTable, field: int, default = 0'f32): float32 =
  let p = t.fieldPos(field)
  if p == 0: default else: t.buf.f32(p)

proc fieldTable*(t: FbTable, field: int): FbTable =
  ## Only call when `has` — an absent table has no position to report.
  let p = t.fieldPos(field)
  if p == 0:
    raise newException(FbError, &"field {field} is absent, so it has no table")
  initFbTable(t.buf, t.buf.offsetAt(p))

proc fieldString*(t: FbTable, field: int): string =
  let p = t.fieldPos(field)
  if p == 0: return ""
  let sp = t.buf.offsetAt(p)
  let n = int(t.buf.u32(sp))
  t.buf.need(sp + 4, n, "string")
  result = newString(n)
  for i in 0 ..< n: result[i] = char(t.buf.bytes[sp + 4 + i])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# VECTORS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

proc fieldVector*(t: FbTable, field: int): FbVector =
  ## An absent vector reads as empty, which is what every caller wants.
  let p = t.fieldPos(field)
  if p == 0:
    return FbVector(buf: t.buf, pos: 0, len: 0)
  let vp = t.buf.offsetAt(p)
  let n = int(t.buf.u32(vp))
  if n < 0:
    raise newException(FbError, &"vector at {vp} has a negative length")
  FbVector(buf: t.buf, pos: vp + 4, len: n)

proc boundsCheck(v: FbVector, i: int) =
  if i < 0 or i >= v.len:
    raise newException(FbError, &"index {i} is outside a {v.len}-element vector")

proc elemPos(v: FbVector, i, size: int): int =
  v.boundsCheck(i)
  let at = v.pos + i * size
  v.buf.need(at, size, "vector element")
  at

proc tableAt*(v: FbVector, i: int): FbTable =
  let at = v.elemPos(i, 4)
  initFbTable(v.buf, v.buf.offsetAt(at))

proc i32At*(v: FbVector, i: int): int32 = v.buf.i32(v.elemPos(i, 4))
proc i64At*(v: FbVector, i: int): int64 = v.buf.i64(v.elemPos(i, 8))
proc f32At*(v: FbVector, i: int): float32 = v.buf.f32(v.elemPos(i, 4))
proc u8At*(v: FbVector, i: int): uint8 = v.buf.u8(v.elemPos(i, 1))

proc toIntSeq*(v: FbVector): seq[int] =
  result = newSeqOfCap[int](v.len)
  for i in 0 ..< v.len: result.add int(v.i32At(i))

proc toFloat64Seq*(v: FbVector): seq[float64] =
  result = newSeqOfCap[float64](v.len)
  for i in 0 ..< v.len: result.add float64(v.f32At(i))

proc toInt32Seq*(v: FbVector): seq[int32] =
  result = newSeqOfCap[int32](v.len)
  for i in 0 ..< v.len: result.add v.i32At(i)

proc byteRange*(v: FbVector): (int, int) =
  ## Start offset and length of a `[ubyte]` vector, so multi-megabyte weight
  ## buffers can be copied once by the caller rather than element by element.
  if v.len == 0: return (0, 0)
  v.buf.need(v.pos, v.len, "byte vector")
  (v.pos, v.len)

proc copyBytes*(v: FbVector): seq[byte] =
  let (start, n) = v.byteRange()
  result = newSeq[byte](n)
  if n > 0:
    copyMem(addr result[0], unsafeAddr v.buf.bytes[start], n)
