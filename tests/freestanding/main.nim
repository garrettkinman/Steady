# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

## Bare-metal build check.
##
## Compiles the runtime and a generated model under `--os:any --mm:none
## --panics:on`: no garbage collector, no OS layer, no heap. There is no
## `main` — this is a static library to be linked into a firmware image, so
## the entry points are plain exported C functions.
##
## Building is only half the check. `nimble freestanding` also greps the
## object files for allocator symbols, because "it compiled" and "it did not
## quietly link malloc" are different claims.

import steady
import ../generated/tiny_cnn as model
import ../generated/branch_net as branch
import ../generated/posit_net as posit

var mult = [1073741824'i32]
var shift = [0'i32]

proc steady_selftest(): int32 {.exportc, cdecl.} =
  ## Exercises a kernel directly, with no allocation anywhere.
  var y: array[2, int8]
  var x = [4'i8, 2'i8]
  var w = [6'i8, 3'i8, 1'i8, 1'i8]
  var b = [0'i32, 0'i32]
  let prm = AffineParams(
    mult: cast[ptr UncheckedArray[int32]](addr mult[0]),
    shift: cast[ptr UncheckedArray[int32]](addr shift[0]),
    channelStride: 0, outZeroPoint: 0, actMin: -128'i32, actMax: 127'i32)
  fullyConnected(AffineI8,
    cast[ptr UncheckedArray[int8]](addr y[0]),
    cast[ptr UncheckedArray[int8]](addr x[0]),
    cast[ptr UncheckedArray[int8]](addr w[0]),
    cast[ptr UncheckedArray[int32]](addr b[0]), prm, 2, 2)
  int32(y[0])

proc steady_model_invoke() {.exportc, cdecl.} =
  ## The generated model, running freestanding.
  model.invoke()

proc steady_model_input(): ptr UncheckedArray[int8] {.exportc, cdecl.} =
  model.input0()

proc steady_model_output(): ptr UncheckedArray[int8] {.exportc, cdecl.} =
  model.output0()

proc steady_arena_size(): int32 {.exportc, cdecl.} =
  int32(model.ArenaSize)

# The second model is here for the ops the first one does not have: padding,
# concatenation, a spatial mean, a table activation, and a softmax whose
# normalising divide is the only 64-bit arithmetic on the target. If any of
# that quietly wanted libm, a float printf or an allocator, this is where it
# shows up — and the placement audit checks that its tables went to flash
# rather than being copied into RAM at startup.

proc steady_branch_invoke() {.exportc, cdecl.} =
  branch.invoke()

proc steady_branch_input(): ptr UncheckedArray[int8] {.exportc, cdecl.} =
  branch.input0()

proc steady_branch_output(): ptr UncheckedArray[int8] {.exportc, cdecl.} =
  branch.output0()

proc steady_branch_arena_size(): int32 {.exportc, cdecl.} =
  int32(branch.ArenaSize)

# The third model is a *real-number* policy: posit(8,0) storage, an int64
# quire, and a 256-entry decode table that belongs in flash like any other
# constant. Nothing in it may reach for libm or a software float — the whole
# point of a fixed-point quire is that a part with no FPU never needs one —
# and the placement audit checks the table's section for the same reason it
# checks an activation table's.

proc steady_posit_invoke() {.exportc, cdecl.} =
  posit.invoke()

proc steady_posit_input(): ptr UncheckedArray[Posit8] {.exportc, cdecl.} =
  posit.input0()

proc steady_posit_output(): ptr UncheckedArray[Posit8] {.exportc, cdecl.} =
  posit.output0()

proc steady_posit_arena_size(): int32 {.exportc, cdecl.} =
  int32(posit.ArenaSize)
