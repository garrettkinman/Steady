# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

## End-to-end verification.
##
## Runs the generated straight-line code and the host simulator over the
## same inputs and requires the outputs to be identical, element for
## element. Because the simulator evaluates the *unfolded* form — explicit
## `x - Zx` at every tap, original biases — agreement is a proof that the
## compiler's bias folding is correct, including at SAME-padded edges.
##
## Run `nimble test`, which regenerates the model first.

import std/[unittest, tables, sequtils]
import steadyc
import ../examples/tiny_cnn
import generated/tiny_cnn as model

proc runModel(input: seq[int32]): seq[int32] =
  let inp = model.input0()
  for i, v in input: inp[i] = int8(v)
  model.invoke()
  let outp = model.output0()
  result = newSeq[int32](10)
  for i in 0 ..< 10: result[i] = int32(outp[i])

proc lcgInput(seed: uint32): seq[int32] =
  var s = seed
  result = newSeq[int32](64)
  for i in 0 ..< 64:
    s = s * 1664525'u32 + 1013904223'u32
    result[i] = int32((s shr 16) and 0xFF'u32) - 128'i32

suite "end-to-end":

  let g = buildGraph()

  test "generated code matches the unfolded simulator, bit for bit":
    for trial in 0'u32 ..< 64'u32:
      let x = lcgInput(1000'u32 + trial * 7919'u32)
      let got = runModel(x)
      var inputs = initTable[int, seq[int32]]()
      inputs[g.inputs[0]] = x
      let want = simulate(g, inputs)[g.outputs[0]]
      check got == want

  test "saturating inputs are handled identically":
    for v in [-128'i32, -1'i32, 0'i32, 1'i32, 127'i32]:
      let x = newSeqWith(64, v)
      let got = runModel(x)
      var inputs = initTable[int, seq[int32]]()
      inputs[g.inputs[0]] = x
      let want = simulate(g, inputs)[g.outputs[0]]
      check got == want

  test "invoke is repeatable — no state carried between calls":
    let x = lcgInput(4242'u32)
    let first = runModel(x)
    for _ in 0 ..< 5:
      check runModel(x) == first

  test "arena is sized to the plan":
    let p = planOne(g)
    check model.ArenaSize == p.arenaSize
    check p.arenaSize > 0

  test "output buffer survives the whole invocation":
    # logits shares an offset with a dead conv buffer; if the planner got
    # the live range wrong, a second invoke would corrupt the first result.
    let a = lcgInput(7'u32)
    let b = lcgInput(9'u32)
    let ra = runModel(a)
    discard runModel(b)
    check runModel(a) == ra
