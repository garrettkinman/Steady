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
import ../examples/branch_net
import generated/tiny_cnn as model
import generated/branch_net as branch

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

  let g = tiny_cnn.buildGraph()

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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The data-movement and non-linear ops, on the same terms: generated code
# against a simulator that computes the unfolded form.
#
# What each op proves here:
#   pad     the copy offsets, and that the fill value is the *zero point*
#   concat  the slice geometry, and that a matching operand is copied while a
#           mismatched one is rescaled
#   mean    the `-count * Zx` fold, against a simulator that subtracts Zx tap
#           by tap
#   tanh    that a table indexed by storage byte agrees with the true function
#           indexed by value — i.e. the two's-complement key encoding
#   softmax the exp table, its Q-format, and the normalising divide

proc runBranch(input: seq[int32]): seq[int32] =
  let inp = branch.input0()
  for i, v in input: inp[i] = int8(v)
  branch.invoke()
  let outp = branch.output0()
  result = newSeq[int32](5)
  for i in 0 ..< 5: result[i] = int32(outp[i])

proc lcgSeq(seed: uint32, n: int): seq[int32] =
  var s = seed
  result = newSeq[int32](n)
  for i in 0 ..< n:
    s = s * 1664525'u32 + 1013904223'u32
    result[i] = int32((s shr 16) and 0xFF'u32) - 128'i32

suite "end-to-end: pad, concat, mean, tanh, softmax":

  let g = branch_net.buildGraph()

  proc simulated(x: seq[int32]): seq[int32] =
    var inputs = initTable[int, seq[int32]]()
    inputs[g.inputs[0]] = x
    simulate(g, inputs)[g.outputs[0]]

  test "generated code matches the unfolded simulator, bit for bit":
    for trial in 0'u32 ..< 64'u32:
      let x = lcgSeq(2000'u32 + trial * 6151'u32, 72)
      check runBranch(x) == simulated(x)

  test "saturating and constant inputs agree too":
    for v in [-128'i32, -3'i32, 0'i32, 1'i32, 127'i32]:
      let x = newSeqWith(72, v)
      check runBranch(x) == simulated(x)

  test "softmax output is a probability distribution":
    # Dequantized with scale 1/256 and zero point -128, the five outputs must
    # sum to one. A table or divide that drifted would show up here even if
    # the simulator drifted with it.
    for trial in 0'u32 ..< 16'u32:
      let got = runBranch(lcgSeq(555'u32 + trial * 977'u32, 72))
      var total = 0.0
      for q in got: total += float64(q + 128) / 256.0
      check abs(total - 1.0) < 0.02
      for q in got: check q >= -128 and q <= 127

  test "invoke is repeatable and the arena is sized to the plan":
    let x = lcgSeq(31337'u32, 72)
    let first = runBranch(x)
    for _ in 0 ..< 5:
      check runBranch(x) == first
    check branch.ArenaSize == planOne(g).arenaSize
