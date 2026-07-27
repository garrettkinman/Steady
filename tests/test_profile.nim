# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

## The profiling emission.
##
## `-d:steadyProfile` makes the emitter assemble the same operator calls a
## second time, as per-op entry points the benchmark harness can time
## individually. Everything the benchmark reports rests on those two
## assemblies being the same program, so that is what this checks: driving the
## ops one at a time, in order, must produce bit-identical output to `invoke`.
##
## It is a cheap test of an expensive-to-notice mistake. A per-op entry point
## that dropped an operator, ran one twice, or reordered a concatenation's
## operand writes would still produce a plausible profile — and would attribute
## time to the wrong operator, which is the one thing this feature exists to
## get right.
##
## Built with `-d:steadyProfile` by `nimble test`; without it there is nothing
## here to test, and the file says so rather than passing vacuously.

import std/[unittest, sequtils]
import generated/tiny_cnn as model
import generated/branch_net as branch

when not defined(steadyProfile):
  {.error: "compile this with -d:steadyProfile; see the nimble test task".}

proc lcg(seed: uint32, n: int): seq[int8] =
  result = newSeq[int8](n)
  var s = seed
  for i in 0 ..< n:
    s = s * 1664525'u32 + 1013904223'u32
    result[i] = int8(int32((s shr 16) and 0xFF'u32) - 128'i32)

template opsMatchInvoke(ops, inElems, outElems: static int,
                        inputFn, outputFn, runWhole, runOne: untyped) =
  ## The accessors are passed in rather than reached through a module alias:
  ## both fixtures export the same names, so the qualification has to happen at
  ## the call site where it is unambiguous.
  for trial in 0'u32 ..< 16'u32:
    let x = lcg(4242'u32 + trial * 7919'u32, inElems)

    let inp = inputFn()
    for i, v in x: inp[i] = v
    runWhole()
    let whole = toSeq(outputFn().toOpenArray(0, outElems - 1))

    # The arena reuses the input buffer, so it has to be refilled — the same
    # contract every caller has.
    let inp2 = inputFn()
    for i, v in x: inp2[i] = v
    for i in 0 ..< ops:
      runOne(i)
    let piecewise = toSeq(outputFn().toOpenArray(0, outElems - 1))

    check whole == piecewise

suite "profiling entry points":

  test "invokeOp in order reproduces invoke exactly":
    opsMatchInvoke(model.OpCount, model.Input0Elems, model.Output0Elems,
                   model.input0, model.output0, model.invoke, model.invokeOp)

  test "and on a graph with padding, concatenation, a mean and a softmax":
    opsMatchInvoke(branch.OpCount, branch.Input0Elems, branch.Output0Elems,
                   branch.input0, branch.output0, branch.invoke, branch.invokeOp)

  test "op metadata describes the graph the model was built from":
    # Both fixtures end in a classifier, so the last op's output is the
    # model's, and every op with multiplies must have counted some.
    check model.OpCount > 0
    check branch.OpCount > 0
    check model.OpProfiles[^1].outElems == model.Output0Elems
    check branch.OpProfiles[^1].outElems == branch.Output0Elems

    var counted = 0
    for p in model.OpProfiles:
      if $p.kind in ["okConv2d", "okDepthwiseConv2d", "okFullyConnected"]:
        check p.macs > 0
        counted += p.macs
      else:
        check p.macs == 0
    check counted == model.TotalMacs
