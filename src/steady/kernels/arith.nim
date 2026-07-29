# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

## Policy-member dispatch.
##
## The same mechanism as `kernels/dispatch.nim`, one level down. That module
## lets a backend replace a whole *kernel*; this one lets it replace the
## *arithmetic* a kernel is built from. They exist separately because they
## answer to different hardware:
##
##   an NPU or a CMSIS-NN port swallows a layer          -> dispatch.nim
##   a fused multiply-accumulate instruction, an ALU      -> here
##
## The distinction matters. A core with a multiply-accumulate instruction does
## not accelerate `conv2d`; it accelerates `mac`. Routing through the op-level seam would
## mean reimplementing seven kernels that differ from the reference only in
## which instruction sits in the innermost loop — exactly the forking this
## design exists to avoid. Overriding `mac` for one policy instead makes
## convolution, depthwise, fully-connected, add, concat-rescale, mean and
## pooling all faster at once, with no kernel touched.
##
## To supply one:
##
##   1. write a module named `steady_arith` that imports `steady/contract`
##      and defines any subset of the members below, with matching signatures
##   2. build with `-d:steadyArith --path:<dir containing it>`
##
## It imports the *contract* rather than the policy, and that is what keeps
## the layering acyclic: an arithmetic backend sits below the defaults it
## replaces, a kernel backend sits above them. Selection is per-member and
## per-policy through `when compiles`, so a backend defining only `mac` for
## `AffineI8` gets exactly that and everything else falls through — no
## registry, no indirect call, nothing at runtime.
##
## The default implementations remain compiled and reachable as `policy.mac`
## and friends, which is what makes a backend differential-testable against
## them rather than merely a replacement for them.
##
## Not overridable here: the associated types (`Store`, `Accum`, `Bias`,
## `Params`) and the block widths' *meaning*. Changing what an accumulator is
## changes what the host emits, so it is a new policy rather than an override
## — see the note at the end of `contract.nim`.
##
## TODO(Lanes): a SIMD unit needs more than a faster scalar `mac` — it needs
## the loop to be vector-shaped. The intended shape is a `Lanes(P)` associated
## constant plus a widened `mac` over lane vectors, with today's scalar
## policies as the degenerate `Lanes = 1` case so the kernels stay one source.
## That is a change to the kernels rather than to this seam, and it is
## deliberately not attempted until there is silicon to measure against.

import ../contract
from ../policy import nil

export contract

when defined(steadyArith):
  from steady_arith import nil

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template viaArith(call, fallback: untyped): untyped =
  ## `compiles` is false for an undeclared `steady_arith`, so the guard works
  ## whether or not a backend was configured. Both branches are expressions
  ## where the member is one, which is why `finish` and `Store`-returning
  ## members work through the same helper as the statement-shaped ones.
  when compiles(call):
    call
  else:
    fallback

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MEMBERS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# One template per member rather than one per member per policy: the policy
# is a parameter, so these four lines cover all four policies and will cover
# the fifth without being touched.

template zeroAccum*(P: typedesc): untyped =
  viaArith(steady_arith.zeroAccum(P), policy.zeroAccum(P))

template mac*(P: typedesc, acc, a, b: untyped) =
  viaArith(steady_arith.mac(P, acc, a, b), policy.mac(P, acc, a, b))

template addBias*(P: typedesc, acc, b: untyped) =
  viaArith(steady_arith.addBias(P, acc, b), policy.addBias(P, acc, b))

template finish*(P: typedesc, acc, prm, ch: untyped): untyped =
  viaArith(steady_arith.finish(P, acc, prm, ch), policy.finish(P, acc, prm, ch))

template lowestStore*(P: typedesc): untyped =
  viaArith(steady_arith.lowestStore(P), policy.lowestStore(P))

template accumulate*(P: typedesc, acc, v: untyped) =
  viaArith(steady_arith.accumulate(P, acc, v), policy.accumulate(P, acc, v))

template divAccum*(P: typedesc, acc, n: untyped): untyped =
  viaArith(steady_arith.divAccum(P, acc, n), policy.divAccum(P, acc, n))

template meanScale*(P: typedesc, acc, count: untyped): untyped =
  viaArith(steady_arith.meanScale(P, acc, count), policy.meanScale(P, acc, count))

template storeOf*(P: typedesc, acc: untyped): untyped =
  viaArith(steady_arith.storeOf(P, acc), policy.storeOf(P, acc))

template addRescaled*(P: typedesc, acc, v, mult, shift, offset: untyped) =
  viaArith(steady_arith.addRescaled(P, acc, v, mult, shift, offset),
           policy.addRescaled(P, acc, v, mult, shift, offset))

template lutIndex*(P: typedesc, v: untyped): untyped =
  ## Only ever instantiated for policies that have one; a store wider than a
  ## byte has no enumerable domain, so `lut1d` simply does not compile for it
  ## and the host rejects the op rather than the kernel branching.
  viaArith(steady_arith.lutIndex(P, v), policy.lutIndex(P, v))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BLOCK WIDTHS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Dispatched like any other member, and for the same reason: how many
# accumulators should be in flight is a fact about the hardware holding them.
# A backend that overrides `mac` with an instruction against a single
# accumulator register almost certainly wants to override these to 1 in the
# same module.
#
# The result has to be a compile-time constant — the kernels bind it to a
# `static int` and unroll against it — so an override must be a literal or a
# constant expression, not a runtime value.

template ocBlockOf*(P: typedesc): int =
  viaArith(steady_arith.OcBlock(P), contract.OcBlock(P))

template dwBlockOf*(P: typedesc): int =
  viaArith(steady_arith.DwBlock(P), contract.DwBlock(P))

template fcBlockOf*(P: typedesc): int =
  viaArith(steady_arith.FcBlock(P), contract.FcBlock(P))
