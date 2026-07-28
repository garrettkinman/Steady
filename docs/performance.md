<!--
 Copyright (c) 2026 Garrett Kinman

 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
-->

# Performance

Methodology and results behind the summary table in the [README](../README.md#performance).

`nimble mcu` builds a firmware image per model, flashes a B-L475E-IOT01A — STM32L475VG, Cortex-M4 at
80 MHz, 96 KB of usable SRAM, weights in internal flash — and reads per-operator **cycle counts**
back over the ST-LINK's virtual COM port. No scheduler, no other process, no frequency scaling, no
interrupt enabled anywhere in the image. Rerunning a measurement reproduces it to the cycle, and
everything below is a before-and-after from it.

It exists because "the reference kernels are unoptimized" is not a measurement and neither is an
argument about which loop to fix.

MAC counts come from the compiler, not from a paper: `macCount` counts the multiply-accumulates the
kernels actually perform, padded taps included, since those are multiplied against `padValue` rather
than skipped. Crediting a SAME-padded layer with only the arithmetic it strictly needs would overstate
its MAC/s by the border.

## Why there is no host benchmark

There was one, and it was removed. Timing these models on a desktop core is easy, reproducible enough
to look authoritative, and measures a machine with a cache hierarchy, a branch predictor, an
out-of-order pipeline, other processes competing for it, and thermal behaviour — none of which the
target has. The risk is not that the numbers are noisy. It is that they are *confidently wrong about
which change matters*, and the compile-time unroll below is the case that proved it: the host priced
it at nothing and the target priced it at 1.40x, because the two builds use different optimization
levels and GCC makes a different decision at each.

A benchmark that disagrees with the target on the ranking of changes is worse than no benchmark,
because it is used to decide what to work on. The board is cheap and the measurement is exact, so
there is no reason to keep a proxy for it.

## What the profile says

`-d:steadyProfile` adds per-operator entry points to the generated module — the same emitted calls
`invoke` contains, reachable one at a time — so the benchmark attributes cycles to an *operator*
rather than to a model. `invoke` itself is unchanged by the flag: there is no hook inside it, because
on this scale the measurement is the thing that has to be trustworthy. That the two assemblies of the
same calls really are the same program is a test rather than a claim: `test_profile` drives the ops
one at a time, in order, and requires bit-identical output to `invoke` on both example models. An
entry point that skipped an operator would otherwise still produce a plausible-looking profile.

The per-operator table is reported from the **cacheless** configuration deliberately — prefetch and
both caches off — because that is what a part without flash accelerators actually does, and it is the
attribution least flattered by the board. `kws`, by operator kind:

| | of the multiplies | of the cycles | cyc/MAC |
|---|---|---|---|
| `Conv2D` (5) | 89% | 77.6% | 29.80 |
| `DepthwiseConv2D` (4) | 11% | 22.0% | 69.55 |

**A depthwise convolution costs 69.6 cycles per MAC against a convolution's 29.8**, and that is not a
defect in the loop nest. A depthwise convolution does nine multiplies per output over nine values
that are `inC` bytes apart, with no channel reduction to amortize anything over; it is a memory-bound
operator wearing an arithmetic operator's clothes. On `person_detect` — MobileNetV1 at 0.25 width, so
thinner channels and proportionally more depthwise — the split moves further toward it.

Not yet measured on target: the share of inference spent in requantization. It is the one part of the
inner loop with a data-dependent branch, it is bit-exactness-critical code where a "faster" tie rule
is wrong across a whole network, and it should be priced with the same stub-substitution method on
the board before anyone optimizes it.

## On the target itself

The kernel work, priced on the board. `kws`, 2.66 MMAC, with the flash accelerators on:

| kernels | flash `.text` | per inference | | |
|---|---|---|---|---|
| before | 36808 | 1169.8 ms | 35.2 cyc/MAC | |
| blocked | 37948 | 753.8 ms | 22.7 cyc/MAC | 1.55x |
| blocked, unrolled at compile time | 38700 | **539.2 ms** | **16.2 cyc/MAC** | **2.17x** |

and `vww`, MobileNetV1 at 96x96, 7.49 MMAC: **2892 ms to 1433 ms, 2.02x**.

The middle row is the whole argument for owning a board. Blocking the kernels is worth 1.55x;
unrolling the block loop *in the Nim source instead of leaving it to the C compiler* is worth a
further **1.40x** on top. GCC unrolls that loop at `-O3` and declines at `-Os`, which is what the
freestanding build uses — so the change is invisible at the optimization level a desktop build would
pick and is the largest single win available at the one that ships. Reading the disassembly is what
found it; the board is what priced it. See `unrolled` in
[src/steady/kernels/reference.nim](../src/steady/kernels/reference.nim).

Two more things the board settles:

- **The flash accelerators are worth more than the kernels.** Prefetch plus the instruction and data
  caches take `kws` from 1137 ms to 539 ms — 2.1x, larger than everything above. The benchmark sweeps
  them rather than assuming a configuration, and reports the cacheless row too, because that is what
  parts without them actually do. The earlier worry that four interleaved weight streams would thrash
  an 8-line data cache did not survive contact: enabling the data cache is worth 21% on `kws` *with*
  the blocked kernels.
- **Depthwise convolution is ~2.3x more expensive per MAC than convolution** with no cache in play at
  all, which settles that it is a property of the operator rather than of a memory hierarchy.

The output checksum — `6962079d` for `vww`, `ab2fdbf1` for `kws` — is the same before and after the
kernel work, and the same in all three flash configurations. A benchmark that cannot tell a fast kernel from a kernel the
linker deleted is measuring its own optimizer, so the firmware checksums its output and the harness
refuses a record whose per-build nonce does not match the image it just flashed. Both of those exist
because the first version of this harness reported two different kernels as identical to the digit,
which is what a stale line in a serial buffer looks like.

## What was changed, and why it is still bit-exact

Four transforms, all in [src/steady/kernels/reference.nim](../src/steady/kernels/reference.nim), all
expressed through `mac` and none of them touching the policy abstraction:

1. **A 1x1 fast path in `conv2d`.** A pointwise convolution's "patch" is one contiguous run of `inC`
   values, so the general path's per-tap work — deriving `iy` and `ix`, testing them against the
   input bounds, re-deriving a weight base — is pure overhead around a reduction that may be eight
   elements long. It needs no bounds test at all: with no padding the host's own shape rule gives
   `(outH - 1) * strideH <= inH - 1`, so every sampled position is inside by construction.
2. **Output-channel blocking in `conv2d`**, four filters per pass over the patch.
3. **Channel blocking in `depthwiseConv2d`**, which is the same transform aimed at memory rather than
   arithmetic: a block of four consecutive output channels reads four *contiguous* activation bytes
   and four contiguous weight bytes, where one channel at a time touched nine cache lines to use one
   byte of each.
4. **Row blocking in `fullyConnected`**, four rows sharing one pass over the input vector. Worth the
   least, because a fully-connected layer reads its weights exactly once no matter what, so
   activation loads are the only thing there is to save — `ad`, which is ten stacked
   fully-connected layers, moves least of the six.

The property that makes this safe is what is *not* done: no output's reduction is ever split into
partial sums. That is the other standard way to break a dependency chain, and it re-associates the
addition — exact for a wrapping int32 accumulator, not exact for float32, fp8, or a posit quire's
rounding. Every accumulator above still sees exactly the taps it saw before, in exactly the order it
saw them, so bit-exactness holds by construction rather than by retesting. Independent accumulators
come from blocking instead, which gives the same instruction-level parallelism for free.

The harness confirms it rather than taking it on trust: across all six comparable models at fifteen
trials each, the optimized kernels reproduce the old ones' agreement with TFLite exactly. The models
that are bit-identical still are, and — the sharper half — the ones that *disagree* disagree by the
same amount, digit for digit: `ad` at 67.417% of output elements equal, `kws` at 98.889%,
`mobilenet_v2` at 99.900%. Those percentages are decided by where fixed-point rounds land on a tie,
so a reassociation anywhere would have moved them.

Costs, stated because they are real: the freestanding Cortex-M4 image grows from 7284 to 8564 bytes
of `.text` for the specialized paths, and a kernel frame holds up to four accumulators. `.data` and
`.bss` are unchanged — no new RAM, and the arena is untouched.

Block widths are `OcBlock`, `DwBlock` and `FcBlock`, defaulting to four for every policy in
[src/steady/contract.nim](../src/steady/contract.nim). Four beat two by about 10%; eight was worse
than four, and the assembly says why — eight accumulators plus eight weights plus eight activations
spilled seven of the accumulators to the stack. That number is a property of the register file
holding `Accum(P)`, which is why it is a policy member rather than a constant of the kernel file: a
posit unit with a single quire register wants one, and an arithmetic backend overrides it without
touching a kernel. Three separate constants is deliberate too — the right width for a scalar core
and for a Helium core are not the same number, and the three kernels want them for different
reasons.

## What did not work

**Pixel tiling in the pointwise path.** Blocking two adjacent output pixels against four filters
turns a 1x1 convolution into a properly register-tiled GEMM and reuses *weights* across pixels, where
the shipped version reuses only activations across filters. On paper it halves weight traffic in the
layers that hold most of the multiplies. Measured, it bought almost nothing for a substantially
larger kernel and a doubled loop nest. Reverted — but it is the right transform for a machine whose
weights come from slow flash, and it was priced before the on-target harness existed, so it is worth
running again on the board rather than dismissed on the strength of the old number.

## Why not Winograd

Winograd F(2x2, 3x3) cuts multiplies by 2.25x, and for this target it is close to the wrong lever
entirely:

- **It does not apply to the layers that cost.** In a MobileNet-class model the overwhelming majority
  of multiplies are in 1x1 pointwise convolutions, which Winograd does not touch. The 3x3s that
  remain are *depthwise*, where there is no channel reduction to amortize the input and output
  transforms over, so the transform overhead eats the saving. The profile above puts numbers on both
  halves of that sentence.
- **It needs RAM.** Transformed tiles are intermediate storage, and RAM is the scarcest thing here —
  the arena for these models is tens of kilobytes, sized to the byte at compile time. Trading it for
  multiplies is backwards on a part with 64 KB of SRAM.
- **It is hostile to int8.** The transforms carry fractional coefficients, so an integer
  implementation needs wider intermediates and its own scaling, and it stops being bit-exact against
  TFLite — a property this project spends real effort maintaining and can currently demonstrate.

Winograd earns its keep on large 3x3 dense convolutions in float or fp16, with a cache and vector
units to hide the transforms. That is a different machine from the one this compiler targets.

## What is left, in order

1. **Data layout plus SIMD, through a vendor backend.** Still the single biggest factor, and now the
   only large one left in reach — and now measurable on the target rather than argued for. On
   Cortex-M4/M7 `SMLAD` does two int8 MACs per cycle, on M55/M85 Helium does sixteen; both need
   weights arranged so the inner loop reads contiguous pairs or quads. Published CMSIS-NN figures are
   around 4-5x over reference kernels, and that is a *layout* result as much as an
   instruction-selection one — which is exactly the split between
   [src/steadyc/backend.nim](../src/steadyc/backend.nim) and the target-side `steady_backend`. The
   blocking above is the same shape a SIMD kernel wants, which is a convenience rather than a
   substitute.
2. **The requantize path**, whose share rises every time the multiplies get faster. Price it first,
   by substituting a stub `finish` and re-running on the board; the number that used to be quoted
   here came from a desktop and has no business steering this. In a vendor backend this is vector
   fixed-point arithmetic; portably it is a handful of instructions per output element that have to
   keep two different gemmlowp tie rules exactly right.
3. **Depthwise blocked over pixels**, keeping a channel's weights in registers across several output
   positions. Depthwise is memory-bound — 70 cycles per MAC against convolution's 28 on the M4 — and
   this is the transform that addresses loads rather than multiplies. The pixel-tiling result above
   is the reason to price it on the board rather than argue for it.
