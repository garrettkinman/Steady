<!--
 Copyright (c) 2026 Garrett Kinman

 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
-->

# Performance

Methodology and results behind the summary table in the [README](../README.md#performance).

`nimble bench` compiles each fixture and times it, whole-inference and per operator. It exists
because "the reference kernels are unoptimized" is not a measurement and neither is an argument about
which loop to fix, and everything here is a before-and-after from it.

MAC counts come from the compiler, not from a paper: `macCount` counts the multiply-accumulates the
kernels actually perform, padded taps included, since those are multiplied against `padValue` rather
than skipped. Crediting a SAME-padded layer with only the arithmetic it strictly needs would overstate
its MAC/s by the border.

## On x86-64

At `-d:danger --passC:-O3` on one core of an i5-1135G7, with arena and flash from the compiler's own
report — before and after the kernel work described below:

| model | arena | flash | MMAC | before | after | | |
|---|---|---|---|---|---|---|---|
| `kws` | 16 KB | 24 KB | 2.66 | 2.05 ms | 1.15 ms | 2.30 GMAC/s | **1.77x** |
| `vww` | 63 KB | 214 KB | 7.49 | 3.78 ms | 2.60 ms | 2.88 GMAC/s | **1.45x** |
| `mobilenet_v2` | 1.5 MB | 3.4 MB | 300.77 | 117.45 ms | 82.83 ms | 3.63 GMAC/s | **1.42x** |
| `fomo` | 78 KB | 19 KB | 5.40 | 3.94 ms | 2.85 ms | 1.89 GMAC/s | **1.38x** |
| `person_detect` | 54 KB | 214 KB | 7.16 | 3.92 ms | 2.84 ms | 2.52 GMAC/s | **1.38x** |
| `resnet8` | 48 KB | 77 KB | 12.50 | 3.83 ms | 2.87 ms | 4.35 GMAC/s | **1.33x** |
| `ad` | 768 B | 265 KB | 0.26 | 0.043 ms | 0.039 ms | 6.77 GMAC/s | **1.10x** |

Both columns are from one session, because this was measured on a laptop that throttles by more than
the effect being measured: the two builds were compiled up front and then run alternately, pinned to
one core, minimum of five. `nimble bench` pins itself the same way, for the same reason — a benchmark
whose run-to-run spread exceeds the changes it is meant to judge is a way of confirming what you
already believed.

FOMO is the interesting row: a real object detector in 78 KB of RAM and 19 KB of flash. MobileNetV2
at 224 is desktop-scale here and is in the suite for coverage, not because 1.5 MB of arena fits
anything this compiler targets.

## What the profile says

`-d:steadyProfile` adds per-operator entry points to the generated module — the same emitted calls
`invoke` contains, reachable one at a time — so the benchmark attributes time to an *operator* rather
than to a model. `invoke` itself is unchanged by the flag: there is no hook inside it, because on
this scale the measurement is the thing that has to be trustworthy. That the two assemblies of the
same calls really are the same program is a test rather than a claim: `test_profile` drives the ops
one at a time, in order, and requires bit-identical output to `invoke` on both example models. An
entry point that skipped an operator would otherwise still produce a plausible-looking profile.

Per operator kind on `vww`, which is the shape of every depthwise-separable model here:

| | of the multiplies | of the time | rate |
|---|---|---|---|
| `Conv2D` (14) | 89% | 63% | 4.01 GMAC/s |
| `DepthwiseConv2D` (13) | 11% | 37% | 0.81 GMAC/s |

That ratio is the whole story of the operator, and it is not a defect in the loop nest. A depthwise
convolution does nine multiplies per output over nine values that are `inC` bytes apart, with no
channel reduction to amortize anything over; it is a memory-bound operator wearing an arithmetic
operator's clothes, and 0.8 against 4.0 GMAC/s is what that costs. The same table on
`person_detect` — MobileNetV1 at 0.25 width, so thinner channels and proportionally more depthwise —
splits the time almost exactly in half between the two.

Two other measurements worth recording, because both redirect effort:

- **Requantization is 13-25% of inference** (22% on `mobilenet_v2`, 25% on `vww`, 13% on `kws`,
  measured by replacing `finish` with a stub and re-running). That is gemmlowp's fixed-point
  multiplier, per output element, and it is bit-exactness-critical code where a "faster" tie rule is
  wrong across a whole network. It is also the one part of the inner loop with a data-dependent
  branch. Left alone deliberately: the available win is small, the risk is the property this project
  spends the most effort on, and SIMD in a vendor backend addresses it properly.
- **`-march=native` is still not the lever.** It is worth 6% on `vww` and `mobilenet_v2` and *costs*
  17% on `kws`. int8-to-int32 widening MACs into several accumulators are a textbook SLP case that
  GCC declines to take, and the fixed-point requantize does not vectorize at all.

| build | `vww` per inference | |
|---|---|---|
| `-d:danger --passC:-O3` | 2.60 ms | 2.88 GMAC/s |
| `-d:release` | 11.56 ms | 0.65 GMAC/s |

Runtime checks cost 4.4x, which is why the freestanding build uses `-d:danger`; on the target those
checks are not merely slow, they have nothing to report to.

## On the target itself

`nimble mcu` builds a firmware image per model, flashes a B-L475E-IOT01A — STM32L475VG, Cortex-M4 at
80 MHz, 96 KB of usable SRAM, weights in internal flash — and reads per-operator **cycle counts**
back over the ST-LINK's virtual COM port. Everything the host benchmark apologises for is absent: no
scheduler, no other process, no frequency scaling, and no interrupt enabled anywhere in the image.
Rerunning a measurement reproduces it to the cycle.

It was worth the trouble, because the host was wrong about which changes mattered. `kws`, 2.66 MMAC,
with the flash accelerators on:

| kernels | flash `.text` | per inference | | |
|---|---|---|---|---|
| before | 36808 | 1169.8 ms | 35.2 cyc/MAC | |
| blocked | 37948 | 753.8 ms | 22.7 cyc/MAC | 1.55x |
| blocked, unrolled at compile time | 38700 | **539.2 ms** | **16.2 cyc/MAC** | **2.17x** |

and `vww`, MobileNetV1 at 96x96, 7.49 MMAC: **2892 ms to 1433 ms, 2.02x** — against 1.45x for the
same change on x86-64.

The middle row is the whole argument for owning a board. Blocking the kernels is worth 1.55x on
target; unrolling the block loop *in the Nim source instead of leaving it to the C compiler* is worth
a further **1.40x on the target and nothing at all on x86-64**. GCC unrolls that loop at `-O3` and
declines at `-Os`, which is what the freestanding build uses, so the host said the change was free
and the target said it was the largest single win available. Reading the disassembly is what found
it; the board is what priced it. See `unrolled` in
[src/steady/kernels/reference.nim](../src/steady/kernels/reference.nim).

Two more things the target says and the host cannot:

- **The flash accelerators are worth more than the kernels.** Prefetch plus the instruction and data
  caches take `kws` from 1137 ms to 539 ms — 2.1x, larger than everything above. The benchmark sweeps
  them rather than assuming a configuration, and reports the cacheless row too, because that is what
  parts without them actually do. The earlier worry that four interleaved weight streams would thrash
  an 8-line data cache did not survive contact: enabling the data cache is worth 21% on `kws` *with*
  the blocked kernels.
- **Depthwise convolution costs 70 cycles per MAC against convolution's 28.** Same ratio the host
  profile shows, on a machine with no cache at all, which settles that it is a property of the
  operator rather than of x86.

The output checksum is the same on both machines — `6962079d` for `vww` from the Cortex-M4 and from
the host benchmark, `ab2fdbf1` for `kws` — and the same before and after the kernel work, and the
same in all three flash configurations. A benchmark that cannot tell a fast kernel from a kernel the
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
   least (`ad` is 1.10x) because a fully-connected layer reads its weights exactly once no matter
   what, so activation loads are the only thing there is to save.

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

Block widths are `OcBlock`, `DwBlock` and `FcBlock` in
[src/steady/kernels/reference.nim](../src/steady/kernels/reference.nim), all four. Four beat two by
about 10%; eight was worse than four, and the assembly says why — eight accumulators plus eight
weights plus eight activations spilled seven of the accumulators to the stack. That number is a
property of the target's register file, so it is the first thing to re-tune when the target changes,
and one constant per kernel is deliberate: the right width for a scalar core and for a Helium core
are not the same number.

## What did not work

**Pixel tiling in the pointwise path.** Blocking two adjacent output pixels against four filters
turns a 1x1 convolution into a properly register-tiled GEMM and reuses *weights* across pixels, where
the shipped version reuses only activations across filters. On paper it halves weight traffic in the
layers that hold most of the multiplies. Measured, it was worth about 2% on `vww` and `fomo` and
nothing on `kws`, for a substantially larger kernel and a doubled loop nest. Reverted. It is the
right transform for a machine whose weights come from slow flash, which is an argument for measuring
it again on a real target rather than for keeping it on the strength of the argument.

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
2. **The requantize path**, at 13-25% of inference and rising as a share every time the multiplies
   get faster. In a vendor backend this is vector fixed-point arithmetic; portably it is a handful of
   instructions per output element that have to keep two different gemmlowp tie rules exactly right.
3. **Depthwise blocked over pixels**, keeping a channel's weights in registers across several output
   positions. Depthwise is memory-bound — 70 cycles per MAC against convolution's 28 on the M4 — and
   this is the transform that addresses loads rather than multiplies. The pixel-tiling result above
   is the reason to try it on the board first: that one was worth 2% on x86 and was dropped, and the
   host would have said the same about the compile-time unroll, which turned out to be worth 1.40x on
   target.
