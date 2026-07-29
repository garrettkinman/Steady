<!--
 Copyright (c) 2026 Garrett Kinman

 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
-->

# Performance

Methodology and results behind the summary table in the [README](../README.md#performance).

`nimble mcu` builds a firmware image per model, flashes a board and reads per-operator **cycle
counts** back over its console. No scheduler, no other process, no frequency scaling, no interrupt
enabled anywhere in the image. Rerunning a measurement reproduces it to the cycle, and everything
below is a before-and-after from it.

Three boards are supported. Unless a section says otherwise, the numbers here are from the first:

- **`stm32l475`** — B-L475E-IOT01A: STM32L475VG, Cortex-M4 at 80 MHz, 96 KB of usable SRAM, weights
  in internal flash, flashed and reset through the on-board ST-LINK and read over its virtual COM
  port. This is the reference part: every before-and-after below was measured on it, and it stays
  the one the kernel work is judged against.
- **`samd51`** — Adafruit ItsyBitsy M4 Express: ATSAMD51G19A, Cortex-M4 at 120 MHz, 192 KB of SRAM,
  4 KB of unified cache. A second part rather than a faster one — see
  [On a second part](#on-a-second-part).
- **`esp32c3`** — ESP32-C3-DevKitM-1: ESP32-C3, RISC-V at 160 MHz, 384 KB of SRAM, weights in a
  serial flash the core reaches only through a page table and a 16 KB cache. A second *instruction
  set*, and the first part where the answer differs — see
  [On a part that is not an ARM](#on-a-part-that-is-not-an-arm).

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
| `Conv2D` (5) | 89% | 77.6% | 29.94 |
| `DepthwiseConv2D` (4) | 11% | 22.0% | 69.74 |

**A depthwise convolution costs 69.7 cycles per MAC against a convolution's 29.9**, and that is not a
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
| blocked, unrolled at compile time | 38700 | **539.6 ms** | **16.3 cyc/MAC** | **2.17x** |

and `vww`, MobileNetV1 at 96x96, 7.49 MMAC: **2892 ms to 1432 ms, 2.02x**.

The middle row is the whole argument for owning a board. Blocking the kernels is worth 1.55x;
unrolling the block loop *in the Nim source instead of leaving it to the C compiler* is worth a
further **1.40x** on top. GCC unrolls that loop at `-O3` and declines at `-Os`, which is what the
freestanding build uses — so the change is invisible at the optimization level a desktop build would
pick and is the largest single win available at the one that ships. Reading the disassembly is what
found it; the board is what priced it. See `unrolled` in
[src/steady/kernels/reference.nim](../src/steady/kernels/reference.nim).

Two more things the board settles:

- **The flash accelerators are worth more than the kernels.** Prefetch plus the instruction and data
  caches take `kws` from 1141 ms to 540 ms — 2.1x, larger than everything above. The benchmark sweeps
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

## On a second part

`--board samd51` runs the same six models on an Adafruit ItsyBitsy M4 Express: ATSAMD51G19A, the
same Cortex-M4 at 120 MHz, 192 KB of SRAM, a 4 KB unified cache (CMCC) and a pair of read buffers
inside the flash controller. It was added to answer one question — how much of the table above is a
fact about these kernels and how much is a fact about ST's memory system.

Mostly the former. With each part's caching fully enabled, the two land within 2–4% of each other in
**cycles** on every model, and in the same order:

| model | STM32L475 @ 80 MHz | SAMD51 @ 120 MHz | ratio |
|---|---|---|---|
| `ad` | 2,413,782 | 2,375,122 | 0.98 |
| `resnet8` | 149,371,445 | 145,826,568 | 0.98 |
| `vww` | 114,544,319 | 111,497,163 | 0.97 |
| `person_detect` | 112,660,040 | 110,233,310 | 0.98 |
| `kws` | 43,169,869 | 42,421,704 | 0.98 |
| `fomo` | 103,255,958 | 99,481,577 | 0.96 |

Two things follow. The wall-clock difference between the parts is very nearly just the clock ratio,
which is the boring and correct answer. And the per-operator attribution above — the depthwise/dense
split in particular — reproduces: with no cache in play at all, `kws` on the SAMD51 costs 104.3
cyc/MAC in `DepthwiseConv2d` against 41.8 in `Conv2d`, a 2.49x ratio against the STM32's 2.33x. A
memory-bound operator is memory-bound on both.

What does *not* transfer is how much the memory system is worth. The SAMD51's 4 KB cache takes `kws`
from 1079.6 ms to 353.5 ms — **3.05x**, against 2.1x for the STM32's prefetch buffer and two caches —
and its cacheless row is the worse of the two in cycles (48.8 cyc/MAC against 34.4), because five
flash wait states at 120 MHz with nothing in front of them is what that costs. The cache is doing
more work because there is more work to do. This is the reason the per-operator table is reported
from the cacheless configuration on both boards: it is the number that is about the kernel.

Every output checksum matches between the two parts.

### What the port cost

A `board.c`, a linker script and a `board.sh` under [../tests/mcu/boards/](../tests/mcu/boards/).
Nothing above that directory knows which part it is talking to; the flash-accelerator sweep in
particular is now the board's to define, because "prefetch, instruction cache, data cache" is
STM32 vocabulary and this part has neither three switches nor those names.

Two things about it are worth writing down.

The board has **no debug probe and no USB-serial bridge**, so the firmware serves its own USB CDC
console — and serves it from a polling loop, not an interrupt, because the one property this image
cannot lose is that nothing fires inside a measured region. The cost is that the device ignores its
host for as long as a measurement takes, which is seconds; the host notices, because setting the line
discipline on a CDC port is a control request that blocks until the device answers it. `board_poll`
exists for that, called between repetitions and never inside one.

And an image that fails to enumerate **hands itself back to the bootloader** after about a minute.
On a part whose only route in is USB, that is the difference between a bad build costing a reflash
and a bad build costing someone walking over to the board to double-tap a button. The first one cost
the latter, which is why the second exists.

## On a part that is not an ARM

`--board esp32c3` runs the same six on an ESP32-C3-DevKitM-1: one RISC-V core (RV32IMC) at 160 MHz,
384 KB of SRAM, 4 MB of flash. It was added to ask the question the SAMD51 could not, because the
SAMD51 agreed: how much of the table above survives changing the instruction set *and* the way the
part gets to its weights.

| model | KB of weights per MMAC | STM32L475 cycles | ESP32-C3 cycles | ratio |
|---|---|---|---|---|
| `fomo` | 4 | 103,255,958 | 109,865,046 | 1.06 |
| `resnet8` | 6 | 149,371,445 | 171,219,533 | 1.15 |
| `kws` | 9 | 43,169,869 | 41,965,806 | 0.97 |
| `vww` | 29 | 114,544,319 | 150,680,790 | 1.32 |
| `person_detect` | 30 | 112,660,040 | 155,059,564 | 1.38 |
| `ad` | 1019 | 2,413,782 | 4,823,075 | 2.00 |

The kernels survive; the memory system does not. Where the SAMD51 was flat at 0.96–0.98 across every
model, this part ranges over 2x — and the first column accounts for the ends of it. This flash is a
serial SPI part read through 16 KB of cache, where both Cortex-M4s read a parallel flash on the
core's own bus. `ad` is ten stacked fully-connected layers: 265 KB of weights, each used once,
0.26 MMAC of arithmetic to hide the reads behind. It pays double. `fomo` and `resnet8`, which reuse
what they load, pay 6–15% for a different ISA compiling the same kernel source.

The middle of that range is not ordered by the first column — `kws` at 9 KB/MMAC beats the STM32
outright while `resnet8` at 6 loses by 15% — so bytes-per-MAC is the explanation at the extremes and
not a model of the part. What it does establish is the shape of the answer: this project's numbers
have been a property of the kernels on every part so far, and the first place that stops being true
is where the weights stop fitting behind the cache.

Two other things this port settles.

**The per-operator ratio holds across the ISA**, and widens exactly where it should. On `kws`, with
each part's caching fully enabled — the only configuration this one has, so the comparison is against
the others' equivalent row rather than their cacheless one:

| | `Conv2d` | `DepthwiseConv2d` | ratio |
|---|---|---|---|
| STM32L475 | 13.52 cyc/MAC | 38.16 | 2.82 |
| SAMD51 | 13.35 | 36.95 | 2.77 |
| ESP32-C3 | 12.58 | 40.74 | 3.24 |

The RISC-V core is the *fastest* of the three at convolution and the slowest at depthwise
convolution, which is the bandwidth story again at operator granularity: depthwise has the lower
arithmetic intensity, so it is the operator that notices a narrower path to weights. A memory-bound
operator is memory-bound on a third architecture, and more so on the part with the least memory
bandwidth to hide behind.

**There is nothing to sweep.** The benchmark asks each board how many flash configurations it has and
this one answers one, because the cache here is not in front of the path to flash, it *is* the path:
the MMU translates into it, nothing addresses flash around it, and the kernels are being fetched
through it as well as the weights. A cacheless row on this part is not a slow configuration, it is
not a configuration.

Every output checksum matches all three parts, on all six models.

### What the port cost

The same three files as the SAMD51, plus a `startup.c` — the shared one is a Cortex-M vector table —
and, above the board directory, a toolchain prefix and a Nim `--cpu` where `arm-none-eabi-` and
`--cpu:arm` had been hardcoded. That is the whole of what a second instruction set cost the harness.

The part itself cost more, and all of it is in `board.c`. Three findings are worth writing down
because each one presents as a hang rather than as an error:

- **The core cannot address flash.** Kernels and weights are linked into two windows that a
  128-entry page table maps onto flash pages, through a cache that is shut off from both buses at
  reset. The table is written to the identity, so a mapped address is its own flash offset and the
  linker script can simply name the offsets `board.sh` writes to — the alternative couples a linker
  script to a flasher by page number, and getting that wrong reads plausible rubbish out of the
  wrong part of flash rather than failing.
- **Nothing that runs before the mapping may call a compiler helper.** Measuring the core clock ahead
  of it hung the board, because the measurement divides a 64-bit product and libgcc's division
  helper is linked into flash with everything else. The constraint is not "our code" — it is any
  code, including the code the compiler emits calls to without being asked.
- **Two standard CSRs are not implemented.** `mie` and `mcycle` both trap as illegal instructions —
  mcause 2, mtval the instruction itself. The cycle counter is a vendor CSR (PCCR) instead, and the
  trap handler that reports this had to be aligned to 256 bytes before it could: this core ignores
  the low eight bits of `mtvec` and forces vectored mode, so a handler at `…02C0` is installed at
  `…0200` and traps land in the middle of some other function. That one presented as a deliberately
  illegal instruction *restarting the firmware* instead of parking it.

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
