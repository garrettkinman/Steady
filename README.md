<!--
 Copyright (c) 2026 Garrett Kinman

 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
-->

# Steady

An ahead-of-time TinyML inference compiler and runtime, in pure Nim, for microcontrollers.

Steady is not an interpreter. A host-side compiler reads a model, plans its memory, resolves its
quantization, and emits straight-line code with every buffer address already resolved. What runs on
the device is the calls, in order, and nothing else — no graph structure, no operator registry, no
dispatch table, no allocator.

> **Status: early but real.** `steadyc model.tflite` compiles actual int8 models — MobileNetV1,
> ResNet-8, DS-CNN — and three of the four reference models in the test suite come out
> **bit-identical to TFLite on every intermediate tensor**. See [Roadmap](#roadmap) for what is and
> is not done.

---

## Why it is shaped this way

The single structural decision: **model-specific work happens on the build machine, never on the
target.**

| Host (`steadyc`) | Target (`steady`) |
|---|---|
| parse, validate, infer shapes | — |
| resolve quantization, fold biases | — |
| lifetime analysis, arena packing | — |
| emit code | run it |
| `seq`, `string`, exceptions | no allocation, no OS, no dependencies |

Everything else follows. Peak RAM is a number you read at compile time. Weights are `const` and live
in flash. Shape errors are host-side messages that name the op, not generic instantiation failures.
There is no interpreter because there is nothing left for one to do.

The cost is that you cannot load a model the binary was not built against. Multiple models in one
binary are fine — they share one arena sized to the worst case — but over-the-air model replacement
is out of scope by construction.

---

## Design principles

**Configurable over the core numeric type.** Kernels are generic over a *numeric policy* that
abstracts what happens between accumulate and store. Integer-affine quantization and real-number
formats (fp8, posits) genuinely differ there and nowhere else.

**No dynamic allocation, ever.** Every activation buffer is a compile-time offset into one static
arena. Verified by linking a real Cortex-M4 image and auditing it for allocator symbols.

**Zero-overhead execution.** Generated `invoke` is a flat sequence of calls.

**Acceleratable without forking.** Backends override individual ops, per policy, by compile-time
name resolution with automatic fallback.

**No external dependencies.** Nim's standard library on the host; nothing at all on the target.

---

## The numeric policy

A policy is an empty tag type plus overloaded templates. It answers four questions — what is stored,
what accumulates, what the bias looks like, what metadata the store step needs — and provides the
operations between them.

```nim
Store(P)   Accum(P)   Bias(P)   Params(P)

zeroAccum(P) -> Accum
mac(P, acc: var Accum, a, b: Store)
addBias(P, acc: var Accum, b: Bias)
finish(P, acc: Accum, prm: Params, ch: int) -> Store
```

Plus one member only 8-bit policies have: `lutIndex(P, v: Store) -> int`, the raw storage byte of a
value. Its presence *is* the statement "this storage type has 256 values and the host can tabulate a
function over it" — see [Non-linear activations](#non-linear-activations).

Three policies ship: `AffineI8`, `RealF32`, `RealFp8`.

**`mac` is a primitive, not sugar for `acc += a * b`.** A posit policy accumulates into a *quire* —
an exact fixed-point accumulator that supports fused multiply-accumulate and nothing else. There is
no `*` on a quire. Routing every kernel through `mac` is what keeps that door open, and it is the
single most expensive thing to retrofit.

`RealFp8` (OCP FP8 E4M3) exists as the **canary**. It is a non-native storage type with software
arithmetic, a widening conversion into a different accumulator, and no scale metadata — every code
path a posit policy will need. All eight kernels compile and run correctly under all three policies
with no `when` branches in kernel source. Two policies would not have proven that; the third forced
it.

### Adding a numeric format

The host and the target need *different* things from a numeric type, which is why posit support does
not require a posit implementation up front:

- **Host** needs `decode: bits -> float64` and `encode: float64 -> bits` — for converting weights,
  generating activation lookup tables, and running the reference harness. A SoftPosit binding plugs
  in here, on a real OS. This is `steadyc/codec.nim`, and it is the entire host-side numeric surface.
- **Target** needs `mac`, `finish`, `zero`, plus `lutIndex` if the format is 8-bit. Nothing else.

---

## Non-linear activations

For any 8-bit format, non-linear activations are 256-entry lookup tables the host generates by
evaluating the true function at each representable value — exact by construction, constant-time, and
requiring no target-side arithmetic or libm at all. The kernel is one line:

```nim
y[i] = table[lutIndex(P, x[i])]
```

The table is keyed on the storage *byte*, not the numeric value, so for int8 the entry for -128 lives
at index 128. That is exactly the sort of thing that works on one policy and silently breaks on the
next, so the end-to-end test indexes the reference implementation by value and requires the two to
agree.

Softmax cannot be a single table — it needs a max subtraction for range, and the max is a runtime
value — so it is a table of `exp(-d)` keyed on the quantized difference `max - x`, which for a
uniform integer store is itself an index in `0 ..< 256`. The host picks the table's fractional bits
from the class count so the runtime's sum cannot overflow an int32, and entry 0 is exactly the unit,
so the normalising divide can never see a zero denominator. Ten classes get a table 2^12 times finer
than a worst-case bound would allow, decided at compile time and costing nothing.

This is where a format's structure decides an op's availability rather than the code branching on it:
`RealF32` has no enumerable domain, so it has no table activations; `RealFp8` has one but is not
uniform, so it has tables and no softmax. Both are host-side validation errors naming the op and the
policy — the kernel set stays policy-generic, and the *op* set is a property of the number format.

The tables are within one LSB of a float64 reference over the whole input domain
(`tests/test_codec.nim` pins that, entry by entry). The softmax was expected to be the *loosest* part
of the compiler, since it evaluates `exp` from a host-built table where TFLite uses gemmlowp's
fixed-point series — so it is worth recording that on three real models it comes out
[bit-identical to TFLite anyway](#against-tflite-itself), including a twelve-class and a ten-class
softmax.

---

## Quantization

Affine int8 follows TFLite exactly, including the parts that are easy to get subtly wrong:

- Q31 multipliers via `QuantizeMultiplier`, per-tensor or per-channel
- `SaturatingRoundingDoublingHighMul` + `RoundingDivideByPOT`, whose **tie rules differ from each
  other** — the first rounds half toward `+inf`, the second half away from zero. Reproducing that
  asymmetry is the point; "fixing" it puts you one LSB off across an entire network.
- ReLU clamps at the output *zero point*, not at integer zero

**Zero points never reach the kernel.** Since TFLite mandates symmetric int8 weights (`Zw = 0`):

```
acc = sum_k w_k * (x_k - Zx)  =  sum_k w_k * x_k  -  Zx * sum_k w_k
                                 ^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^^
                                 the kernel          folded into the int32 bias
```

Padded taps are multiplied against an explicit `padValue` — `Zx` for affine, `0` for real formats —
rather than skipped. This keeps the folded bias valid at the edges without special-casing boundary
pixels, keeps the inner loop branch-free, and keeps the kernel free of quantization semantics.

The same folding argument recurs wherever a zero point would otherwise reach the target. A spatial
mean gets `-count * Zx` as a scalar correction in accumulator units — the same trick as a folded
bias, in the same units, for the same reason. `Pad` fills with the *quantized* pad constant, so a
TFLite PAD's "fill with 0.0" arrives as the zero point without the caller or the kernel spelling that
out. Concatenation copies operands verbatim when they already carry the output's quantization, which
is what the converter usually produces, and rescales through the add kernel's two-stage multiplier
when they do not. (TFLite's reference concatenation rescales in float; doing it in Q31 can differ by
an LSB when the quantizations differ, and is exact when they agree, because then there is no
arithmetic at all.)

---

## Verification

The end-to-end test runs generated code and a host simulator over the same inputs and requires
**bit-identical** output. The simulator deliberately evaluates the *unfolded* form — explicit
`x - Zx` at every tap, original biases — so agreement is a proof that the folding transform is
correct, including at SAME-padded edges. An independent implementation of the same optimization
would prove nothing.

Two models are checked this way: a straight-line CNN, and a second one built to be awkward —
explicit padding, two branches concatenated where one operand needs rescaling and the other does not,
a spatial mean into a different quantization, a table activation, a softmax.

Requantization arithmetic is shared between runtime and simulator, so that is covered separately by
vectors pinned to the published gemmlowp definitions. Activation tables and the softmax
normalisation are shared the same way, and are pinned the same way — against a float64 reference,
entry by entry.

The bare-metal check links a real Cortex-M4 image containing both models and audits it:

```
==> Image size
   text    data     bss     dec
   7284       4     904    8192
==> Allocator audit
    no allocator linked in
==> Placement audit
    weights in flash (.rodata)
    activation and exp tables in flash (.rodata)
    arena in RAM (.bss)
```

That last check matters more than it looks. Nim `const` cannot have its address taken and a
module-level `let` lands in `.data` — copied into RAM at startup, which on a part with 64 KB of SRAM
and 512 KB of flash is exactly backwards. Weights are therefore emitted as `const` C arrays and
linked in, which also gives explicit control over alignment and section placement. Activation tables
are constants on the same terms, and are audited the same way: a 256-byte table shadowed into RAM
would work perfectly and cost 256 bytes of SRAM for nothing.

The packaging check is separate again, and answers a separate question — see
[Using it from C](#using-it-from-c).

### Against TFLite itself

The simulator proves the folding transform, and the unit tests pin the tables and the fixed-point
arithmetic against the mathematics. Neither can prove agreement with TFLite, because both halves are
ours. `nimble models` does that: it takes seven real int8 models, compiles each with `steadyc`, and compares
against TFLite's own **reference** kernels — not the optimized path, which delegates to XNNPACK and
requantizes int8 differently from anything a microcontroller runs. Five are fetched against pinned
checksums (`nimble fetch`); MobileNetV2 and FOMO have no canonical published file, so
[tests/models/convert.py](tests/models/convert.py) builds them.

| model | what it covers | agreement |
|---|---|---|
| `vww` | MobileNetV1 96x96, float I/O boundary stripped | **every tensor exact** |
| `fomo` | object detector: a 12x12 grid of per-cell softmaxes | **every tensor exact** |
| `resnet8` | residual adds whose operands differ in zero point | **every tensor exact** |
| `kws` | depthwise-separable CNN, 12-class softmax | **every tensor exact** |
| `mobilenet_v2` | 1000-class softmax, 3.4 MB of weights, 10 residual adds | exact but for the mean, below |
| `ad` | ten stacked fully-connected layers | 1 LSB per layer, below |
| `person_detect` | MobileNetV1 0.25; compiles, but modern TFLite refuses to load it | compile only |

Every intermediate tensor is compared, not just the output, so a divergence is attributed to an
*operator* rather than to a model. Fifty-two depthwise-separable layers, ten residual adds whose
operands carry different zero points, pooling, reshape, matmul, and softmax over both a vector and a
144-cell grid: exact to the last LSB.

Two models diverge, and the harness is built to say exactly where and why. Both cases are the same
cause: desktop TFLite evaluates int8 `FULLY_CONNECTED` and its generic `MEAN` reducer in **float**,
while its convolutions use gemmlowp fixed point. Steady is fixed point throughout — the arithmetic
CMSIS-NN and TFLite Micro use, and the only kind available on a part with no FPU. Where a fixed-point
round lands on a tie the two answers can differ by one. So the harness does not average that into a
tolerance; it requires that **the first divergent tensor is one of those two operators**, and fails if
a convolution ever drifts.

`ad` stacks ten fully-connected layers with small requantization shifts, so its divergence compounds
to 2-4 LSBs by the output. `mobilenet_v2` diverges only at its global average pool, on 4 channels of
1280 — and there the direction is the opposite of what a tolerance implies. Measured against the
exact real-valued mean, **our answer is the correctly rounded one on every channel where the two
disagree** (mean error 0.4805 against TFLite's 0.5195); their float path mis-rounds just past a tie.
That op is also where MobileNetV2 paid for itself: `MEAN` originally divided the accumulator in the
kernel and then requantized, which disagreed on 130 channels rather than 4. Folding `1/count` into
the requantization multiplier means the reduction rounds exactly once — more accurate, and one fewer
division in the kernel.

That harness also earned its keep immediately: it found an overflow in `roundingDivideByPOT`, where
the mask was built as `1'i32 shl exponent` and an exponent of 31 does not fit an int32 — gemmlowp
writes `1ll <<` for exactly this reason. A MobileNet channel whose weights are all but zero produces
a multiplier that small. Any bounds-checked build trapped on it, and `-d:danger` only worked by
accident of two's-complement wrapping.

---

## Usage

```sh
steadyc model.tflite -o generated          # what you will usually do
steadyc model.tflite -o generated --dump   # and this when it goes wrong
```

```
model 'vww'  policy AffineI8  ops 31
arena:
  buffers        31
  peak RAM       64512 bytes
  without reuse  259472 bytes
  saved          75.1%
  flash (const)  219064 bytes
```

Or drive the compiler as a library, which is what the importer itself does:

```nim
import steadyc

var g = initGraph("my_model", pkAffineI8)
let x = g.addInput("input", @[1, 8, 8, 1], dtInt8, inQuant)
# ... add consts, ops ...
g.validate                      # shapes and dtypes checked here, on the host

let p = planOne(g)
echo p.report([g])
emitModel(g, p, "generated", "my_model")
emitCApi(g, p, "generated", "my_model")   # optional: C header + staticlib shim
```

```
arena:
  buffers        4
  peak RAM       320 bytes
  without reuse  400 bytes
  saved          20.0%
  flash (const)  732 bytes
```

Then, on the target:

```nim
import generated/my_model

let inp = input0()
for i in 0 ..< 64: inp[i] = sample[i]
invoke()
let logits = output0()
```

One contract worth stating plainly, because the arena is packed: **an input buffer is scratch, not
storage.** Once the model has read it those bytes are reusable, and the planner does reuse them, so
fill every input before every invoke. Output buffers stay valid until the next invoke, and no pointer
ever moves — they are fixed offsets into a static array.

See [examples/tiny_cnn.nim](examples/tiny_cnn.nim) for a complete model, and
[examples/branch_net.nim](examples/branch_net.nim) for one that exercises padding, concatenation, a
spatial mean, a table activation and softmax.

### The TFLite importer

[src/steadyc/tflite.nim](src/steadyc/tflite.nim) populates the IR from a `.tflite` file, reading the
flatbuffer through [a reader written for the purpose](src/steadyc/flatbuffers.nim) — about 200 lines,
no `flatc`, no generated accessors, no dependency. Schema field indices are named constants so they
can be read against `schema.fbs` line by line, because a wrong index does not fail loudly: it reads a
neighbouring field, and a stride that arrives as a dilation produces a model that runs and is wrong.

Two things it refuses to do:

- **Guess.** Only operator codes the importer is sure of are mapped; anything else is an error naming
  the code. Mapping an op that cannot be tested is the same silent-wrongness failure as a bad field
  index.
- **Load uint8 models.** The TF1-era "quant" files use *asymmetric* weights, which breaks the
  bias-folding identity the kernels rest on — it leaves a `Zw * sum(x)` term that is data-dependent
  and cannot be folded on the host, so it would mean a row-sum pass inside every matmul kernel. The
  error says so and says to re-convert as int8.

Boundary `QUANTIZE`/`DEQUANTIZE` ops are stripped rather than implemented. TF2's "full integer with
float I/O" conversion wraps the graph in them; implementing them would put two storage types in one
graph, which is what one-policy-per-graph exists to avoid, and would make the device do arithmetic
the caller can do for free. So the quantized tensor becomes the model's own input, and its scale and
zero point are published in the generated C header. One anywhere other than the boundary is an error,
because that is a genuine mid-graph type change.

### Using it from C

`emitCApi` writes two more files: `<stem>_api.nim`, a set of `{.exportc, cdecl.}` wrappers, and
`<stem>.h`, the header a C caller includes and the only thing it needs to know.

```sh
nim c --app:staticlib --os:any --mm:none --panics:on --noMain -d:danger \
      --path:<steady>/src generated/my_model_api.nim
```

```c
#include "my_model.h"

steady_my_model_init();
int8_t *in = steady_my_model_input0();
for (size_t i = 0; i < STEADY_MY_MODEL_INPUT0_ELEMS; i++) in[i] = quantize(sample[i]);
steady_my_model_invoke();
const int8_t *out = steady_my_model_output0();
```

Element counts, shapes, byte sizes, the arena size, and each input's and output's scale and zero
point are all `#define`d, so a C caller holding real values can quantize them without guessing and
can size its own arrays from the header. There is no context handle and nothing to free: the arena is
a static array inside the archive.

`nimble staticlib` builds the library, greps the archive for the entry points, compiles a C program
that includes nothing but the header, links the two with plain `gcc`, and diffs its output against the
same input run through the Nim module. "It linked" and "it still computes the same thing" are
different claims and only the second one is interesting.

### Hardware acceleration

Write a module named `steady_backend` exposing any subset of the kernel entry points, then build
with `-d:steadyBackend --path:<dir>`. Overriding is per-op *and* per-policy — a backend defining
only an int8 `conv2d` gets used for int8 convolutions and ignored everywhere else, because the
`when compiles(...)` test fails for other instantiations and falls through. Nothing is forked and no
stubs are required. See [tests/fixtures/steady_backend.nim](tests/fixtures/steady_backend.nim).

A faster kernel is usually not the whole story: CMSIS-NN wants particular weight orderings, an NPU
wants weights pre-tiled, and both want it arranged at build time. That half is the host-side hook in
[src/steadyc/backend.nim](src/steadyc/backend.nim). A `HostBackend` sees every constant on its way to
flash — tagged with what it is (weights, folded bias, requant multipliers, activation table) and which
op produced it — and may return different bytes, a different shape, or a different dtype; the emitter
re-derives the array length from whatever comes back.

```nim
proc relayout(g: Graph, c: BackendConst): BackendConst =
  result = c
  if c.role == crWeights and c.opKind == okFullyConnected:
    result.data = transposed(c.data, c.dtype.byteWidth, c.shape[0], c.shape[1])
    result.shape = @[c.shape[1], c.shape[0]]

emitModel(g, p, "generated", "my_model",
          backend = initHostBackend("cmsis-nn", "cmsis-nn/int8-hwc", relayout))
```

Biases are offered *after* folding, because folded bytes are what actually reach flash.

Reordered weights are wrong unless the target kernel reads the new order, and no build step can check
that for you. So a backend declares a layout tag, the emitter writes it into the generated module as
`WeightLayout`, and target-side code can refuse a layout it does not implement:

```nim
static: assert model.WeightLayout == "cmsis-nn/int8-hwc"
```

---

## Operators

Implemented: `Conv2D`, `DepthwiseConv2D`, `FullyConnected`, `MaxPool2D`, `AveragePool2D`, `Add`,
`Clamp` (ReLU / ReLU6 / ReLUN1To1), `Reshape` (free — resolved by aliasing), `Pad`, `Concatenation`,
`Mean` (over H and W, i.e. global average), `Logistic`, `Tanh`, `Softmax`.

Three of them do not exist under every policy, and the host says so rather than the kernels branching:

| | `AffineI8` | `RealF32` | `RealFp8` |
|---|---|---|---|
| everything except the three below | yes | yes | yes |
| `Logistic`, `Tanh` | yes | — | yes |
| `Softmax` | yes | — | — |

The reasoning is in [Non-linear activations](#non-linear-activations): a table needs an enumerable
storage domain, and softmax needs a uniform one on top of that.

`Softmax` normalises over the last axis once per row, so it covers both a classifier's single vector
and a detector's grid — FOMO's output is 144 independent 3-class distributions, and one call does all
of them.

Other deliberate limits, each a host-side error naming the op rather than a wrong answer: `Pad` is
spatial only, `Mean` reduces H and W only, `Add` does not broadcast, batch is always 1.

The deliberate bias is toward doing a small set completely rather than a large set approximately.

---

## Kernel performance

The reference kernels are correct and unoptimized. Numbers first, because the repo has no benchmark
task yet and that is the more embarrassing half of the sentence. MobileNetV1 at 96x96 — the `vww`
fixture, about 7.5 MMAC per inference — on one x86-64 core:

| build | per inference | |
|---|---|---|
| `-d:release` | 15.1 ms | 0.5 GMAC/s |
| `-d:danger` | 3.7 ms | 2.0 GMAC/s |
| `-d:danger --passC:-O3 --passC:-march=native` | 3.8 ms | 2.0 GMAC/s |

Two things fall out. Runtime checks cost 4x, which is why the freestanding build uses `-d:danger`; on
the target those checks are not just slow, they have nothing to report to. And `-march=native` buys
nothing at all — the loop nest as written does not auto-vectorize, and that is where the headroom is.

Across the fixtures, at `-d:danger` on the same host, with arena sizes from the compiler's own report:

| model | arena | per inference | |
|---|---|---|---|
| `fomo` | 78 KB | 3.8 ms | 3.2 GMAC/s |
| `vww` | 63 KB | 3.7 ms | 2.0 GMAC/s |
| `mobilenet_v2` | 1.5 MB | 112 ms | 2.7 GMAC/s |

FOMO is the interesting row: a real object detector in 78 KB of RAM and 19 KB of flash. MobileNetV2 at
224 is a desktop-scale model here and is in the suite for coverage and scale, not because 1.5 MB of
arena fits anything this compiler targets.

### Why not Winograd

Winograd F(2x2, 3x3) cuts multiplies by 2.25x, and for this target it is close to the wrong lever
entirely:

- **It does not apply to the layers that cost.** In a MobileNet-class model the overwhelming majority
  of multiplies are in 1x1 pointwise convolutions, which Winograd does not touch. The 3x3s that
  remain are *depthwise*, where there is no channel reduction to amortize the input and output
  transforms over, so the transform overhead eats the saving.
- **It needs RAM.** Transformed tiles are intermediate storage, and RAM is the scarcest thing here —
  the arena for these models is tens of kilobytes, sized to the byte at compile time. Trading it for
  multiplies is backwards on a part with 64 KB of SRAM.
- **It is hostile to int8.** The transforms carry fractional coefficients, so an integer
  implementation needs wider intermediates and its own scaling, and it stops being bit-exact against
  TFLite — a property this project spends real effort maintaining and can currently demonstrate.

Winograd earns its keep on large 3x3 dense convolutions in float or fp16, with a cache and vector
units to hide the transforms. That is a different machine from the one this compiler targets.

### The levers that do matter, in order

1. **Data layout plus SIMD, through a vendor backend.** The single biggest factor, and the reason both
   backend hooks exist. On Cortex-M4/M7 `SMLAD` does two int8 MACs per cycle, on M55/M85 Helium does
   sixteen; both need weights arranged so the inner loop reads contiguous pairs or quads. Published
   CMSIS-NN figures are around 4-5x over reference kernels, and that is a *layout* result as much as
   an instruction-selection one — which is exactly the split between
   [`steadyc/backend.nim`](src/steadyc/backend.nim) and the target-side `steady_backend`.
2. **Portable structure in the reference kernels themselves**, worth having because it costs no
   dependency and helps every target: several accumulators in flight to break the serial dependency on
   `acc`, and output-channel blocking so one pass over an input patch feeds two or four filters
   instead of re-reading the patch per filter. Both are expressible through `mac` without touching the
   policy abstraction.
3. **A 1x1 fast path.** A stride-1 pointwise convolution is a GEMM whose "patch" is already
   contiguous: no padding logic, no per-tap bounds arithmetic, no im2col. Given where the multiplies
   are, this is the highest-value specialization in the portable set.
4. **Depthwise blocked over pixels**, keeping a channel's nine weights in registers across several
   output positions. Depthwise is memory-bound, so the win is in loads, not multiplies.

None of that should be done on the strength of the paragraph above. The first commit here is the
benchmark task, reporting per-operator time and MMAC/s per model, so each change afterwards is
justified by a measurement instead of by an argument.

---

## Building

```sh
nimble test           # full suite, regenerates the example models first
nimble freestanding   # bare-metal build + link audit (needs arm-none-eabi-gcc)
nimble staticlib      # --app:staticlib build + C consumer, diffed against Nim
nimble fetch          # download the checksummed .tflite fixtures
nimble models         # differential harness against TFLite's reference kernels
nimble ci             # all of the above
```

`freestanding` needs `arm-none-eabi-gcc`; `models` needs the fixtures and a reference interpreter:

```sh
python3 -m venv .venv && .venv/bin/pip install ai-edge-litert numpy
nimble fetch                                    # five published models
.venv/bin/pip install tensorflow-cpu            # only to build the other two
.venv/bin/python tests/models/convert.py        # mobilenet_v2 and fomo
```

Both skip themselves, loudly, when their optional dependency is missing. A missing tool is not a
failure, but quietly reporting success would be.

Requires Nim ≥ 2.2.0.

---

## Roadmap

Done:

- [x] Numeric policy abstraction, validated across three dtypes
- [x] Reference kernels — policy-generic, destination-passing, runtime shapes
- [x] Backend override with per-op/per-policy fallback
- [x] IR, shape inference, validation
- [x] Lifetime analysis and arena packing, multi-model
- [x] Quantization resolution and bias folding
- [x] Code emission with flash-resident weights
- [x] Bit-exact verification against an unfolded simulator
- [x] `--os:any --mm:none` build, linked and audited on Cortex-M4
- [x] Tier-1 data movement: `Pad`, `Concatenation`, `Mean` (spatial)
- [x] Host-side numeric codec, LUT activations (`Logistic`, `Tanh`) and int8 `Softmax`
- [x] Host-side backend hook for build-time constant layout, with a layout tag the target can assert
- [x] `--app:staticlib` packaging with a C header, verified by linking a C consumer against it

- [x] TFLite importer, with its own flatbuffer reader — no `flatc`, no dependency
- [x] `steadyc` command-line driver
- [x] Differential harness against TFLite's reference kernels on five real int8 models

Next, roughly in order:

- [ ] **Performance.** The reference kernels are correct and unoptimized, and there is no benchmark in
      the repo, which is the more embarrassing half of that sentence. First a benchmark task that
      reports per-op time and MMAC/s, then the portable structural work: multiple accumulators to
      break the dependency chain, output-channel blocking so an input patch is read once instead of
      once per filter, and a 1x1 fast path — MobileNet-class models spend most of their multiplies
      there. Winograd is the wrong lever for this target; see
      [Kernel performance](#kernel-performance).
- [ ] **A real CMSIS-NN backend**, target kernels and host layout together. Both hooks it needs now
      exist, and this is where the large factors are: SMLAD and Helium, not better loop nests.
- [ ] Wider `Pad` (channel padding), `Mean` over other axes, broadcasting `Add`
- [ ] One shared arena across models emitted separately — the planner already sizes for it, but each
      generated module still declares its own array
- [ ] An ONNX importer, which the IR was shaped to allow and nothing else blocks

Deliberately out of scope: runtime model loading, training, batch sizes above 1.

---

## License

MIT — see [LICENSE](LICENSE).
