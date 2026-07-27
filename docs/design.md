<!--
 Copyright (c) 2026 Garrett Kinman

 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
-->

# Design

Background to the summary in the [README](../README.md): the numeric policy, non-linear activations,
quantization, the TFLite importer, and the extension points.

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
([tests/test_codec.nim](../tests/test_codec.nim) pins that, entry by entry). The softmax was expected
to be the *loosest* part of the compiler, since it evaluates `exp` from a host-built table where
TFLite uses gemmlowp's fixed-point series — so it is worth recording that on three real models it
comes out bit-identical to TFLite anyway, including a twelve-class and a ten-class softmax.

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

## Verification, in detail

The end-to-end test runs generated code and a host simulator over the same inputs and requires
**bit-identical** output. The simulator deliberately evaluates the *unfolded* form — explicit
`x - Zx` at every tap, original biases — so agreement is a proof that the folding transform is
correct, including at SAME-padded edges. An independent implementation of the same optimization would
prove nothing.

Two models are checked this way: a straight-line CNN, and a second one built to be awkward — explicit
padding, two branches concatenated where one operand needs rescaling and the other does not, a
spatial mean into a different quantization, a table activation, a softmax.

Requantization arithmetic is shared between runtime and simulator, so that is covered separately by
vectors pinned to the published gemmlowp definitions. Activation tables and the softmax normalisation
are shared the same way, and are pinned the same way — against a float64 reference, entry by entry.

### The bare-metal audit

`nimble freestanding` links a real Cortex-M4 image containing both models and audits it:

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

### Against TFLite itself

The simulator proves the folding transform, and the unit tests pin the tables and the fixed-point
arithmetic against the mathematics. Neither can prove agreement with TFLite, because both halves are
ours. `nimble models` does that, against TFLite's own **reference** kernels — not the optimized path,
which delegates to XNNPACK and requantizes int8 differently from anything a microcontroller runs.
Five models are fetched against pinned checksums (`nimble fetch`); MobileNetV2 and FOMO have no
canonical published file, so [tests/models/convert.py](../tests/models/convert.py) builds them. The
result table is in the [README](../README.md#verification).

Every intermediate tensor is compared, not just the output, so a divergence is attributed to an
*operator* rather than to a model. Fifty-two depthwise-separable layers, ten residual adds whose
operands carry different zero points, pooling, reshape, matmul, and softmax over both a vector and a
144-cell grid: exact to the last LSB.

`ad` stacks ten fully-connected layers with small requantization shifts, so its divergence compounds
to 2-4 LSBs by the output. `mobilenet_v2` diverges only at its global average pool, on 4 channels of
1280 — and there the direction is the opposite of what a tolerance implies. Measured against the
exact real-valued mean, **our answer is the correctly rounded one on every channel where the two
disagree** (mean error 0.4805 against TFLite's 0.5195); their float path mis-rounds just past a tie.
That op is also where MobileNetV2 paid for itself: `MEAN` originally divided the accumulator in the
kernel and then requantized, which disagreed on 130 channels rather than 4. Folding `1/count` into
the requantization multiplier means the reduction rounds exactly once — more accurate, and one fewer
division in the kernel.

The harness also earned its keep immediately: it found an overflow in `roundingDivideByPOT`, where
the mask was built as `1'i32 shl exponent` and an exponent of 31 does not fit an int32 — gemmlowp
writes `1ll <<` for exactly this reason. A MobileNet channel whose weights are all but zero produces
a multiplier that small. Any bounds-checked build trapped on it, and `-d:danger` only worked by
accident of two's-complement wrapping.

## The TFLite importer

[src/steadyc/tflite.nim](../src/steadyc/tflite.nim) populates the IR from a `.tflite` file, reading
the flatbuffer through [a reader written for the purpose](../src/steadyc/flatbuffers.nim) — about 200
lines, no `flatc`, no generated accessors, no dependency. Schema field indices are named constants so
they can be read against `schema.fbs` line by line, because a wrong index does not fail loudly: it
reads a neighbouring field, and a stride that arrives as a dilation produces a model that runs and is
wrong.

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

## Packaging for C

`emitCApi` writes two more files: `<name>_api.nim`, a set of `{.exportc, cdecl.}` wrappers, and
`<name>.h`, the header a C caller includes and the only thing it needs to know. Usage is in the
[README](../README.md#from-c).

`nimble staticlib` builds the library, greps the archive for the entry points, compiles a C program
that includes nothing but the header, links the two with plain `gcc`, and diffs its output against
the same input run through the Nim module. "It linked" and "it still computes the same thing" are
different claims and only the second one is interesting.

## Hardware acceleration

Write a module named `steady_backend` exposing any subset of the kernel entry points, then build with
`-d:steadyBackend --path:<dir>`. Overriding is per-op *and* per-policy — a backend defining only an
int8 `conv2d` gets used for int8 convolutions and ignored everywhere else, because the
`when compiles(...)` test fails for other instantiations and falls through. Nothing is forked and no
stubs are required. See [tests/fixtures/steady_backend.nim](../tests/fixtures/steady_backend.nim).

A faster kernel is usually not the whole story: CMSIS-NN wants particular weight orderings, an NPU
wants weights pre-tiled, and both want it arranged at build time. That half is the host-side hook in
[src/steadyc/backend.nim](../src/steadyc/backend.nim). A `HostBackend` sees every constant on its way
to flash — tagged with what it is (weights, folded bias, requant multipliers, activation table) and
which op produced it — and may return different bytes, a different shape, or a different dtype; the
emitter re-derives the array length from whatever comes back.

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
