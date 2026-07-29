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

One policy ships: `AffineI8` — int8 storage, int32 accumulation, a Q31 multiplier and shift on the
way out.

**`mac` is a primitive, not sugar for `acc += a * b`,** and that is the reason the contract exists
at all rather than the kernels simply being written in int8. Some accumulators support fused
multiply-accumulate and nothing else; some cores have one instruction that does the multiply and the
add together, which is where a CMSIS-NN backend gets its speed. Routing every kernel through `mac`
keeps both available, and it is the single most expensive thing to retrofit — it is a change to
every kernel rather than the addition of a file.

Three other policies once shipped — `RealF32`, `RealFp8` (OCP FP8 E4M3), and `RealP8`, posit(8,0)
over an exact int64 quire — and were removed. Two reasons, recorded here because "the compiler is
generic over a numeric policy and has one" is a fair thing to ask about:

- **Nothing could feed them.** No interchange format carries fp8 or posit weights, so a model under
  those policies could only be built through the library API by hand. The seven real models in the
  test suite are all int8, and so is every `.tflite` anyone would bring.
- **The posit one was not the format its name claimed.** The 2022 Posit Standard fixes `es = 2` and
  a quire of `16n` bits, so a conforming posit8 needs a 128-bit accumulator. What was implemented
  was posit(8,0) with an int64 accumulator — ample for *that* format, and genuinely interesting
  because `es = 0` makes every value an integer multiple of 2^-6, so products are multiples of
  2^-12, an int64 just counts them, and `mac` is two table loads and one 16x16 multiply with no
  floating point anywhere. But that cheapness is a property of `es = 0`, not of posits: under the
  standard, products span 2^-48 to 2^48 and `mac` becomes a four-word shifted accumulate with carry
  propagation. The result was real; it was a result about posit(8,0).

What stands in their place as evidence that the seam is real is
[tests/fixtures/steady_arith.nim](../tests/fixtures/steady_arith.nim), which substitutes the
arithmetic under the shipping policy and requires the end-to-end models to produce identical bits.

A format with hardware behind it would be welcome back. It would want `Accum(P)` made overridable
first — see [Arithmetic — `steady_arith`](#arithmetic--steady_arith).

### Adding a numeric format

The host and the target need *different* things from a numeric type, which is what keeps the cost of
one down:

- **Host** needs `decode: bits -> float64` and `encode: float64 -> bits` — for converting weights,
  generating activation lookup tables, and running the reference harness. This is
  `steadyc/codec.nim`, and it is the entire host-side numeric surface.
- **Target** needs `mac`, `finish`, `zero`, plus `lutIndex` if the format is 8-bit. Nothing else.

When there were four policies, adding one was a new module for the format, one policy block, and a
case arm in each of six `case` statements the compiler pointed at by refusing to compile without
them. No kernel changed, no planner change, no emitter restructuring. That is the cost to expect,
plus a way to get weights in — which is the part that turned out to matter.

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

This is where a format's structure decides an op's availability rather than the code branching on it.
A table activation needs an enumerable storage domain, and softmax needs a uniform one on top of that
— int8 has both. A policy whose store is wider than a byte simply does not define `lutIndex`, so
`lut1d` fails to instantiate and the host rejects the op by name during validation. The kernel set
stays policy-generic and the *op* set is a property of the number format, checked on the host rather
than branched on in a kernel.

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

The same two models are then run a second time with the *arithmetic* substituted — `mac` replaced
and the block widths forced from 4 to 1, so the matmul family takes its unblocked path — and required
to produce identical bits. That covers two claims at once: that the arithmetic seam is genuinely
substitutable, and that each accumulator sees the same taps in the same order whatever the blocking.

### The bare-metal audit

`nimble freestanding` links a real Cortex-M4 image containing both models and audits it:

```
==> Image size
   text    data     bss     dec
  16552       4    1480   18036
==> Allocator audit
    no allocator linked in
==> Placement audit
    weights in flash (.rodata)
    activation and exp tables in flash (.rodata)
    no soft-float routines linked in
    arena in RAM (.bss)
```

The soft-float check is the *host* half of the compiler under audit. Every scale and zero point was
resolved into an integer multiplier and shift at build time, so nothing on the target should need a
float; if `__aeabi_dadd` and friends ever became reachable, something is being evaluated at runtime
that was supposed to have been evaluated at compile time — on a part with no FPU.

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

There are two target-side seams, at different heights, because different hardware attaches at
different heights. Both resolve with `when compiles(...)`, so both are free at runtime, and a backend
may use either or both.

### Whole kernels — `steady_backend`

Write a module named `steady_backend` exposing any subset of the kernel entry points, then build with
`-d:steadyBackend --path:<dir>`. Overriding is per-op *and* per-policy — a backend defining only an
int8 `conv2d` gets used for int8 convolutions and ignored everywhere else, because the
`when compiles(...)` test fails for other instantiations and falls through. Nothing is forked and no
stubs are required. See [tests/fixtures/steady_backend.nim](../tests/fixtures/steady_backend.nim).

This is the seam for an NPU or a CMSIS-NN port: hardware that swallows a whole layer.

### Arithmetic — `steady_arith`

An arithmetic unit does not accelerate `conv2d`; it accelerates `mac`. A Cortex-M4's SMLAD is one
instruction in an innermost loop, not a layer. Routing that through the op-level seam would mean
reimplementing seven kernels that differ from the reference only in which instruction sits in that
loop, which is exactly the forking this design exists to avoid.

So the policy members dispatch too. Write a module named `steady_arith` defining any subset of
`mac`, `addBias`, `finish`, `zeroAccum`, `accumulate`, `divAccum`, `meanScale`, `storeOf`,
`addRescaled`, `lowestStore`, `lutIndex` — per policy — and build with `-d:steadyArith --path:<dir>`.
Overriding `mac` makes convolution, depthwise, fully-connected, add, concat-rescale, mean and pooling
all faster at once, with no kernel touched.

The block widths dispatch through the same seam. `OcBlock`, `DwBlock` and `FcBlock` decide how many
accumulators are in flight, which is a fact about the register file holding them: four is right for a
core carrying int32 accumulators, a unit with a *single* accumulator register wants one — blocking by
four would spill it to memory every iteration and lose more than the load reuse gains — and a vector
unit wants considerably more.

An arithmetic backend imports `steady/contract` rather than `steady/policy`, and that is what keeps
the layering acyclic — it has to name `AffineI8` to overload on it, and the module it overrides
cannot import the module overriding it:

```
contract.nim        tags, associated types, params, block widths
  steady_arith      (optional) hardware arithmetic — below the policy
policy.nim          default members
kernels/arith.nim   per-member dispatch
kernels/reference   sequenced loops
  steady_backend    (optional) whole-kernel override — above them
```

The defaults stay compiled and reachable whatever a backend does, which is what makes a backend
differential-testable rather than merely a substitute for the thing it replaces.
[tests/fixtures/steady_arith.nim](../tests/fixtures/steady_arith.nim) is the worked example: it
replaces `mac` with arithmetic identical to the default and sets every block width to 1, so the
kernels take their unblocked path. `nimble test` then runs the whole suite — including the
end-to-end models against a simulator that knows nothing about either — and requires every result to
come out bit-identical. What is under test is that the seam changes nothing.

With one policy shipping, this fixture is also the only standing demonstration that the seam is real.
It was worth keeping for that reason alone: without it, `mac` quietly stops being a primitive and
becomes a spelling of `+=`.

**Not overridable, and the known limitation**: the associated types. A backend may replace the
arithmetic over an accumulator; it cannot say the accumulator *is* something else — an architectural
register of another width, say — because that changes what the host emits too, so it is a new policy
rather than an override. That is precisely the case this seam was built to serve and the one place it
does not reach. Making `Accum(P)` opaque and overridable is the fix, and it is not a large change; it
has not been made because there is no hardware here to test it against.

**Not yet done**: SIMD. A vector unit needs the loop to be vector-shaped, not just the primitive to
be faster, and the intended shape is a `Lanes(P)` associated constant plus a widened `mac` over lane
vectors, with today's scalar policy as the degenerate `Lanes = 1` case so the kernels stay one
source. That is a change to the kernels rather than to this seam. Blocked activation layouts are the
other half of it and would reach further — the host layout hook rewrites constants only, so a vector
unit wanting NC8HW8 activations would need shape inference and the arena planner to agree.

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
