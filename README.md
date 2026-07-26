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

> **Status: early.** The compiler, runtime, planner and verification harness work end to end, but
> the TFLite importer is not written yet — graphs are built directly against the IR. See
> [Roadmap](#roadmap) for what is and is not done.

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
  in here, on a real OS.
- **Target** needs `mac`, `finish`, `zero`. Nothing else.

For any 8-bit format, non-linear activations become 256-entry lookup tables the host generates by
evaluating the true function at each representable value — exact, constant-time, and requiring no
target-side arithmetic or libm at all.

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

---

## Verification

The end-to-end test runs generated code and a host simulator over the same inputs and requires
**bit-identical** output. The simulator deliberately evaluates the *unfolded* form — explicit
`x - Zx` at every tap, original biases — so agreement is a proof that the folding transform is
correct, including at SAME-padded edges. An independent implementation of the same optimization
would prove nothing.

Requantization arithmetic is shared between runtime and simulator, so that is covered separately by
vectors pinned to the published gemmlowp definitions.

The bare-metal check links a real Cortex-M4 image and audits it:

```
==> Image size
   text    data     bss     dec
   2956       4     328    3288
==> Allocator audit
    no allocator linked in
==> Placement audit
    weights in flash (.rodata)
    arena in RAM (.bss)
```

That last check matters more than it looks. Nim `const` cannot have its address taken and a
module-level `let` lands in `.data` — copied into RAM at startup, which on a part with 64 KB of SRAM
and 512 KB of flash is exactly backwards. Weights are therefore emitted as `const` C arrays and
linked in, which also gives explicit control over alignment and section placement.

---

## Usage

```nim
import steadyc

var g = initGraph("my_model", pkAffineI8)
let x = g.addInput("input", @[1, 8, 8, 1], dtInt8, inQuant)
# ... add consts, ops ...
g.validate                      # shapes and dtypes checked here, on the host

let p = planOne(g)
echo p.report([g])
emitModel(g, p, "generated", "my_model")
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

See [examples/tiny_cnn.nim](examples/tiny_cnn.nim) for a complete model.

### Hardware acceleration

Write a module named `steady_backend` exposing any subset of the kernel entry points, then build
with `-d:steadyBackend --path:<dir>`. Overriding is per-op *and* per-policy — a backend defining
only an int8 `conv2d` gets used for int8 convolutions and ignored everywhere else, because the
`when compiles(...)` test fails for other instantiations and falls through. Nothing is forked and no
stubs are required. See [tests/fixtures/steady_backend.nim](tests/fixtures/steady_backend.nim).

Note that a serious accelerator usually needs more than a faster kernel — CMSIS-NN wants particular
weight orderings, an NPU wants weights pre-tiled, both at build time. That hook belongs on the host
side and is not built yet.

---

## Operators

Implemented: `Conv2D`, `DepthwiseConv2D`, `FullyConnected`, `MaxPool2D`, `AveragePool2D`, `Add`,
`Clamp` (ReLU / ReLU6 / ReLUN1To1), `Reshape` (free — resolved by aliasing).

The deliberate bias is toward doing a small set completely rather than a large set approximately.

---

## Building

```sh
nimble test           # full suite, regenerates the example model first
nimble freestanding   # bare-metal build + link audit (needs arm-none-eabi-gcc)
nimble ci             # both
```

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

Next, roughly in order:

- [ ] **TFLite importer** — the IR is frontend-neutral and everything downstream is already written
      against it; this is volume, not design
- [ ] **Softmax and LUT-based activations** — needs the host-side codec plumbing, best done as one
      coherent piece
- [ ] **TFLite differential harness in CI** — the simulator proves the folding transform, but only
      TFLite itself can prove agreement with TFLite
- [ ] Remaining Tier-1 ops: `Pad`, `Concatenation`, `Mean`
- [ ] Host-side backend hook for build-time weight layout transformation
- [ ] A real CMSIS-NN backend
- [ ] `--app:staticlib` packaging with a clean C header, so the output drops into Keil / IAR /
      CubeIDE / Zephyr without Nim in the loop

Deliberately out of scope: runtime model loading, training, batch sizes above 1.

---

## License

MIT — see [LICENSE](LICENSE).
