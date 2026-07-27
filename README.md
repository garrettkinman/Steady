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

> **Status: early but real.** `steadyc model.tflite` compiles actual int8 models — MobileNetV1/V2,
> ResNet-8, DS-CNN, FOMO — and five of the seven reference models in the test suite come out
> bit-identical to TFLite on every intermediate tensor. See [Roadmap](#roadmap).

## Quick start

Requires Nim ≥ 2.2.0 and nothing else.

```sh
nimble build                                 # builds ./steadyc

steadyc <model>.tflite -o <outdir>           # compile a model
steadyc <model>.tflite -o <outdir> --dump    # ...and inspect the imported graph
```

Angle brackets are placeholders, and both are yours to choose. `<outdir>` is any directory and
defaults to `generated`. `<model>` names the model: the file's stem, unless `-n <name>` overrides it.
That name is the generated module's name, the stem of every file written, and the prefix on every
exported C symbol — so pick something a C header can live with. `steadyc --help` lists the rest.

Compiling `my_model.tflite` into `generated/` therefore writes:

| file | what it is |
|---|---|
| `my_model.nim` | the model — `invoke`, `input0`, `output0`, the arena, the sizes |
| `my_model_weights.c` / `.h` | the constants, as `const` C arrays so they land in flash |
| `my_model.h` | the C API header (skip with `--no-capi`) |
| `my_model_api.nim` | `{.exportc, cdecl.}` shims for `--app:staticlib` (skip with `--no-capi`) |

Every compile also reports what the device will cost — here, MobileNetV1 96x96:

```
model 'vww'  policy AffineI8  ops 31
arena:
  buffers        31
  peak RAM       64512 bytes
  without reuse  259472 bytes
  saved          75.1%
  flash (const)  219064 bytes
```

On the target:

```nim
import generated/my_model

let inp = input0()
for i in 0 ..< 64: inp[i] = sample[i]
invoke()
let logits = output0()
```

The generated module also publishes what the host knew and the device would otherwise have to be
told: `ArenaSize`, `Input0Elems`, `Output0Elems`, and `TotalMacs`. They are `const`, so a caller can
size its own buffers or check a cycle budget without a magic number.

One contract worth stating plainly, because the arena is packed: **an input buffer is scratch, not
storage.** Once the model has read it those bytes are reusable, and the planner does reuse them, so
fill every input before every invoke. Output buffers stay valid until the next invoke, and no pointer
ever moves — they are fixed offsets into a static array.

### From C

```sh
nim c --app:staticlib --os:any --mm:none --panics:on --noMain -d:danger -d:useMalloc \
      --path:<steady>/src --out:libmy_model.a generated/my_model_api.nim
```

`<steady>` is wherever this repository is checked out, and `--out` names the archive — without it Nim
names the file after the module, `libmy_model_api.a`. The rest of the line is fixed. Then the header
is the only thing the caller needs to know:

```c
#include "my_model.h"

steady_my_model_init();
int8_t *in = steady_my_model_input0();
for (size_t i = 0; i < STEADY_MY_MODEL_INPUT0_ELEMS; i++) in[i] = quantize(sample[i]);
steady_my_model_invoke();
const int8_t *out = steady_my_model_output0();
```

Element counts, shapes, byte sizes, the arena size, and each input's and output's scale and zero
point are `#define`d, so a C caller holding real values can quantize them without guessing. There is
no context handle and nothing to free: the arena is a static array inside the archive.

### As a library

The importer itself drives the compiler this way:

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

See [examples/tiny_cnn.nim](examples/tiny_cnn.nim) for a complete model, and
[examples/branch_net.nim](examples/branch_net.nim) for one that exercises padding, concatenation, a
spatial mean, a table activation and softmax.

## How it works

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

### Design principles

- **Configurable over the core numeric type.** Kernels are generic over a *numeric policy* that
  abstracts what happens between accumulate and store. Integer-affine quantization and real-number
  formats (fp8, posits) genuinely differ there and nowhere else. Three policies ship: `AffineI8`,
  `RealF32`, `RealFp8`.
- **No dynamic allocation, ever.** Every activation buffer is a compile-time offset into one static
  arena, verified by linking a real Cortex-M4 image and auditing it for allocator symbols.
- **Zero-overhead execution.** Generated `invoke` is a flat sequence of calls.
- **Acceleratable without forking.** Backends override individual ops, per policy, by compile-time
  name resolution with automatic fallback.
- **No external dependencies.** Nim's standard library on the host; nothing at all on the target.

[docs/design.md](docs/design.md) covers the numeric policy, quantization and bias folding, table
activations, the TFLite importer, and the backend hooks in detail.

## Operators

Implemented: `Conv2D`, `DepthwiseConv2D`, `FullyConnected`, `MaxPool2D`, `AveragePool2D`, `Add`,
`Clamp` (ReLU / ReLU6 / ReLUN1To1), `Reshape` (free — resolved by aliasing), `Pad`, `Concatenation`,
`Mean` (over H and W, i.e. global average), `Logistic`, `Tanh`, `Softmax`.

Three do not exist under every policy, and the host says so rather than the kernels branching:

| | `AffineI8` | `RealF32` | `RealFp8` |
|---|---|---|---|
| everything except the three below | yes | yes | yes |
| `Logistic`, `Tanh` | yes | — | yes |
| `Softmax` | yes | — | — |

A table activation needs an enumerable storage domain, and softmax needs a uniform one on top of
that; the op set is therefore a property of the number format rather than a `when` branch in a
kernel. `Softmax` normalises over the last axis once per row, so one call covers both a classifier's
single vector and a detector's grid — FOMO's output is 144 independent 3-class distributions.

Other deliberate limits, each a host-side error naming the op rather than a wrong answer: `Pad` is
spatial only, `Mean` reduces H and W only, `Add` does not broadcast, batch is always 1. The bias is
toward doing a small set completely rather than a large set approximately.

## Verification

Three layers, each answering a question the others cannot.

**Against a simulator.** The end-to-end test runs generated code and a host simulator over the same
inputs and requires bit-identical output. The simulator deliberately evaluates the *unfolded* form —
explicit `x - Zx` at every tap, original biases — so agreement proves the folding transform correct,
including at SAME-padded edges. Requantization arithmetic, activation tables and softmax
normalisation are shared between runtime and simulator, so they are pinned separately against the
published gemmlowp definitions and a float64 reference, entry by entry.

**Against a linker.** `nimble freestanding` links a real Cortex-M4 image containing both example
models and audits it: no allocator symbols, weights and tables in `.rodata`, arena in `.bss`. That
last check matters — a module-level `let` lands in `.data` and is copied into RAM at startup, which
on a part with 64 KB of SRAM and 512 KB of flash is exactly backwards.

**Against TFLite itself.** `nimble models` compiles seven real int8 models with `steadyc` and
compares every intermediate tensor against TFLite's own **reference** kernels — not the optimized
path, which delegates to XNNPACK and requantizes int8 differently from anything a microcontroller
runs. Comparing every tensor rather than just the output attributes a divergence to an *operator*
rather than to a model.

| model | what it covers | agreement |
|---|---|---|
| `vww` | MobileNetV1 96x96, float I/O boundary stripped | every tensor exact |
| `fomo` | object detector: a 12x12 grid of per-cell softmaxes | every tensor exact |
| `resnet8` | residual adds whose operands differ in zero point | every tensor exact |
| `kws` | depthwise-separable CNN, 12-class softmax | every tensor exact |
| `mobilenet_v2` | 1000-class softmax, 3.4 MB of weights, 10 residual adds | exact but for the mean |
| `ad` | ten stacked fully-connected layers | 1 LSB per layer |
| `person_detect` | MobileNetV1 0.25; compiles, but modern TFLite refuses to load it | compile only |

Both divergences have the same cause: desktop TFLite evaluates int8 `FULLY_CONNECTED` and its
generic `MEAN` reducer in float, while its convolutions use gemmlowp fixed point. Steady is fixed
point throughout — the arithmetic CMSIS-NN and TFLite Micro use, and the only kind available on a
part with no FPU. So the harness does not average that into a tolerance; it requires that the first
divergent tensor is one of those two operators, and fails if a convolution ever drifts.

Where `mobilenet_v2` diverges, on 4 channels of 1280, the direction is the opposite of what a
tolerance implies: measured against the exact real-valued mean, ours is the correctly rounded answer
on every channel where the two disagree (mean error 0.4805 against TFLite's 0.5195).

## Performance

`nimble bench` compiles each fixture and times it, whole-inference and per operator. MAC counts come
from the compiler's own `macCount`, padded taps included, since those are multiplied against
`padValue` rather than skipped.

At `-d:danger --passC:-O3` on one core of an i5-1135G7, before and after the kernel work in
[docs/performance.md](docs/performance.md):

| model | arena | flash | MMAC | before | after | | |
|---|---|---|---|---|---|---|---|
| `kws` | 16 KB | 24 KB | 2.66 | 2.05 ms | 1.15 ms | 2.30 GMAC/s | **1.77x** |
| `vww` | 63 KB | 214 KB | 7.49 | 3.78 ms | 2.60 ms | 2.88 GMAC/s | **1.45x** |
| `mobilenet_v2` | 1.5 MB | 3.4 MB | 300.77 | 117.45 ms | 82.83 ms | 3.63 GMAC/s | **1.42x** |
| `fomo` | 78 KB | 19 KB | 5.40 | 3.94 ms | 2.85 ms | 1.89 GMAC/s | **1.38x** |
| `person_detect` | 54 KB | 214 KB | 7.16 | 3.92 ms | 2.84 ms | 2.52 GMAC/s | **1.38x** |
| `resnet8` | 48 KB | 77 KB | 12.50 | 3.83 ms | 2.87 ms | 4.35 GMAC/s | **1.33x** |
| `ad` | 768 B | 265 KB | 0.26 | 0.043 ms | 0.039 ms | 6.77 GMAC/s | **1.10x** |

FOMO is the interesting row: a real object detector in 78 KB of RAM and 19 KB of flash. MobileNetV2
at 224 is desktop-scale here and is in the suite for coverage.

`nimble mcu` runs the same models on a B-L475E-IOT01A (STM32L475VG, Cortex-M4 at 80 MHz) and reads
per-operator **cycle counts** back over the ST-LINK's virtual COM port, with no scheduler, no
frequency scaling and no interrupt enabled anywhere in the image. On `kws` the kernel work is worth
2.17x and on `vww` 2.02x, against 1.45x for the same change on x86-64 — and the flash accelerators
are worth more than the kernels are. Output checksums match the host benchmark's, and the harness
refuses a record whose per-build nonce does not match the image it just flashed.

Full methodology, the per-operator profile, the four kernel transforms and why they stay bit-exact,
what was tried and reverted, and what is left: [docs/performance.md](docs/performance.md).

## Building

```sh
nimble test           # full suite, regenerates the example models first
nimble freestanding   # bare-metal build + link audit (needs arm-none-eabi-gcc)
nimble staticlib      # --app:staticlib build + C consumer, diffed against Nim
nimble fetch          # download the checksummed .tflite fixtures
nimble models         # differential harness against TFLite's reference kernels
nimble ci             # all of the above
nimble bench          # per-operator benchmark on the same models
nimble mcu            # the same models on a Cortex-M4, in cycles
```

`bench` is deliberately not part of `ci`: a timing is not a pass or a fail, and a number measured on
a shared build machine is not worth failing a build over. It takes one model name to work on a
single one (`tests/bench/check.sh vww`), writes a per-operator TSV to `build/bench/<model>/` for
diffing two revisions of a kernel, and adds a `-d:release` column with `STEADY_BENCH_RELEASE=1`.

`models` needs the fixtures and a reference interpreter. Five are fetched against pinned checksums;
MobileNetV2 and FOMO have no canonical published file, so `convert.py` builds them:

```sh
python3 -m venv .venv && .venv/bin/pip install ai-edge-litert numpy
nimble fetch                                    # five published models
.venv/bin/pip install tensorflow-cpu            # only to build the other two
.venv/bin/python tests/models/convert.py        # mobilenet_v2 and fomo
```

`mcu` needs `arm-none-eabi-gcc`, `pyocd`, and a B-L475E-IOT01A on USB. The probe needs a udev rule to
be usable without root:

```
SUBSYSTEM=="usb", ATTR{idVendor}=="0483", ATTR{idProduct}=="374b", MODE="0666", TAG+="uaccess"
```

Porting it to another Cortex-M part is [tests/mcu/board.c](tests/mcu/board.c) and
[tests/mcu/stm32l475.ld](tests/mcu/stm32l475.ld) — a clock, a UART, a 32-bit timer and a memory map,
about a hundred lines with no vendor HAL.

Every optional check skips itself, loudly, when its dependency is missing. A missing tool is not a
failure, but quietly reporting success would be.

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
- [x] Differential harness against TFLite's reference kernels on seven real int8 models
- [x] Per-operator benchmark and the portable kernel work it justified: 1.1x to 1.8x on x86-64 and
      up to 2.2x on a Cortex-M4, with the bit-exactness unchanged, digit for digit
- [x] On-target benchmark on real hardware: cycle counts per operator from an STM32L475, with the
      flash accelerators as a swept variable and the output checksum matching the host's

Next, roughly in order:

- [ ] **A real CMSIS-NN backend**, target kernels and host layout together. Both hooks it needs now
      exist, and this is where the large remaining factors are: SMLAD and Helium, not better loop
      nests.
- [ ] Wider `Pad` (channel padding), `Mean` over other axes, broadcasting `Add`
- [ ] One shared arena across models emitted separately — the planner already sizes for it, but each
      generated module still declares its own array
- [ ] An ONNX importer, which the IR was shaped to allow and nothing else blocks

Deliberately out of scope: runtime model loading, training, batch sizes above 1.

## License

MIT — see [LICENSE](LICENSE).
