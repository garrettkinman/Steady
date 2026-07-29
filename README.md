<!--
 Copyright (c) 2026 Garrett Kinman

 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
-->

# Steady

[![CI](https://github.com/garrettkinman/Steady/actions/workflows/ci.yml/badge.svg)](https://github.com/garrettkinman/Steady/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Nim](https://img.shields.io/badge/nim-%E2%89%A5%202.2.0-f3d400.svg)](https://nim-lang.org)

An ahead-of-time TinyML inference compiler and runtime, in pure Nim, for microcontrollers.

Steady is not an interpreter. A host-side compiler reads a model, plans its memory, resolves its
quantization, and emits straight-line code with every buffer address already resolved. What runs on
the device is the calls, in order, and nothing else — no graph structure, no operator registry, no
dispatch table, no allocator.

> **Status: early but real.** `steadyc model.tflite` compiles actual int8 models — MobileNetV1/V2,
> ResNet-8, DS-CNN, FOMO — and four of the seven reference models in the test suite come out
> bit-identical to TFLite on every intermediate tensor, with the two divergences attributed to a
> named operator rather than averaged into a tolerance. See [Verification](#verification).

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

See [examples/tiny_cnn.nim](examples/tiny_cnn.nim) for a complete model and
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

- **Arithmetic is a seam, not a hardcoding.** Kernels are generic over a *numeric policy* that
  abstracts what happens between accumulate and store, and `mac` is a primitive rather than sugar
  for `acc += a * b`. One policy ships — `AffineI8`, int8 storage and int32 accumulation — because
  that is what interchange formats carry. The abstraction is there for what plugs into it: SMLAD on
  a Cortex-M4, a vector unit's widened accumulate, an accumulator a hardware unit owns. Kernels
  written around `+=` close that door, and reopening it means rewriting every kernel rather than
  adding a file.
- **No dynamic allocation, ever.** Every activation buffer is a compile-time offset into one static
  arena, verified by linking a real Cortex-M4 image and auditing it for allocator symbols.
- **Zero-overhead execution.** Generated `invoke` is a flat sequence of calls.
- **Acceleratable without forking, at two heights.** Backends override individual *ops* per policy —
  the seam an NPU or CMSIS-NN wants — or individual *policy members* like `mac` and `finish`, which
  is the seam an arithmetic unit wants. Overriding `mac` speeds up every kernel at once without
  touching one. Both resolve by compile-time name resolution with automatic fallback, and the block
  widths dispatch the same way, since how many accumulators belong in flight is a fact about the
  register file holding them. The test suite runs the whole end-to-end path with the arithmetic
  substituted and requires bit-identical results, which is what keeps "the abstraction is free"
  from being an assertion.
- **No external dependencies.** Nim's standard library on the host; nothing at all on the target.

[docs/design.md](docs/design.md) covers the numeric policy, quantization and bias folding, table
activations, the TFLite importer, and the backend hooks in detail.

## Operators

Implemented: `Conv2D`, `DepthwiseConv2D`, `FullyConnected`, `MaxPool2D`, `AveragePool2D`, `Add`,
`Clamp` (ReLU / ReLU6 / ReLUN1To1), `Reshape` (free — resolved by aliasing), `Pad`, `Concatenation`,
`Mean` (over H and W, i.e. global average), `Logistic`, `Tanh`, `Softmax`.

`Logistic` and `Tanh` are lookup tables the host builds by evaluating the true function at all 256
representable inputs, so the device does a load rather than arithmetic — no libm, no software float,
no fixed-point series expansion. That works because int8 has an enumerable storage domain, and it is
a host-side check rather than a `when` branch in a kernel: a policy whose store is wider than a byte
gets a compile-time rejection naming the op. `Softmax` normalises over the last axis once per row, so
one call covers both a classifier's single vector and a detector's grid — FOMO's output is 144
independent 3-class distributions.

Other deliberate limits, each a host-side error naming the op rather than a wrong answer: `Pad` is
spatial only, `Mean` reduces H and W only, `Add` does not broadcast, batch is always 1. The bias is
toward doing a small set completely rather than a large set approximately.

## Verification

Three layers, each answering a question the others cannot.

**Against a simulator.** The end-to-end test runs generated code and a host simulator over the same
inputs and requires bit-identical output. The simulator deliberately evaluates the *unfolded* form —
explicit `x - Zx` at every tap, original biases — so agreement proves the folding transform correct,
including at SAME-padded edges. The same models are then run again with the *arithmetic* replaced —
`mac` substituted and the block widths forced to 1 — and required to produce identical bits, which
is what makes "the numeric seam costs nothing" a test rather than a claim. Requantization arithmetic,
activation tables and softmax normalisation are shared between runtime and simulator, so they are
pinned separately against the published gemmlowp definitions and a float64 reference, entry by
entry.

**Against a linker.** `nimble freestanding` links a real Cortex-M4 image containing both example
models and audits it: no allocator symbols, weights and tables in `.rodata`, arena in `.bss`, and no
soft-float routines anywhere in the image — the host resolved every scale into a multiplier and a
shift, so a float on the target would mean something was left to runtime that should not have been.
The placement check matters more than it looks: a module-level `let` lands in `.data` and is copied
into RAM at startup, which on a part with 64 KB of SRAM and 512 KB of flash is exactly backwards.

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

Measured where it matters: on the part. `nimble mcu` builds a firmware image per model, flashes a
board and reads per-operator **cycle counts** back over its console — no scheduler, no other process,
no frequency scaling, no interrupt enabled anywhere in the image. Rerunning a measurement reproduces
it to the cycle. The reference board is a B-L475E-IOT01A (STM32L475VG, Cortex-M4 at 80 MHz, 96 KB
usable SRAM, weights in internal flash); [a second part](#the-same-compiler-on-a-second-part) and
[a third that is not an ARM at all](#and-a-part-that-is-not-a-cortex-m) are below.

MAC counts come from the compiler's own `macCount`, padded taps included, since those are multiplied
against `padValue` rather than skipped. Arena and flash are the compiler's own report. Cycles are
with the flash accelerators on, which is the configuration a real product would ship:

| model | arena | flash | MMAC | cycles | ms | cyc/MAC | MMAC/s |
|---|---|---|---|---|---|---|---|
| `ad` | 768 B | 265 KB | 0.26 | 2,413,782 | 30.2 | 9.14 | 8.8 |
| `resnet8` | 48 KB | 77 KB | 12.50 | 149,371,445 | 1867.1 | 11.95 | 6.7 |
| `vww` | 63 KB | 214 KB | 7.49 | 114,544,319 | 1431.8 | 15.29 | 5.2 |
| `person_detect` | 54 KB | 214 KB | 7.16 | 112,660,040 | 1408.3 | 15.74 | 5.1 |
| `kws` | 16 KB | 24 KB | 2.66 | 43,169,869 | 539.6 | 16.25 | 4.9 |
| `fomo` | 78 KB | 19 KB | 5.40 | 103,255,958 | 1290.7 | 19.12 | 4.2 |

FOMO is the interesting row: a real object detector in 78 KB of RAM and 19 KB of flash. MobileNetV2
is absent because it does not fit — 1.5 MB of arena against 96 KB of SRAM — and stays in the
correctness suite rather than this one.

### The same compiler on a second part

`--board samd51` runs the same six on an Adafruit ItsyBitsy M4 Express — ATSAMD51G19A, the same
Cortex-M4 at 120 MHz with 192 KB of SRAM and a 4 KB unified cache, reached over nothing but a USB
cable. Cycles with everything enabled:

| model | MMAC | cycles | ms | cyc/MAC | MMAC/s | vs STM32, cycles |
|---|---|---|---|---|---|---|
| `ad` | 0.26 | 2,375,122 | 19.8 | 8.99 | 13.3 | 0.98x |
| `resnet8` | 12.50 | 145,826,568 | 1215.2 | 11.66 | 10.3 | 0.98x |
| `vww` | 7.49 | 111,497,163 | 929.1 | 14.89 | 8.1 | 0.97x |
| `person_detect` | 7.16 | 110,233,310 | 918.6 | 15.40 | 7.8 | 0.98x |
| `kws` | 2.66 | 42,421,704 | 353.5 | 15.97 | 7.5 | 0.98x |
| `fomo` | 5.40 | 99,481,577 | 829.0 | 18.42 | 6.5 | 0.96x |

The last column is the point. Two parts with different vendors, different flash controllers and
different things in front of flash land within 2–4% of each other **in cycles** — so the wall-clock
difference is very nearly just the clock ratio, and the ranking of the models is identical. That is
the result worth having from a second board: it says the numbers in the table above are a property of
these kernels on this core, not of one vendor's memory system.

Where the two genuinely differ is how much that memory system is worth. The SAMD51's 4 KB cache is
**3.05x on `kws`** against no cache at all, where the STM32's prefetch buffer and two caches are 2.1x;
its cacheless row is correspondingly worse (48.8 cyc/MAC against 34.4), because five flash wait states
at 120 MHz with nothing in front of them is what that costs. Every output checksum is identical on
the two parts, which is the claim the whole compiler rests on and now has two witnesses.

### And a part that is not a Cortex-M

`--board esp32c3` runs the same six on an ESP32-C3-DevKitM-1: a single RISC-V core (RV32IMC) at
160 MHz, 384 KB of SRAM, and 4 MB of flash the core **cannot address**. It reaches flash through a
128-entry page table into a 16 KB cache, and nothing is mapped when the image starts — so this port
programs that table before it can call a kernel or read a weight.

The six models, and the same models' cycle counts on the STM32 for scale:

| model | KB/MMAC | MMAC | cycles | ms | cyc/MAC | MMAC/s | vs STM32, cycles |
|---|---|---|---|---|---|---|---|
| `resnet8` | 6 | 12.50 | 171,219,533 | 1070.1 | 13.70 | 11.7 | 1.15x |
| `kws` | 9 | 2.66 | 41,965,806 | 262.3 | 15.80 | 10.1 | 0.97x |
| `ad` | 1019 | 0.26 | 4,823,075 | 30.1 | 18.26 | 8.8 | 2.00x |
| `vww` | 29 | 7.49 | 150,680,790 | 941.8 | 20.12 | 8.0 | 1.32x |
| `fomo` | 4 | 5.40 | 109,865,046 | 686.7 | 20.34 | 7.9 | 1.06x |
| `person_detect` | 30 | 7.16 | 155,059,564 | 969.1 | 21.66 | 7.4 | 1.38x |

The last column is where this board earns its place, and it says something the second one could not.
The SAMD51 landed within 2–4% of the STM32 on every model, which is what "same core, different
vendor" should look like. A different instruction set lands within 15% on the models that are
compute-bound and **2x worse on `ad`** — and the first column says why. `ad` needs roughly one byte
of weights per MAC: ten stacked fully-connected layers use each int8 weight exactly once, so there is
no reuse for a cache to find. The ESP32-C3's flash is a serial part read over SPI through 16 KB of
cache, where both Cortex-M4s read a parallel flash on the core's own bus. Models that reuse their
weights do not notice. A model that streams 265 KB of them per inference notices twice over.

That is worth more than the agreement was. Two Cortex-M4s agreeing says the numbers are a property
of these kernels; a third part disagreeing *only where bandwidth is the limit* says which numbers are
a property of the kernels and which were a property of having flash on the bus.

In wall-clock terms the extra clock mostly pays for it: this part is faster than *both* Cortex-M4s on
three of the six — `kws` in 262 ms against 353 on the SAMD51 and 540 on the STM32 — and faster than
the 80 MHz STM32 on all six, if only by a hair on `ad`. Cycles are what this table is for, but which
part finishes first is a different question and worth answering separately.

There is no cacheless row, and that is structural rather than an omission. On both Cortex-M4s the
accelerators in front of flash are an optimisation: switch them off and the core still reads flash,
only slowly, which is what makes "everything off" the honest number for a part without them. Here the
cache is not in front of the path to flash, it *is* the path — disabling it does not produce a slow
read, it produces no read at all, and the kernels are being fetched through it too. The benchmark asks
each board how many configurations it has, so this one answers one.

**Every output checksum is identical to both Cortex-M4s' on all six models.** Two instruction sets,
three vendors, one set of answers, bit for bit — and this third witness shares no arithmetic
hardware with either of the other two.

The kernel work in [docs/performance.md](docs/performance.md) is worth **2.17x on `kws`** and
**2.02x on `vww`** against the unoptimized reference kernels, and the flash accelerators are worth
more again — prefetch plus both caches take `kws` from 1141 ms to 540 ms. The harness sweeps them
rather than assuming a configuration, and reports the cacheless row too, because that is what parts
without them actually do.

The firmware checksums its output and the harness refuses a record whose per-build nonce does not
match the image it just flashed. A benchmark that cannot tell a fast kernel from a kernel the linker
deleted is measuring its own optimizer.

There is deliberately **no host benchmark.** A number from a desktop core says very little about this
target and is easy to mistake for something that does: it has a cache hierarchy, a branch predictor,
an out-of-order pipeline, other processes, and thermal behaviour the part does not. The one time the
two disagreed, the host was wrong about which change mattered — see the compile-time unroll in
[docs/performance.md](docs/performance.md).

Full methodology, the per-operator profile, the four kernel transforms and why they stay bit-exact,
what was tried and reverted, and what is left: [docs/performance.md](docs/performance.md).

## Building

### Repository layout

| path | what is in it |
|---|---|
| [src/steady/](src/steady/) | the target runtime — numeric policy contract, kernels, dispatch |
| [src/steadyc/](src/steadyc/) | the host compiler — IR, TFLite importer, quantization, arena planner, emitter |
| [examples/](examples/) | models built through the library API; `nimble gen` regenerates them |
| [tests/mcu/boards/](tests/mcu/boards/) | one directory per supported board, and nothing part-specific above it |
| [docs/](docs/) | [design](docs/design.md) and [performance](docs/performance.md) in detail |

### Tasks

```sh
nimble test           # full suite, regenerates the example models first
nimble freestanding   # bare-metal build + link audit (needs arm-none-eabi-gcc)
nimble staticlib      # --app:staticlib build + C consumer, diffed against Nim
nimble fetch          # download the checksummed .tflite fixtures
nimble models         # differential harness against TFLite's reference kernels
nimble ci             # all of the above
nimble mcu            # the same models on a Cortex-M4, in cycles
```

`mcu` is deliberately not part of `ci`: it needs hardware on a USB port, and a timing is not a pass
or a fail. It takes one model name to work on a single one (`tests/mcu/check.sh vww`) and leaves the
raw serial capture and a per-operator table under `build/mcu/<model>/` for diffing two revisions of
a kernel.

`models` needs the fixtures and a reference interpreter. Five are fetched against pinned checksums;
MobileNetV2 and FOMO have no canonical published file, so `convert.py` builds them:

```sh
python3 -m venv .venv && .venv/bin/pip install ai-edge-litert numpy
nimble fetch                                    # five published models
.venv/bin/pip install tensorflow-cpu            # only to build the other two
.venv/bin/python tests/models/convert.py        # mobilenet_v2 and fomo
```

`mcu` needs a board on USB and the cross compiler that board asks for. Three are supported, it
reports which one it found rather than assuming, and with more than one attached it asks — their
numbers are not comparable:

| `--board` | part | toolchain | how it is reached |
|---|---|---|---|
| `stm32l475` | STM32L475VG @ 80 MHz, 96 KB SRAM (B-L475E-IOT01A) | `arm-none-eabi-` | ST-LINK, via `pyocd` |
| `samd51` | ATSAMD51G19A @ 120 MHz, 192 KB SRAM (Adafruit ItsyBitsy M4 Express) | `arm-none-eabi-` | its own USB |
| `esp32c3` | ESP32-C3 @ 160 MHz, 384 KB SRAM (ESP32-C3-DevKitM-1) | `riscv32-esp-elf-` | its boot ROM, via `esptool` |

The ST-LINK needs `pyocd` and a udev rule to be usable without root:

```
SUBSYSTEM=="usb", ATTR{idVendor}=="0483", ATTR{idProduct}=="374b", MODE="0666", TAG+="uaccess"
```

The ItsyBitsy needs neither, and nothing but the cable: it is flashed by copying a UF2 onto its
bootloader's mass-storage volume, and the harness asks for that bootloader the way every
Arduino-compatible board does, by opening the port at 1200 baud and dropping DTR. It has no debug
probe and no USB-serial bridge, so the firmware serves its own CDC console —
[tests/mcu/boards/samd51/usb_cdc.c](tests/mcu/boards/samd51/usb_cdc.c), polled rather than
interrupt-driven, because an interrupt landing inside a measured region is exactly the noise a cycle
counter is supposed to be free of. An image that fails to enumerate hands itself back to the
bootloader after a minute, which is what keeps a bad build from being unrecoverable on a part with no
other way in.

The DevKitM-1 needs Espressif's RISC-V toolchain and `esptool`, and neither a probe nor a udev rule:
it is programmed by its own boot ROM over the same USB bridge the console arrives on. The harness
finds the toolchain under `~/.espressif` if an esp-idf install put it there, which is where it is not
on `PATH`. Nothing else of esp-idf is used or needed — no SDK, no second-stage bootloader, no
FreeRTOS. The image is three pieces at three flash offsets, because on this part flash is not
addressable until software has built a page table for it; see
[tests/mcu/boards/esp32c3/board.ld](tests/mcu/boards/esp32c3/board.ld).

Porting to a fourth is [tests/mcu/boards/](tests/mcu/boards/): a `board.c` implementing the entry
points the benchmark calls, a linker script, and a `board.sh` saying how to find the board and get an
image onto it. A clock, a console, a 32-bit counter, a memory map and whatever sits in front of flash
— no vendor HAL in any of the three that exist. The RISC-V port also brought its own `startup.c` and
taught the harness a toolchain prefix and a Nim `--cpu`, which is the whole of what a second
instruction set cost above the board directory.

Every optional check skips itself, loudly, when its dependency is missing. A missing tool is not a
failure, but quietly reporting success would be.

## Roadmap

Done:

- [x] Numeric policy abstraction — `mac` as a primitive, so arithmetic is substitutable
- [x] Reference kernels — policy-generic, destination-passing, runtime shapes
- [x] Backend override with per-op/per-policy fallback
- [x] Policy-member dispatch — `mac`, `finish` and the block widths overridable, so an arithmetic
      unit plugs in without a kernel changing; verified by running the whole suite with substituted
      arithmetic and requiring bit-identical results
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
- [x] On-target benchmark on real hardware: per-operator cycle counts from an STM32L475, with the
      flash accelerators as a swept variable and a per-build nonce the harness refuses to mismatch
- [x] A second board behind the same harness — an ATSAMD51G19A reached over nothing but USB, with a
      polled CDC console the firmware serves itself so that no interrupt exists to land inside a
      measurement. Same checksums, and cycle counts within 4% of the first part's
- [x] A third board that is not an ARM: an ESP32-C3, RISC-V, whose flash the core cannot address
      until the firmware has built the page table for it. Same checksums on all six models across
      two instruction sets, and the first result the harness has produced that is *not* a property
      of the kernels — where a model is bandwidth-bound rather than compute-bound, serial flash
      costs it 2x
- [x] The portable kernel work that benchmark justified: up to 2.2x on a Cortex-M4, with the
      bit-exactness unchanged, digit for digit

Next, roughly in order:

- [ ] **A real CMSIS-NN backend**, target kernels and host layout together. Both hooks it needs now
      exist, and this is where the large remaining factors are: SMLAD and Helium, not better loop
      nests.
- [ ] `Lanes(P)` for SIMD: a widened `mac` over lane vectors with scalar policies as the degenerate
      case, so the kernels stay one source. Wants blocked activation layouts alongside it, which the
      host layout hook does not yet reach.
- [ ] Wider `Pad` (channel padding), `Mean` over other axes, broadcasting `Add`
- [ ] One shared arena across models emitted separately — the planner already sizes for it, but each
      generated module still declares its own array
- [ ] An ONNX importer, which the IR was shaped to allow and nothing else blocks

Deliberately out of scope: runtime model loading, training, batch sizes above 1.

**Alternative number formats**, too, and that one is worth a sentence because the compiler is
visibly parameterised over a numeric policy and ships exactly one. Float32, OCP fp8 E4M3 and
posit(8,0)-over-an-int64-quire were all implemented here and all removed. No interchange format can
carry them, so none could be driven from a `.tflite`; and the posit one was not the format its name
claimed, since the 2022 standard fixes es = 2 and a 16n-bit quire — a conforming posit8 needs a
128-bit accumulator, while the cheap `mac` that made it interesting depended entirely on es = 0.
The seam they were built on stays, because that is what a CMSIS-NN or SIMD backend plugs into; see
[src/steady/contract.nim](src/steady/contract.nim). A format with hardware behind it would be
welcome back, and would want `Accum(P)` made overridable first.

## License

MIT — see [LICENSE](LICENSE).
