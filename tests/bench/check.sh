#!/usr/bin/env bash
# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
#
# Kernel benchmark: real models, compiled by steadyc, timed per operator.
#
# This exists because "the reference kernels are unoptimized" is not a
# measurement, and neither is an argument about which lever matters. Every
# optimization in src/steady/kernels/reference.nim is supposed to be justified
# by a before-and-after from this script.
#
# Two builds are timed, because the difference between them is itself a result:
#
#   -d:danger    what the freestanding build uses. No bounds checks, no
#                overflow checks. On the target those checks would have nowhere
#                to report to anyway.
#   -d:release   the same code with them, quantifying what they cost. Opt in
#                with STEADY_BENCH_RELEASE=1, since it doubles the build.
#
# Per-operator attribution needs -d:steadyProfile, which adds `invokeOp` to the
# generated module and changes `invoke` not at all.
#
#   tests/bench/check.sh                     every fetched fixture
#   tests/bench/check.sh vww kws             just these
#   STEADY_BENCH_RELEASE=1 tests/bench/check.sh vww
#
# TSVs land in build/bench/<model>/bench.tsv, one row per operator, for
# comparing two revisions of a kernel without reading two terminal dumps.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIR="$ROOT/tests/models"
BUILD="$ROOT/build/bench"
CC_FLAGS="${STEADY_BENCH_CFLAGS:---passC:-O3}"

# Every fixture is benchmarkable, including person_detect: modern TFLite
# refuses to *load* it, which stops the differential harness and not this one.
DEFAULT_MODELS="vww fomo resnet8 kws ad mobilenet_v2 person_detect"
MODELS=("${@:-}")
if [ -z "${MODELS[0]:-}" ]; then
  read -r -a MODELS <<<"${STEADY_BENCH_MODELS:-$DEFAULT_MODELS}"
fi

if ! ls "$DIR"/*.tflite >/dev/null 2>&1; then
  echo "==> no models in tests/models; run tests/models/fetch.sh to enable this check"
  echo "    (mobilenet_v2 and fomo additionally need tests/models/convert.py)"
  exit 0
fi

# Pin to one core where the platform allows it. A run that migrates between
# cores mid-measurement varies by more than most changes this is meant to
# detect, which would make the whole exercise circular. STEADY_BENCH_CPU
# overrides the choice; the default is the last core, on the theory that core 0
# fields more interrupts than the others.
PIN=""
if command -v taskset >/dev/null 2>&1 && command -v nproc >/dev/null 2>&1; then
  PIN="taskset -c ${STEADY_BENCH_CPU:-$(( $(nproc) - 1 ))}"
fi

echo "==> Building the compiler"
nim c --hints:off -d:release --path:"$ROOT/src" \
      --out:"$BUILD/steadyc" "$ROOT/src/steadyc.nim"

echo "==> Host"
echo "    $(nim --version | head -1)"
if [ -r /proc/cpuinfo ]; then
  echo "    $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')"
fi
echo "    flags: -d:danger $CC_FLAGS"

for name in "${MODELS[@]}"; do
  model="$DIR/$name.tflite"
  [ -f "$model" ] || { echo "==> $name: not fetched, skipping"; continue; }

  echo
  echo "==> $name"
  out="$BUILD/$name"
  rm -rf "$out"
  mkdir -p "$out"

  "$BUILD/steadyc" "$model" -o "$out" -n "$name" --no-capi | sed 's/^/    /'
  sed "s/@MODEL@/$name/g" "$ROOT/tests/bench/bench.nim.in" > "$out/bench.nim"

  nim c --hints:off -d:danger -d:steadyProfile $CC_FLAGS \
        --path:"$ROOT/src" --nimcache:"$out/cache_danger" \
        --out:"$out/bench_danger" "$out/bench.nim"
  $PIN "$out/bench_danger" "$out/bench.tsv"

  if [ "${STEADY_BENCH_RELEASE:-0}" = "1" ]; then
    nim c --hints:off -d:release -d:steadyProfile $CC_FLAGS \
          --path:"$ROOT/src" --nimcache:"$out/cache_release" \
          --out:"$out/bench_release" "$out/bench.nim"
    echo "    -d:release (runtime checks on):"
    $PIN "$out/bench_release" "$out/bench_release.tsv" | head -3 | sed 's/^/  /'
  fi
done

echo
echo "benchmark complete; per-op TSVs under build/bench/<model>/"
