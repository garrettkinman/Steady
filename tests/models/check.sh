#!/usr/bin/env bash
# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
#
# Differential harness: real models, compiled by steadyc, checked against
# TFLite's own interpreter.
#
# The end-to-end test proves the compiler agrees with its own simulator, which
# is what makes bias folding trustworthy. It cannot prove agreement with
# TFLite, because both halves of it are ours. This can, and it is the only test
# here that needs something outside the repo:
#
#   tests/models/fetch.sh   the models, verified against pinned checksums
#   .venv                   ai-edge-litert, the reference interpreter
#
# Both are optional. Without them this exits 0 with a note, the way the
# freestanding check skips when there is no ARM toolchain — a missing optional
# dependency is not a failure, but silently reporting success would be.
#
# Per-model tolerance is in int8 LSBs of the *final output*. Two of the five are
# required to be bit-identical end to end; the other two are bounded, for a
# reason the per-tensor pass below pins down precisely rather than assuming.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIR="$ROOT/tests/models"
BUILD="$ROOT/build/models"
PY="$ROOT/.venv/bin/python"
TRIALS="${STEADY_TRIALS:-5}"

# name|tolerance|note
#
# The softmax models get 1 LSB: this compiler evaluates softmax from a
# host-generated exp table, TFLite from gemmlowp's fixed-point exp, and the two
# are within an LSB of each other and of the mathematics. Everything before the
# softmax is required to be exact, which is what `ad` (no softmax) pins.
# name|tolerance|note   (tolerance -1 means compile only, no reference run)
#
# vww and resnet8 are required to be bit-identical, which is the real
# regression guard: 27 depthwise-separable layers and 3 residual adds with
# differing operand zero points, exact to the last LSB.
#
# kws and ad are bounded instead. Both end in a fully-connected layer whose
# right shift is small, and desktop TFLite requantizes int8 FULLY_CONNECTED
# through float while its convolutions use gemmlowp fixed point. Ours is fixed
# point throughout — the same arithmetic CMSIS-NN and TFLite Micro use, and the
# only kind available on a part without an FPU. Where a two-stage fixed-point
# round lands on a tie the two answers differ by one LSB, and a softmax on top
# of a 1-LSB logit amplifies that. `ad` stacks ten such layers, so its
# divergence compounds; the per-tensor pass below is what keeps this honest by
# failing if the first divergence is ever anything but a fully-connected output.
#
# mobilenet_v2 is bit-exact through all 52 convolutions and all 10 residual
# adds, and diverges by one LSB at its global average pool — 4 channels out of
# 1280. Its *output* tolerance is 2 rather than 1 because a softmax turns a
# 1-LSB logit difference into up to 2 LSBs of the 1/256 probability grid; the
# per-tensor pass below reports the tighter and more meaningful bound, and the
# argmax agrees on every trial. That one is worth spelling out, because the direction is the opposite of
# what a tolerance implies: measured against the exact real-valued mean, our
# answer is the correctly rounded one on every channel where the two disagree
# (mean error 0.4805 against TFLite's 0.5195). TFLite's generic MEAN reducer
# computes in float and mis-rounds just past a tie. We fold 1/count into the
# requantization multiplier so the reduction rounds exactly once; doing the
# division in the kernel first, as this compiler originally did, disagreed on
# 130 channels instead of 4.
#
# person_detect is compile-only: modern TFLite refuses to load it, because its
# 2019 converter wrote quantized_dimension 3 on rank-1 bias tensors. This
# compiler ignores that field for biases and handles the model correctly, but
# there is no reference to compare against.
MODELS=(
  "vww|0|MobileNetV1 96x96 RGB, float boundary stripped: bit-exact"
  "mobilenet_v2|2|MobileNetV2 1.0 224, 1000-class softmax, 3.4 MB of weights"
  "fomo|0|FOMO detector: a 12x12 grid of per-cell softmaxes"
  "resnet8|0|residual adds with differing operand zero points: bit-exact"
  "kws|7|depthwise-separable CNN; final FC then softmax"
  "ad|3|10 stacked fully-connected layers, no softmax"
  "person_detect|-1|MobileNetV1 0.25 96x96; rejected by modern TFLite"
)

if ! ls "$DIR"/*.tflite >/dev/null 2>&1; then
  echo "==> no models in tests/models; run tests/models/fetch.sh to enable this check"
  echo "    (mobilenet_v2 and fomo additionally need tests/models/convert.py)"
  exit 0
fi
if [ ! -x "$PY" ] || ! "$PY" -c "import ai_edge_litert" 2>/dev/null; then
  echo "==> no reference interpreter; create one with:"
  echo "        python3 -m venv .venv && .venv/bin/pip install ai-edge-litert numpy"
  exit 0
fi

echo "==> Building the compiler"
nim c --hints:off -d:release --path:"$ROOT/src" \
      --out:"$BUILD/steadyc" "$ROOT/src/steadyc.nim"

failed=0
for spec in "${MODELS[@]}"; do
  IFS='|' read -r name tol note <<<"$spec"
  model="$DIR/$name.tflite"
  [ -f "$model" ] || { echo "==> $name: not fetched, skipping"; continue; }

  echo
  echo "==> $name — $note"
  out="$BUILD/$name"
  rm -rf "$out"
  mkdir -p "$out"

  "$BUILD/steadyc" "$model" -o "$out" -n "$name" | sed 's/^/    /'

  sed "s/@MODEL@/$name/g" "$DIR/runner.nim.in" > "$out/runner.nim"
  nim c --hints:off -d:release --path:"$ROOT/src" \
        --nimcache:"$out/nimcache" --out:"$out/runner" "$out/runner.nim"

  if [ "$tol" -lt 0 ]; then
    echo "    compiled; no reference comparison for this file"
    continue
  fi

  "$out/runner" gen "$model" "$out/io.txt" "$TRIALS"
  "$PY" "$DIR/reference.py" outputs "$model" "$out/io.txt" "$out/ref.txt"
  if ! "$out/runner" check "$out/io.txt" "$out/ref.txt" "$tol"; then
    failed=1
  fi

  # Per-operator attribution: where does agreement stop, and is it somewhere we
  # accept? This is what turns "the outputs differ a bit" into a statement.
  "$PY" "$DIR/reference.py" tensors "$model" "$out/io.txt" "$out/all.txt"
  if ! "$out/runner" tensors "$model" "$out/io.txt" "$out/all.txt"; then
    failed=1
  fi
done

echo
if [ "$failed" -ne 0 ]; then
  echo "differential harness FAILED"
  exit 1
fi
echo "differential harness passed"
