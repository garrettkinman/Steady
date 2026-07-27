#!/usr/bin/env bash
# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
#
# Fetches the real int8 models the differential harness runs against.
#
# The files are not committed: they are someone else's artefacts, they are
# large, and a checksum is a better record of *which* file we tested against
# than a copy of it. Every download is verified against a pinned SHA-256, so a
# silently re-published model fails the fetch instead of quietly changing what
# the test suite means.
#
# All five are int8 with per-channel weights, and between them they cover every
# operator the importer maps except Pad, Concatenation, Logistic and Tanh —
# which examples/branch_net.nim covers bit-exactly against the simulator.

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TINY=https://github.com/mlcommons/tiny/raw/master/benchmark/training
TFLM=https://github.com/tensorflow/tflite-micro/raw/main/tensorflow/lite/micro/models

# name|url|sha256
MODELS=(
  "person_detect|$TFLM/person_detect.tflite|808cfdfc0cf3a6fa6f6fa26bfa379ea97c16d5db7334637766e39c3408502e9d"
  "kws|$TINY/keyword_spotting/trained_models/kws_ref_model.tflite|aeea436800704fce17b17292e4412630ad856e9d777c044c64ef748a880bd0ae"
  "ad|$TINY/anomaly_detection/trained_models/ad01_int8.tflite|87cf24194ef93d1d9b11a591d805526b98008e351655d29883c825c9c106ba24"
  "vww|$TINY/visual_wake_words/trained_models/vww_96_int8.tflite|597a384c8c2c8a1276f04702f25013b7838f2f814f1ca7c174d295b73e3d6b7b"
  "resnet8|$TINY/image_classification/trained_models/pretrainedResnet_quant.tflite|3c002613d1b2475eb51dd78dfb85a546c8ae658dee71cf6ade43b022fe205415"
)

for spec in "${MODELS[@]}"; do
  IFS='|' read -r name url want <<<"$spec"
  dest="$DIR/$name.tflite"
  if [ -f "$dest" ]; then
    got="$(sha256sum "$dest" | cut -d' ' -f1)"
    if [ "$got" = "$want" ]; then
      echo "    $name.tflite ok"
      continue
    fi
    echo "==> $name.tflite has the wrong checksum; re-fetching"
  fi
  echo "==> fetching $name.tflite"
  curl -sSL --fail -o "$dest.part" "$url"
  got="$(sha256sum "$dest.part" | cut -d' ' -f1)"
  if [ "$got" != "$want" ]; then
    rm -f "$dest.part"
    echo "FAIL: $name.tflite hashed $got, expected $want" >&2
    exit 1
  fi
  mv "$dest.part" "$dest"
  echo "    $name.tflite verified"
done
