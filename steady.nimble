# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Package

version       = "0.1.0"
author        = "Garrett Kinman"
description   = "A production TinyML inference compiler and runtime in pure Nim, for microcontrollers."
license       = "MIT"
srcDir        = "src"
installExt    = @["nim"]

# A hybrid package: `steadyc` is both the compiler library and, built as a
# binary, the command-line driver. `src/steadyc.nim` is the library when
# imported and the CLI when it is the main module.
bin           = @["steadyc"]


# Dependencies

requires "nim >= 2.2.0"


# Tasks

import std/os

task gen, "Regenerate the example models into tests/generated":
  exec "nim c -r --hints:off --path:src examples/tiny_cnn.nim tests/generated"
  exec "nim c -r --hints:off --path:src examples/branch_net.nim tests/generated"

task test, "Run the full test suite":
  genTask()
  for f in ["test_fp8", "test_quant", "test_codec", "test_arena",
            "test_backend", "test_tflite", "test_dispatch", "test_e2e"]:
    echo "\n=== " & f & " ==="
    exec "nim c -r --hints:off --path:src tests/" & f & ".nim"
  # Re-run the dispatch tests against the partial backend fixture; the
  # override paths only exist when -d:steadyBackend is set.
  echo "\n=== test_dispatch (with backend) ==="
  exec "nim c -r --hints:off -d:steadyBackend --path:src --path:tests/fixtures " &
       "--nimcache:build/dispatch_be tests/test_dispatch.nim"

task freestanding, "Verify the runtime builds and links for a bare-metal target":
  genTask()
  exec "bash tests/freestanding/check.sh"

task staticlib, "Package a model as a C static library and consume it from C":
  genTask()
  exec "bash tests/capi/check.sh"

task fetch, "Download the real .tflite fixtures (checksummed)":
  exec "bash tests/models/fetch.sh"

task models, "Differential harness: real models against TFLite's own kernels":
  # Skips itself, loudly, when the fixtures or the reference interpreter are
  # absent — see tests/models/check.sh.
  exec "bash tests/models/check.sh"

task ci, "Everything":
  testTask()
  freestandingTask()
  staticlibTask()
  modelsTask()

task clean, "Remove build artefacts":
  rmDir "tests/generated"
  rmDir "build"
  rmDir "nimcache"
  rmFile "steadyc"
