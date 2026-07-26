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

# No `bin` yet: `steadyc` is a library, and the CLI driver arrives with the
# TFLite importer. Until then, graphs are built against the IR directly and
# emitted by a small host program — see examples/tiny_cnn.nim.


# Dependencies

requires "nim >= 2.2.0"


# Tasks

import std/os

task gen, "Regenerate the example model into tests/generated":
  exec "nim c -r --hints:off --path:src examples/tiny_cnn.nim tests/generated"

task test, "Run the full test suite":
  genTask()
  for f in ["test_fp8", "test_quant", "test_arena", "test_dispatch", "test_e2e"]:
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

task ci, "Everything":
  testTask()
  freestandingTask()

task clean, "Remove build artefacts":
  rmDir "tests/generated"
  rmDir "build"
  rmDir "nimcache"
