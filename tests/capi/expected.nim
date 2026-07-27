# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

## Reference output for the C consumer.
##
## Runs the same model over the same input through the Nim module, in the same
## print format. `tests/capi/check.sh` diffs the two, so the packaging check
## proves the library *behaves* the same through the C boundary rather than
## merely linking.

import std/strutils
import ../generated/tiny_cnn as model

when isMainModule:
  let inp = model.input0()
  for i in 0 ..< 64:
    inp[i] = int8(i - 32)
  model.invoke()
  let outp = model.output0()

  var best = 0
  for i in 1 ..< 10:
    if outp[i] > outp[best]: best = i

  var lines: seq[string]
  for i in 0 ..< 10:
    lines.add $int(outp[i])
  lines.add "argmax " & $best
  echo lines.join("\n")
