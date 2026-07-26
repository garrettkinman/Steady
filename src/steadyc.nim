# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

## Host-side model compiler.
##
## Runs on the build machine, never on the target. Takes a graph, validates
## it, resolves quantization, plans the arena, and emits code.
##
## Nothing in here is subject to the target's constraints — it uses `seq`,
## `string` and exceptions freely. Keeping the two halves strictly separate
## is what lets the device-side code be as small as it is.

import steadyc/[ir, quant, arena, emit, sim]

export ir, quant, arena, emit, sim
