# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

## Target-side runtime.
##
## This is everything that ships to the device: numeric policies, portable
## kernels, and the compile-time backend dispatch layer. It allocates
## nothing, has no dependencies beyond the Nim system module, and is
## intended to build under `--mm:none --os:any`.
##
## Generated model code imports `steady/kernels/dispatch` directly; this
## module is the convenience surface for handwritten code and tests.

import steady/[fp8, policy]
import steady/kernels/dispatch

export fp8, policy, dispatch
