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

## One import covers the chain: `dispatch` re-exports `arith` (the
## dispatching policy members), which re-exports `contract` (the tags,
## associated types and params).
##
## Deliberately *not* `policy`. Its members are the undispatched defaults, and
## having both those and `arith`'s in scope would make every call site
## ambiguous. Code that wants a default specifically — a kernel backend
## calling `AffineI8.finish`, say — imports `steady/policy` directly.

import steady/kernels/dispatch

export dispatch
