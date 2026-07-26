# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

## Required by `--os:any`: the Nim runtime has no OS to report failures to,
## so the application supplies these two hooks.
##
## A real firmware image would route `rawoutput` to a UART or an RTT channel
## and make `panic` trigger a watchdog reset or drop into a fault handler.
## Deliberately allocation-free and return-free.

proc rawoutput(s: string) =
  discard

proc panic(s: string) {.noreturn.} =
  while true:
    discard
