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

import steadyc/[ir, quant, codec, arena, backend, emit, capi, sim, tflite]

export ir, quant, codec, arena, backend, emit, capi, sim, tflite

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# COMMAND-LINE DRIVER
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# `steadyc model.tflite -o generated` is the whole interface. Importing this
# module as a library is unaffected: none of the below exists unless it is the
# main module.

when isMainModule:
  import std/[os, parseopt, strformat, strutils]

  const Usage = """
steadyc """ & ToolVersion & """ — ahead-of-time TinyML compiler

  steadyc <model.tflite> [options]

Options:
  -o, --out:DIR      where to write the generated files (default: generated)
  -n, --name:NAME    model name and file stem (default: the file's base name)
      --align:N      arena alignment in bytes, a power of two (default: 8)
      --no-capi      skip the C header and static-library shim
      --dump         print the imported graph, one operator per line
  -q, --quiet        do not print the resource report
  -h, --help         this text

Emits <stem>.nim, <stem>_weights.c/.h and, unless --no-capi, <stem>.h plus
<stem>_api.nim for packaging as a static library.
"""

  proc fnv1a64(bytes: openArray[byte]): uint64 =
    ## Provenance, not security: a stamp in the generated header that changes
    ## when the input model changes. Spelled out rather than pulled in, so it
    ## cannot drift with a library version.
    result = 0xcbf29ce484222325'u64
    for b in bytes:
      result = (result xor uint64(b)) * 0x100000001b3'u64

  proc fail(msg: string) {.noreturn.} =
    stderr.writeLine "steadyc: " & msg
    quit 1

  var
    modelPath = ""
    outDir = "generated"
    stem = ""
    alignment = DefaultAlignment
    wantCApi = true
    wantDump = false
    quiet = false

  # parseopt only recognises `-o:dir`, but `-o dir` is what people type, so an
  # option still waiting for its value claims the next argument.
  var pending = ""

  proc takeValue(opt, val: string) =
    case opt
    of "o", "out": outDir = val
    of "n", "name": stem = val
    of "align":
      try: alignment = parseInt(val)
      except ValueError: fail &"--align needs an integer, got '{val}'"
    else: discard

  for kind, key, val in getopt(commandLineParams()):
    case kind
    of cmdArgument:
      if pending.len > 0:
        takeValue(pending, key)
        pending = ""
      elif modelPath.len > 0:
        fail &"unexpected second input '{key}'; one model at a time"
      else:
        modelPath = key
    of cmdLongOption, cmdShortOption:
      if pending.len > 0:
        fail &"--{pending} needs a value"
      case key
      of "o", "out", "n", "name", "align":
        if val.len > 0: takeValue(key, val) else: pending = key
      of "no-capi": wantCApi = false
      of "dump": wantDump = true
      of "q", "quiet": quiet = true
      of "h", "help":
        echo Usage
        quit 0
      else:
        fail &"unknown option '{key}' (try --help)"
    of cmdEnd: discard

  if pending.len > 0:
    fail &"--{pending} needs a value"

  if modelPath.len == 0:
    echo Usage
    quit 1

  if stem.len == 0:
    stem = modelPath.splitFile.name
  let ident = cIdent(stem)
  if ident != stem:
    # The stem becomes a Nim module name and part of every C symbol.
    fail &"'{stem}' is not usable as a file stem; try --name:{ident}"

  try:
    let content = readFile(modelPath)
    var bytes = newSeq[byte](content.len)
    if content.len > 0:
      copyMem(addr bytes[0], unsafeAddr content[0], content.len)
    let stamp = &"fnv1a64:{fnv1a64(bytes):016x}"

    var g = importTflite(bytes, stem, modelPath)
    let plan = planOne(g, alignment)

    if wantDump:
      stdout.write g.summary
    if not quiet:
      echo &"model '{g.name}'  policy {g.policy.policyName}  ops {g.ops.len}"
      stdout.write plan.report([g])

    emitModel(g, plan, outDir, stem, sourceHash = stamp)
    if wantCApi:
      emitCApi(g, plan, outDir, stem)
    if not quiet:
      echo &"wrote {outDir}/{stem}.nim and companions"
  except IOError as e:
    fail &"cannot read '{modelPath}': {e.msg}"
  except FbError as e:
    fail &"'{modelPath}' is not a well-formed flatbuffer: {e.msg}"
  except TfliteError as e:
    fail e.msg
  except IrError as e:
    fail &"the imported graph does not validate: {e.msg}"
  except QuantError as e:
    fail &"quantization: {e.msg}"
  except ArenaError as e:
    fail &"arena planning: {e.msg}"
