# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

"""Reference values from TFLite itself.

The only part of the test suite that is not Nim, and the only part with an
external dependency, because the claim being checked is "this agrees with
TFLite" and nothing but TFLite can settle it.

Two modes:

    outputs   final output per trial — the end-to-end gate
    tensors   every intermediate tensor for one trial, keyed by name, so a
              divergence can be attributed to an operator instead of a model

`BUILTIN_REF` selects TFLite's reference kernels. That matters more than it
looks: the optimized path delegates to XNNPACK, whose int8 requantization is
its own, and a microcontroller runs neither of them — it runs the reference
integer semantics, which is what this compiler targets.

The division of labour is deliberate. This script does not choose the inputs
and does not know the model's quantization: the Nim side writes both, because
after the importer strips a boundary QUANTIZE the graph's input quantization is
a property of a tensor that was internal to the file. Here we take integer codes,
present them to the interpreter in whatever domain it wants, and hand back codes
in the domain the compiled model produces. Both conversions are exact.
"""

import sys

import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType


def read_io(path):
    meta, trials = {}, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, rest = line.partition(" ")
            if key == "trial":
                trials.append(np.array([int(v) for v in rest.split()], dtype=np.int32))
            else:
                meta[key] = rest
    return meta, trials


def make_interpreter(model_path, all_tensors=False):
    return Interpreter(
        model_path=model_path,
        experimental_op_resolver_type=OpResolverType.BUILTIN_REF,
        experimental_preserve_all_tensors=all_tensors,
    )


def feed(interp, inp, q, in_scale, in_zero):
    if inp["dtype"] in (np.int8, np.int16):
        value = q.astype(inp["dtype"]).reshape(inp["shape"])
    elif inp["dtype"] == np.float32:
        # The model quantizes internally; feeding the exact real values these
        # int8 codes stand for makes that step a round trip.
        value = (in_scale * (q.astype(np.float32) - in_zero)).reshape(inp["shape"])
        value = value.astype(np.float32)
    else:
        raise SystemExit(f"unsupported interpreter input dtype {inp['dtype']}")
    interp.set_tensor(inp["index"], value)


def to_codes(values, dtype, scale, zero):
    if dtype in (np.int8, np.int16):
        return values.astype(np.int32)
    if dtype == np.float32:
        # Invert the trailing DEQUANTIZE the importer stripped.
        q = np.round(values.astype(np.float64) / scale).astype(np.int64) + zero
        return np.clip(q, -128, 127).astype(np.int32)
    raise SystemExit(f"unsupported interpreter output dtype {dtype}")


def ports(interp, meta, key):
    """A model's inputs or outputs in the order `io.txt` lists them, by name,
    with how many values each holds"""
    details = {d["name"]: d for d in (interp.get_input_details() if key == "input"
                                      else interp.get_output_details())}
    names = meta[f"{key}_names"].split()
    sizes = [int(v) for v in meta[f"{key}_sizes"].split()]
    return [(details[n], k) for n, k in zip(names, sizes)]


def run_length(interp, meta, trial):
    """One trial of a model taking a length: its first value the length,
    every input's open dimension resized to it"""
    n, q = int(trial[0]), trial[1:]
    details = {d["name"]: d for d in interp.get_input_details()}
    order = [details[name] for name in meta["input_names"].split()]
    for d in order:
        shape = [n if v < 0 else v for v in d["shape_signature"]]
        interp.resize_tensor_input(d["index"], shape)
    interp.allocate_tensors()
    at = 0
    details = {d["name"]: d for d in interp.get_input_details()}
    for name in meta["input_names"].split():
        d = details[name]
        size = int(np.prod(d["shape"]))
        interp.set_tensor(d["index"], q[at:at + size].astype(d["dtype"]).reshape(d["shape"]))
        at += size
    interp.invoke()
    outs = {d["name"]: d for d in interp.get_output_details()}
    return np.concatenate([interp.get_tensor(outs[name]["index"]).reshape(-1).astype(np.int32)
                           for name in meta["output_names"].split()])


def run_ports(interp, meta, q):
    """One trial of a model with several inputs or outputs, whose values all
    go in and come out as they are, integers at both ends"""
    at = 0
    for d, n in ports(interp, meta, "input"):
        interp.set_tensor(d["index"], q[at:at + n].astype(d["dtype"]).reshape(d["shape"]))
        at += n
    interp.invoke()
    return np.concatenate([interp.get_tensor(d["index"]).reshape(-1).astype(np.int32)
                           for d, _ in ports(interp, meta, "output")])


def mode_outputs(model_path, io_path, out_path):
    meta, trials = read_io(io_path)
    interp = make_interpreter(model_path)
    interp.allocate_tensors()
    if "bound" in meta:
        with open(out_path, "w") as f:
            f.write("# reference kernels, each trial at its own length\n")
            for q in trials:
                f.write("trial " + " ".join(str(int(v)) for v in run_length(interp, meta, q)) + "\n")
        print(f"    {len(trials)} reference outputs, at their own lengths")
        return
    if "input_names" in meta:
        with open(out_path, "w") as f:
            f.write("# reference kernels, several inputs and outputs\n")
            for q in trials:
                f.write("trial " + " ".join(str(int(v)) for v in run_ports(interp, meta, q)) + "\n")
        print(f"    {len(trials)} reference outputs")
        return
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    rows = []
    for q in trials:
        feed(interp, inp, q, float(meta["in_scale"]), int(meta["in_zero"]))
        interp.invoke()
        got = interp.get_tensor(out["index"]).reshape(-1)
        rows.append(to_codes(got, out["dtype"], float(meta["out_scale"]),
                             int(meta["out_zero"])))

    with open(out_path, "w") as f:
        f.write(f"# reference kernels, I/O {inp['dtype'].__name__} -> "
                f"{out['dtype'].__name__}\n")
        for r in rows:
            f.write("trial " + " ".join(str(int(v)) for v in r) + "\n")
    print(f"    {len(rows)} reference outputs "
          f"({inp['dtype'].__name__} -> {out['dtype'].__name__} at the interpreter)")


def mode_tensors(model_path, io_path, out_path):
    meta, trials = read_io(io_path)
    interp = make_interpreter(model_path, all_tensors=True)
    interp.allocate_tensors()
    if "bound" in meta:
        run_length(interp, meta, trials[0])
    elif "input_names" in meta:
        run_ports(interp, meta, trials[0])
    else:
        inp = interp.get_input_details()[0]
        feed(interp, inp, trials[0], float(meta["in_scale"]), int(meta["in_zero"]))
        interp.invoke()

    written = 0
    with open(out_path, "w") as f:
        for d in interp.get_tensor_details():
            if d["dtype"] not in (np.int8, np.int16):
                continue
            try:
                v = interp.get_tensor(d["index"]).reshape(-1)
            except (ValueError, IndexError):
                continue        # a tensor the runtime did not materialise
            f.write(d["name"] + "\t" + " ".join(str(int(x)) for x in v) + "\n")
            written += 1
    print(f"    {written} intermediate tensors for trial 0")


def main():
    if len(sys.argv) < 5:
        raise SystemExit("usage: reference.py outputs|tensors <model> <io.txt> <out>")
    mode = sys.argv[1]
    if mode == "outputs":
        mode_outputs(*sys.argv[2:5])
    elif mode == "tensors":
        mode_tensors(*sys.argv[2:5])
    else:
        raise SystemExit(f"unknown mode '{mode}'")


if __name__ == "__main__":
    main()
