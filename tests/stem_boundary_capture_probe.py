"""Capture unchanged native stem operands after a complete original exact gate."""

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
from merlin.common.digest import sha256_file
from merlin.llvmlower.abi import HostModel
from merlin.llvmlower.codegen import mlir_runtime_c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    base, bundle = args.base.resolve(), args.bundle.resolve()
    build = base / "build_direct"
    manifest = json.loads((bundle / "stem_pool.json").read_text())
    shape = manifest["shape"]
    symbol = manifest["symbol"] + "_kernel"
    catalog = json.loads((build / "device_catalog/device_catalog.json").read_text())
    sources = list(catalog["native_oracle_sources"])
    original = bundle / "native_oracle.c"
    assert str(original) in sources
    text = original.read_text()
    signature = f"void {symbol}(int8_t*a,int8_t*b,int8_t*c){{"
    assert text.count(signature) == 1
    text = text.replace(signature, signature.replace(symbol, symbol + "_original"))
    a_count = (shape["h"] + 6) * (shape["w"] + 6) * 3
    b_count = 147 * shape["cout"]
    c_count = ((shape["h"] + 3) // 4) * ((shape["w"] + 3) // 4) * shape["cout"]

    def dump(name, pointer, count):
        path = json.dumps(str(work / (name + ".bin")))
        return f'{{FILE*f=fopen({path},"wb");if(!f||fwrite({pointer},1,{count},f)!={count}||fclose(f))__builtin_trap();}}'

    text = (
        "#include <stdio.h>\n"
        + text
        + f"""
void {symbol}(int8_t*a,int8_t*b,int8_t*c){{
 {dump("a", "a", a_count)} {dump("b", "b", b_count)}
 {symbol}_original(a,b,c);
 {dump("expected", "c", c_count)}
}}
"""
    )
    instrumented = work / "instrumented_oracle.c"
    instrumented.write_text(text)
    sources[sources.index(str(original))] = str(instrumented)
    model_object, reference = base / "host/model.o", base / "host/reference.c"
    shim, runtime = build / "device/device_catalog_shim.c", Path(mlir_runtime_c())
    shared = work / "model.so"
    command = [
        "cc",
        "-O3",
        "-march=native",
        "-fPIC",
        "-shared",
        str(model_object),
        str(reference),
        *sources,
        str(shim),
        str(runtime),
        "-lm",
        "-o",
        str(shared),
    ]
    subprocess.run(command, check=True, capture_output=True)
    spec = importlib.util.spec_from_file_location(
        "whole_probe", Path(__file__).with_name("fused_whole_model_probe.py")
    )
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    inputs = probe.forward_args(base / "capture", build)
    golden = np.load(base / "capture/golden.npy")
    output = np.zeros_like(golden)
    HostModel.load(str(shared))(
        [(v.ctypes.data, v.shape) for v in inputs]
        + [(output.ctypes.data, output.shape)]
    )
    assert np.array_equal(output.view(np.uint32), golden.view(np.uint32))
    np.save(work / "output.npy", output)
    files = [
        model_object,
        reference,
        shim,
        runtime,
        original,
        instrumented,
        bundle / "stem_pool.json",
        base / "capture/golden.npy",
        *map(Path, sources),
    ]
    result = {
        "schema": "source_bound_native_stem_fixture_v1",
        "kernel_symbol": symbol,
        "shape": shape,
        "whole_output_words": int(output.size),
        "all_original_f32_bits_exact": True,
        "original_gate": {
            "atol": 0.0,
            "rtol": 0.0,
            "criterion": "all original binary32 output words exact",
        },
        "source_pins": {str(p): sha256_file(p) for p in files},
        "compiler_argv": command,
        "source_numeric_proof": manifest["scalar_transition_proof"],
        "pool_proof": manifest["pool_proof"],
        "a_sha256": sha256_file(work / "a.bin"),
        "b_sha256": sha256_file(work / "b.bin"),
        "expected_sha256": sha256_file(work / "expected.bin"),
        "scope": "Original consumed packed RGB halo/weights/output captured around unchanged scalar stem oracle during full original native replay; instrumentation performs binary file writes only.",
    }
    (work / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        "STEM_ORIGINAL_FIXTURE_PASS", output.size, a_count, b_count, c_count, flush=True
    )


if __name__ == "__main__":
    main()
