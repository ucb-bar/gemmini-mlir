"""Complete source PV pair with exact prepared coefficients, no hardware claim."""

import json
import subprocess
from pathlib import Path

from merlin.perf.layer_bench import build_program
from mlir_oot.host_outward_fp import emit_fixed_outward_f64_header
from mlir_oot.no_fsm_audit import audit_elf
from source_integer_pv_native import HERE, CLANG, CORE, sha, save
from source_integer_pv_target import OUT as PRIOR, GCC, ALLOCATOR


def main():
    native = HERE / "out/artifacts/probes/source-integer-pv-prepared-native-20261007"
    assert json.loads((native / "qualification.json").read_text())["status"] == "pass"
    previous = json.loads((PRIOR / "qualification.json").read_text())
    for path, expected in previous["pins"].items():
        assert sha(path) == expected, path
    for directed in (False, True):
        destination = PRIOR.parent / ("source-integer-pv-prepared-directed-target-20261007" if directed else "source-integer-pv-prepared-target-20261007")
        assert not destination.exists()
        destination.mkdir()
        capability = PRIOR / "target_capability.h"
        header = destination / "target_capability.h"
        header.write_bytes(capability.read_bytes())
        if directed:
            header.write_text(header.read_text() + emit_fixed_outward_f64_header(name="pv_bounds", host_isa="rv64gc", narrow_f32=True))
        source = destination / "executor.c"
        source.write_bytes((native / "context_00/executor.c").read_bytes())
        obj = destination / "executor.o"
        argv = list(previous["commands"][0])
        for index, argument in enumerate(argv):
            if argument == str(PRIOR / "target_capability.h"):
                argv[index] = str(header)
            elif argument == str(PRIOR / "executor.d"):
                argv[index] = str(destination / "executor.d")
            elif argument == str(PRIOR / "executor.c"):
                argv[index] = str(source)
            elif argument == str(PRIOR / "executor.o"):
                argv[index] = str(obj)
        complete = subprocess.run(argv, capture_output=True, text=True)
        (destination / "compile.stdout").write_text(complete.stdout)
        (destination / "compile.stderr").write_text(complete.stderr)
        assert complete.returncode == 0, complete.stderr
        shared_objects = [PRIOR / "data.o", PRIOR / "source.o", PRIOR / "main.o",
                          *[PRIOR / f"product_{k}/kernel.o" for k in (8, 16, 24)], ALLOCATOR]
        program = build_program([obj, *shared_objects], destination / "build", target="gemmini", max_loaded_bytes=None)
        audit = audit_elf(program.elf.read_bytes())
        save(destination / "nofsm.json", audit)
        assert audit["status"] == "pass"
        spike = [str(GCC.with_name("spike")), "--isa=rv64gc", "--extension=gemmini", "-m0x80000000:0x400000000", str(program.elf)]
        complete = subprocess.run(spike, capture_output=True, text=True, timeout=180)
        (destination / "spike.stdout").write_text(complete.stdout)
        (destination / "spike.stderr").write_text(complete.stderr)
        assert complete.returncode == 0 and "INTEGER_PV_COMPLETE" in complete.stdout, complete.stderr
        rows = []
        for line in complete.stdout.splitlines():
            if line.startswith("INTEGER_PV_ROW "):
                fields = dict(field.split("=", 1) for field in line.split()[1:])
                rows.append({key: int(value, 16 if key == "digest" else 10) for key, value in fields.items()})
        assert [(row["id"], row["sample"]) for row in rows] == [(0, 0), (1, 1), (1, 2), (0, 3)]
        pins = {str(path): sha(path) for path in [Path(__file__), native / "qualification.json",
                   PRIOR / "qualification.json", PRIOR / "target_capability.h", *shared_objects,
                   CORE / "src/merlin/llvmlower/prepared_scaled_integer_observer.py",
                   HERE / "mlir_oot/host_outward_fp.py", CLANG, GCC, GCC.with_name("spike"),
                   *[path for path in destination.rglob("*") if path.is_file()]]}
        for name in (destination / "executor.d").read_text().replace("\\\n", " ").split(":", 1)[1].split():
            path = Path(name)
            if path.is_file():
                pins[str(path.resolve())] = sha(path)
        record = {
            "schema": "source_integer_pv_prepared_complete_target_v1", "status": "pass",
            "explicit_alternative": "Prepared immutable source-column coefficients" + (" plus existing explicit directed binary64 target capability" if directed else " with unchanged portable adjacency"),
            "parent_immutable_receipt": {"path": str(PRIOR / "qualification.json"), "sha256": sha(PRIOR / "qualification.json")},
            "elf_path": str(program.elf), "elf_sha256": sha(program.elf), "zeroFSM": True,
            "unchanged_object_boundary": [str(path) for path in shared_objects], "original_i8_exact": 16384,
            "actual_integer_readout_words_exact": 98304, "cost_scope": previous["cost_scope"],
            "all5rounding_output_and_dirtyguards": True, "commands": [argv, spike], "rows": rows,
            "mean_retired_instructions": {"source": sum(row["instructions"] for row in rows if row["id"] == 0) / 2,
                                          "candidate": sum(row["instructions"] for row in rows if row["id"] == 1) / 2},
            "cycles": "UNKNOWN: strict functional Spike counts are not hardware cycles", "whole_model": "No normal route or whole gate",
            "token_usage_available": False, "pins": pins,
        }
        save(destination / "qualification.json", record)
        print(json.dumps({"directed": directed, "receipt": str(destination / "qualification.json"),
                          "mean_retired_instructions": record["mean_retired_instructions"], "elf_sha256": sha(program.elf)}), flush=True)


if __name__ == "__main__":
    main()
