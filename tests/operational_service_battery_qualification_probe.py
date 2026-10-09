"""Close source-bound gather or Gemmini operational service measurements.

This export does not fit costs or infer physical DDR traffic. CPU21's released
parser and source remain unchanged; the same strict report format is reused.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import inspect
import json
import shlex
import subprocess
import tempfile
from pathlib import Path

from mlir_oot.executed_features import _symbol_ranges, census
from mlir_oot.no_fsm_audit import audit_elf
from mlir_oot.service_battery import (
    feature_rows,
    import_stock_report,
    parse_service_report,
)


def pin(path):
    path = Path(path).resolve()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def compiler_dependencies(commands, output):
    """Pin actual compiler/include/library inputs from captured build argv."""
    paths = set()
    for index, row in enumerate(commands):
        argv = row["argv"]
        if row["returncode"] != 0:
            raise ValueError("the selected final build command failed")
        paths.add(Path(argv[0]).resolve())
        if "-c" in argv:
            end = argv.index("-o")
            source = Path(argv[argv.index("-c") + 1])
            command = [value for value in argv[:end] if value != "-c"] + ["-M"]
            run = subprocess.run(command, check=True, capture_output=True, text=True)
            (output / f"dependencies_{index}.d").write_text(run.stdout)
            paths.add(source.resolve())
            dependencies = run.stdout.replace("\\\n", " ").split(":", 1)[1]
            for value in shlex.split(dependencies):
                paths.add(Path(value).resolve())
        else:
            for value in argv:
                if Path(value).is_file():
                    paths.add(Path(value).resolve())
        for library in ("libgcc.a", "libm.a", "libc.a"):
            result = subprocess.run(
                [argv[0], "-print-file-name=" + library],
                check=True,
                capture_output=True,
                text=True,
            )
            path = Path(result.stdout.strip())
            if not path.is_file():
                raise ValueError("compiler library input is unresolved")
            paths.add(path.resolve())
        for program in ("cc1", "as", "ld"):
            result = subprocess.run(
                [argv[0], "-print-prog-name=" + program],
                check=True,
                capture_output=True,
                text=True,
            )
            path = Path(result.stdout.strip())
            if not path.is_file():
                raise ValueError("compiler stage input is unresolved")
            paths.add(path.resolve())
    result = [pin(path) for path in sorted(paths)]
    (output / "dependency_pins.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def export(args):
    work, output = args.workdir.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    primitive = args.kind == "primitive"
    manifest_path = (
        work / ("generated_v5" if primitive else "generated") / "manifest.json"
    )
    source = manifest_path.with_name("battery.c")
    built_path = work / ("built_v5.json" if primitive else "built.json")
    commands_path = work / (
        "build_commands_v5.json" if primitive else "build_commands.json"
    )
    manifest, built = (
        json.loads(manifest_path.read_text()),
        json.loads(built_path.read_text()),
    )
    elf = Path(built["elf"])
    if pin(elf)["sha256"] != built["elf_sha256"]:
        raise ValueError("selected ELF has changed")
    expected_schema = (
        "gemmini_primitive_service_battery_v1"
        if primitive
        else "rv64gc_gather_service_battery_v1"
    )
    if manifest["schema"] != expected_schema:
        raise ValueError("another source protocol was selected")
    held = [case["id"] for case in manifest["cases"] if case["partition"] == "heldout"]
    declared_held = ([1, 4, 7] if primitive else [1])
    if args.generator is not None:
        declared_held = manifest.get("heldout_ids")
        if not isinstance(declared_held, list) or not declared_held or any(type(x) is not int for x in declared_held):
            raise ValueError("explicit source partition IDs required for another generator")
    if held != declared_held:
        raise ValueError("predeclared middle partitions changed")
    commands = json.loads(commands_path.read_text())
    compile_rows = [row for row in commands if "-c" in row["argv"]]
    if not any(str(source) in row["argv"] for row in compile_rows):
        raise ValueError("actual compile argv does not use the selected source")
    for row in commands:
        argv = row["argv"]
        if (
            argv.index("-fno-fast-math") <= argv.index("-ffast-math")
            or "-ffp-contract=off" not in argv
        ):
            raise ValueError("actual compile/link floating-point flags changed")
    audit = audit_elf(elf.read_bytes())
    if audit["status"] != "pass":
        raise ValueError("final ELF failed the zero-FSM audit")
    if not primitive and audit["custom_funct_counts"]:
        raise ValueError("CPU gather unexpectedly contains accelerator commands")
    generator_path = (
        Path(__file__)
        .resolve()
        .with_name(
            "primitive_service_battery_probe.py"
            if primitive
            else "gather_service_battery_probe.py"
        )
    )
    if args.generator is not None:
        generator_path = args.generator.resolve()
    spec = importlib.util.spec_from_file_location(
        "qualified_service_generator", generator_path
    )
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    with tempfile.TemporaryDirectory(prefix="qualified_service_") as temporary:
        regenerated = Path(temporary) / "source"
        generator.generate(regenerated)
        for name in ("battery.c", "manifest.json"):
            if (
                regenerated.joinpath(name).read_bytes()
                != source.parent.joinpath(name).read_bytes()
            ):
                raise ValueError(
                    "independent source/expected-value regeneration changed"
                )
    symbols = ["empty"] + (
        ["resident_gemm", "requested_load", "i32_readback"] if primitive else ["gather"]
    )
    ranges = _symbol_ranges(elf, symbols, "readelf")
    sizes = {row["symbol"]: row["end"] - row["start"] for row in ranges}
    stdout = work / ("spike_v5.stdout" if primitive else "spike.stdout")
    histogram = work / ("spike_v5.stderr" if primitive else "spike.stderr")
    report = parse_service_report(stdout.read_text(), manifest)
    spike = Path(built["spike"]["command"][0])
    before = {"elf": pin(elf), "engine": pin(spike)}
    run = subprocess.run(
        built["spike"]["command"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if (
        run.returncode
        or run.stdout != stdout.read_text()
        or run.stderr != histogram.read_text()
    ):
        raise ValueError(
            "strict execution and histogram did not reproduce byte exactly"
        )
    if before != {"elf": pin(elf), "engine": pin(spike)}:
        raise ValueError("source ELF or engine changed during strict execution")
    strict_path = output / "strict_spike.json"
    strict_path.write_text(
        json.dumps(
            {
                "command": built["spike"]["command"],
                "returncode": run.returncode,
                "before_and_after": before,
                "stdout_stderr_reproduced_byte_exact": True,
            },
            indent=2,
        )
        + "\n"
    )
    engine = {
        "kind": "spike_functional",
        "path": str(spike),
        "sha256": pin(spike)["sha256"],
        "isa": "rv64gc",
        "extension": "gemmini",
        "mcycle_interpretation": "retired instruction proxy, not hardware cycles",
    }
    parser_path = Path(__file__).resolve().parents[1] / "mlir_oot/service_battery.py"
    refs = {
        "stdout": pin(stdout),
        "elf": pin(elf),
        "manifest": pin(manifest_path),
        "parser": pin(parser_path),
        "engine": pin(spike),
        "histogram": pin(histogram),
        "qualification": pin(strict_path),
    }

    def observation(decoded, current_engine, current_refs):
        provenance = {
            "elf_sha256": pin(elf)["sha256"],
            "manifest_sha256": pin(manifest_path)["sha256"],
            "stdout_sha256": current_refs["stdout"]["sha256"],
            "scope": "same fenced common kernel-call windows",
        }
        rows = feature_rows(
            decoded,
            manifest,
            engine=current_engine,
            provenance=provenance,
            function_sizes=sizes,
        )
        if primitive:
            for row in rows:
                case = None if row["id"] == -1 else manifest["cases"][row["id"]]
                added = {
                    "primitive_commands": case["roi_primitive_commands"]
                    if case
                    else {},
                    "array_work_padded_rows": case["array_work_padded_rows"]
                    if case
                    else 0,
                    "requested_dma_load_bytes": case["roi_requested_dma_load_bytes"]
                    if case
                    else 0,
                    "requested_dma_store_bytes": case["roi_requested_dma_store_bytes"]
                    if case
                    else 0,
                }
                row["features"].update(added)
                row["unknown_features"]["CPU_accelerator_overlap"] = (
                    "UNKNOWN; complete-finish composite window includes CPU issue and accelerator stalls"
                )
                for key in added:
                    row["feature_status"]["/features/" + key] = {
                        "status": "source_and_executed_function_bound",
                        "provenance": provenance,
                    }
        return {"report": decoded, "features": rows}

    observations = {"spike": observation(report, engine, refs)}
    measurements = {"spike": refs}
    counts = {}
    for symbol in symbols:
        scope = "all_invocations_of_" + symbol
        counts[symbol] = census(
            elf,
            histogram,
            symbols=[symbol],
            scope_id=scope,
            execution_binding={
                "elf_sha256": pin(elf)["sha256"],
                "histogram_sha256": pin(histogram)["sha256"],
                "scope_id": scope,
                "engine": engine,
            },
        )
        if counts[symbol]["status"] != "pass":
            raise ValueError("function execution census did not close")
        if primitive:
            expected = {}
            for case in manifest["cases"]:
                if case["kernel"] == symbol:
                    for name, value in case["roi_primitive_commands"].items():
                        expected[name] = (
                            expected.get(name, 0) + value * manifest["repetitions"]
                        )
            actual = {
                key: value
                for key, value in counts[symbol]["features"][
                    "primitive_commands"
                ].items()
                if value
            }
            if actual != expected:
                raise ValueError(
                    (symbol, "executed command totals disagree", actual, expected)
                )
    if args.gsim_directory:
        directory = args.gsim_directory.resolve()
        gsim_path = directory / "gsim.json"
        receipt = json.loads(gsim_path.read_text())
        console = directory / "gsim.stdout"
        if (
            receipt["returncode"]
            or not receipt["finish"]["done"]
            or receipt["stdout_sha256"] != pin(console)["sha256"]
        ):
            raise ValueError("GSIM is partial or its console changed")
        gsim_report = parse_service_report(console.read_text(), manifest)
        if [row["instructions"] for row in gsim_report["rows"]] != [
            row["instructions"] for row in report["rows"]
        ]:
            raise ValueError("GSIM executes different window instruction counts")
        engine_path = Path(receipt["engine"]["path"])
        if pin(engine_path)["sha256"] != receipt["engine"]["binary_sha256"]:
            raise ValueError("GSIM engine pin changed")
        gsim_engine = {
            "kind": "gsim_rtl",
            "provenance": receipt["engine"],
            "load_path": receipt["load_path"],
            "memory_regime": "elaborated GSIM harness; distinct from stock FireSim",
        }
        gsim_refs = {
            "stdout": pin(console),
            "elf": pin(elf),
            "manifest": pin(manifest_path),
            "parser": pin(parser_path),
            "engine": pin(engine_path),
            "qualification": pin(gsim_path),
        }
        observations["gsim"] = observation(gsim_report, gsim_engine, gsim_refs)
        measurements["gsim"] = gsim_refs
    if args.stock_record:
        stock_report, record = import_stock_report(
            args.stock_record, manifest, elf=elf, parser_path=parser_path
        )
        if [row["instructions"] for row in stock_report["rows"]] != [
            row["instructions"] for row in report["rows"]
        ]:
            raise ValueError("stock executes different window instruction counts")
        stock_engine = {
            "kind": "firesim_stock",
            "job_id": record["job_id"],
            "hw_config": record["hw_config"],
            "hwdb_sha256": record["hwdb_sha256"],
            "bitstream_sha256": record["bitstream_sha256"],
            "memory_regime": "actual stock FireSim; no GSIM slopes transplanted",
        }
        stock_refs = {
            "stdout": pin(record["uart"]),
            "elf": pin(elf),
            "manifest": pin(manifest_path),
            "parser": pin(parser_path),
            "qualification": pin(args.stock_record),
        }
        for name in (
            "expected_elf_receipt",
            "staged_elf_receipt",
            "staged_bitstream_receipt",
            "runworkload_receipt",
        ):
            stock_refs[name] = pin(record[name])
        observations["firesim"] = observation(stock_report, stock_engine, stock_refs)
        measurements["firesim"] = stock_refs
    dependency_pins = compiler_dependencies(commands, output)
    paths = {Path(value["path"]) for value in dependency_pins}
    paths.update(
        {
            source,
            manifest_path,
            built_path,
            commands_path,
            elf,
            stdout,
            histogram,
            parser_path,
            Path(__file__).resolve(),
            Path(__file__).resolve().parents[1]
            / "tests"
            / (
                "primitive_service_battery_probe.py"
                if primitive
                else "gather_service_battery_probe.py"
            ),
        }
    )
    paths.add(generator_path)
    if hasattr(generator, "PARENT"):
        dependency = generator.PARENT.resolve()
        if pin(dependency)["sha256"] != manifest.get("source_generator_dependency_sha256"):
            raise ValueError("generator dependency changed")
        paths.add(dependency)
    from merlin.perf.layer_bench import build_program, run_on_gsim
    from merlin.targetgen.elf_lanes import executable_sections

    paths.update(
        Path(inspect.getfile(function)).resolve()
        for function in (build_program, run_on_gsim, executable_sections)
    )
    paths.update(
        Path(inspect.getfile(function)).resolve()
        for function in (audit_elf, census, _symbol_ranges, feature_rows)
    )
    oot = Path(__file__).resolve().parents[1]
    paths.update(
        oot / name
        for name in (
            "mlir_oot/cpu_opcode_census.py",
            "mlir_oot/tables/rtl_facts.py",
            "support/gemmini_gsim/backend.py",
            "support/gemmini_gsim/provider.yaml",
        )
    )
    if args.gsim_directory:
        paths.add(Path(receipt["engine"]["receipt"]["receipt_path"]))
        paths.add(work / "run_gsim.py")
    for values in measurements.values():
        paths.update(Path(value["path"]) for value in values.values())
    paths.update(path for path in output.iterdir() if path.is_file())
    paths.update(path for path in elf.parent.iterdir() if path.is_file())
    envelope = {
        "schema": "operational_service_battery_qualified_v1",
        "captured_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "built": built,
        "manifest": manifest,
        "observations": observations,
        "measurement_refs": measurements,
        "executed_function_census": counts,
        "function_ranges": ranges,
        "function_sizes": sizes,
        "nofsm_audit": audit,
        "flat_artifact_pins": [pin(path) for path in sorted(paths)],
        "scope_limitations": [
            "Middle sizes and all their repeated observations held out together; no fitting here",
            "Empty-window overhead remains separate, without automatic subtraction",
            "Function census aggregates varied-size invocations; no per-case alias division",
            "Requested payloads/64B regions do not imply physical DDR traffic, cache geometry or misses",
            "Operational command streams include CPU issue and completion stalls, not pure device rates",
        ],
        "token_usage_available": False,
    }
    target = output / "qualified.json"
    target.write_text(json.dumps(envelope, indent=2) + "\n")
    print(
        json.dumps(
            {
                "qualified": pin(target),
                "engines": list(observations),
                "counter_rows": len(report["rows"]),
                "pins": len(envelope["flat_artifact_pins"]),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("gather", "primitive"), required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gsim-directory", type=Path)
    parser.add_argument("--stock-record", type=Path)
    parser.add_argument("--generator", type=Path)
    export(parser.parse_args())
