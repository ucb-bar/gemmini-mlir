"""Reproduce a sealed source-bound experiment; this is not a production selector."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import inspect
import json
import shutil
import subprocess
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reclose(receipt: dict) -> None:
    for name, expected in receipt["pins"].items():
        path = Path(name)
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Qualification input changed: {name}")


def require_identity(actual: Path, expected: Path) -> None:
    if actual.read_bytes() != expected.read_bytes():
        raise ValueError(f"Reproduction differs: {actual} versus {expected}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        required=True,
        type=Path,
        help="Frozen fused_reconstruction directory (candidate is control here)",
    )
    parser.add_argument(
        "--qualification",
        required=True,
        type=Path,
        help="Sealed fused_encoder_radix_current2069_qualification.json",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="New, nonexistent output directory"
    )
    parser.add_argument(
        "--emit-only",
        action="store_true",
        help="Close dependencies and generated C without executing compilers",
    )
    args = parser.parse_args()
    baseline = args.baseline.resolve()
    output = args.output.resolve()
    qualification = args.qualification.resolve()
    if output.exists():
        raise ValueError(
            "Output must not already exist; historical artifacts are immutable"
        )
    receipt = json.loads(qualification.read_text())
    if receipt["schema"] != "fused_encoder_radix_composition_qualification_v1":
        raise ValueError("Wrong source-bound qualification schema")
    reclose(receipt)
    from merlin.common.paths import data_path
    from merlin.llvmlower.fused_encoded_witness import prepare_fused_encoded_witness

    from mlir_oot.no_fsm_audit import audit_elf

    helper = Path(inspect.getsourcefile(prepare_fused_encoded_witness)).resolve()
    helper_rows = [
        row
        for row in receipt["source_contract"]["source_rows"]
        if Path(row["original"]).name == helper.name
    ]
    if len(helper_rows) != 1 or digest(helper) != helper_rows[0]["sha256"]:
        raise ValueError("Loaded generic transformation differs from sealed source")
    expected = Path(receipt["candidate_elf"]).parent.parent
    # The baseline and qualification must identify the exact same controlled ELF.
    require_identity(baseline / "candidate/model.elf", Path(receipt["control_elf"]))
    for role in ("target_numeric", "native_numeric"):
        source = baseline / "candidate" / role
        for p in source.iterdir():
            if p.suffix in (".h", ".c"):
                require_identity(p, expected / "control" / role / p.name)
        require_identity(
            source / "bf16_radix_pack.h",
            data_path("runtime", "c") / "bf16_radix_pack.h",
        )
    output.mkdir(parents=True)
    commands = []
    for role in ("target_numeric", "native_numeric"):
        source = baseline / "candidate" / role
        for arm in ("control", "candidate"):
            destination = output / arm / role
            destination.mkdir(parents=True)
            for p in source.iterdir():
                if p.suffix in (".h", ".c"):
                    shutil.copyfile(p, destination / p.name)
            if arm == "candidate":
                provider = destination / "provider.c"
                provider.write_text(prepare_fused_encoded_witness(provider.read_text()))
            require_identity(
                destination / "provider.c", expected / arm / role / "provider.c"
            )
            if args.emit_only:
                continue
            sealed_source = expected / "control" / role
            compile_receipt = json.loads((sealed_source / "compile.json").read_text())
            reclose(compile_receipt)
            for original in compile_receipt["commands"]:
                command = [
                    arg.replace(str(sealed_source), str(destination))
                    for arg in original
                ]
                subprocess.run(command, check=True)
                commands.append(command)
            artifact = "provider.o" if role == "target_numeric" else "provider.so"
            require_identity(destination / artifact, expected / arm / role / artifact)
    if not args.emit_only:
        original_link = json.loads((expected / "control/build.json").read_text())[
            "link"
        ]
        for arm in ("control", "candidate"):
            destination = output / arm
            replacements = {
                str(expected / "control/model.elf"): str(destination / "model.elf"),
                str(expected / "control/target_numeric/provider.o"): str(
                    destination / "target_numeric/provider.o"
                ),
            }
            command = [replacements.get(arg, arg) for arg in original_link]
            subprocess.run(command, check=True)
            commands.append(command)
            require_identity(destination / "model.elf", expected / arm / "model.elf")
            if audit_elf((destination / "model.elf").read_bytes())["status"] != "pass":
                raise ValueError("Final instruction audit failed")
        library = ctypes.CDLL(str(output / "candidate/native_numeric/provider.so"))
        library.group_provider_workspace_bytes.restype = ctypes.c_size_t
        if (
            library.group_provider_workspace_bytes()
            != receipt["native"]["workspace_bytes"]
        ):
            raise ValueError("Actual workspace ABI differs")
    result = {
        "scope": "Frozen complete-group experiment reproduction; not a new normal RMS4 seal",
        "qualification": str(qualification),
        "qualification_sha256": digest(qualification),
        "generic_helper": str(helper),
        "generic_helper_sha256": digest(helper),
        "generated_source_byte_identity": True,
        "compiled_identity": not args.emit_only,
        "commands": commands,
        "pins": {str(p): digest(p) for p in output.rglob("*") if p.is_file()},
    }
    (output / "reproduction.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "generated_source_byte_identity": True,
                "compiled_identity": not args.emit_only,
            }
        )
    )


if __name__ == "__main__":
    main()
