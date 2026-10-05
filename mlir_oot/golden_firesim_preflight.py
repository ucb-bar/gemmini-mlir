"""Pin a stock FireSim job's ELF, workload, HWDB entry and bitstream.

Writes a one-entry immutable HWDB artifact for `firesim-queue
runworkload-full --hwdb-config-artifact`.  It also audits the final ELF before
the queue can consume FPGA time.  The queue still owns slot/setup readiness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse, unquote

import yaml

from .no_fsm_audit import audit_elf


def _sha(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def preflight(chipyard: Path, workload: str, elf: Path, hardware_key: str,
              expected_bitstream_sha256: str, output: Path) -> dict:
    deploy = chipyard.resolve() / "sims/firesim/deploy"
    workload_path = deploy / "workloads" / (workload + ".json")
    config_path = deploy / "config_hwdb.yaml"
    definition = json.loads(workload_path.read_text())
    if definition.get("benchmark_name") != workload or not definition.get("common_bootbinary"):
        raise ValueError("workload name or common_bootbinary disagrees with its JSON")
    hwdb = yaml.safe_load(config_path.read_text())
    if hardware_key not in hwdb:
        raise ValueError(f"hardware key {hardware_key!r} is absent from HWDB")
    entry = hwdb[hardware_key]
    uri = urlparse(entry["bitstream_tar"])
    if uri.scheme != "file" or uri.netloc not in ("", "localhost"):
        raise ValueError("the stock bitstream must be a local file URI")
    bitstream = Path(unquote(uri.path)).resolve(strict=True)
    actual_bitstream_sha = _sha(bitstream)
    if actual_bitstream_sha != expected_bitstream_sha256.lower():
        raise ValueError("bitstream SHA disagrees with the pinned stock reference")
    elf = elf.resolve(strict=True)
    audit = audit_elf(elf.read_bytes())
    if audit["status"] != "pass":
        raise ValueError("final linked ELF fails the zero-FSM audit")
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / "stock_one_entry_hwdb.yaml"
    artifact.write_text(yaml.safe_dump({hardware_key: entry}, sort_keys=False))
    result = {
        "schema": "golden_firesim_preflight_v1",
        "chipyard": str(chipyard.resolve()),
        "workload": workload,
        "bootbinary": definition["common_bootbinary"],
        "workload_json_sha256": _sha(workload_path),
        "hardware_key": hardware_key,
        "bitstream_path": str(bitstream),
        "bitstream_sha256": actual_bitstream_sha,
        "hwdb_artifact": str(artifact.resolve()),
        "hwdb_artifact_sha256": _sha(artifact),
        "elf": str(elf),
        "elf_sha256": _sha(elf),
        "elf_nofsm_audit": audit,
    }
    (output / "firesim_preflight.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chipyard", type=Path, required=True)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--elf", type=Path, required=True)
    ap.add_argument("--hardware-key", required=True)
    ap.add_argument("--expected-bitstream-sha256", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = preflight(args.chipyard, args.workload, args.elf, args.hardware_key,
                       args.expected_bitstream_sha256, args.output)
    print(json.dumps({k: result[k] for k in ("bitstream_sha256", "elf_sha256",
                                             "hwdb_artifact", "hwdb_artifact_sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
