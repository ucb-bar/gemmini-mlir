"""Read-only review of the current main-based SDPA mask topic publication."""

import hashlib
import json
import os
import subprocess
import urllib.request
import zipfile
from pathlib import Path

from review_closed_i8_stock_capsule_probe import require, sha


def review():
    root = Path(__file__).resolve().parents[1]
    core = Path("/scratch/agustin/tmp/model2mlir-sdpa-mask-dtype-20261007")
    directory = core / "out/artifacts/mask_dtype_v1"
    packet = Path("/scratch/agustin/tmp/gemmini-residual-domain-20261007/docs/perf_records/model2mlir_sdpa_mask_pr5_dtype_followup.json")
    q = json.loads(packet.read_text())
    validation = json.loads((directory / "validation.json").read_text())
    publication = json.loads((directory / "pull_request.json").read_text())
    output = root / "docs/perf_records/root_model2mlir_sdpa_PR5_dtype_publication_review_20261007.json"
    require(not output.exists(), "review output must be fresh")
    require(q["status"] == validation["status"] == "PASS" and len(q["flat_file_pins"]) == 319,
            "qualified delivery extent differs")
    for pin in q["flat_file_pins"]:
        require(sha(pin["path"]) == pin["sha256"], "changed SDPA publication evidence: " + pin["path"])

    def git(*args):
        return subprocess.check_output(["git", "-C", str(core), *args], text=True).strip()

    head, base = q["head"], q["main_unchanged"]
    require(git("rev-parse", "HEAD") == head and git("merge-base", base, head) == base and
            git("rev-parse", "HEAD^") == q["previous_reviewed_head"], "clean topic ancestry differs")
    changed = git("diff", "--name-only", base, head).splitlines()
    require(set(changed) == {"m2m/ir/decompositions.py", "m2m/ir/torchmlir_decomps.py",
                             "tests/test_sdpa_mask_semantics.py", "tests/test_sdpa_mask_dtypes.py"},
            "unexpected topic files")
    identity = json.loads((directory / "module_identity.json").read_text())
    wheel = directory / "wheel/m2m-0.0.1-py3-none-any.whl"
    require(identity["count"] == 59 and identity["all_three_byte_identical"], "package identity extent differs")
    with zipfile.ZipFile(wheel) as archive:
        for row in identity["modules"]:
            payload = archive.read(row["module"])
            require(Path(row["source"]).read_bytes() == Path(row["installed"]).read_bytes() == payload,
                    "source/wheel/installed module mismatch")
    require(validation["source_tests"] == validation["installed_tests"] == 69 and
            "69 passed" in (directory / "source_closed.log").read_text() and
            "69 passed" in (directory / "installed_closed.log").read_text(), "native/package test outcome differs")
    credential = subprocess.run(["git", "-C", str(core), "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n", capture_output=True, text=True, timeout=20,
        check=True, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    fields = dict(line.split("=", 1) for line in credential.stdout.splitlines() if "=" in line)
    request = urllib.request.Request("https://api.github.com/repos/ucb-bar/model2MLIR/pulls/5",
        headers={"Authorization": "Bearer " + fields["password"], "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        pr = json.load(response)
    remote = dict(reversed(line.split()) for line in git("ls-remote", "origin", "refs/heads/main",
                  "refs/heads/" + publication["head"]["ref"]).splitlines())
    require(pr["state"] == "open" and not pr["merged"] and pr["head"]["sha"] == head and
            pr["base"]["ref"] == "main" and pr["base"]["sha"] == base and
            remote["refs/heads/main"] == base and remote["refs/heads/" + pr["head"]["ref"]] == head and
            hashlib.sha256(pr["body"].encode()).hexdigest() == publication["exact_body_sha256"] ==
            sha(directory / "pr_body.md"), "actual remote head/main/body differs")
    result = {
        "schema": "root_model2mlir_sdpa_mask_dtype_PR5_review_v1",
        "status": "SOURCE_WHEEL_INSTALLED_AND_ACTUAL_REMOTE_REVIEWED_MAIN_BASED_TOPIC",
        "PR": pr["html_url"], "base_main": base, "head": head, "previous_head": q["previous_reviewed_head"],
        "flat_pins_reclosed": 319, "source_wheel_installed_modules_exact": 59,
        "source_tests": 69, "outside_checkout_installed_tests": 69, "changed_files": changed,
        "source_contract": "SDPA all-negative-infinity rows zero, genuine NaN/+inf retained, exact boolean/f32/query-dtype mask admission and opmath casts; ordinary softmax unchanged.",
        "actual_model_failure_attribution": "UNKNOWN; no empty Tiny causal rows or causal attribution to current Smol/Tiny failures established",
        "previous_delivery_tree_preserved": True, "main_push_force_or_merge": False,
        "pins": {str(p.resolve()): sha(p) for p in (packet, directory / "validation.json",
                 directory / "pull_request.json", directory / "module_identity.json", wheel,
                 directory / "pr_body.md", Path(__file__))},
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "PR", "head", "source_tests", "outside_checkout_installed_tests")}))


if __name__ == "__main__":
    review()
