"""Read-only package/source/remote review of the generic reconstruction topic."""

import hashlib
import json
import os
import subprocess
import urllib.request
import zipfile
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def review():
    root = Path(__file__).resolve().parents[1]
    core = Path("/scratch/agustin/tmp/merlin-fused-radix-readout-main-20261007")
    directory = core / "out/artifacts/delivery/fused-radix-readout-926a1eace"
    qualified_path = directory / "qualified_delivery.json"
    publication_path = directory / "publication.json"
    qualified = json.loads(qualified_path.read_text())
    publication = json.loads(publication_path.read_text())
    for path, expected in qualified["pins"].items():
        if sha(path) != expected:
            raise ValueError("changed delivery dependency: " + path)
    head = subprocess.check_output(["git", "-C", str(core), "rev-parse", "HEAD"], text=True).strip()
    if head != qualified["core_head"] or head != publication["head"]:
        raise ValueError("reviewed source head differs")
    changed = subprocess.check_output(
        ["git", "-C", str(core), "diff", "--name-only", qualified["base_main"], head], text=True
    ).splitlines()
    if set(changed) != {
        "src/merlin/llvmlower/radix_integer_reconstruct.py",
        "src/merlin/llvmlower/source_attention_frontier.py",
        "src/merlin/llvmlower/AGENT.md",
        "merlin/tests/ir/test_radix_fused_integer_reconstruct.py",
        "merlin/tests/runtime/test_source_attention_frontier.py",
        "docs/design/agent_compiler_performance.md",
    }:
        raise ValueError("unexpected topic files")
    installed = Path(json.loads((directory / "installed_identity.json").read_text())["frontier"]).parents[2]
    wheel = directory / "wheel/merlin-0.0.1-py3-none-any.whl"
    code_count = data_count = bundled_count = 0
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if ".dist-info/" in name or name.endswith("/"):
                continue
            payload = archive.read(name)
            source = (core / "merlin" / name.removeprefix("merlin/_data/")) if name.startswith("merlin/_data/") else core / "src" / name
            if source.read_bytes() != payload or (installed / name).read_bytes() != payload:
                raise ValueError("source/wheel/installed bytes differ: " + name)
            if name.endswith(".py"):
                code_count += 1
            elif name.startswith("merlin/_data/"):
                data_count += 1
            else:
                bundled_count += 1
    if (code_count, data_count, bundled_count) != (958, 151, 50):
        raise ValueError("qualified package extent differs")
    log = (directory / "installed_pytest_resource_fix.log").read_text()
    if "56 passed, 890 deselected" not in log:
        raise ValueError("installed native/refusal tests failed")
    body_path = directory / "pr_body.md"
    if sha(body_path) != publication["body_sha256"]:
        raise ValueError("published body changed")
    credential = subprocess.run(
        ["git", "-C", str(core), "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n", text=True, capture_output=True,
        check=True, timeout=20, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    fields = dict(line.split("=", 1) for line in credential.stdout.splitlines() if "=" in line)
    request = urllib.request.Request(
        "https://api.github.com/repos/ucb-bar/merlin/pulls/49",
        headers={"Authorization": "Bearer " + fields["password"],
                 "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        pr = json.load(response)
    refs = subprocess.check_output(
        ["git", "ls-remote", "origin", "refs/heads/main", "refs/heads/" + publication["branch"]],
        cwd=core, text=True,
    )
    heads = {line.split()[1]: line.split()[0] for line in refs.splitlines()}
    if (pr["head"]["sha"] != head or heads["refs/heads/" + publication["branch"]] != head
            or pr["base"]["ref"] != "main" or pr["base"]["sha"] != qualified["base_main"]
            or heads["refs/heads/main"] != qualified["base_main"] or pr["state"] != "open"
            or pr["merged"] or hashlib.sha256(pr["body"].encode()).hexdigest() != publication["body_sha256"]):
        raise ValueError("actual remote publication differs")
    result = {
        "schema": "root_fused_radix_PR49_delivery_publication_review_v1",
        "status": "SOURCE_WHEEL_INSTALLED_AND_REMOTE_REVIEWED_MAIN_BASED_TOPIC",
        "PR": pr["html_url"], "head": head, "base": qualified["base_main"],
        "changed_files": changed, "source_wheel_installed_modules_byteexact": code_count,
        "packaged_data_resources_byteexact": data_count, "bundled_source_resources_byteexact": bundled_count,
        "source_tests": qualified["source_tests"], "installed_tests": qualified["outside_checkout_installed_tests"],
        "retained_failures": qualified["retained_failures"], "inherited_gate_limits": qualified["gate_limitations"],
        "defaults_enabled": False, "direct_main_push_force_or_merge": False,
        "scope": "Explicit generic completed signed-i32 plane storage schedule; one defined exact integer sum followed by original binary64 conversion. Allocation/cache/write-address costs require separate runtime qualification. Source and installed tests do not claim whole hardware performance.",
        "hardware_cycles": "UNKNOWN", "whole_default_promotion": False,
        "root_initial_review_refusal": "Wrong checkout resource root; refused before receipt. Corrected mapping to canonical merlin/ resource tree; package/source bytes unchanged.",
        "pins": {str(path): sha(path) for path in
                 (qualified_path, publication_path, wheel, body_path,
                  directory / "installed_identity.json", directory / "installed_pytest_resource_fix.log", Path(__file__))},
    }
    output = root / "docs/perf_records/root_merlin_fused_radix_PR49_delivery_review_20261007.json"
    if output.exists():
        raise ValueError("review output must be fresh")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "PR", "head")}))


if __name__ == "__main__":
    review()
