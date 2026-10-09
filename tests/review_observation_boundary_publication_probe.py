"""Close the target-independent typed boundary topic and actual publication."""

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
    core = Path("/scratch/agustin/tmp/merlin-typed-observation-boundary-pr-20261007")
    directory = core / "out/artifacts/observation-boundary-publication"
    q = json.loads((directory / "validation.json").read_text())
    published = json.loads((directory / "pull_request.json").read_text())
    output = root / "docs/perf_records/root_merlin_observation_boundary_PR51_review_20261007.json"
    require(not output.exists(), "review output must be fresh")
    for path, expected in q["pins"].items():
        require(sha(path) == expected, "changed typed boundary delivery: " + path)

    def git(*args):
        return subprocess.check_output(["git", "-C", str(core), *args], text=True).strip()

    require(git("rev-parse", "HEAD") == q["head"] and git("rev-parse", "HEAD^") == q["base"] and
            q["head"] == published["head"] and q["base"] == published["base"], "one-topic main ancestry differs")
    changed = git("diff", "--name-only", q["base"], q["head"]).splitlines()
    require(set(changed) == {"src/merlin/llvmlower/observation_boundary.py",
                             "merlin/tests/ir/test_observation_boundary.py", "src/merlin/llvmlower/AGENT.md"},
            "unexpected typed boundary topic files")
    installed_root = Path(q["installed_module"]).parents[2]
    wheel = directory / "merlin-0.0.1-py3-none-any.whl"
    names = ["merlin/llvmlower/" + name + ".py" for name in
             ("observation_boundary", "quantized_consumer_frontier", "ordered_fma_groups")]
    with zipfile.ZipFile(wheel) as archive:
        for name in names:
            require((core / "src" / name).read_bytes() == archive.read(name) == (installed_root / name).read_bytes(),
                    "source/wheel/installed typed boundary dependency differs")
    require(q["source_tests"] == q["installed_tests"] == 52 and
            "52 passed" in (directory / "source_tests.log").read_text() and
            "52 passed" in (directory / "installed_tests.log").read_text() and
            "All structure checks passed." in (directory / "structure.log").read_text(),
            "source/package/structure gates failed")
    require(q["numerical_permission"] == "NONE", "source analysis granted an unproved numeric relaxation")
    credential = subprocess.run(["git", "-C", str(core), "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n", text=True, capture_output=True, check=True,
        timeout=20, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    fields = dict(line.split("=", 1) for line in credential.stdout.splitlines() if "=" in line)
    request = urllib.request.Request("https://api.github.com/repos/ucb-bar/merlin/pulls/51",
        headers={"Authorization": "Bearer " + fields["password"], "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        pr = json.load(response)
    remote = dict(reversed(line.split()) for line in git("ls-remote", "origin", "refs/heads/main",
                  "refs/heads/" + published["branch"]).splitlines())
    require(pr["state"] == "open" and not pr["merged"] and pr["head"]["sha"] == q["head"] and
            pr["base"]["ref"] == "main" and pr["base"]["sha"] == q["base"] and
            remote["refs/heads/main"] == q["base"] and remote["refs/heads/" + published["branch"]] == q["head"] and
            hashlib.sha256(pr["body"].encode()).hexdigest() == published["body_sha256"] == sha(directory / "pr_body.md"),
            "actual main/head/body publication differs")
    result = {
        "schema": "root_merlin_typed_observation_boundary_PR51_review_v1",
        "status": "SOURCE_DEPENDENCIES_WHEEL_INSTALLED_AND_ACTUAL_REMOTE_REVIEWED",
        "PR": pr["html_url"], "head": q["head"], "base_main": q["base"],
        "delivery_pins_reclosed": len(q["pins"]), "source_wheel_installed_dependencies_exact": names,
        "source_tests": 52, "installed_tests": 52, "structure_gate": "PASS", "changed_files": changed,
        "scope": "Pure typed source DAG, complete external-use observations and immutable source/context witness validation; no cloning/rewiring or numeric relaxation.",
        "numerical_permission": "NONE", "target_or_workload_selector": False,
        "whole_model_performance": "UNKNOWN", "main_merge_or_default_enablement": False,
        "pins": {str(p.resolve()): sha(p) for p in (directory / "validation.json", directory / "pull_request.json",
                                                  directory / "pr_body.md", wheel, Path(__file__))},
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "PR", "head", "source_tests", "installed_tests")}))


if __name__ == "__main__":
    review()
