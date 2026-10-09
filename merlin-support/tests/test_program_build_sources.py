"""Exact relocated OOT preparation against the active install; no tools execute."""

import copy
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    from merlin.common import paths
    from merlin.llvmlower import toolchain
    from merlin.runtime.backends import base

    original = Path(__file__).resolve().parents[1]
    provider = tmp_path / "relocated-provider"
    for directory in ("build_support", "resources"):
        shutil.copytree(original / directory, provider / directory, ignore=shutil.ignore_patterns("__pycache__"))
    (provider / "build_support/unimported_sibling.py").write_text("VALUE = 1\n")
    # Synthetic transitive optional owner: importing the inert namespace must be
    # recorded even when its distribution is separate from the core source root.
    initializer = provider / "build_support/__init__.py"
    initializer.write_text(initializer.read_text() + "\nimport merlin_experiments\n")
    backend = provider / "backend"
    backend.mkdir()
    worker = backend / "gemmini_program_build.py"
    shutil.copyfile(original / "backend/gemmini_program_build.py", worker)
    module_name = "build_source_fixture_" + tmp_path.name.replace("-", "_")
    spec = importlib.util.spec_from_file_location(module_name, worker)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    operator = tmp_path / "operator"
    operator.mkdir()
    # The ACTIVE INSTALL's bundled data, resolved before the operator's repository is substituted: an
    # editable checkout serves it from its tree (it has no wheel `_data` bundle), so resolving it after
    # the substitution would look for a wheel that is not installed.
    installed_data = paths.data_path("schemas").parent
    monkeypatch.setattr(paths, "data_path", lambda *parts: installed_data.joinpath(*parts))
    monkeypatch.setattr(paths, "repo_root", lambda: operator)
    monkeypatch.delenv("MERLIN_QUANT_FORMATS", raising=False)
    tool = tmp_path / "unexecuted-tool"
    tool.write_text("synthetic compiler identity only")
    resources = provider / "resources/gemmini-rocc-tests"
    recipe = SimpleNamespace(
        compiler=tool,
        include_roots=(resources / "include",),
        support_sources=(resources / "riscv-tests/benchmarks/common/syscalls.c",),
        link_script=resources / "riscv-tests/benchmarks/common/test.ld",
        load_address=0,
        cflags=(),
        ldflags=(),
        require_kernel_stack_frame=lambda: SimpleNamespace(record=lambda: {}),
    )
    monkeypatch.setattr(base, "harness_build_recipe", lambda target: recipe)
    monkeypatch.setattr(toolchain, "clang", lambda: str(tool))
    monkeypatch.setattr(toolchain, "mlir_translate", lambda: str(tool))
    service = module._prepare_build_service(None)
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"schema": "short_complete_program_build_v2", "build_service": service, "files": {}}))
    pins = {row["source"]: row["sha256"] for row in service["dependencies"]}
    build = module.ShortProgramBuild((), str(request), json.dumps(pins))
    return module, provider, service, build


def test_installed_shared_and_complete_provider_sources_are_bound(prepared):
    from merlin.common.paths import module_source_path

    module, provider, service, build = prepared
    sources = {Path(row["source"]) for row in service["dependencies"] if row["kind"] == "python"}
    assert module_source_path("merlin.targetgen.contract.compile").resolve() in sources
    assert module_source_path("merlin.common.source_membership").resolve() in sources
    assert module_source_path("merlin_experiments").resolve() in sources
    assert set((provider / "build_support").rglob("*.py")) <= sources
    assert service["pure_package_init"] == str(provider / "build_support/__init__.py")
    assert any(row["kind"] == "header" for row in service["dependencies"])
    assert any(row["kind"] == "support" for row in service["dependencies"])
    assert len([row for row in service["dependencies"] if row["kind"] == "format_data"]) == 2
    build.revalidate()
    for row in service["dependencies"]:
        if row["kind"] == "format_data":
            assert (
                Path(row["destination"])
                == Path(service["namespace_root"]) / "merlin/schemas" / Path(row["source"]).name
            )
    assert module._worker_import_roots(service)


@pytest.mark.parametrize(
    "relative",
    [
        "build_support/measurement.py",
        "build_support/unimported_sibling.py",
        "resources/gemmini-rocc-tests/include/gemmini.h",
        "resources/gemmini-rocc-tests/riscv-tests/benchmarks/common/syscalls.c",
    ],
)
def test_existing_build_identity_refuses_provider_source_or_resource_mutation(prepared, relative):
    module, provider, service, build = prepared
    leaf = provider / relative
    leaf.write_bytes(leaf.read_bytes() + b"\nmutation\n")
    with pytest.raises(ValueError, match="short build input/worker changed"):
        build.revalidate()


def test_added_unimported_sibling_refuses_existing_identity(prepared):
    module, provider, service, build = prepared
    (provider / "build_support/new_sibling.py").write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="membership changed"):
        build.revalidate()


@pytest.mark.parametrize("case", ["empty", "relative", "escape", "mapping", "duplicate", "unbound", "unknown"])
def test_import_root_refuses_malformed_or_unbound_mapping(prepared, case):
    module, provider, service, build = prepared
    service = copy.deepcopy(service)
    roots = service["python_import_roots"]
    if case == "empty":
        roots.clear()
    elif case == "relative":
        roots[0]["source"] = "relative"
    elif case == "escape":
        roots[0]["destination"] += "/../escape"
    elif case == "mapping":
        roots[0]["destination"] = str(provider / "different")
    elif case == "duplicate":
        roots.append(dict(roots[0]))
    elif case == "unbound":
        roots[0] = {"source": str(provider), "destination": str(provider)}
        service["dependencies"] = [
            row for row in service["dependencies"] if not Path(row["source"]).is_relative_to(provider)
        ]
    else:
        roots[0]["extra"] = "unsupported"
    with pytest.raises(ValueError, match="Python import root"):
        module._worker_import_roots(service)


def test_legacy_import_root_decoding_is_not_rewritten(prepared):
    module, provider, service, build = prepared
    service = dict(service)
    service.pop("python_import_roots")
    assert module._worker_import_roots(service) == [str(Path(service["namespace_root"]) / "merlin/python")]


def test_actual_worker_validates_installed_format_grants_without_tools(prepared, monkeypatch):
    module, provider, service, build = prepared
    for row in service["dependencies"]:
        if row["kind"] == "format_data":
            destination = Path(row["destination"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(row["source"], destination)
    original = module._validate_format_data

    class ValidatedBeforeToolExecution(Exception):
        pass

    def validate_then_stop(*args):
        assert original(*args) == service["format_data_relation"]["validation"]
        raise ValidatedBeforeToolExecution

    monkeypatch.setattr(module, "_validate_format_data", validate_then_stop)
    monkeypatch.setattr(sys, "path", list(sys.path))
    with pytest.raises(ValidatedBeforeToolExecution):
        module._worker(build.request_path)
