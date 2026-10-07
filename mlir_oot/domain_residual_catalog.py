"""Bind source-proven operand intervals to exact residual coefficient choices.

This explicit catalog transformation consumes sealed full-domain producer
contracts and actual typed SSA/view chains. It never grants ranges from symbols,
provenance, captured values or a caller-supplied domain attribute alone. Scalar
range and admitted-pair proofs are shared Merlin; coefficient representation,
primitive resource selection, declarations and object emission belong here.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from merlin.llvmlower import quantized_affine_domain as domain
from merlin.llvmlower import quantized_affine_pair as pair
from merlin.llvmlower.typed_integer_domains import (
    SignedByteDomain,
    trace_signed_byte_domain,
)
from xdsl.dialects import func
from xdsl.dialects.builtin import StringAttr, TensorType, i8

from .captured_residual_bundle import (
    binding_attributes,
    canonical,
    compile_adapter,
    sha,
    wide_adapter,
    wide_oracle,
)
from .direct_conv_binding import serialize
from .frontend.parse import parse_module
from .golden_device_compile import compile_module
from .golden_wide_resadd import build as build_kernel
from .golden_wide_resadd import tables
from .no_fsm_audit import audit_elf


def products(predictor):
    return sum((predictor[key] + 126) // 127 for key in ("p", "q"))


def _sealed_source_module(base, record, source=None):
    for name, key in (
        ("residual.o", "object_sha256"),
        ("rewritten.mlir", "rewritten_sha256"),
        ("native_oracle.c", "native_oracle_sha256"),
    ):
        if sha(base / name) != record[key]:
            raise ValueError("sealed residual artifact changed: " + name)
    if (
        record["implementation"] != "wide_integer"
        or record["numeric_policy"]["max_output_lsb"] != 0
    ):
        raise ValueError("unchanged exact wide-integer source catalog required")
    module = parse_module(
        (base / "rewritten.mlir").read_text()
        if source is None
        else Path(source).read_text()
    )
    module.verify()
    declarations = {
        op.sym_name.data: op
        for op in module.body.block.ops
        if isinstance(op, func.FuncOp) and not op.body.blocks
    }
    call_groups = {}
    for op in module.walk():
        if isinstance(op, func.CallOp):
            call_groups.setdefault(op.callee.root_reference.data, []).append(op)
    if any(
        len(call_groups.get(route["symbol"], [])) != 1 for route in record["routes"]
    ):
        raise ValueError(
            "one actual source call per sealed residual route required; shared-call domains unsupported"
        )
    calls = {
        name: values[0] for name, values in call_groups.items() if len(values) == 1
    }
    facts = []
    for route in record["routes"]:
        symbol = route["symbol"]
        if symbol not in declarations or symbol not in calls:
            raise ValueError("actual typed source call/declaration closure missing")
        declaration, call = declarations[symbol], calls[symbol]
        if (
            hashlib.sha256(canonical(route["proof"]).encode()).hexdigest()
            != route["proof_sha256"]
        ):
            raise ValueError("sealed numeric certificate changed")
        coefficients = {
            key: route["proof"]["coefficients"][key] for key in ("p", "q", "scale")
        }
        full = pair.derive(**route["proof"]["source"], **coefficients)
        if (
            full["mismatched_pairs"]
            or route["proof"]["pairs"] != 65536
            or not route["proof"]["exact"]
        ):
            raise ValueError(
                "producer range requires complete source-equivalent arithmetic"
            )
        for key, value in binding_attributes(
            route, record["source_sha256"], record["capture_receipt_sha256"]
        ).items():
            if declaration.attributes.get(key) != value:
                raise ValueError(
                    "typed producer numeric/source declaration differs: " + key
                )
        shape = TensorType(i8, [route["m"], route["n"]])
        if (
            len(call.arguments) != 3
            or len(call.results) != 1
            or any(value.type != shape for value in (*call.arguments, *call.results))
        ):
            raise ValueError("typed producer ABI differs from sealed source contract")
        if tuple(declaration.function_type.inputs) != (shape,) * 3 or tuple(
            declaration.function_type.outputs
        ) != (shape,):
            raise ValueError("declared tensor ABI differs from sealed producer")
        attrs = declaration.properties.get("arg_attrs")
        if attrs is None or [
            getattr(row.data.get("bufferization.access"), "data", None) for row in attrs
        ] != ["read", "read", "write"]:
            raise ValueError("producer effects differ")
        work = base / symbol
        for name, expected in (
            ("kernel.gemmini.mlir", route["compilation"]["target_ir_sha256"]),
            ("kernel.o", route["compilation"]["object_sha256"]),
            ("adapter.c", route["adapter_compilation"]["source_sha256"]),
            ("adapter.o", route["adapter_compilation"]["object_sha256"]),
        ):
            if sha(work / name) != expected:
                raise ValueError("sealed producer implementation changed")
        certificate = domain.source_result_interval(route["proof"]["source"])
        certificate.update(
            producer_symbol_for_binding_audit_only=symbol,
            source_sha256=record["source_sha256"],
            numeric_proof_sha256=route["proof_sha256"],
            implementation_sha256=route["compilation"]["object_sha256"],
        )
        facts.append(
            SignedByteDomain(
                call.results[0],
                certificate["minimum"],
                certificate["maximum"],
                certificate,
            )
        )
    return module, declarations, calls, facts


def inspect_choices(base, proposals, *, source=None):
    """Reprove actual operand domains and scalar equality before any mutation."""
    base = Path(base)
    record = json.loads((base / "residual.json").read_text())
    module, declarations, calls, facts = _sealed_source_module(base, record, source)
    candidates = {}
    for proposal in proposals:
        source = pair.derive(**proposal["source"], **proposal["predictor"])["source"]
        key = canonical(source)
        candidates.setdefault(key, []).append(dict(proposal["predictor"]))
    choices = []
    for route in record["routes"]:
        call = calls[route["symbol"]]
        traces = [
            trace_signed_byte_domain(value, facts) for value in call.arguments[:2]
        ]
        intervals = [[trace["minimum"], trace["maximum"]] for trace in traces]
        old = products(route["proof"]["coefficients"])
        best = None
        for predictor in candidates.get(canonical(route["proof"]["source"]), []):
            if any(
                type(predictor[key]) is not int or not 1 <= predictor[key] <= 32767
                for key in ("p", "q")
            ):
                raise ValueError(
                    "positive bounded target diagonal coefficient required"
                )
            if products(predictor) >= old:
                continue
            try:
                certificate = domain.derive(
                    route["proof"]["source"], predictor, intervals
                )
            except ValueError:
                continue
            if best is None or products(predictor) < products(best["predictor"]):
                best = {"predictor": predictor, "certificate": certificate}
        choices.append(
            {
                "symbol_for_binding_audit_only": route["symbol"],
                "source": route["proof"]["source"],
                "m": route["m"],
                "n": route["n"],
                "input_domain_traces": traces,
                "operand_intervals": intervals,
                "old_products": old,
                "choice": best,
            }
        )
    return record, module, declarations, choices


def build(base, proposals, llvm_bin, output):
    base, llvm_bin, output = map(Path, (base, llvm_bin, output))
    record, module, declarations, choices = inspect_choices(base, proposals)
    output.mkdir(parents=True, exist_ok=False)
    record = copy.deepcopy(record)
    objects, native, changed = [], [], []
    for route, choice in zip(record["routes"], choices, strict=True):
        work = output / route["symbol"]
        selected = choice["choice"]
        if selected is None:
            shutil.copytree(base / route["symbol"], work)
        else:
            certificate = selected["certificate"]
            route["proof"] = {
                "proof": "complete admitted source-domain pair proof",
                "source": certificate["source"],
                "coefficients": certificate["predictor"],
                "primitive": {"readout": certificate["predictor"]["scale"]},
                "pairs": certificate["admitted_pairs"],
                "exact": True,
                "max_output_lsb_error": 0,
                "domain_certificate": certificate,
                "source_domain_traces": choice["input_domain_traces"],
            }
            route["proof_sha256"] = hashlib.sha256(
                canonical(route["proof"]).encode()
            ).hexdigest()
            route["numeric_policy"] = dict(
                route["numeric_policy"],
                domain="all admitted source-proven signed-byte operand pairs",
            )
            coeff = route["proof"]["coefficients"]
            schedule = route.get("device_schedule", {})
            device = build_kernel(
                route["m"],
                coeff["p"],
                coeff["q"],
                coeff["scale"],
                relu=route["proof"]["source"]["relu"],
                prefetch_m=schedule.get("prefetch_m", False),
                banked_accumulators=schedule.get("banked_accumulators", False),
            )
            device.body.block.first_op.properties["sym_name"] = StringAttr(
                route["kernel"]
            )
            resources = device.attributes.get("gemmini.residual_m_prefetch")
            if schedule.get("prefetch_m", False):
                route["device_schedule"] = {
                    "prefetch_m": True,
                    "banked_accumulators": schedule.get("banked_accumulators", False),
                    "active": resources is not None,
                    "resources": {
                        key: value.value.data for key, value in resources.data.items()
                    }
                    if resources is not None
                    else {},
                }
            route["compilation"] = compile_module(device, llvm_bin, work)
            c = wide_adapter(route, route["symbol"], route["kernel"])
            (work / "adapter.c").write_text(c)
            route["adapter_compilation"] = compile_adapter(
                work / "adapter.c", work / "adapter.o", llvm_bin
            )
            route["coefficient_table_sha256"] = hashlib.sha256(
                tables(coeff["p"], coeff["q"])
            ).hexdigest()
            declarations[route["symbol"]].attributes.update(
                binding_attributes(
                    route, record["source_sha256"], record["capture_receipt_sha256"]
                )
            )
            changed.append(dict(choice, new_products=products(coeff)))
        objects.extend([work / "kernel.o", work / "adapter.o"])
        native.append(
            wide_adapter(route, route["symbol"], route["kernel"])
            + wide_oracle(route, route["kernel"])
        )
    text = serialize(module, list(declarations.values()))
    parse_module(text).verify()
    (output / "rewritten.mlir").write_text(text)
    (output / "native_oracle.c").write_text("\n".join(native))
    linker = llvm_bin / "ld.lld"
    if not linker.is_file():
        linker = Path(shutil.which("ld.lld"))
    command = [str(linker), "-r", *map(str, objects), "-o", str(output / "residual.o")]
    subprocess.run(command, check=True, capture_output=True)
    audit = audit_elf((output / "residual.o").read_bytes())
    if audit["status"] != "pass":
        raise ValueError("source-domain object contains forbidden instructions")
    record.update(
        rewritten_sha256=sha(output / "rewritten.mlir"),
        object_sha256=sha(output / "residual.o"),
        native_oracle_sha256=sha(output / "native_oracle.c"),
        nofsm_audit=audit,
        source_domain_derivation={
            "base_manifest": str(base / "residual.json"),
            "base_manifest_sha256": sha(base / "residual.json"),
            "changed": changed,
            "all_typed_choices": choices,
            "linker_argv": command,
            "linker_sha256": sha(linker),
            "scope": "exact source-proven domains, no sampled range or workload selector; cycle cost UNKNOWN",
        },
    )
    (output / "residual.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def rebind_joint_context(original_bundle, joint_bundle, output):
    """Retain selected bytes while validating an unchanged source-route context."""
    from .joint_residual_catalog import _validate_record

    original_bundle, joint_bundle, output = map(
        Path, (original_bundle, joint_bundle, output)
    )
    original = json.loads((original_bundle / "residual.json").read_text())
    selected = json.loads((joint_bundle / "joint.json").read_text())
    # Equality of each original_route, source proof, ABI and resource plan is
    # required. A changed selected arithmetic route cannot be rebound this way.
    _validate_record(original, selected)
    for file, key in (
        ("joint.o", "object_sha256"),
        ("native_oracle.c", "native_oracle_sha256"),
    ):
        if sha(joint_bundle / file) != selected[key]:
            raise ValueError("retained joint artifact identity changed")
    output.mkdir(parents=True, exist_ok=False)
    for path in joint_bundle.iterdir():
        if path.is_file() and path.name != "joint.json":
            shutil.copyfile(path, output / path.name)
    selected["original_manifest_path"] = str(
        (original_bundle / "residual.json").resolve()
    )
    selected["original_manifest_sha256"] = sha(original_bundle / "residual.json")
    selected["unchanged_joint_context_rebinding"] = {
        "previous_manifest": str((joint_bundle / "joint.json").resolve()),
        "previous_manifest_sha256": sha(joint_bundle / "joint.json"),
        "selected_code_and_numeric_routes_byte_unchanged": True,
    }
    (output / "joint.json").write_text(json.dumps(selected, indent=2) + "\n")
    return selected


def merlin_callbacks(
    llvm_bin, original_bundle, domain_bundle, base_callbacks, *, joint_bundle=None
):
    """Opt into actual source-domain choices before normal catalog preparation."""
    from .residual_mixed_catalog import merlin_callbacks as residual_callbacks

    original_bundle, domain_bundle = map(Path, (original_bundle, domain_bundle))
    json.loads((original_bundle / "residual.json").read_text())
    selected = json.loads((domain_bundle / "residual.json").read_text())
    derivation = selected["source_domain_derivation"]
    if derivation["base_manifest_sha256"] != sha(original_bundle / "residual.json"):
        raise ValueError("domain selection original manifest changed")
    changed = {
        row["symbol_for_binding_audit_only"]: row for row in derivation["changed"]
    }
    proposals = [
        {"source": row["source"], "predictor": row["choice"]["predictor"]}
        for row in changed.values()
    ]
    routes = {row["symbol"]: row for row in selected["routes"]}
    wrapped_prepare, wrapped_build = residual_callbacks(
        llvm_bin, domain_bundle, base_callbacks, joint_bundle=joint_bundle
    )
    manifest_pin = sha(domain_bundle / "residual.json")

    def prepare(source, work):
        if sha(domain_bundle / "residual.json") != manifest_pin:
            raise ValueError("source-domain catalog changed")
        # Validate the actual incoming model, not just the sealed proposal. The
        # source may have other legal provider rewrites; every residual producer
        # and each relevant typed input-view chain must still close exactly.
        _, module, declarations, actual = inspect_choices(
            original_bundle, proposals, source=source
        )
        actual_selected = {
            row["symbol_for_binding_audit_only"]: row
            for row in actual
            if row["choice"] is not None
        }
        if set(actual_selected) != set(changed):
            raise ValueError("actual source-domain eligibility changed")
        for symbol, choice in actual_selected.items():
            if (
                choice["choice"] != changed[symbol]["choice"]
                or choice["input_domain_traces"]
                != changed[symbol]["input_domain_traces"]
            ):
                raise ValueError("actual typed producer/domain source path differs")
        for symbol in changed:
            declarations[symbol].attributes.update(
                binding_attributes(
                    routes[symbol],
                    selected["source_sha256"],
                    selected["capture_receipt_sha256"],
                )
            )
        module.verify()
        work = Path(work)
        work.mkdir(parents=True, exist_ok=True)
        target = work / "source_domain_residual_input.mlir"
        target.write_text(serialize(module, list(declarations.values())))
        (work / "source_domain_selection.json").write_text(
            json.dumps(
                {
                    "schema": "actual_source_domain_residual_selection_v1",
                    "input_sha256": sha(source),
                    "rewritten_sha256": sha(target),
                    "base_manifest_sha256": sha(original_bundle / "residual.json"),
                    "selected_manifest_sha256": manifest_pin,
                    "changed": actual_selected,
                    "source_ranges_from_samples": False,
                    "alias_or_lifetime_changes": False,
                },
                indent=2,
            )
            + "\n"
        )
        return wrapped_prepare(target, work)

    return prepare, wrapped_build
