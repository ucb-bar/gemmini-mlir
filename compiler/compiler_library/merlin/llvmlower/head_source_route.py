"""Original-owner head refolding through the existing per-slice device ABI.

Every unmatched group is routed normally. Batch means a stock descriptor-shim
loop over the same matrix artifact, never a larger single device execution.
Source membership and physical execution remain independent audit obligations.
"""

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .borrowed_effects import BorrowedBufferEffects
from .sibling_source_route import _sha, _text


@dataclass(frozen=True)
class HeadBatchSourceRoute:
    source_ir_sha256: str
    effects: BorrowedBufferEffects
    shared_input_siblings: bool = False

    def record(self):
        if (type(self.source_ir_sha256) is not str or len(self.source_ir_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.source_ir_sha256)
                or type(self.effects) is not BorrowedBufferEffects
                or type(self.shared_input_siblings) is not bool):
            raise ValueError("head route needs original source identity and typed synchronous effects")
        self.effects.__post_init__()
        return {"kind": "independent_heads", **asdict(self)}


@dataclass(frozen=True)
class HeadProjection:
    source: object
    module: object
    batching: object
    members: tuple


def project_heads(module):
    """Refold exact head coordinates; retain all original member identities.

    Fresh aggregate provenance is only a handle for normal group construction.
    It grants no original source membership: the caller independently joins the
    complete original member roster and original supporting-operation demands.
    """
    from xdsl.dialects.builtin import ArrayAttr, StringAttr

    from merlin.transforms.head_batching import plan_independent_head_batching

    plan = plan_independent_head_batching(module)
    if not plan._replacements:
        raise ValueError("source has no complete independent head family")
    projected = plan.candidate.clone()
    copied = dict(zip(plan.candidate.walk(), projected.walk(), strict=True))
    rows = []
    for group, _init, batch, _slices in plan._replacements:
        original = []
        for head, root in enumerate(group.members):
            region = root.attributes.get("prov.region_id")
            nodes = root.attributes.get("prov.source_node_ids")
            if (type(region) is not StringAttr or not region.data or type(nodes) is not ArrayAttr
                    or not nodes or any(type(node) is not StringAttr for node in nodes)):
                raise ValueError("head members need complete original source provenance")
            original.append({"head": head, "source_region_id": region.data,
                             "source_node_ids": [node.data for node in nodes],
                             "geometry_m_k_n": [group.problem.m, group.problem.k, group.problem.n]})
        if len({r["source_region_id"] for r in original}) != len(original):
            raise ValueError("head members have ambiguous original source identities")
        region = "independent_heads_" + _sha(json.dumps(original, sort_keys=True))
        root = copied[batch]
        root.attributes.update({"prov.region_id": StringAttr(region), "prov.family": StringAttr("contraction"),
                                "prov.op": StringAttr("matmul"), "prov.source_node_ids": ArrayAttr(
                                    StringAttr(node) for row in original for node in row["source_node_ids"])})
        rows.append({"projected_region_id": region, "members": original,
                     "batch_extent": group.problem.products,
                     "geometry_m_k_n": [group.problem.m, group.problem.k, group.problem.n]})
    projected.verify()
    plan.revalidate()
    return HeadProjection(module, projected, plan, tuple(rows))


@dataclass(frozen=True)
class RoutedHeads:
    source: object
    module: object
    core: object
    interface: str
    rewrite: object
    projection: HeadProjection
    receipt: dict
    integrity_sha256: str
    component_host: object = None


def _snapshot(host):
    from .sibling_component_route import _snapshot as component_snapshot

    return {"source": _sha(_text(host.source)), "module": _sha(_text(host.module)),
            "core": _sha(_text(host.core)), "interface": _sha(host.interface),
            "rewrite": host.rewrite.to_sidecar(), "receipt": host.receipt,
            "projection_source": _sha(_text(host.projection.source)),
            "projection": _sha(_text(host.projection.module)), "members": host.projection.members,
            "component_host": component_snapshot(host.component_host) if host.component_host is not None else None}


def component_selection(selected):
    from .sibling_source_route import SiblingSourceRoute

    selected.record()
    if not selected.shared_input_siblings:
        raise ValueError("head selection did not request checked sibling components")
    return SiblingSourceRoute(selected.source_ir_sha256, selected.effects, partial=True)


def require_host(host, selected):
    if (type(host) is not RoutedHeads or type(selected) is not HeadBatchSourceRoute
            or _sha(_text(host.source)) != selected.source_ir_sha256
            or _sha(json.dumps(_snapshot(host), sort_keys=True, allow_nan=False)) != host.integrity_sha256):
        raise ValueError("head source, member projection, host or device custody changed")
    host.projection.batching.revalidate()
    if host.component_host is not None:
        from .sibling_component_route import require_components

        require_components(host.component_host, component_selection(selected))
        if host.component_host.rewrite != host.rewrite or _text(host.component_host.module) != _text(host.module):
            raise ValueError("head and sibling routes lost their common actual host or rewrite")
    elif selected.shared_input_siblings:
        raise ValueError("joint head route lost its checked sibling components")
    host.module.verify()
    selected.record()


def prepare_host(device, selected, source, directory):
    from merlin.compile.mesh import _mesh_tile_binding
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.targetgen import corpus_spec as CS

    from .device_offload import rewrite_groups_to_device
    from .exact_offload import _package_sha256

    selected.record()
    actual = parse_mlir_file(source)
    original, preparation = actual, None
    if _sha(_text(actual)) != selected.source_ir_sha256:
        from .prepacked_device_capture import original_sibling_source

        original, preparation = original_sibling_source(device, selected, actual_source=source, work=directory.parent)
    projection = project_heads(actual)
    module = projection.module.clone()
    # Same normal source/group owner for both matched and unmatched groups.
    # Only the checked source projection changes; no contraction falls back to
    # scalar host execution to make this route applicable.
    component_host = None
    if selected.shared_input_siblings:
        from .sibling_component_route import build_components

        component_host = build_components(device, component_selection(selected), projection.module,
                                          original=original, preparation=preparation)
        module, rewrite = component_host.module, component_host.rewrite
    else:
        rewrite = rewrite_groups_to_device(
            module, device.device, select=device.select, capture=device.capture, model=device.model,
            require_complete_groups=getattr(device, "require_complete_groups", False),
        )
    by_region = {row.get("source_region_id"): row for row in rewrite.to_sidecar()["routed"]}
    matched = []
    for family in projection.members:
        row = by_region.get(family["projected_region_id"])
        m, k, n = family["geometry_m_k_n"]
        if (row is None or row.get("parallel") != [family["batch_extent"], m, n]
                or row.get("reduction") != [k] or row.get("dtypes") != ["i8", "i8", "i32"]):
            raise ValueError("normal group route did not retain complete independent head geometry")
        symbol = row["symbol"]
        call = rewrite.call_buffers[symbol]
        if ([r["shape"] for r in call] != [[family["batch_extent"], m, k],
                                          [family["batch_extent"], k, n],
                                          [family["batch_extent"], m, n]]
                or [r["dtype"] for r in call] != ["i8", "i8", "i32"]):
            raise ValueError("normal head call lost original complete ranked operands/results")
        matched.append({**family, "physical_symbol": symbol, "physical_group": row["group"]})
    binding = _mesh_tile_binding(
        device.device, device.operand_dtype, device.accum_dtype, numeric_policy=device.numeric_policy)
    interfaces = {}
    for symbol, entry in rewrite.entries.items():
        if component_host is not None:
            interfaces[symbol] = rewrite.expected_interfaces[symbol]
        else:
            _capsule, interface = CS.build(entry, binding)
            interfaces[symbol] = {"mlir": interface, "sha256": _sha(interface)}
    rewrite = replace(rewrite, model_sha256=selected.source_ir_sha256,
                      package_sha256=_package_sha256(Path(device.package_dir)),
                      expected_interfaces=interfaces)
    receipt = {"schema": "merlin.original_head_source_projection.v1",
               "source_ir_sha256": selected.source_ir_sha256, "prepared_source_ir_sha256": _sha(_text(actual)),
               "projected_source_ir_sha256": _sha(_text(projection.module)), "head_families": matched,
               "weight_preparation": preparation, "unmatched_source_routed_normally": True,
               "original_readouts_preserved": True, "matrix_products": sum(r["batch_extent"] for r in matched),
               "batch_abi_calls": len(matched), "accelerator_execution_count_reduced": False,
               "shared_input_components": len(component_host.components) if component_host is not None else 0,
               "effect_or_placement_grant": False}
    interface = next(iter(interfaces.values()))["mlir"]
    host = RoutedHeads(original, module, projection.module, interface, rewrite, projection, receipt, "", component_host)
    host = replace(host, integrity_sha256=_sha(json.dumps(_snapshot(host), sort_keys=True, allow_nan=False)))
    require_host(host, selected)
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "host.mlir"
    path.write_text(_text(module) + "\n")
    (directory / "core.mlir").write_text(_text(projection.module) + "\n")
    (directory / "interface.mlir").write_text(interface)
    (directory / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return path, host
