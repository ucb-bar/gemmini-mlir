"""Source readers and declaration-only dataflow for the authored backend.

No tensor values or target instructions are manufactured here.  In particular,
the original parsed command list and complete linalg inventory remain available
to later stages, including commands a particular lowering does not yet support.
"""
from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path
from typing import Any

_PACKAGE = Path(__file__).resolve().parent
_LIBRARY = _PACKAGE / "compiler_library"
if str(_LIBRARY) not in sys.path:
    sys.path.insert(0, str(_LIBRARY))

from merlin.targetgen.oot_starterkit.iface import (  # noqa: E402
    is_linalg_on_tensors,
    parse_interface,
    parse_linalg,
)
from merlin.targetgen.contract import interface_emit  # noqa: E402
from merlin.compile.logical_abi import (  # noqa: E402
    declared_output_dtypes,
    logical_abi,
    logical_interface,
)


def logical_args(cb: dict[str, Any], *, contract_path=None) -> list[dict[str, Any]]:
    """Return the public logical pointer signature, in the declared ABI order.

    The explicit contract path avoids environment-dependent contract discovery.
    A caller packaging its own contract may override it; no guessed argument
    ordering substitutes for a missing contract.
    """
    packaged_contract = _PACKAGE / "data" / "mlir_oot_backend_contract.yaml"
    path = contract_path or (packaged_contract if packaged_contract.is_file()
                             else _PACKAGE.parent / "mlir_oot_backend_contract.yaml")
    buffers = logical_interface(cb, logical_abi(path))
    return [
        {
            "index": index,
            "name": buf.name,
            "kind": buf.kind,
            "shape": list(buf.shape),
            "dtype": buf.dtype,
            "access": "read" if buf.kind == "input" else (
                "readwrite" if buf.kind == "inout" else "write"
            ),
        }
        for index, buf in enumerate(buffers)
    ]


def describe_commands(cb: dict[str, Any]) -> list[dict[str, Any]]:
    """One ordered description per source command, retaining every operand.

    Shapes and product geometry are convenience declarations.  Unknown operations
    remain explicit records so a lowering must implement them or diagnose them.
    Packing handles resolve to their source tensors without removing packing ops.
    """
    tensors = cb.get("tensors") or {}
    dtypes = declared_output_dtypes(cb)
    handles: dict[str, str] = {}
    products: dict[str, dict[str, Any]] = {}
    records = []

    def source(name):
        seen = set()
        while name in handles and name not in seen:
            seen.add(name)
            name = handles[name]
        return name

    for index, command in enumerate(cb.get("commands") or []):
        opcode = command["opcode"]
        operands = copy.deepcopy(command.get("operands") or {})
        attrs = copy.deepcopy(command.get("attributes") or {})
        resolved = {key: source(name) for key, name in operands.items()}
        category = {
            "RES_PACK": "packing", "EVICT": "eviction",
            "MATMUL": "product", "MATMUL_RESIDENT": "product",
            "BATCHED_MATMUL": "product", "CONV2D": "product",
            "ATTENTION_QK": "product", "ATTENTION_PV": "product",
            "COMMIT": "readout", "MOVEMENT": "movement",
            "RESIDUAL_ADD": "residual", "BIAS_ADD": "elementwise",
        }.get(opcode, "other")
        record = {
            "index": index, "opcode": opcode, "category": category,
            "operands": operands, "resolved_operands": resolved,
            "attributes": attrs,
            "tensor_specs": {key: copy.deepcopy(tensors[name])
                             for key, name in resolved.items() if name in tensors},
        }
        dst = operands.get("dst")
        if dst is not None:
            record["output"] = dst
            if dst in dtypes:
                record["output_dtype"] = dtypes[dst]
        if opcode == "RES_PACK":
            handles[dst] = resolved["src"]
        if opcode in ("MATMUL", "MATMUL_RESIDENT", "BATCHED_MATMUL"):
            lhs = resolved.get("lhs", resolved.get("a"))
            rhs = resolved.get("rhs", resolved.get("w"))
            a = (tensors.get(lhs) or {}).get("shape", [])
            w = (tensors.get(rhs) or {}).get("shape", [])
            if len(a) >= 2 and len(w) >= 2:
                record["extents"] = {"m": a[-2], "k": a[-1], "n": w[-1]}
                record["batch_shape"] = list(a[:-2])
            products[dst] = record
        elif opcode == "CONV2D":
            record["geometry"] = {key: copy.deepcopy(attrs[key]) for key in
                                  ("kernel", "stride", "padding", "dilation", "layout")
                                  if key in attrs}
            products[dst] = record
        elif opcode == "COMMIT":
            producer = products.get(operands.get("src"))
            if producer is not None:
                record["product_index"] = producer["index"]
                if "extents" in producer:
                    record["extents"] = copy.deepcopy(producer["extents"])
        records.append(record)
    return records


def normalize_source(text: str) -> dict[str, Any]:
    """Parse either public grammar and retain its complete source-derived model."""
    if not isinstance(text, str):
        raise TypeError("source must be MLIR text")
    identity = hashlib.sha256(text.encode("utf-8")).hexdigest()
    # The provenance marker is optional on ordinary prepared tensor sources.
    # Ask the canonical tensor parser whenever the residency grammar contains
    # no actual operations; this also accepts builtin.module's custom spelling
    # without rewriting any source text or relying on function-name heuristics.
    if is_linalg_on_tensors(text) or not interface_emit.op_mnemonics(text):
        inventory = parse_linalg(text)
        return {"grammar": "linalg", "source_sha256": identity,
                "inventory": inventory}
    attrs = interface_emit._parse_attr_block(interface_emit._module_attrs(text))
    version = attrs.get("merlin_iface.version")
    if version != interface_emit.GRAMMAR_VERSION:
        raise ValueError(f"unsupported merlin_iface.version {version!r}; expected "
                         f"{interface_emit.GRAMMAR_VERSION!r}")
    cb = parse_interface(text)
    args = logical_args(cb)
    return {"grammar": "merlin_iface", "source_sha256": identity,
            "cb": cb, "logical_args": args,
            "operations": describe_commands(cb)}
