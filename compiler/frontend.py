"""Input readers for the frozen interface and ordinary tensor-function MLIR.

The reviewed, target-neutral grammar reader owns operation and attribute parsing.
This adapter supplies strict version checks, logical-name resolution, and complete
result declarations needed by our backend. It does not choose target schedules.
"""
from __future__ import annotations

from pathlib import Path
import re
import sys
from typing import Any

# Keep imports independent of the invoking process's current working directory.
_LIBRARY = Path(__file__).resolve().parent / "compiler_library"
if str(_LIBRARY) not in sys.path:
    sys.path.insert(0, str(_LIBRARY))

from merlin.targetgen.contract import interface_emit as _interface
from merlin.targetgen.oot_starterkit.iface import parse_linalg

InterfaceGrammarError = _interface.InterfaceGrammarError
_OP_START = re.compile(r"^(?:%[^=\s]+\s*=\s*)?merlin_iface\.[A-Za-z_][A-Za-z_0-9]*\b")


def _without_comments(text: str) -> str:
    """Remove MLIR line comments while keeping quoted strings intact."""
    result = []
    for line in text.splitlines():
        quoted = escaped = False
        for i, char in enumerate(line):
            if not quoted and line[i:i + 2] == "//":
                line = line[:i]
                break
            if char == '"' and not escaped:
                quoted = not quoted
            escaped = char == "\\" and not escaped
        result.append(line)
    return "\n".join(result)


def _interface_statements(text: str) -> tuple[dict[str, Any], list[str]]:
    """Feed complete operations to the library's line-oriented grammar reader.

    The public grammar permits line breaks in types and attribute dictionaries;
    these are whitespace between tokens, so retain them as spaces. Unknown
    operations are retained here and rejected by the canonical mnemonic table.
    """
    text = _without_comments(text)
    attrs_body = _interface._module_attrs(text)
    attrs = _interface._parse_attr_block(attrs_body)
    # Start after the module's body-opening brace, not its attribute dictionary.
    module = text.find("module")
    if module < 0:
        raise InterfaceGrammarError("expected an interface module")
    first_brace = text.find("{", module)
    if first_brace < 0:
        raise InterfaceGrammarError("interface module has no body")
    if "attributes" in text[module:first_brace]:
        _, after_attrs = _interface._braced(text, first_brace)
        if after_attrs < 0:
            raise InterfaceGrammarError("unclosed interface module attributes")
        first_brace = text.find("{", after_attrs)
    if first_brace < 0:
        raise InterfaceGrammarError("interface module has no body")
    body, end = _interface._braced(text, first_brace)
    if end < 0:
        raise InterfaceGrammarError("unclosed interface module body")
    statements: list[str] = []
    current: list[str] = []
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        if _OP_START.match(line):
            if current:
                statements.append(" ".join(current))
            current = [line]
        elif current:
            current.append(line)
        else:
            raise InterfaceGrammarError(f"unexpected statement in interface module: {line!r}")
    if current:
        statements.append(" ".join(current))
    return attrs, statements


def parse_interface_source(text: str) -> dict[str, Any]:
    """Read interface v0.1 into a complete logical command-buffer dictionary.

    Leaf and output names stay verbatim, including when their SSA identifiers
    differ. All command attributes (including source provenance) survive. Version
    and malformed operand errors fail closed before any target lowering begins.
    """
    module_attrs, statements = _interface_statements(text)
    version = module_attrs.get("merlin_iface.version")
    if version != "0.1":
        raise InterfaceGrammarError(f"unsupported merlin_iface.version {version!r}; expected '0.1'")
    abi = module_attrs.get("merlin_iface.abi_version", "0.1")
    if abi != "0.1":
        raise InterfaceGrammarError(f"unsupported merlin_iface.abi_version {abi!r}; expected '0.1'")
    normalized = "module attributes " + _interface._fmt_attrs(module_attrs) + " {\n"
    normalized += "\n".join(statements) + "\n}"
    # The canonical parser diagnoses unknown mnemonics. Check structural
    # residency operand counts first so malformed input yields a grammar error
    # rather than indexing an absent operand inside that parser.
    for statement in statements:
        op = _interface._op_line(statement)
        if op is None:
            raise InterfaceGrammarError(f"unreadable interface operation: {statement!r}")
        count = {"tensor": 0, "resident_pack": 1, "matmul": 2, "commit": 1, "evict": 1}.get(op["mnemonic"])
        if count is not None and len(op["operands"]) != count:
            raise InterfaceGrammarError(
                f"merlin_iface.{op['mnemonic']} expects {count} operands, got {len(op['operands'])}"
            )
    if len(_interface.op_mnemonics(normalized)) != len(statements):
        raise InterfaceGrammarError("expected one complete interface operation per statement")
    cb = _interface.parse_interface_mlir(normalized)
    if not cb["target"]:
        raise InterfaceGrammarError("interface module must declare merlin_iface.target")

    aliases: dict[str, str] = {}
    tensor_order: list[str] = []
    command_index = 0
    for statement in statements:
        op = _interface._op_line(statement)
        if op is None:
            raise InterfaceGrammarError(f"unreadable interface operation: {statement!r}")
        mnemonic = op["mnemonic"]
        attrs = _interface._parse_attr_block(op["attrs_body"])
        result = op["result"]
        count = {"tensor": 0, "resident_pack": 1, "matmul": 2, "commit": 1, "evict": 1}.get(mnemonic)
        if count is not None and len(op["operands"]) != count:
            raise InterfaceGrammarError(f"merlin_iface.{mnemonic} expects {count} operands, got {len(op['operands'])}")
        if mnemonic != "evict" and not result:
            raise InterfaceGrammarError(f"merlin_iface.{mnemonic} requires an SSA result")
        if result and result in aliases:
            raise InterfaceGrammarError(f"duplicate SSA result %{result}")
        for operand in op["operands"]:
            if operand not in aliases:
                raise InterfaceGrammarError(f"merlin_iface.{mnemonic} uses undefined SSA value %{operand}")
        logical = attrs.get("name", result) if mnemonic == "tensor" or mnemonic == "commit" or mnemonic in _interface._NAMED_OP_OPERAND_KEYS else result
        if result:
            aliases[result] = logical
        if mnemonic in {"tensor", "commit"} or mnemonic in _interface._NAMED_OP_OPERAND_KEYS:
            if logical not in tensor_order:
                tensor_order.append(logical)
        if mnemonic == "tensor":
            # The canonical reader parses storage encodings and scale metadata;
            # preserve additional declared metadata, including source identity.
            cb["tensors"][logical].update({k: v for k, v in attrs.items() if k != "name"})
            continue
        command = cb["commands"][command_index]
        command_index += 1
        operand_keys = [key for key in command["operands"] if key != "dst"]
        for key, source in zip(operand_keys, op["operands"]):
            command["operands"][key] = aliases[source]
        if "dst" in command["operands"]:
            command["operands"]["dst"] = logical
        if attrs:
            command.setdefault("attributes", {}).update({k: v for k, v in attrs.items() if k != "name"})
        if mnemonic == "commit":
            shape, dtype = _interface._shape_dtype(_interface._last_type(op["tail"]))
            spec = {"shape": shape, "dtype": dtype, "role": "output"}
            old = cb["tensors"].get(logical)
            if old is not None and (old["shape"], old["dtype"]) != (shape, dtype):
                raise InterfaceGrammarError(f"inconsistent declaration for output {logical!r}")
            cb["tensors"][logical] = spec
    # The canonical reader discovers whole-op outputs during its first pass,
    # while COMMIT outputs are declared by this adapter. Preserve actual source
    # result encounter order across both forms, rather than placing every
    # COMMIT result after every whole-op result as an artifact of parser phases.
    cb["tensors"] = {name: cb["tensors"][name] for name in
                     dict.fromkeys([*tensor_order, *cb["tensors"]])}
    params = module_attrs.get("merlin_iface.params", module_attrs.get("params", {}))
    if not isinstance(params, dict):
        raise InterfaceGrammarError("module params must be a dictionary")
    if params:
        cb["params"] = dict(params)
    provenance = {k: v for k, v in module_attrs.items() if k.startswith("prov.")}
    if provenance:
        cb.setdefault("params", {}).update(provenance)
    return cb


def source_kind(text: str) -> str:
    """Select the grammar; parsing still independently validates its contents."""
    clean = _without_comments(text)
    if "merlin_iface.version" in clean or any(_OP_START.match(line.strip()) for line in clean.splitlines()):
        return "merlin_iface"
    return "linalg-on-tensors"


def parse_source(text: str) -> dict[str, Any]:
    """Read an interface command buffer, or a verified linalg workload inventory.

    A linalg inventory has ``level == 'linalg-on-tensors'`` and requires authored
    lowering. It must never be interpreted as an empty command buffer.
    """
    if source_kind(text) == "merlin_iface":
        return parse_interface_source(text)
    return parse_linalg(text)
