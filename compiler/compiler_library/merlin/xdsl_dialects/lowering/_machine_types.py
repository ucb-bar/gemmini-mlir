"""Plain scalar spellings used by fusion, independently of quantized formats.

Signless MLIR integers retain the signed interpretation of the integer source
contraction matcher. Explicit ``ui``/``u``/``uint`` spellings are unsigned. IEEE
machine floats have fixed semantics; packed and named low-precision formats
continue through the optional quant-format registry.
"""
from __future__ import annotations


def machine_scalar(name: str | None) -> tuple[str, int, bool, str] | None:
    """Return (kind, bits, signed, canonical token), without consulting a registry."""
    if not name:
        return None
    name = str(name)
    floats = {
        "f16": 16, "float16": 16,
        "f32": 32, "float32": 32,
        "f64": 64, "float64": 64,
    }
    if name in floats:
        bits = floats[name]
        return "float", bits, True, "float" + str(bits)
    for prefix, signed in (
        ("uint", False), ("int", True), ("ui", False),
        ("si", True), ("u", False), ("i", True),
    ):
        if name.startswith(prefix):
            width = name[len(prefix):]
            if width.isdecimal() and int(width) > 0:
                bits = int(width)
                return "integer", bits, signed, ("int" if signed else "uint") + str(bits)
    return None
