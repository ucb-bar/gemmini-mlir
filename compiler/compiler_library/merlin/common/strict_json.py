"""Bounded, unambiguous JSON parsing for records used as authority."""

from __future__ import annotations

import json
import math
from typing import Any

MAX_RECORD_BYTES = 128 * 1024 * 1024


def loads(raw: bytes | str, *, max_bytes: int = MAX_RECORD_BYTES) -> Any:
    """Reject duplicate keys and non-finite numbers before interpreting claims.

    The bound applies to encoded input bytes. This reader grants no provenance or
    admission; consumers still validate their independently selected identities.
    """
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
        raise ValueError("JSON reader limit must be a nonnegative integer")
    if not isinstance(raw, (bytes, str)):
        raise TypeError("JSON record must be bytes or text")
    if len(raw) > max_bytes or (isinstance(raw, str) and len(raw.encode("utf-8")) > max_bytes):
        raise ValueError("JSON record exceeds the bounded reader limit")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON value: {value}")

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"non-finite JSON value: {value}")
        return parsed

    return json.loads(raw, object_pairs_hook=unique_pairs, parse_constant=reject_constant, parse_float=finite_float)
