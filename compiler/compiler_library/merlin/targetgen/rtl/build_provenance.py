"""Bind an existing reproducible-build attestation to a later execution.

This checks byte identity only. The attester that produced the supplied record
owns the actual source-to-binary rebuild; this module never infers that a file
named like a simulator was built from selected RTL. Field paths are explicit so
target-specific receipt layouts stay outside core.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from merlin.common.digest import is_sha256, sha256_file


def _document(path: Path) -> Mapping:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"receipt must be a JSON object: {path}")
    return doc


def _digest_field(doc: Mapping, fields: tuple[str, ...]) -> str:
    if not fields or any(not isinstance(field, str) or not field for field in fields):
        raise ValueError("digest field path must be nonempty strings")
    value = doc
    for field in fields:
        if not isinstance(value, Mapping) or field not in value:
            raise ValueError(f"receipt lacks digest field {'.'.join(fields)}")
        value = value[field]
    if not is_sha256(value):
        raise ValueError(f"receipt has invalid SHA-256 at {'.'.join(fields)}")
    return value


def bind_reproduced_binary(
    *,
    attestation_path: Path,
    attested_execution_path: Path,
    current_execution_path: Path,
    source_selection_path: Path,
    executable_path: Path,
    attestation_prior_receipt_field: tuple[str, ...],
    attestation_source_field: tuple[str, ...],
    attestation_tested_binary_field: tuple[str, ...],
    attestation_rebuilt_binary_field: tuple[str, ...],
    execution_source_field: tuple[str, ...],
    execution_binary_field: tuple[str, ...],
) -> dict:
    """Require both executions to name the same selected source and binary bytes.

    The attestation must report exact binary reproduction and still bind its
    original execution receipt. A later execution can use a different program
    ELF; this check does not assert that its numerics or program are equivalent.
    """
    attestation_path = Path(attestation_path).resolve(strict=True)
    attested_execution_path = Path(attested_execution_path).resolve(strict=True)
    current_execution_path = Path(current_execution_path).resolve(strict=True)
    source_selection_path = Path(source_selection_path).resolve(strict=True)
    executable_path = Path(executable_path).resolve(strict=True)
    attestation = _document(attestation_path)
    prior = _document(attested_execution_path)
    current = _document(current_execution_path)
    if attestation.get("status") != "reproduced_exact_binary":
        raise ValueError("attestation does not report an exact reproduced binary")
    prior_digest = sha256_file(attested_execution_path)
    source_digest = sha256_file(source_selection_path)
    binary_digest = sha256_file(executable_path)
    obligations = (
        (_digest_field(attestation, attestation_prior_receipt_field), prior_digest, "attested receipt"),
        (_digest_field(attestation, attestation_source_field), source_digest, "attested selected source"),
        (_digest_field(attestation, attestation_tested_binary_field), binary_digest, "tested binary"),
        (_digest_field(attestation, attestation_rebuilt_binary_field), binary_digest, "rebuilt binary"),
        (_digest_field(prior, execution_source_field), source_digest, "prior execution source"),
        (_digest_field(current, execution_source_field), source_digest, "current execution source"),
        (_digest_field(prior, execution_binary_field), binary_digest, "prior execution binary"),
        (_digest_field(current, execution_binary_field), binary_digest, "current execution binary"),
    )
    for declared, actual, label in obligations:
        if declared != actual:
            raise ValueError(f"{label} digest differs from selected bytes")
    return {
        "schema": "merlin.reproduced-binary-execution-binding.v1",
        "status": "bound_exact_binary",
        "attestation_receipt_sha256": sha256_file(attestation_path),
        "attested_execution_receipt_sha256": prior_digest,
        "current_execution_receipt_sha256": sha256_file(current_execution_path),
        "source_selection_sha256": source_digest,
        "executable_sha256": binary_digest,
        "claim": "current execution names the exact selected source and executable reproduced by the attestation",
        "limits": [
            "This verifies receipt and byte identity, not the attester's build procedure.",
            "It does not re-run either execution or verify numerical results or source-to-FIRRTL generation.",
        ],
    }
