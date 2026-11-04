"""Optional exact slice preparation for the authored whole-source entry.

Call on the original parsed module after recording its source operations. The
returned original operation objects let the caller retain source attribution;
text reparsing must not replace the original source identity or ABI observers.
Ordinary owned-host policy selection is separately permission gated.
"""

from merlin.llvmlower.static_slice_composition import compose_static_slices

ENABLED = False


def prepare(module, *, source_sha256):
    if not isinstance(source_sha256, str) or len(source_sha256) != 64 or any(
        ch not in "0123456789abcdef" for ch in source_sha256
    ):
        raise ValueError("layout preparation requires the exact original source SHA256")
    operations = tuple(module.walk())
    report = compose_static_slices(module) if ENABLED else {
        "schema": "merlin.static_slice_composition.v1", "compositions": []
    }
    report["source_sha256"] = source_sha256
    report["enabled"] = ENABLED
    report["operation_bindings"] = tuple(
        (operations[row["removed_source_op_index"]], operations[row["retained_source_op_index"]])
        for row in report["compositions"]
    )
    return report
