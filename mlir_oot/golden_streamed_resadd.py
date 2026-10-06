"""Explicit residual prediction with correction of completed output panels.

The shared certificate proves the correction relation, not target timing or
the primitive predictor. The provider must compile the returned correction C
under the source floating contract and close its object together with the
kernel. Inputs and output must be disjoint and survive the entire call.
"""

import hashlib
import json

from xdsl.dialects.builtin import StringAttr

from .golden_wide_resadd import _build_prefetch, tables


def build(m, certificate, *, correction_symbol):
    """Bind an unchanged full-domain certificate to bounded target scheduling."""
    from merlin.llvmlower.quantized_affine_pair import derive, emit_correction

    try:
        expected = derive(**certificate["source"], **certificate["predictor"])
    except (KeyError, TypeError, ValueError) as failure:
        raise ValueError(
            "complete signed-byte source certificate required"
        ) from failure
    if certificate != expected:
        raise ValueError("unchanged complete signed-byte source certificate required")
    if (
        not isinstance(correction_symbol, str)
        or not correction_symbol.isascii()
        or not correction_symbol.isidentifier()
        or correction_symbol == "gemmini_golden_wide_resadd"
    ):
        raise ValueError("distinct correction function identifier required")
    predictor = certificate["predictor"]
    code = emit_correction(
        certificate, correction_symbol, packed_prefix=True, output_value_guard=True
    )
    module = _build_prefetch(
        m,
        **predictor,
        relu=certificate["source"]["relu"],
        banked_accumulators=True,
        correction_symbol=correction_symbol,
    )
    proof_sha = hashlib.sha256(
        json.dumps(certificate, sort_keys=True).encode()
    ).hexdigest()
    module.attributes["gemmini.correction_certificate_sha256"] = StringAttr(proof_sha)
    module.attributes["gemmini.correction_source_sha256"] = StringAttr(
        hashlib.sha256(code.encode()).hexdigest()
    )
    module.attributes["gemmini.correction_obligations"] = StringAttr(
        "Complete target predictor qualification; linked exact generated helper; "
        "source RN-even/gradual-underflow/separate operations; immutable disjoint "
        "A/B and fresh private C; Rocket/Gemmini FENCE closes prior stores; "
        "CPU helper writes only the completed panel and issues no RoCC commands"
    )
    module.verify()
    return module, code, tables(predictor["p"], predictor["q"])
