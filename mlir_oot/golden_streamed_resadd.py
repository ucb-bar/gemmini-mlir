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


def panel_storage(m, *, coherence_granule_bytes, dma_max_request_bytes):
    """Prove byte and transaction-block disjointness for completed panels.

    The provider must bind both byte granularities to its selected hardware.
    This arithmetic proof establishes no allocator or input lifetime fact.
    The ranked adapter checks the physical output alignment and offset; normal
    compilation must separately retain the fresh-writer and borrowed effects.
    """
    values = (m, coherence_granule_bytes, dma_max_request_bytes)
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("positive integer panel and hardware bounds required")
    if m % 16 or m * 64 >= 1 << 63:
        raise ValueError("bounded complete sixteen-row output panels required")
    if any(value & (value - 1) for value in values[1:]):
        raise ValueError("power-of-two hardware transaction granularities required")
    alignment = max(coherence_granule_bytes, dma_max_request_bytes)
    if 1024 % alignment:
        raise ValueError("output panels must end on every transaction block boundary")
    return {
        "output_bytes": m * 64,
        "panel_bytes": 1024,
        "panels": m // 16,
        "required_output_alignment": alignment,
        "required_output_offset": 0,
        "coherence_granule_bytes": coherence_granule_bytes,
        "dma_max_request_bytes": dma_max_request_bytes,
        "completed_panel": "[1024*j,1024*(j+1))",
        "pending_panel": "[1024*(j+1),1024*(j+2))",
        "proof": "aligned base and panel endpoints partition every declared block; "
        "no completed CPU panel shares a declared coherence or DMA block with "
        "a pending device panel",
    }


def ranked_adapter(route, *, coherence_granule_bytes, dma_max_request_bytes):
    """Emit checked ordinary ranked ABI around the streamed primitive.

    Existing descriptor shape, byte overlap and overflow checks are reused.
    No allocation, input mutation, escaping pointer or extra output is added.
    Unknown/unaligned/subview output owners refuse before issuing commands.
    The external correction helper is separately compiled under the original
    source floating contract and called by the target panel schedule.
    """
    from merlin.llvmlower.quantized_affine_pair import emit_correction

    from .joint_residual_catalog import _identifier, _proof, _single_adapter

    _proof(route["proof"], single_output_guard=True)
    if route.get("single_output_guard") is not True or route.get("n") != 64:
        raise ValueError("single fully written Mx64 output contract required")
    symbol, kernel = (_identifier(route[key]) for key in ("symbol", "kernel"))
    storage = panel_storage(
        route["m"],
        coherence_granule_bytes=coherence_granule_bytes,
        dma_max_request_bytes=dma_max_request_bytes,
    )
    correction = symbol + "_correct"
    prefix = emit_correction(route["proof"], correction, output_value_guard=True)
    ordinary = _single_adapter(route)
    if not ordinary.startswith(prefix):
        raise ValueError("original correction and ranked adapter composition changed")
    suffix = ordinary[len(prefix) :]
    call = f" {correction}(pa,pb,pc,{storage['output_bytes']});\n"
    if suffix.count(call) != 1:
        raise ValueError("one original completed correction call required")
    suffix = suffix.replace(call, "")
    command = f" {kernel}(pa,pb,pc,{symbol}_coefficients);"
    if suffix.count(command) != 1:
        raise ValueError("one source-bound target producer call required")
    alignment = storage["required_output_alignment"]
    suffix = suffix.replace(
        command,
        " if(c->offset!=0 || c->allocated!=c->aligned || "
        f"((uintptr_t)pc&{alignment - 1})) __builtin_trap();\n" + command,
    )
    return "#include <stdint.h>\n#include <stddef.h>\n" + suffix, storage


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
