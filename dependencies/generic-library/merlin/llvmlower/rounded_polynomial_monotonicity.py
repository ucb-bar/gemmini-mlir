"""Explicit finite-domain rounded-DAG monotonicity evidence consumption.

This API binds a caller-supplied exhaustive theorem; hashes do not prove that
theorem. Native/target evaluator equivalence and gradual underflow remain
caller obligations. It never changes non-word enclosures or arbitrary plans.
"""

import hashlib
import struct
from dataclasses import dataclass

from merlin.llvmlower.source_numeric_capability import SourceNumericContract


@dataclass(frozen=True)
class RoundedPolynomialMonotonicity:
    plan_words: tuple[int, ...]
    checked_negative_words: int
    evidence_sha256: str
    evaluator_sha256: str
    header_sha256: str
    gradual_underflow: bool
    complete_domain: bool
    evaluator_equivalent: bool

    def validate(self) -> None:
        if (
            type(self.plan_words) is not tuple
            or len(self.plan_words) != 8
            or any(type(x) is not int or not 0 <= x < 2**32 for x in self.plan_words)
        ):
            raise ValueError("eight source binary32 plan words required")
        cutoff = self.plan_words[0]
        if not 0x80000000 < cutoff < 0xFF800000:
            raise ValueError("finite negative cutoff required")
        if type(self.checked_negative_words) is not int or self.checked_negative_words != cutoff - 0x80000000 + 1:
            raise ValueError("complete cutoff through negative-zero domain required")
        if any(x is not True for x in (self.gradual_underflow, self.complete_domain, self.evaluator_equivalent)):
            raise ValueError("complete evaluator/environment proof required")
        for digest in (self.evidence_sha256, self.evaluator_sha256, self.header_sha256):
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("immutable proof identities required")
        if any(
            not (-float("inf") < struct.unpack("!f", struct.pack("!I", x))[0] < float("inf")) for x in self.plan_words
        ):
            raise ValueError("finite source plan required")


def consume_rounded_polynomial_monotonicity(
    header: str,
    proof: RoundedPolynomialMonotonicity | None,
    contract: SourceNumericContract | None = None,
) -> str:
    """Default byte identity; only prepared word-budget expansion is removed.

    Runtime checks retain the original valid environment, exact plan words and
    zero upper endpoint. Any other plan/domain/rounding mode uses the old budget.
    Source values below cutoff remain the original zero branch.
    """
    if proof is None:
        return header
    proof.validate()
    if not isinstance(contract, SourceNumericContract) or not all(
        x is True
        for x in (
            contract.round_to_nearest_even,
            contract.fused_single_rounding,
            contract.errno_unobserved,
            contract.nontrapping,
            contract.exception_flags_unobserved,
            contract.standard_bitcast_copy,
            contract.copy_interposition_unobserved,
            contract.standard_floor_values,
            contract.floor_interposition_unobserved,
        )
    ):
        raise ValueError("source rounding/effect/floor equivalence contract required")
    if hashlib.sha256(header.encode()).hexdigest() != proof.header_sha256:
        raise ValueError("changed consumer source")
    anchor = "  out.rounding_error=total_error;out.word_budget=(uint32_t)budget;out.fast_valid=1;"
    if header.count(anchor) != 1:
        raise ValueError("unsupported preparation consumer")
    declarations = "\n#include <fenv.h>\n#include <string.h>\n"
    words = ",".join(f"0x{x:08x}u" for x in proof.plan_words)
    check = """
  /* Caller-proved exhaustive rounded source DAG, not real-polynomial alone. */
  if (fegetround()==FE_TONEAREST && upper==0.0f) {
    const uint32_t expected[8]={WORDS};
    const float actual[8]={p->cutoff,p->scale,p->coefficients[0],
      p->coefficients[1],p->coefficients[2],p->coefficients[3],
      p->bit_multiplier,p->bit_bias};
    int same=1;
    for(int j=0;j<8;j++){uint32_t bits;memcpy(&bits,&actual[j],sizeof(bits));
      if(bits!=expected[j])same=0;}
    if(same)out.word_budget=0;
  }
""".replace("WORDS", words)
    return declarations + header.replace(anchor, anchor + check)
