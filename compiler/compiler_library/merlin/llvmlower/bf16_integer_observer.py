"""Exact BF16 ties-even integer observations under an explicit pure source contract."""

from dataclasses import dataclass

from merlin.common.paths import data_path


@dataclass(frozen=True)
class BF16IntegerObserverContract:
    stable_rne: bool
    nontrapping: bool
    exception_flags_unobserved: bool
    integer_value_only_observation: bool
    pure_quantizer_no_library_effects: bool
    standard_bitcast_copy: bool
    copy_interposition_unobserved: bool

    def validate(self) -> None:
        if any(type(value) is not bool or not value for value in vars(self).values()):
            raise ValueError("complete pure BF16 integer-observation contract required")


def c_header(contract: BF16IntegerObserverContract) -> str:
    """Specialize the owned quantizer; keep its complete scale/product DAG.

    Source math.roundeven is pure. This permission does not remove effects from
    an arbitrary interposed nearbyintf call. The only observed result is the
    final clamped integer, with bounds inside [-128,127].
    """
    if not isinstance(contract, BF16IntegerObserverContract):
        raise ValueError("typed pure BF16 integer-observation contract required")
    contract.validate()
    source = (data_path("runtime", "c") / "bf16_quant_frontier.h").read_text()
    original = """ float product=merlin_frontier_bf16(x*inverse);
 /* nearbyintf implements source ties-even under the admitted RNE mode. */
 float rounded=merlin_frontier_bf16(nearbyintf(product));
 rounded=merlin_frontier_bf16(rounded+0.0f);
 return (int)MERLIN_SOURCE_F32_MAX((float)p.lower,MERLIN_SOURCE_F32_MIN((float)p.upper,rounded));"""
    helper = r"""/* All finite BF16 values in the unsaturated domain have exponent<=6.
 * Their rounded integers lie in [-128,128] and are exactly BF16. Subsequent
 * BF16 rounding and +0 have the same integer observation. Values outside
 * the domain saturate; standard min/max maps a NaN to the upper clamp. */
static inline int merlin_bf16_rne_clamped_integer_word(uint16_t word,int lower,int upper){
 unsigned magnitude=word&0x7fffu,biased=magnitude>>7;
 int value;
 if(magnitude>0x7f80u)return upper;
 if(biased>=134u)return word&0x8000u?lower:upper;
 if(biased<126u)value=0;
 else{
  unsigned significand=(magnitude&127u)|128u,rounded;
  if(biased==126u)rounded=significand>128u;
  else{
   unsigned shift=134u-biased,integral=significand>>shift;
   unsigned remainder=significand&((1u<<shift)-1u),half=1u<<(shift-1u);
   rounded=integral+(remainder>half||(remainder==half&&(integral&1u)));
  }
  value=word&0x8000u?-(int)rounded:(int)rounded;
 }
 return value<lower?lower:value>upper?upper:value;
}
"""
    replacement = """ float product=merlin_frontier_bf16(x*inverse);
 uint32_t bits;MERLIN_SOURCE_BITCAST_COPY(&bits,&product,sizeof(bits));
 return merlin_bf16_rne_clamped_integer_word((uint16_t)(bits>>16),p.lower,p.upper);"""
    anchor = "typedef struct { float divisor, epsilon; int lower, upper; } merlin_bf16_quant_plan;"
    if source.count(original) != 1 or source.count(anchor) != 1:
        raise ValueError("owned complete BF16 quantizer source changed; refused")
    return source.replace(anchor, helper + anchor).replace(original, replacement)


def prepare_bf16_integer_observer(source: str, *, contract: BF16IntegerObserverContract) -> str:
    """Insert before the original scale/observer users and optional point rows.

    Apply this after point-row specialization, if selected. Its copied owned
    runtime header preserves the exact original row refusal/publication rules.
    """
    include = '#include "bf16_quant_frontier.h"'
    if not source.count(include) or "merlin_bf16_rne_clamped_integer_word" in source:
        raise ValueError("owned BF16 observer include changed; refused")
    # Repeated includes share the authoritative header guard. The first owned
    # include defines the chosen implementation for every later observer user.
    return source.replace(include, c_header(contract), 1)
