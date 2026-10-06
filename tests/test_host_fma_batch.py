"""Target capability must retain the explicit generic source obligations."""

from dataclasses import replace

import pytest

from merlin.llvmlower.source_fma_batch import SourceFmaBatchContract
from mlir_oot.host_fma_batch import SourceFmaBatchCapability


CONTRACT = SourceFmaBatchContract(8, *([True] * 9))
CAPABILITY = SourceFmaBatchCapability('rv64gc', 'lp64d', CONTRACT)


@pytest.mark.parametrize('change', [
    {'isa': 'unknown'}, {'abi': 'unknown'}, {'contract': None},
    {'contract': replace(CONTRACT, finite_operands_and_results=False)},
    {'contract': replace(CONTRACT, exception_flags_unobserved=False)},
    {'contract': replace(CONTRACT, operand_order_preserved=False)},
    {'contract': replace(CONTRACT, private_disjoint_storage=False)},
])
def test_unsupported_target_or_source_proof_refuses(change):
    with pytest.raises(ValueError):
        replace(CAPABILITY, **change).header()
