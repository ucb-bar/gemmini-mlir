from pathlib import Path
import pytest
from mlir_oot.direct_conv_bundle import build
from mlir_oot.frontend.parse import parse_module


def test_empty_direct_composition_is_explicit(tmp_path):
    llvm=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
    if not (llvm/'clang').is_file():pytest.skip('target compiler unavailable')
    source=tmp_path/'source.mlir';source.write_text('builtin.module { func.func @identity(%a: tensor<4xi8>) -> tensor<4xi8> {func.return %a : tensor<4xi8>} }')
    with pytest.raises(ValueError,match='no structurally proven'):build(source,llvm,tmp_path/'default')
    result=build(source,llvm,tmp_path/'composed',allow_empty=True)
    assert result['routes']==[] and result['nofsm_audit']['status']=='pass'
    parse_module((tmp_path/'composed/rewritten.mlir').read_text()).verify()
