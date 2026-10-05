from pathlib import Path
import pytest
from mlir_oot.late_quant_rne import rewrite
SOURCE=Path(__file__).parent.joinpath('fixtures/bounded_quant_rne.ll').read_text()

def test_complete_chain_and_other_uses_are_preserved():
    source=SOURCE.replace('  ret i8 %v17','  %other = add i8 %v3, 2\n  ret i8 %v17')
    new,proof=rewrite(source)
    assert len(proof['routes'])==1
    assert 'fcvt.w.s $0, $1, rne' in new
    assert '%other = add i8 %v3, 2' in new
    assert '%v3 = fptosi' in new

@pytest.mark.parametrize('old,new',[
 ('-1.280000e+02','-1.290000e+02'),('1.270000e+02','1.280000e+02'),
 ('and i8 %v3, 1','and i8 %v3, 2'),('5.000000e-01','4.000000e-01'),
 ('fcmp olt','fcmp ole'),('i8 -1, i8 1','i8 1, i8 -1'),
 ('fsub float %v2, %v4','fsub float %x, %v4'),
 ('@llvm.minimum.f32','@llvm.minnum.f32')])
def test_refuse_unproved_chain(old,new):
    source=SOURCE.replace(old,new)
    assert source!=SOURCE
    rewritten,proof=rewrite(source)
    assert not proof['routes'] and rewritten==source

def test_function_scopes_are_independent():
    source=SOURCE+SOURCE[:SOURCE.index('declare')].replace('@quant','@quant2')
    rewritten,proof=rewrite(source)
    assert len(proof['routes'])==2


@pytest.mark.parametrize('marker',['attributes #0 = { strictfp }','declare float @llvm.experimental.constrained.roundeven.f32(float, metadata)'])
def test_strict_fp_refuses(marker):
    source=SOURCE+'\n'+marker+'\n'
    new,proof=rewrite(source)
    assert new==source and not proof['routes'] and proof['refusal']


def test_native_preserves_existing_intrinsic_declaration_attributes():
    source=SOURCE+'declare float @llvm.roundeven.f32(float) #0\nattributes #0 = { nounwind }\n'
    new,proof=rewrite(source,native_oracle=True)
    assert len(proof['routes'])==1
    assert new.count('declare float @llvm.roundeven.f32')==1
    assert 'declare float @llvm.roundeven.f32(float) #0' in new
