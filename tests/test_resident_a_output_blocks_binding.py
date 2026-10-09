"""Separate complete-A lifetime survives normal source-bound input borrowing."""
from dataclasses import asdict
import hashlib

import pytest
from xdsl.dialects.builtin import StringAttr

from mlir_oot.captured_requant_bundle import build, scalar_oracle
from mlir_oot.dense_schedule import select_resident_a_output_blocks
from mlir_oot.frontend.parse import parse_module
from mlir_oot.golden_gemm import GoldenGemm, Shape
from mlir_oot.segmented_input_binding import adapter_source, derive
from test_segmented_input_binding import SOURCE, fixture


def test_actual_selected_source_target_hash_and_segmented_adapter_are_preserved():
    original, _, route, _ = fixture()
    source = SOURCE
    for before, after in (
        ('1x3x5x2', '1x47x47x2'), ('1x2x2x2', '1x23x23x2'),
        ('1,2,2,2', '1,23,23,2'), ('8xi8', '1058xi8'),
        ('4x2xi8', '529x2xi8'), ('4,2>', '529,2>'),
        ('4x3xi8', '529x67xi8'), ('2x3xi8', '2x67xi8'),
    ):
        source = source.replace(before, after)
    module = parse_module(source)
    declarations = {op.sym_name.data: op for op in original.body.block.ops if op.name == 'func.func'}
    for op in module.body.block.ops:
        if op.name == 'func.func' and not op.body.blocks:
            op.properties['arg_attrs'] = declarations[op.sym_name.data].properties['arg_attrs']
    control = GoldenGemm(Shape(529, 67, 2, bm=34, bn=1, cache_a=True, reuse_b=True,
                               wide_b=True, wide_store=True, scale=.25))
    generator, decision = select_resident_a_output_blocks(control)
    assert decision['applied'] and generator.shape.bm == 16 and generator.shape.bn == 4
    emitted = generator.build()
    emitted.body.block.first_op.properties['sym_name'] = StringAttr(route['kernel'])
    route['schedule'] = asdict(generator.shape)
    route['dense_resident_output_block_decision'] = decision
    route['compilation']['target_ir_sha256'] = hashlib.sha256((str(emitted) + '\n').encode()).hexdigest()
    bindings, refused = derive(module, [route])
    assert len(bindings) == 1 and not refused
    binding = bindings[0]
    assert binding.generator.cached_a_output_blocks
    assert binding.generator.shape == generator.shape
    assert binding.contract.address.rows == 529
    adapter_source(binding.generator.shape, binding.contract.accepted_symbol, 'bound_kernel',
                   binding.contract.address, binding.owner_shape, cached_a_output_blocks=True)
    scalar_oracle(binding.generator.shape, 'bound_kernel', False,
                  input_view=binding.contract.address, cached_a_output_blocks=True)
    route['dense_resident_output_block_decision'] = dict(decision, applied=False)
    bindings, refused = derive(module, [route])
    assert not bindings and refused


@pytest.mark.parametrize('channels', [False, 0, 5, 1.5])
def test_normal_option_validation_refuses_before_output_creation(tmp_path, channels):
    output = tmp_path / 'not_created'
    with pytest.raises(ValueError, match='channel tiles'):
        build(tmp_path / 'absent', tmp_path / 'absent', output,
              dense_resident_output_channel_tiles=channels)
    assert not output.exists()
