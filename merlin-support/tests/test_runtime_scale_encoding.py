"""Symbolic CONFIG_ST encoding, source identity and mutation boundaries."""

import json
from dataclasses import replace

import pytest

from mlir_oot.core_aten.emit import build_module, encode, instruction_tree, verify_forms
from mlir_oot.core_aten.ops import Kernel
from mlir_oot.core_aten.catalog import _stream_digest
from mlir_oot.core_aten.readout_encoding import config_st, flags, runtime_scale_slot, unpack
from mlir_oot.core_aten.reroll import Rec, iter_expand, reroll
from mlir_oot.core_aten.runtime_scale import (
    RuntimeScaleRef, bind_scale_operand, validate_runtime_record, validate_scale_ref,
)
from mlir_oot.core_aten.stream import ACC, Arg, CompactInstructions, Ins, InstructionLoop, Stream
from mlir_oot.frontend.parse import parse_module
from mlir_oot.tables import isa


def scale_record(index=1):
    return dict(schema='merlin.readout_scale_operand.v1', kind='runtime_input', dtype='f32',
                storage_shape=[1], source=dict(func_arg=index, element_index=0),
                granularity='tensor', domain='finite_f32')


def scalar_source(dtype='f32', extent='1'):
    return parse_module(f'''builtin.module {{ func.func @forward(%a:tensor<1xi8>,
        %first:tensor<{extent}x{dtype}>,%second:tensor<1xf32>) {{ func.return }} }}''')


def runtime_stream():
    return Stream([Arg('out', 'i8', (3, 17), 'output'),
                   Arg('scale0', 'f32', (1,), 'input'), Arg('scale1', 'f32', (1,), 'input')])


def test_runtime_config_rs2_is_symbolic_and_stride_is_static():
    stream = runtime_stream()
    stream.config_st(17, scale_operand=RuntimeScaleRef(1))
    ins, = stream.ins
    rec = encode(ins)
    assert rec == Rec((isa.K_CONFIG, None, RuntimeScaleRef(1, 0)), config_st(17, 0, 0)[1:])
    assert ins.arg is None and ins.off == 0
    assert ins.a('multiplier_bits') is None and ins.a('acc_scale') is None
    assert unpack(rec.values[1], 'ConfigMvoutRs2', 'acc_scale') == 0
    assert unpack(rec.values[1], 'ConfigMvoutRs2', 'stride') == 17
    assert runtime_scale_slot() == (32, 32)
    assert json.loads(json.dumps(rec.key))[2] == [1, 0]
    serialized = json.dumps(dict(scale_operand=scale_record()))
    assert 'multiplier_bits' not in serialized and 'device_scale' not in serialized
    assert verify_forms(stream) == 1


@pytest.mark.parametrize('compact', [False, True])
def test_runtime_scale_roundtrips_through_reroll(compact):
    stream = runtime_stream()
    stream.config_st(17, scale_operand=RuntimeScaleRef(1))
    first = stream.ins[0]
    stream.config_st(17, scale_operand=RuntimeScaleRef(2))
    second = stream.ins[1]
    dma = Ins('mvout', tuple(sorted(dict(local=ACC(0), cols=1, rows=1).items())), 0, 7)
    body = (first, second, dma)
    if compact:
        stream.ins = CompactInstructions([InstructionLoop(8, body, ((0, 17),))])
    else:
        stream.ins = [i._replace(off=i.off + 17 * n) if i.arg is not None else i
                      for n in range(8) for i in body]
    expected = [encode(i) for i in stream.ins]
    assert list(iter_expand(instruction_tree(stream.ins))) == expected
    assert list(iter_expand(reroll(expected))) == expected
    configs = [r for r in expected if len(r.key) == 3]
    assert {r.key[2] for r in configs} == {RuntimeScaleRef(1), RuntimeScaleRef(2)}
    assert [r.values[0] for r in expected if r.key == (isa.K_MVOUT, 0)] == [7 + n * 17 for n in range(8)]


@pytest.mark.parametrize('sources', [dict(scale=1., multiplier_bits=0x3f800000),
    dict(scale=1., scale_operand=RuntimeScaleRef(1)),
    dict(multiplier_bits=0x3f800000, scale_operand=RuntimeScaleRef(1)),
    dict(scale=1., multiplier_bits=0x3f800000, scale_operand=RuntimeScaleRef(1))])
def test_config_st_rejects_ambiguous_scale_sources(sources):
    with pytest.raises(ValueError, match='ambiguous'):
        runtime_stream().config_st(17, **sources)


@pytest.mark.parametrize('arg', [Arg('s', 'f32', (1,), 'const'), Arg('s', 'f32', (1,), 'output'),
    Arg('s', 'f64', (1,), 'input'), Arg('s', 'i32', (1,), 'input'),
    Arg('s', 'f32', (2,), 'input'), Arg('s', 'f32', (True,), 'input'), Arg('s', 'f32', (), 'input')])
def test_runtime_ref_rejects_wrong_argument_abi(arg):
    with pytest.raises(ValueError):
        Stream([arg]).config_st(17, scale_operand=RuntimeScaleRef(0))


@pytest.mark.parametrize('ref', [RuntimeScaleRef(-1), RuntimeScaleRef(3), RuntimeScaleRef(True),
    RuntimeScaleRef(1, 1), RuntimeScaleRef(1, 4), RuntimeScaleRef(1, False), (1, 0)])
def test_runtime_ref_rejects_index_and_byte_extent_mutation(ref):
    with pytest.raises(ValueError):
        runtime_stream().config_st(17, scale_operand=ref)


def test_runtime_scale_rejects_short_declared_extent():
    class ShortArg:
        role, dtype, shape, nbytes = 'input', 'f32', (1,), 3
    with pytest.raises(ValueError, match='four-byte'):
        validate_scale_ref(RuntimeScaleRef(0), [ShortArg()])


@pytest.mark.parametrize('field,value', [('stride', -1), ('stride', 1 << 32), ('stride', True),
    ('acc_act', 2), ('scale_operand', None), ('acc_scale', 1.0), ('multiplier_bits', 0), ('pool_stride', 1), ('unknown', 0)])
def test_runtime_config_rejects_field_mutation(field, value):
    stream = runtime_stream()
    stream.config_st(17, scale_operand=RuntimeScaleRef(1))
    attrs = dict(stream.ins[0].attrs, **{field: value})
    stream.ins[0] = stream.ins[0]._replace(attrs=tuple(sorted(attrs.items())))
    with pytest.raises(ValueError):
        build_module(stream, 'bad_runtime_config')


@pytest.mark.parametrize('word,bit', [(0, 0), (0, 4), (0, 16), (0, 63), (1, 32), (1, 63)])
def test_runtime_record_rejects_scale_and_reserved_bits(word, bit):
    stream = runtime_stream()
    stream.config_st(17, scale_operand=RuntimeScaleRef(1))
    rec = encode(stream.ins[0])
    words = list(rec.values)
    words[word] ^= 1 << bit
    with pytest.raises(ValueError):
        validate_runtime_record(replace(rec, values=tuple(words)), stream.args)


@pytest.mark.parametrize('arg,off', [(0, 0), (None, 4)])
def test_runtime_config_never_uses_dma_fields(arg, off):
    stream = runtime_stream()
    stream.config_st(17, scale_operand=RuntimeScaleRef(1))
    with pytest.raises(ValueError):
        encode(stream.ins[0]._replace(arg=arg, off=off))


def test_runtime_pooled_scale_and_loop_advancement_reject():
    stream = runtime_stream()
    with pytest.raises(ValueError, match='pooled runtime'):
        stream.config_st(17, scale_operand=RuntimeScaleRef(1), pool={})
    stream.config_st(17, scale_operand=RuntimeScaleRef(1))
    stream.ins = CompactInstructions([InstructionLoop(4, tuple(stream.ins), ((1, 4),))])
    with pytest.raises(ValueError, match='advance'):
        build_module(stream, 'bad_loop')


@pytest.mark.parametrize('scale', [1., .25, -.5, -0.])
def test_constant_config_and_readout_encoding_are_unchanged(scale):
    stream = runtime_stream()
    stream.config_st(17, 1, scale)
    f, rs1, rs2 = isa.config_st(stride=17, acc_act=1, acc_scale=scale)
    assert encode(stream.ins[-1]) == Rec((f, None), (rs1, rs2))
    stream.config_st(17, 1, multiplier_bits=isa.f32_bits(scale))
    assert encode(stream.ins[-1]) == Rec((f, None), (rs1, rs2))
    stream.config_st(68)
    assert encode(stream.ins[-1]).values[1] == 0x3f80000000000044
    module, _ = build_module(stream, 'constant_scale')
    assert all(op.name != 'llvm.load' for op in module.walk())
    assert all(op.constraints.data == 'r,r,~{memory}' for op in module.walk()
               if op.name == 'llvm.inline_asm')
    assert 'gemmini.runtime_scale' not in module.attributes


def test_source_scale_binding_uses_ssa_identity_and_first_use_order():
    module = scalar_source()
    operands, args = [], []
    first = bind_scale_operand(scale_record(2), module, operands, args)
    second = bind_scale_operand(scale_record(1), module, operands, args)
    assert (first, second) == (RuntimeScaleRef(0), RuntimeScaleRef(1))
    assert bind_scale_operand(scale_record(2), module, operands, args) == first
    assert [a.name for a in args] == ['readout_scale0', 'readout_scale1']
    assert operands[0][1] is not operands[1][1]
    cloned = module.clone()
    assert bind_scale_operand(scale_record(2), cloned, operands, args) == RuntimeScaleRef(2)
    assert len(args) == 3  # Equal printed names/types in a clone are different SSA.


@pytest.mark.parametrize('mutation', ['index', 'element', 'bool_index', 'bool_element', 'shape',
    'bool_shape', 'dtype', 'domain', 'extra', 'source_extra'])
def test_source_scale_binding_rejects_record_mutation(mutation):
    record = scale_record()
    if mutation == 'index': record['source']['func_arg'] = 7
    if mutation == 'element': record['source']['element_index'] = 1
    if mutation == 'bool_index': record['source']['func_arg'] = True
    if mutation == 'bool_element': record['source']['element_index'] = False
    if mutation == 'shape': record['storage_shape'] = [2]
    if mutation == 'bool_shape': record['storage_shape'] = [True]
    if mutation == 'dtype': record['dtype'] = 'f64'
    if mutation == 'domain': record['domain'] = 'any'
    if mutation == 'extra': record['unknown'] = 0
    if mutation == 'source_extra': record['source']['unknown'] = 0
    with pytest.raises(ValueError):
        bind_scale_operand(record, scalar_source(), [], [])


@pytest.mark.parametrize('dtype,extent', [('f64', '1'), ('f16', '1'), ('i32', '1'), ('f32', '2')])
def test_source_scale_binding_checks_actual_type(dtype, extent):
    with pytest.raises(ValueError):
        bind_scale_operand(scale_record(), scalar_source(dtype, extent), [], [])


def test_existing_scale_binding_is_revalidated():
    module = scalar_source()
    value = next(op for op in module.walk() if op.name == 'func.func').body.blocks[0].args[1]
    with pytest.raises(ValueError):
        bind_scale_operand(scale_record(), module, [('s', value)], [Arg('s','i32',(1,),'input')])
    with pytest.raises(ValueError):
        bind_scale_operand(scale_record(), module, [('s', value), ('t', value)], [])


def test_runtime_reference_mutation_changes_digest():
    stream = runtime_stream()
    stream.config_st(17, scale_operand=RuntimeScaleRef(1))
    kernel = Kernel('runtime', stream)
    original = _stream_digest(kernel)
    ins = stream.ins[0]
    stream.ins[0] = ins._replace(attrs=tuple(sorted(dict(ins.attrs, scale_operand=RuntimeScaleRef(2)).items())))
    assert _stream_digest(kernel) != original
    # Compact digests authenticate even invalid offsets before admission rejects them.
    stream.ins = CompactInstructions([ins])
    original = _stream_digest(kernel)
    stream.ins.nodes[0] = ins._replace(attrs=tuple(sorted(dict(ins.attrs, scale_operand=RuntimeScaleRef(1, 4)).items())))
    assert _stream_digest(kernel) != original
    with pytest.raises(ValueError):
        verify_forms(stream)


def test_rerolled_runtime_configs_keep_distinct_static_strides():
    stream = runtime_stream()
    for stride in range(17, 25):
        stream.config_st(stride, scale_operand=RuntimeScaleRef(1))
    module, tree = build_module(stream, 'affine_static_strides')
    assert list(iter_expand(tree)) == [encode(ins) for ins in stream.ins]
    assert any(op.name == 'llvm.load' for op in module.walk())


@pytest.mark.parametrize('relu', [False, True])
def test_runtime_lint_records_activation_order_obligation_and_keeps_dma_hazards(relu):
    from mlir_oot.core_aten.lint import HazardError, lint

    stream = runtime_stream()
    stream.config_st(17, int(relu), scale_operand=RuntimeScaleRef(1))
    stream.mvout(stream.at('out'), ACC(0), 1, 1)
    report = lint(stream)
    expected = [dict(instruction_index=0, scale_operand=[1, 0],
                     requires='relu_scale_order_parity_or_nonnegative_test_input')] if relu else []
    assert report.get('runtime_oracle_obligations', []) == expected
    stream.mvout(stream.at('out'), ACC(0), 1, 1)
    with pytest.raises(HazardError, match='DRAM WAW'):
        lint(stream)
    stream = runtime_stream()
    stream.config_st(17, 1, -.5)
    with pytest.raises(HazardError, match='C7'):
        lint(stream)


@pytest.mark.parametrize('mutation', ['reference', 'stride', 'missing_reference'])
def test_runtime_lint_rejects_invalid_config(mutation):
    from mlir_oot.core_aten.lint import lint

    stream = runtime_stream()
    stream.config_st(17, 1, scale_operand=RuntimeScaleRef(1))
    attrs = dict(stream.ins[0].attrs)
    if mutation == 'reference':
        attrs['scale_operand'] = RuntimeScaleRef(0)
    elif mutation == 'stride':
        attrs['stride'] = 1 << 32
    else:
        attrs['scale_operand'] = None
    stream.ins[0] = stream.ins[0]._replace(attrs=tuple(sorted(attrs.items())))
    with pytest.raises(ValueError):
        lint(stream)


def selected_runtime_facts(monkeypatch, mutation):
    from pathlib import Path
    from mlir_oot.core_aten import readout_encoding

    facts = json.loads((Path(__file__).parent / 'fixtures/runtime_scale/rtl-facts.json').read_text())
    bundle = next(row for row in facts['facts']['interfaces']
                  if row['name'] == 'register_bundle_layouts')['bundles']['ConfigMvoutRs2']
    fields = bundle['fields']
    if mutation == 'relocate':
        fields['acc_scale']['offset'], fields['stride']['offset'] = 0, 32
    elif mutation == 'overlap':
        fields['acc_scale']['offset'] = 16
    elif mutation == 'outside':
        fields['acc_scale']['offset'] = 64
    elif mutation == 'narrow_bundle':
        bundle['width'] = 63
    monkeypatch.setattr(readout_encoding, 'load_facts', lambda target: facts)
    return facts


def test_runtime_scale_transport_follows_relocated_slot(monkeypatch):
    from mlir_oot.core_aten.readout_encoding import runtime_scale_facts
    from test_runtime_scale_emission import PATH

    facts = selected_runtime_facts(monkeypatch, 'relocate')
    assert runtime_scale_slot() == (0, 32)
    transport = runtime_scale_facts(facts, PATH)
    assert (transport['offset'], transport['width']) == (0, 32)
    s = runtime_stream()
    s.config_st(17, scale_operand=RuntimeScaleRef(1))
    assert encode(s.ins[0]).values[1] == 17 << 32
    validate_runtime_record(encode(s.ins[0]), s.args)


@pytest.mark.parametrize('mutation,reason', [('overlap', 'overlaps'), ('outside', 'inside'),
                                           ('narrow_bundle', 'inside')])
def test_runtime_scale_transport_refuses_invalid_slot(monkeypatch, mutation, reason):
    from mlir_oot.core_aten.readout_encoding import runtime_scale_facts
    from test_runtime_scale_emission import PATH

    facts = selected_runtime_facts(monkeypatch, mutation)
    with pytest.raises(ValueError, match=reason):
        runtime_scale_slot()
    with pytest.raises(ValueError, match=reason):
        runtime_scale_facts(facts, PATH)
