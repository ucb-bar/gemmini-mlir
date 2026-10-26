"""The same semantic backstop protects every catalog compiler and source kind."""

import importlib
import json
from fractions import Fraction

import numpy as np
import pytest

import test_composition_readout as readout


def _linear_source():
    return '''builtin.module { func.func @forward(%x:tensor<256xi8>)->tensor<256xi8> {
      %s=arith.constant dense<1.0>:tensor<f32>
      %z=arith.constant dense<0>:tensor<i64>
      %d="quant_ext.dequantize_per_tensor"(%x,%s,%z)<{quant_min=-128:i64,quant_max=127:i64}>:
        (tensor<256xi8>,tensor<f32>,tensor<i64>)->tensor<256xf32>
      %e=tensor.empty():tensor<256xf32>
      %v=linalg.generic {indexing_maps=[affine_map<(d)->(d)>,affine_map<(d)->(d)>],iterator_types=["parallel"]}
        ins(%d:tensor<256xf32>) outs(%e:tensor<256xf32>) {
        ^bb0(%a:f32,%o:f32): %two=arith.constant 2.0:f32 %m=arith.mulf %a,%two:f32
        linalg.yield %m:f32 } ->tensor<256xf32>
      %q="quant_ext.quantize_per_tensor"(%v,%s,%z)<{quant_min=-128:i64,quant_max=127:i64,output_dtype="int8"}>:
        (tensor<256xf32>,tensor<f32>,tensor<i64>)->tensor<256xi8>
      func.return %q:tensor<256xi8> } }'''


@pytest.mark.parametrize('mutation', ['multipliers', 'constants', 'stride', 'rare_readout', 'rare_unsigned_readout', 'lane_readout'])
def test_closed_linear_arithmetic_mutations_never_compile(monkeypatch, tmp_path, mutation):
    api = readout.provider()
    catalog = api._core_aten_module(readout.PACKAGE, 'catalog')
    original = catalog.match_all
    source = _linear_source()
    if mutation.startswith('rare_'):
        source = source.replace('256x', '1x')
    if mutation == 'rare_unsigned_readout':
        source = (source.replace('quant_min=-128:i64,quant_max=127:i64', 'quant_min=0:i64,quant_max=255:i64')
                  .replace('output_dtype="int8"', 'output_dtype="uint8"')
                  .replace('(%x,%s,%z)<{quant_min=0:i64,quant_max=255:i64}>',
                           '(%x,%s,%z)<{quant_min=0:i64,quant_max=255:i64,input_dtype="uint8"}>'))
    assert api.admit_numeric_schedule(source, package=readout.PACKAGE, numeric_policy=readout.POLICY)['status'] == 'proven'

    def corrupt(*args, **kwargs):
        matches, declines = original(*args, **kwargs)
        for match in matches:
            stream = match.kernel.stream
            if mutation.startswith('rare_'):
                for host in stream.readouts:
                    host.terms[:] = [(name, np.asarray(factor, object) * Fraction(251, 250))
                                     for name, factor in host.terms]
            elif mutation == 'lane_readout':
                for host in stream.readouts:
                    terms = []
                    for name, factor in host.terms:
                        lanes = np.broadcast_to(np.asarray(factor, object), (256,)).copy()
                        lanes[0] *= Fraction(251, 250)
                        terms.append((name, lanes))
                    host.terms[:] = terms
            elif mutation == 'multipliers':
                for host in stream.readouts:
                    host.terms[:] = [(name, Fraction(0)) for name, _ in host.terms]
            elif mutation == 'constants':
                for data in stream.consts.values():
                    data[...] = 0
            else:
                stream.ins = [i._replace(attrs=tuple(sorted({**dict(i.attrs), 'stride': 0}.items())))
                              if i.kind == 'config_ld' else i for i in stream.ins]
        return matches, declines

    monkeypatch.setattr(catalog, 'match_all', corrupt)
    proof = api.admit_numeric_schedule(source, package=readout.PACKAGE, numeric_policy=readout.POLICY)
    assert proof['status'] == 'declined' and 'semantic gate' in proof['reason']
    compiled = []
    monkeypatch.setattr(catalog, 'compile_to_object', lambda *a, **k: compiled.append(True))
    prepared = tmp_path / 'model.mlir'
    prepared.write_text(source)
    manifest, obj = api.core_aten_catalog(prepared, tmp_path / 'out', package=readout.PACKAGE,
                                        numeric_policy=readout.POLICY)
    assert obj is None and not compiled
    assert json.loads(manifest.read_text())['status'] == 'declined'


def _case(kind):
    if kind == 'linear':
        return _linear_source(), None
    if kind == 'movement':
        return '''builtin.module { func.func @forward(%x:tensor<17xi8>)->tensor<17xi8> {
          %y=tensor.cast %x:tensor<17xi8> to tensor<17xi8> func.return %y:tensor<17xi8> } }''', None
    if kind == 'limb':
        import importlib.util
        import sys
        sys.path.insert(0, str(readout.PACKAGE / 'tests/core_aten'))
        path = readout.PACKAGE / 'tests/core_aten/test_limbs.py'
        spec = importlib.util.spec_from_file_location('semantic_gate_limb_fixture', path)
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        return fixture.annotate(fixture.linear_source(), indices=(1,), exponents=(-8,)), None
    if kind in ('os', 'k_concat'):
        from test_composition_schedules import source, options, weights_of
        return source(3, 17, 5, weights=weights_of(17, 5), coefficient=3 if kind == 'k_concat' else None)[0], options()
    if kind == 'shared':
        from test_shared_input_fusion import source, options
        return source(5, 16, (17, 19), constant=True)[0], options()
    if kind == 'operand_sum':
        from test_scaled_operand_sum import staged, options, GRID
        return staged(17, 20, 'per_operand', GRID[:2], GRID[2:])[0], options()
    if kind == 'pooled':
        from test_pooled_store import fused_source, options
        return fused_source(n=1, h=4, w=4, k=8, c=5)[0], options()
    return readout.source(raw=kind == 'raw')[0], readout.options()


@pytest.mark.parametrize('kind', ['linear', 'movement', 'limb', 'raw', 'readout', 'shared', 'operand_sum', 'pooled', 'os', 'k_concat'])
def test_every_kind_reaches_the_same_gate(monkeypatch, tmp_path, kind):
    api = readout.provider()
    catalog = api._core_aten_module(readout.PACKAGE, 'catalog')
    gate = api._core_aten_module(readout.PACKAGE, 'semantic_gate')
    source, options = _case(kind)
    policy = readout.POLICY
    if kind in ('os', 'k_concat'):
        from test_composition_schedules import COMPOSED
        policy = COMPOSED
    proof = api.admit_numeric_schedule(source, package=readout.PACKAGE, numeric_policy=policy,
                                       readout_options=options)
    assert proof['status'] == 'proven', proof
    assert all(row['trials'] >= 6 for row in proof['semantic_gate'])
    original = gate.execute

    def wrong_execution(stream, buffers, **kwargs):
        original(stream, buffers, **kwargs)
        output = next(i for i, arg in enumerate(stream.args) if arg.role == 'output')
        buffers[output].flat[0] ^= 1

    monkeypatch.setattr(gate, 'execute', wrong_execution)
    refused = api.admit_numeric_schedule(source, package=readout.PACKAGE, numeric_policy=policy,
                                         readout_options=options)
    assert refused['status'] == 'declined' and 'semantic gate' in refused['reason']
    # A valid older proof must not bypass execution at compilation time.
    compiled = []
    monkeypatch.setattr(catalog, 'compile_to_object', lambda *a, **k: compiled.append(True))
    prepared = tmp_path / 'model.mlir'
    prepared.write_text(source)
    manifest = catalog.compile_catalog(prepared, tmp_path / 'out', numeric_schedule_proof=proof,
                                        readout_options=options)
    assert manifest['status'] == 'declined' and 'semantic gate' in manifest['reason'] and not compiled


@pytest.mark.parametrize('error', [NotImplementedError, ZeroDivisionError])
def test_unsupported_or_undefined_source_evaluation_refuses(monkeypatch, error):
    from mlir_oot.core_aten import catalog, semantic_gate
    _, manifest, matches = catalog.build(_linear_source())
    monkeypatch.setattr(semantic_gate.SourceEvaluator, '_eval_op',
                        lambda *args: (_ for _ in ()).throw(error('no source evaluator')))
    with pytest.raises(ValueError, match='no source evaluator'):
        semantic_gate.certify(matches, manifest['bindings'])


def test_legacy_compiler_requires_proof_and_rechecks_semantics(monkeypatch, tmp_path):
    api = readout.provider()
    catalog = api._core_aten_module(readout.PACKAGE, 'catalog')
    backend = importlib.import_module(catalog.__package__.rsplit('.', 1)[0] + '.golden_device_catalog')
    gate = api._core_aten_module(readout.PACKAGE, 'semantic_gate')
    # A source whose region ABI is raw integer also has a legacy ABI.
    source = readout.source(m=4, k=6, n=5, raw=True)[0]
    policy = json.loads(json.dumps(readout.POLICY))
    policy['numerical_semantics'].pop('readout')
    proof = api.admit_numeric_schedule(source, package=readout.PACKAGE, numeric_policy=policy)
    assert proof['status'] == 'proven', proof
    prepared = tmp_path / 'model.mlir'
    prepared.write_text(source)
    compiled = []
    monkeypatch.setattr(backend, 'compile_module', lambda *a: compiled.append(True))
    with pytest.raises(ValueError, match='requires'):
        backend.compile_catalog(prepared, tmp_path, tmp_path / 'out')
    original = gate.execute

    def wrong(stream, buffers, **kwargs):
        original(stream, buffers, **kwargs)
        buffers[2].flat[0] ^= 1

    monkeypatch.setattr(gate, 'execute', wrong)
    with pytest.raises(ValueError, match='semantic gate'):
        backend.compile_catalog(prepared, tmp_path, tmp_path / 'out', numeric_schedule_proof=proof)
    assert not compiled


@pytest.mark.parametrize('bitwise', [False, True])
@pytest.mark.parametrize('size', [1, 2, 3, 5, 17])
def test_boundary_values_span_trials_even_for_tiny_tensors(size, bitwise):
    from mlir_oot.core_aten import semantic_gate as gate
    from mlir_oot.core_aten.stream import Arg, Stream
    stream = Stream([Arg('x', 'i8', (size,), 'input')], bitwise=bitwise)
    rng = np.random.default_rng(0x5eed)
    seen = set()
    for trial in range(gate.TRIALS + gate._boundary_trials(stream) - 1):
        _, inputs = gate._buffers(stream, {'x': (-128, 127)}, trial, rng)
        seen.update(inputs['x'].flat)
    assert {-128, 127, 0, -1, 1} <= seen


def _pointwise_match(bits, size, inputs, body=None):
    from types import SimpleNamespace
    from mlir_oot.frontend.parse import parse_module
    from mlir_oot.core_aten.stream import Arg, Stream
    names = [f'x{i}' for i in range(inputs)]
    tensor = f'tensor<{size}xi{bits}>'
    args = ','.join(f'%{n}:{tensor}' for n in names)
    maps = ','.join(['affine_map<(d)->(d)>'] * (inputs + 1))
    source = f'''builtin.module {{ func.func @forward({args})->{tensor} {{
      %e=tensor.empty():{tensor}
      %y=linalg.generic {{indexing_maps=[{maps}],iterator_types=["parallel"]}}
        ins({','.join('%'+n for n in names)}:{','.join([tensor]*inputs)}) outs(%e:{tensor}) {{
        ^bb0({','.join('%a'+str(i)+':i'+str(bits) for i in range(inputs))},%o:i{bits}):
        {body or f'linalg.yield %a0:i{bits}'} }} ->{tensor}
      func.return %y:{tensor} }} }}'''
    module = parse_module(source)
    fn = next(op for op in module.walk() if op.name == 'func.func')
    result = next(op for op in module.walk() if op.name == 'func.return').operands[0]
    stream = Stream([Arg(n, f'i{bits}', (size,), 'input') for n in names])
    return SimpleNamespace(root=module, operands=list(zip(names, fn.body.blocks[0].args)),
                           result=('y', result), kernel=SimpleNamespace(stream=stream))


@pytest.mark.parametrize('bits,size,inputs', [(8, 1, 1), (8, 17, 1), (8, 3, 2), (16, 3, 1), (1, 1, 2)])
def test_elementwise_trials_cover_entire_joint_storage_domain(bits, size, inputs):
    from itertools import product
    from mlir_oot.core_aten import semantic_gate as gate
    match = _pointwise_match(bits, size, inputs)
    ranges = {n: (0, 1) if bits == 1 else (-(1 << (bits-1)), (1 << (bits-1))-1)
              for n, _ in match.operands}
    trials = gate._trial_buffers(match, ranges)
    finite = gate.TRIALS + gate._boundary_trials(match.kernel.stream) - 1
    seen = [set() for _ in range(size)]
    count = 0
    for trial, (_, values) in enumerate(trials):
        if trial < finite:
            continue
        count += 1
        for lane, point in enumerate(zip(*(values[n].flat for n, _ in match.operands))):
            seen[lane].add(point)
    assert count == 1 << (bits * inputs)
    low, high = next(iter(ranges.values()))
    expected = set(product(range(low, high + 1), repeat=inputs))
    assert all(codes == expected for codes in seen)


@pytest.mark.parametrize('kind', ['raw', 'pooled', 'movement'])
def test_non_elementwise_sources_do_not_use_packed_exhaustion(kind):
    from mlir_oot.core_aten import catalog, semantic_gate as gate
    source, options = _case(kind)
    _, _, matches = catalog.build(source, readout_options=options)
    assert matches
    assert all(gate._elementwise_domain(match) is None for match in matches)


@pytest.mark.parametrize('kind', ['linear', 'limb', 'operand_sum'])
def test_elementwise_source_kinds_use_exhaustion(kind):
    from mlir_oot.core_aten import catalog, semantic_gate as gate
    source, options = _case(kind)
    _, _, matches = catalog.build(source, readout_options=options)
    assert matches
    assert any(gate._elementwise_domain(match) is not None for match in matches)


@pytest.mark.parametrize('mutation', ['load_config_mutation', 'truncated_store'])
def test_joint_domain_packing_requires_a_decoded_stream_proof(monkeypatch, mutation):
    from mlir_oot.core_aten import catalog, semantic_gate as gate
    from test_scaled_operand_sum import _tamper
    source, options = _case('operand_sum')
    _, manifest, matches = catalog.build(source, readout_options=options)
    for match in matches:
        match.kernel.stream.ins = _tamper(match.kernel.stream, mutation)

    def unexpected_execution(*args, **kwargs):
        pytest.fail('unproved joint-domain stream reached concrete execution')

    monkeypatch.setattr(gate, 'execute', unexpected_execution)
    with pytest.raises(ValueError):
        gate.certify(matches, manifest['bindings'], readout_options=options)


@pytest.mark.parametrize('mutation', [None, 'readout_lane', 'load_offset', 'store_stride', 'nonzero_seed'])
def test_dma_uniformity_checks_readout_parameters_and_decoded_lane_indices(mutation):
    from mlir_oot.core_aten import catalog, semantic_gate as gate
    from test_scaled_operand_sum import dequantized
    source = dequantized(sa=0.0250686146, sb=0.0202060752, so=0.0302009135)
    _, _, matches = catalog.build(source)
    match, = matches
    stream = match.kernel.stream
    assert gate._uniform_dma_readout(stream)
    if mutation == 'readout_lane':
        stream.readouts[0].terms[0][1].flat[0] *= Fraction(251, 250)
    elif mutation == 'load_offset':
        first = next(i for i, ins in enumerate(stream.ins) if ins.kind == 'mvin' and ins.arg is not None)
        stream.ins[first] = stream.ins[first]._replace(off=stream.ins[first].off + 1)
    elif mutation == 'store_stride':
        stream.ins = [ins._replace(attrs=tuple(sorted({**dict(ins.attrs), 'stride': 0}.items())))
                      if ins.kind == 'config_st' else ins for ins in stream.ins]
    elif mutation == 'nonzero_seed':
        stream.const('nonzero', np.ones(16, np.int32))
        first = next(i for i, ins in enumerate(stream.ins) if ins.kind == 'mvin' and ins.arg is None)
        stream.ins[first] = stream.ins[first]._replace(arg=stream._index['nonzero'],
            attrs=tuple(sorted({**dict(stream.ins[first].attrs), 'zero': 0}.items())))
    assert gate._uniform_dma_readout(stream) == (mutation is None)


def test_domain_budget_and_index_dependent_body_do_not_claim_elementwise():
    from mlir_oot.core_aten import semantic_gate as gate
    assert gate._elementwise_domain(_pointwise_match(8, 17, 3)) is None
    match = _pointwise_match(8, 17, 1)
    from xdsl.dialects.linalg.ops import IndexOp
    body = next(op for op in match.root.walk() if op.name == 'linalg.generic').body.blocks[0]
    body.insert_op_before(IndexOp(0), body.last_op)
    assert gate._elementwise_domain(match) is None


def test_narrow_integer_output_buffers_can_be_poisoned():
    from mlir_oot.core_aten import semantic_gate as gate
    from mlir_oot.core_aten.stream import Arg, Stream
    stream = Stream([Arg('out', 'i16', (1,), 'output')])
    buffers, _ = gate._buffers(stream, {}, 0, np.random.default_rng(0))
    assert buffers[0].tobytes() == b'\x5a\x5a'


def test_scalar_control_flow_is_pointwise_but_nested_indices_are_not():
    from mlir_oot.core_aten import semantic_gate as gate
    from xdsl.dialects.linalg.ops import IndexOp
    match = _pointwise_match(8, 17, 1, '''%c=arith.constant true
      %v=scf.if %c -> (i8) { scf.yield %a0:i8 } else { scf.yield %a0:i8 }
      linalg.yield %v:i8''')
    assert gate._elementwise_domain(match) is not None
    body = next(op for op in match.root.walk() if op.name == 'scf.if').regions[0].blocks[0]
    body.insert_op_before(IndexOp(0), body.last_op)
    assert gate._elementwise_domain(match) is None


@pytest.mark.parametrize('low,high', [(-127, 127), (1, 254)])
def test_declared_ranges_do_not_narrow_runtime_codes(low, high):
    from mlir_oot.core_aten import catalog, semantic_gate as gate
    source = _linear_source().replace('quant_min=-128:i64,quant_max=127:i64',
                                      f'quant_min={low}:i64,quant_max={high}:i64')
    if low >= 0:
        source = (source.replace('output_dtype="int8"', 'output_dtype="uint8"')
                  .replace('dense<0>:tensor<i64>', 'dense<128>:tensor<i64>')
                  .replace(f'(%x,%s,%z)<{{quant_min={low}:i64,quant_max={high}:i64}}>',
                           f'(%x,%s,%z)<{{quant_min={low}:i64,quant_max={high}:i64,input_dtype="uint8"}}>'))
    _, _, matches = catalog.build(source)
    match, = matches
    name, = [n for n, _ in match.operands]
    ranges = gate._input_ranges(match, {})
    assert ranges[name] == (-128, 127)
    seen = [set() for _ in range(256)]
    for _, inputs in gate._trial_buffers(match, ranges):
        for codes, value in zip(seen, inputs[name].flat):
            codes.add(int(value))
    assert all(codes == set(range(-128, 128)) for codes in seen)


def test_narrowed_qdq_copy_refuses_raw_storage_code_minus_128(monkeypatch, tmp_path):
    api = readout.provider()
    catalog = api._core_aten_module(readout.PACKAGE, 'catalog')
    gate = api._core_aten_module(readout.PACKAGE, 'semantic_gate')
    source = '''builtin.module { func.func @forward(%x:tensor<1xi8>)->tensor<1xi8> {
      %s=arith.constant dense<1.0>:tensor<f32>
      %z=arith.constant dense<0>:tensor<i64>
      %d="quant_ext.dequantize_per_tensor"(%x,%s,%z)<{quant_min=-127:i64,quant_max=127:i64}>:
        (tensor<1xi8>,tensor<f32>,tensor<i64>)->tensor<1xf32>
      %v=tensor.cast %d:tensor<1xf32> to tensor<1xf32>
      %q="quant_ext.quantize_per_tensor"(%v,%s,%z)<{quant_min=-127:i64,quant_max=127:i64,output_dtype="int8"}>:
        (tensor<1xf32>,tensor<f32>,tensor<i64>)->tensor<1xi8>
      func.return %q:tensor<1xi8> } }'''
    _, manifest, matches = catalog.build(source)
    match, = matches
    name, = [n for n, _ in match.operands]
    raw = {name: np.array([-128], np.int8)}
    assert gate._source_reference(match)[0](raw)['y'].item() == -127
    from mlir_oot.core_aten.stream import simulate
    assert simulate(match.kernel.stream, raw)['y'].item() == -128
    proof = api.admit_numeric_schedule(source, package=readout.PACKAGE, numeric_policy=readout.POLICY)
    assert proof['status'] == 'declined' and 'semantic gate' in proof['reason']
    compiled = []
    monkeypatch.setattr(catalog, 'compile_to_object', lambda *a, **k: compiled.append(True))
    prepared = tmp_path / 'model.mlir'
    prepared.write_text(source)
    result, obj = api.core_aten_catalog(prepared, tmp_path / 'out', package=readout.PACKAGE,
                                      numeric_policy=readout.POLICY)
    assert obj is None and not compiled
    assert json.loads(result.read_text())['status'] == 'declined'


def test_e2e_driver_refuses_corrupt_movement_with_zero_fixture(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(readout.PACKAGE / 'tests/core_aten'))
    monkeypatch.syspath_prepend(str(readout.PACKAGE / 'tests/core_aten/e2e'))
    import run_e2e
    from mlir_oot.core_aten import catalog
    original = catalog.match_all
    source, _ = _case('movement')
    (tmp_path / 'model.mlir').write_text(source)
    np.savez(tmp_path / 'inputs.npz', arg0=np.zeros(17, np.int8))
    np.save(tmp_path / 'expected.npy', np.zeros(17, np.int8))

    def corrupt(*args, **kwargs):
        matches, declines = original(*args, **kwargs)
        for match in matches:
            stream = match.kernel.stream
            stream.ins = [i._replace(attrs=tuple(sorted({**dict(i.attrs), 'stride': 0}.items())))
                          if i.kind == 'config_ld' else i for i in stream.ins]
        return matches, declines

    monkeypatch.setattr(catalog, 'match_all', corrupt)
    compiled = []
    monkeypatch.setattr(catalog, 'compile_to_object', lambda *a, **k: compiled.append(True))
    with pytest.raises(run_e2e.DeviceRefusal, match='compilation refused'):
        run_e2e.prepare(tmp_path, 'probe', tmp_path / 'work')
    assert not compiled
