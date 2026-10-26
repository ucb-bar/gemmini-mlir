"""Checked byte origins, storage-wide bounds, and clean centered-i32 refusals."""
import numpy as np
import pytest

from mlir_oot.core_aten.catalog import build
from mlir_oot.core_aten.stream import simulate
from exact_oracle import evaluate


def source(k=3, *, unsigned=True, zp=255, runtime=False, truncate=False):
    extension = 'arith.extui' if unsigned else 'arith.extsi'
    maps = 'affine_map<(d0,d1)->(d0,d1)>'
    inputs = f'%a: tensor<1x{k}xi8>, %b: tensor<{k}x1xi8>'
    if runtime: inputs += f', %z: tensor<1x{k}xi32>'
    a = f'''%ea = tensor.empty() : tensor<1x{k}xi32>
      %wa = linalg.generic {{indexing_maps=[{maps},{maps}], iterator_types=["parallel","parallel"]}}
        ins(%a:tensor<1x{k}xi8>) outs(%ea:tensor<1x{k}xi32>) {{
        ^bb0(%x:i8,%y:i32): %v = {extension} %x : i8 to i32
        linalg.yield %v : i32
      }} -> tensor<1x{k}xi32>'''
    if not runtime:
        a += f'\n%z = arith.constant dense<{zp}> : tensor<1x{k}xi32>'
    a += f'''\n%ca = linalg.generic {{indexing_maps=[{maps},{maps},{maps}], iterator_types=["parallel","parallel"]}}
      ins(%wa,%z:tensor<1x{k}xi32>,tensor<1x{k}xi32>) outs(%ea:tensor<1x{k}xi32>) {{
      ^bb1(%x:i32,%point:i32,%y:i32): %v = arith.subi %x,%point : i32
      linalg.yield %v : i32
    }} -> tensor<1x{k}xi32>
    %eb = tensor.empty() : tensor<{k}x1xi32>
    %wb = linalg.generic {{indexing_maps=[{maps},{maps}], iterator_types=["parallel","parallel"]}}
      ins(%b:tensor<{k}x1xi8>) outs(%eb:tensor<{k}x1xi32>) {{
      ^bb2(%x:i8,%y:i32): %v = arith.extsi %x : i8 to i32
      linalg.yield %v : i32
    }} -> tensor<{k}x1xi32>'''
    if truncate:
        a = a.replace('%v = arith.subi %x,%point : i32', '%d = arith.subi %x,%point : i32\n%t = arith.trunci %d : i32 to i8\n%v = arith.extsi %t : i8 to i32')
    return f'''builtin.module {{ func.func @forward({inputs}) -> tensor<1x1xi32> {{
      {a}
      %c = arith.constant 0 : i32
      %seed = tensor.splat %c : tensor<1x1xi32>
      %acc = linalg.matmul ins(%ca,%wb:tensor<1x{k}xi32>,tensor<{k}x1xi32>) outs(%seed:tensor<1x1xi32>) -> tensor<1x1xi32>
      %empty = tensor.empty() : tensor<1x1xi32>
      %out = linalg.generic {{indexing_maps=[{maps},{maps}],iterator_types=["parallel","parallel"]}}
        ins(%acc:tensor<1x1xi32>) outs(%empty:tensor<1x1xi32>)
        attrs = {{prov.quant.arithmetic="exact",prov.quant.accumulator="i32",prov.quant.numerics="integer_readout",
                  prov.quant.scale_numerator=["1"],prov.quant.scale_denominator=["1"]}} {{
        ^bb3(%x:i32,%y:i32): linalg.yield %x : i32
      }} -> tensor<1x1xi32>
      func.return %out : tensor<1x1xi32>
    }} }}'''


@pytest.mark.parametrize('unsigned,zp',[(True,0),(True,128),(True,255),(False,-128),(False,127)])
def test_full_storage_extremes_never_narrow(unsigned,zp):
    text=source(unsigned=unsigned,zp=zp)
    _,manifest,matches=build(text)
    match,=[m for m in matches if m.numeric.get('integer_readout')]
    for code in (-128,-1,0,127):
        feed=dict(x0=np.full((1,3),code,np.int8),x1=np.full((3,1),-128,np.int8))
        assert np.array_equal(simulate(match.kernel.stream,feed)['out'],evaluate(match,feed)['out'])
    assert not manifest['declines']


@pytest.mark.parametrize('options,reason',[
    (dict(runtime=True),'unbound block argument'),
    (dict(truncate=True),'unsupported limb data operation: arith.trunci'),
    (dict(zp=1000),'not a bounded integer constant'),
    (dict(k=70000),'no nonoverflow i32 bound'),
])
def test_unproved_centered_form_declines_cleanly(options,reason):
    _,manifest,matches=build(source(**options))
    assert not any(m.numeric.get('integer_readout') for m in matches)
    assert any(reason in d['reason'] for d in manifest['declines']),manifest['declines']


def repeated_centered_source():
    from xdsl.dialects import func
    from xdsl.dialects.builtin import FunctionType
    from mlir_oot.frontend.parse import parse_module
    module=parse_module(source())
    fn=next(o for o in module.walk() if o.name=='func.func')
    contraction=next(o for o in fn.walk() if o.name=='linalg.matmul')
    reader=next(u.operation for u in contraction.results[0].uses)
    ret=next(o for o in fn.walk() if o.name=='func.return')
    clone=contraction.clone()
    read_clone=reader.clone({contraction.results[0]:clone.results[0]})
    fn.body.block.insert_ops_before([clone,read_clone],ret)
    replacement=func.ReturnOp(ret.operands[0],read_clone.results[0])
    fn.body.block.insert_op_before(replacement,ret);fn.body.block.erase_op(ret)
    fn.properties['function_type']=FunctionType.from_lists(list(fn.function_type.inputs),[replacement.operands[0].type]*2)
    module.verify()
    return str(module)


def test_repeated_centered_bindings_share_one_unambiguous_schedule_proof():
    from conftest import provider, PACKAGE, POLICY
    from mlir_oot.core_aten.offload import rewrite
    text = repeated_centered_source()
    proof=provider().admit_numeric_schedule(text,package=PACKAGE,numeric_policy=POLICY)
    assert proof['status']=='proven',proof
    assert len(proof['bindings'])==2 and len(proof['kernels'])==1
    _, manifest, matches = build(text)
    shared = set(matches[0].root.operands[:2])
    assert all(value.owner not in match.consumed for value in shared for match in matches)
    rewritten = rewrite(text, manifest)
    assert rewritten.count('func.call @ca_') == 2
    assert 'linalg.matmul' not in rewritten


def test_shared_centered_matches_still_decline_swapped_results(monkeypatch):
    from conftest import provider, PACKAGE, POLICY
    api = provider()
    catalog = api._core_aten_module(PACKAGE, 'catalog')
    original = catalog.build

    def swapped(*args, **kwargs):
        device, manifest, matches = original(*args, **kwargs)
        matches[0].result, matches[1].result = matches[1].result, matches[0].result
        rows = manifest['bindings']
        rows[0]['result'], rows[1]['result'] = rows[1]['result'], rows[0]['result']
        return device, manifest, matches

    monkeypatch.setattr(catalog, 'build', swapped)
    proof = api.admit_numeric_schedule(repeated_centered_source(), package=PACKAGE, numeric_policy=POLICY)
    assert proof['status'] == 'declined'
    assert 'bound results' in proof['reason']


@pytest.mark.parametrize('overflow,reason',[
    ('wrap_to_16_bits','provider padded mesh K exceeds selected internal MAC bound'),
    (None,'integer mesh has unsupported selected MAC/accumulator arithmetic')],
    ids=['mesh_k_exceeds_mac_bound','inconsistent_mac_policy'])
def test_selected_16_bit_mac_policy_declines_cleanly(overflow,reason):
    from copy import deepcopy
    from conftest import provider,PACKAGE,POLICY
    policy=deepcopy(POLICY)
    policy['numerical_semantics']['internal_arithmetic']['mac_result_bits']=16
    if overflow:
        policy['numerical_semantics']['internal_arithmetic']['mac_result_overflow']=overflow
    proof=provider().admit_numeric_schedule(source(),package=PACKAGE,numeric_policy=policy)
    assert proof['status']=='declined' and not proof['kernels'] and not proof['bindings']
    assert proof['reason']==reason


def test_centered_i32_source_routes_to_proved_physical_byte_datapath(tmp_path):
    from conftest import synthetic_route
    route=synthetic_route(source(),tmp_path,('i32','i32','i32'))
    assert route and callable(route.region_offload) and route.catalog_builder is None
    assert route.operand_dtype=='i8' and route.accum_dtype=='i32'
    # The route must honor a refusal when proving the prepared bytes; it must
    # not compile simply because the unprepared source had a match.
    from copy import deepcopy
    import json
    from conftest import POLICY
    policy=deepcopy(POLICY)
    policy['numerical_semantics']['internal_arithmetic']['mac_result_bits']=16
    policy['numerical_semantics']['internal_arithmetic']['mac_result_overflow']='wrap_to_16_bits'
    (tmp_path/'spec-int8-admission.json').write_text(json.dumps(dict(selection=dict(spec=policy))))
    result=route.region_offload(tmp_path/'model.mlir',tmp_path/'refused')
    assert result[:3]==(None,None,None)
    manifest=json.loads(result[3].read_text())
    assert manifest['status']=='declined' and not manifest['kernels']
    assert 'compilation' not in manifest


def _admission_with(monkeypatch, text, mutate):
    """Admit ``text`` with ``mutate(stream)`` applied to every emitted kernel."""
    from conftest import provider, PACKAGE, POLICY
    api = provider()
    backend = api._core_aten_module(PACKAGE, 'catalog')
    original = backend.build

    def altered(*args, **kwargs):
        module, manifest, matches = original(*args, **kwargs)
        for match in matches:
            if match.numeric.get('integer_readout'):
                mutate(match.kernel.stream)
        return module, manifest, matches

    monkeypatch.setattr(backend, 'build', altered)
    return api.admit_numeric_schedule(text, package=PACKAGE, numeric_policy=POLICY)


def _mutated_value(text, mutate, feed):
    _, _, matches = build(text)
    match, = [m for m in matches if m.numeric.get('integer_readout')]
    mutate(match.kernel.stream)
    return int(simulate(match.kernel.stream, feed)['out'][0, 0])


def _centered_operands(k, b):
    return dict(x0=np.arange(1, k + 1, dtype=np.int8).reshape(1, k), x1=np.asarray(b, np.int8).reshape(k, 1))


def test_centered_operand_load_stride_is_replayed(monkeypatch):
    """B's operand stride 1 -> 0 re-reads B[0]: 1*4+2*4+3*4 = 24, not 32."""
    from conftest import provider, PACKAGE, POLICY
    text = source(k=3, unsigned=False, zp=0)
    feed = _centered_operands(3, [4, 5, 6])

    def mutate(stream):
        stream.ins = [i._replace(attrs=tuple((k, 0 if k == 'stride' else v) for k, v in i.attrs))
                      if i.kind == 'config_ld' and i.a('load_id') == 1 and i.a('stride') == 1 else i
                      for i in stream.ins]

    assert _mutated_value(text, lambda s: None, feed) == 32
    assert _mutated_value(text, mutate, feed) == 24
    assert provider().admit_numeric_schedule(text, package=PACKAGE, numeric_policy=POLICY)['status'] == 'proven'
    proof = _admission_with(monkeypatch, text, mutate)
    assert proof['status'] == 'declined', proof


def test_centered_scratchpad_store_stride_is_replayed(monkeypatch):
    """A zero CONFIG_ST stride around two gather stores: 140633, not 12529."""
    from mlir_oot.core_aten.readout_encoding import flags
    text = source(k=33, unsigned=False, zp=0)
    feed = _centered_operands(33, np.arange(1, 34))

    def mutate(stream):
        current, out = None, []
        for ins in stream.ins:
            if ins.kind == 'config_st':
                current = ins
            if (ins.kind == 'mvout' and not ins.a('local') & flags()['acc_i8'] and ins.a('rows') == 2):
                zero = current._replace(attrs=tuple((k, 0 if k == 'stride' else v) for k, v in current.attrs))
                out.extend([zero, ins, current])
            else:
                out.append(ins)
        stream.ins = out

    assert _mutated_value(text, lambda s: None, feed) == 12529
    assert _mutated_value(text, mutate, feed) == 140633
    proof = _admission_with(monkeypatch, text, mutate)
    assert proof['status'] == 'declined', proof
