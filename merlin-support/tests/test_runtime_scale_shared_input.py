"""Source-bound shared-input transport, typed result coverage and store state."""
import numpy as np
import pytest

from merlin.runtime.exact_readout import f32_readout, form_multiplier
from mlir_oot.core_aten import shared_input
from mlir_oot.core_aten.catalog import build, authenticate_catalog
from mlir_oot.core_aten.stream import DIM, simulate
from test_shared_input_fusion import source, options, provider, POLICY, PACKAGE


def test_constant_mixed_i8_i32_results_have_distinct_typed_buffers():
    text,weights,scales=source(15,17,(17,19),constant=False,raw=(False,True),relu=(False,))
    _,manifest,matches=build(text,readout_options=options())
    match,=matches
    assert len(match.result_buffers)==2,manifest['declines']
    authenticate_catalog(manifest,matches)
    stream=match.kernel.stream
    assert [(a.name,a.dtype,a.shape) for a in stream.args if a.role=='output']==[
        ('out0','i8',(15,17)),('out1','i32',(15,19))]
    a=np.arange(15*17,dtype=np.int8).reshape(15,17)
    inputs=dict(a=a,w0=weights[0],w1=weights[1])
    actual=simulate(stream,inputs)
    raw=a.astype(np.int64)@weights[1].astype(np.int64)
    narrow=f32_readout(a.astype(np.int64)@weights[0].astype(np.int64),form_multiplier(scales[0],7),'i8')
    assert actual['out0'].dtype==np.int8 and actual['out1'].dtype==np.int32
    np.testing.assert_array_equal(actual['out0'],narrow)
    np.testing.assert_array_equal(actual['out1'],raw)
    reference=shared_input.source_reference(match,options())(inputs)
    for name in ('out0','out1'):
        np.testing.assert_array_equal(reference[name],actual[name])
    assert shared_input.admit(match,readout_options=options())['realization']['status']=='verified_finite_shared_input_effects'
    proof=provider().admit_numeric_schedule(text,package=PACKAGE,numeric_policy=POLICY,readout_options=options())
    assert proof['status']=='proven',proof
    assert proof['kernels'][0]['quantization']=='device'


def test_mixed_siblings_share_actual_input_loads():
    text,_,_=source(15,17,(17,19),constant=False,raw=(False,True),relu=(False,))
    _,manifest,matches=build(text,readout_options=options())
    match,=matches
    assert match.result_buffers
    loads=[i for i in match.kernel.stream.ins if i.kind=='mvin' and i.arg==0]
    assert len(loads)==-(-15//DIM)*-(-17//DIM)
    assert len(manifest['kernels'])==len(manifest['bindings'])==1
    shared_input.admit(match,readout_options=options())


def test_narrow_raw_narrow_store_state_is_independent():
    text,_,_=source(15,17,(17,19,5),constant=False,raw=(False,True,False),relu=(True,False,False))
    _,_,matches=build(text,readout_options=options())
    match,=matches
    proof=shared_input.admit(match,readout_options=options())
    assert len(proof['members'])==3
    stream=match.kernel.stream
    config=None
    seen=[]
    for ins in stream.ins:
        if ins.kind=='config_st':
            config=ins
        if ins.kind=='mvout':
            name=stream.args[ins.arg].name
            if name=='out1':
                assert config.a('stride')==19*4 and config.a('acc_act')==0
                assert np.float32(config.a('acc_scale'))==1
            seen.append(name)
    assert set(seen)=={'out0','out1','out2'}


def runtime_source(m=15,k=17,widths=(17,19),raw=(False,True),relu=False,shared_scale=False):
    """Exact composite source; scale is always an original f32[1] block argument."""
    text,weights,_=source(m,k,widths,constant=False,raw=True)
    scales=[i for i,r in enumerate(raw) if not r]
    params=', '.join(f'%gain{i}:tensor<1xf32>' for i in ([scales[0]] if shared_scale else scales))
    text=text.replace(') -> (',f', {params}) -> (',1)
    result_types=[f'tensor<{m}x{n}x{"i32" if r else "i8"}>' for n,r in zip(widths,raw)]
    start=text.index(' -> (')+len(' -> (')
    end=text.index(') {',start)
    text=text[:start]+', '.join(result_types)+text[end:]
    readouts=[]
    for i in scales:
        n=widths[i]
        activation='''%zero=arith.constant 0.0:f32
          %active=arith.maximumf %converted,%zero:f32''' if relu else ''
        active='%active' if relu else '%converted'
        gain=scales[0] if shared_scale else i
        readouts.append(f'''%dest{i}=tensor.empty():tensor<{m}x{n}xi8>
        %q{i}=linalg.generic {{indexing_maps=[affine_map<(m,n)->(m,n)>,
          affine_map<(m,n)->(0)>,affine_map<(m,n)->(m,n)>],iterator_types=["parallel","parallel"]}}
          ins(%mm{i},%gain{gain}:tensor<{m}x{n}xi32>,tensor<1xf32>)
          outs(%dest{i}:tensor<{m}x{n}xi8>) {{
          ^bb0(%acc:i32,%scale:f32,%unused:i8):
          %converted=arith.sitofp %acc:i32 to f32
          {activation}
          %product=arith.mulf {active},%scale:f32
          %rounded=math.roundeven %product:f32
          %lo=arith.constant -128.0:f32
          %hi=arith.constant 127.0:f32
          %clo=arith.maximumf %rounded,%lo:f32
          %chi=arith.minimumf %clo,%hi:f32
          %code=arith.fptosi %chi:f32 to i8
          linalg.yield %code:i8 }} -> tensor<{m}x{n}xi8>''')
    text=text[:text.index('func.return')]+ '\n'.join(readouts)+ '\nfunc.return '+', '.join(
        f'%mm{i}' if r else f'%q{i}' for i,r in enumerate(raw))+':'+', '.join(result_types)+'\n}}'
    return text,weights


def runtime_options():
    import json
    import os
    from pathlib import Path

    selected=options()
    # Vendored kextdiag RTL facts, sha256 a8d92927fad2f4aa56602b64672f2cf694e3d8ae5094e80413795fb8d270abe3.
    path=Path(os.environ.get('RT_FACTS', Path(__file__).resolve().parent/'fixtures/runtime_scale/rtl-facts.json'))
    selected['facts']=json.loads(path.read_text())
    return selected


def runtime_match(*,raw=(False,True),relu=False,shared_scale=False):
    text,weights=runtime_source(widths=(17,19) if len(raw)==2 else (17,19,5),raw=raw,
                                relu=relu,shared_scale=shared_scale)
    selected=runtime_options()
    _,manifest,matches=build(text,readout_options=selected)
    assert len(matches)==1 and matches[0].kernel.meta.get('shared_input'),manifest['declines']
    return text,weights,selected,manifest,matches[0]


def test_mixed_i8_i32_results_have_distinct_typed_buffers():
    from mlir_oot.core_aten.catalog import _stream_digest
    from mlir_oot.core_aten.stream_proof import execute
    text,weights,selected,manifest,match=runtime_match()
    stream=match.kernel.stream
    digest=_stream_digest(match.kernel)
    a=(np.arange(15*17,dtype=np.int16)%9-4).astype(np.int8).reshape(15,17)
    raw_reference=a.astype(np.int64)@weights[1].astype(np.int64)
    for scale in (0.,.25,.5,.3,-.5):
        buffers={}
        guards={}
        for i,arg in enumerate(stream.args):
            if arg.name=='a': data=a
            elif arg.name.startswith('w'): data=weights[int(arg.name[1:])]
            elif arg.dtype=='f32': data=np.array([scale],np.float32)
            else:
                storage=np.full(arg.nbytes+32,0x5a,np.uint8)
                guards[i]=storage
                data=storage[16:-16].view(np.int8 if arg.dtype=='i8' else np.int32).reshape(arg.shape)
            buffers[i]=data
        execute(stream,buffers)
        for i,storage in guards.items():
            assert np.all(storage[:16]==0x5a) and np.all(storage[-16:]==0x5a)
        np.testing.assert_array_equal(buffers[stream._index['out1']],raw_reference)
        np.testing.assert_array_equal(buffers[stream._index['out0']],f32_readout(
            a.astype(np.int64)@weights[0].astype(np.int64),np.float32(scale),'i8'))
        assert _stream_digest(match.kernel)==digest
    shared_input.admit(match,readout_options=selected)
    authenticate_catalog(manifest,[match])
    assert manifest['kernels'][0]['runtime_scale_arguments']==['readout_scale0']
    import copy
    changed=copy.deepcopy(manifest)
    changed['kernels'][0].pop('runtime_scale_arguments')
    with pytest.raises(ValueError,match='runtime scale arguments'):
        authenticate_catalog(changed,[match])
    proof=provider().admit_numeric_schedule(text,package=PACKAGE,numeric_policy=POLICY,readout_options=selected)
    assert proof['status']=='proven',proof
    assert proof['kernels'][0]['quantization']=='device'


def test_two_runtime_siblings_keep_distinct_scale_ssa():
    from mlir_oot.core_aten.runtime_scale import RuntimeScaleRef
    _,weights,selected,manifest,match=runtime_match(raw=(False,False))
    stream=match.kernel.stream
    refs={i.a('scale_operand') for i in stream.ins if i.kind=='config_st'}
    assert refs=={RuntimeScaleRef(stream._index['readout_scale0']),RuntimeScaleRef(stream._index['readout_scale1'])}
    a=np.ones((15,17),np.int8)
    result=simulate(stream,dict(a=a,w0=weights[0],w1=weights[1],
                                readout_scale0=np.array([.25],np.float32),
                                readout_scale1=np.array([.5],np.float32)))['out']
    want=np.concatenate([f32_readout(a.astype(np.int64)@w.astype(np.int64),np.float32(s),'i8').reshape(-1)
                         for w,s in zip(weights,(.25,.5))])
    np.testing.assert_array_equal(result,want)
    shared_input.admit(match,readout_options=selected)
    first,second=sorted(refs)
    for index,ins in enumerate(stream.ins):
        if ins.a('scale_operand')==first:
            stream.ins[index]=ins._replace(attrs=tuple((key,second if key=='scale_operand' else value)
                                                       for key,value in ins.attrs))
    # Admission checks source identity before its trials; equal-valued trials cannot mask this swap.
    with pytest.raises(ValueError,match='proved multiplier bits'):
        shared_input.admit(match,readout_options=selected)


def test_runtime_narrow_raw_narrow_store_state_is_independent():
    _,_,selected,_,match=runtime_match(raw=(False,True,False),relu=True)
    stream=match.kernel.stream
    shared_input.admit(match,readout_options=selected)
    current=None
    for ins in stream.ins:
        if ins.kind=='config_st': current=ins
        if ins.kind=='mvout':
            name=stream.args[ins.arg].name
            if name=='out1':
                assert current.a('acc_scale')==1. and current.a('acc_act')==0
                assert current.a('stride')==19*4
            else:
                assert current.a('scale_operand').arg_index==stream._index[
                    'readout_scale0' if name=='out0' else 'readout_scale1']
    # Change only the effective raw configuration, keeping all arithmetic untouched.
    for index,ins in enumerate(stream.ins):
        if ins.kind=='config_st' and ins.a('stride')==19*4:
            stream.ins[index]=ins._replace(attrs=tuple((key,.5 if key=='acc_scale' else value)
                                                      for key,value in ins.attrs))
    with pytest.raises(ValueError,match='full-width sibling store'):
        shared_input.admit(match,readout_options=selected)


@pytest.mark.parametrize('region',['integer_region','shared_input'])
def test_runtime_oracle_obligations_are_authenticated_and_surfaced(region):
    import copy
    from test_runtime_scale_emission import source as integer_source
    selected=runtime_options()
    text=integer_source(relu=True) if region=='integer_region' else runtime_match(relu=True)[0]
    _,manifest,matches=build(text,readout_options=selected)
    kernel,=manifest['kernels']
    obligations=kernel['runtime_oracle_obligations']
    assert [row['requires'] for row in obligations]==['relu_scale_order_parity_or_nonnegative_test_input']
    authenticate_catalog(manifest,matches)
    for mutation in ('drop','retarget'):
        changed=copy.deepcopy(manifest)
        if mutation=='drop':
            changed['kernels'][0].pop('runtime_oracle_obligations')
        else:
            changed['kernels'][0]['runtime_oracle_obligations'][0]['scale_operand'][0]+=1
        with pytest.raises(ValueError,match='runtime oracle obligations'):
            authenticate_catalog(changed,matches)
    proof=provider().admit_numeric_schedule(text,package=PACKAGE,numeric_policy=POLICY,readout_options=selected)
    assert proof['status']=='proven',proof
    row,=proof['kernels']
    assert row['runtime_oracle_obligations']==obligations
