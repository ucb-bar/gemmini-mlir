"""Source-bound optional two-store adapter; no sampled numeric eligibility."""
from dataclasses import replace
from merlin.llvmlower.enclosed_readout import synthesize, emit_pair_scan
from .golden_flat_conv import GoldenFlatConv
from .golden_resident_conv import GoldenResidentConv
from .readout_store_plan import PairedReadoutPlan
from .direct_conv_binding import emit_c_adapter


def choose(generator, proof):
    resident = type(generator) is GoldenResidentConv and (
        generator.flat_spatial_planes or generator.source_stride
    )
    if type(generator) is not GoldenFlatConv and not resident:
        return generator, None, dict(applied=False, refusal='selected producer has no paired store implementation')
    result = synthesize(proof['source_scales'], proof['accumulator_min'], proof['accumulator_max'], relu=proof['output_min']==0)
    if not result['accepted']:
        return generator, None, dict(applied=False, refusal=result)
    p=result['certificate']; plan=PairedReadoutPlan(tuple(p['source_scales']),tuple(p['store_scales']),p['lo'],p['hi'],p['relu'])
    if resident:
        candidate=generator.with_emission_options(store_plan=plan)
        for name in ('resident_stripe_decision','source_stride_decision','flat_resident_decision','weight_issue_decision','resident_spatial_tail_decision'):
            if hasattr(generator,name):setattr(candidate,name,getattr(generator,name))
    else:
        candidate=generator.with_emission_options(store_plan=plan)
    return candidate, plan, dict(applied=True, proof=p, synthesis_radius=result['radius'],
        host_copy_policy='compiler_builtin',storage='fresh caller-owned byte scratch and byte output, guarded disjointness, stable through exact decoder')


def adapter(schedule,symbol,kernel,plan,*,checked_alignment=False):
    if type(checked_alignment) is not bool:
        raise ValueError('paired scan checked alignment requires boolean selection')
    plan.require_conv_producer(schedule)
    m,n=schedule.oh*schedule.ow,schedule.cout
    out_bytes=m*n
    input_bytes=(schedule.h+(2 if schedule.explicit_halo else 0))*(schedule.w+(2 if schedule.explicit_halo else 0))*schedule.cin
    weight_bytes=9*schedule.cin*schedule.cout
    if max(out_bytes,input_bytes,weight_bytes)>=(1<<63):raise ValueError('descriptor extent exceeds signed address domain')
    text=emit_c_adapter(replace(schedule,output_dtype='i8'),symbol,kernel)
    signature=f'extern void {kernel}(int8_t*,int8_t*,int8_t*);'
    assert text.count(signature)==1
    text=text.replace(signature,f'extern void {kernel}(int8_t*,int8_t*,int8_t*,int8_t*);')
    text=text.replace('memref2 *c)', 'memref2 *scratch,memref2 *c)')
    check=f'if(!scratch->aligned || scratch->offset<0 || scratch->sizes[0]!={m} || scratch->sizes[1]!={n} || scratch->strides[0]!={n} || scratch->strides[1]!=1)__builtin_trap();'
    text=text.replace(' if (',' '+check+'\n if (',1)
    needle=f'\n {kernel}('; assert text.count(needle)==1
    guard=''
    for name,size in [('a',input_bytes),('b',weight_bytes),('scratch',out_bytes),('c',out_bytes)]:
        guard+=f'uintptr_t {name}_base=(uintptr_t){name}->aligned; if({name}_base>UINTPTR_MAX-(uintptr_t){name}->offset)__builtin_trap(); uintptr_t {name}_start={name}_base+(uintptr_t){name}->offset; if({name}_start>UINTPTR_MAX-{size})__builtin_trap(); uintptr_t {name}_end={name}_start+{size};\n'
    for a,b in [('scratch','c'),('scratch','a'),('scratch','b'),('c','a'),('c','b')]:
        guard+=f'if(!({a}_end<={b}_start || {b}_end<={a}_start))__builtin_trap();\n'
    text=text.replace(needle,'\n '+guard+needle,1)
    tail='(int8_t*)c->aligned+c->offset);';assert text.count(tail)==1
    text=text.replace(tail,'(int8_t*)c->aligned+c->offset,(int8_t*)scratch->aligned+scratch->offset);')
    decoder=symbol+'_pair_decode'
    text=text.replace('*r=*c;',f'{decoder}((unsigned char*)c->aligned+c->offset,(const unsigned char*)scratch->aligned+scratch->offset,{out_bytes});*r=*c;')
    options={'checked_alignment':True} if checked_alignment else {}
    return emit_pair_scan(plan.certificate(),decoder,copy_policy='compiler_builtin',**options)+text


def native_oracle(schedule,kernel,plan,scalar_oracle):
    """Independent exact-i32 convolution followed by both primitive readouts."""
    plan.require_conv_producer(schedule)
    count=schedule.oh*schedule.ow*schedule.cout
    source=scalar_oracle(schedule,kernel+'_accumulator_reference',True)
    stores=''
    low=0 if plan.relu else -128
    for name,scale in zip(('first','second'),plan.store_scales):
        stores+=f'{{volatile float v=(float)acc[i];v=v*{float(scale).hex()}f;v=nearbyintf(v);if(v<{low})v={low};if(v>127)v=127;{name}[i]=(int8_t)v;}}'
    return '#include <stdlib.h>\n#include <math.h>\n'+source+f'''void {kernel}(int8_t*a,int8_t*b,int8_t*first,int8_t*second){{
 int32_t*acc=(int32_t*)malloc({count}*sizeof(int32_t));if(!acc)abort();
 {kernel}_accumulator_reference(a,b,acc);
 for(size_t i=0;i<{count};i++){{{stores}}}free(acc);
}}
'''


def prepare_paired_scratch(module, manifest):
    """Rebind legacy fresh scratch to the selected provider's byte-write ABI.

    Validate every selected declaration and all caller ownership before mutation.
    This changes only an uninitialized private tensor.empty's element type;
    it never reinterprets a live value or an external allocation.
    """
    import hashlib
    import json
    from xdsl.dialects import tensor
    from xdsl.dialects.builtin import DictionaryAttr, FunctionType, IntegerAttr, StringAttr, TensorType, i8, i64
    from xdsl.rewriter import Rewriter

    pending = []
    for route in manifest['routes']:
        selection = route.get('paired_readout', {})
        if not selection.get('applied'):
            continue
        proof = selection['proof']
        plan = PairedReadoutPlan(tuple(proof['source_scales']), tuple(proof['store_scales']), proof['lo'], proof['hi'], proof['relu'])
        original = route['integer_readout']
        if (list(plan.source_scales) != original['source_scales'] or plan.accumulator_min != original['accumulator_min'] or plan.accumulator_max != original['accumulator_max'] or plan.relu != (original['output_min'] == 0)):
            raise ValueError('paired scratch source numeric contract changed')
        if plan.certificate() != proof:
            raise ValueError('paired scratch certificate changed')
        declarations = [op for op in module.walk() if op.name == 'func.func' and op.sym_name.data == route['symbol']]
        calls = [op for op in module.walk() if op.name == 'func.call' and op.callee.root_reference.data == route['symbol']]
        if len(declarations) != 1 or len(calls) != 1:
            raise ValueError('paired scratch requires one bound declaration and call')
        declaration, call = declarations[0], calls[0]
        types = declaration.function_type.inputs.data
        if len(types) != 4 or tuple(value.type for value in call.operands) != tuple(types):
            raise ValueError('paired scratch call ABI differs from declaration')
        if not all(isinstance(t, TensorType) for t in types) or types[2].get_shape() != types[3].get_shape() or str(types[3].get_element_type()) != 'i8':
            raise ValueError('paired scratch tensor geometry differs')
        dtype = str(types[2].get_element_type())
        if dtype not in ('i8', 'i32'):
            raise ValueError('unsupported paired scratch element type')
        count = 1
        for dim in types[2].get_shape():
            if dim <= 0:
                raise ValueError('paired scratch must have positive static shape')
            count *= dim
        attr = declaration.attributes.get('gemmini.integer_readout')
        expected_sha = hashlib.sha256(json.dumps(route['integer_readout'], sort_keys=True).encode()).hexdigest()
        if attr is None or attr.data['proof_sha256'].data != expected_sha or attr.data['source_sha256'].data != manifest['source_sha256'] or attr.data['scratch_bytes'].value.data != count * (4 if dtype == 'i32' else 1) or attr.data['scratch_ownership'].data != 'caller_owned_unique':
            raise ValueError('paired scratch source/proof/extent changed')
        for index in (2, 3):
            value = call.operands[index]
            uses = list(value.uses)
            if not isinstance(value.owner, tensor.EmptyOp) or len(uses) != 1 or uses[0].operation is not call:
                raise ValueError('paired scratch/output requires sole-use fresh tensor.empty')
        if call.operands[2].owner is call.operands[3].owner:
            raise ValueError('paired scratch aliases output')
        existing = declaration.attributes.get('gemmini.paired_readout')
        if existing is not None and json.loads(existing.data) != proof:
            raise ValueError('paired scratch store proof changed')
        pending.append((declaration, call, types, count, proof, dtype))
    for declaration, call, types, count, proof, dtype in pending:
        new_type = TensorType(i8, types[2].get_shape())
        if dtype == 'i32':
            Rewriter.replace_op(call.operands[2].owner, tensor.EmptyOp([], new_type))
            declaration.properties['function_type'] = FunctionType.from_lists([types[0], types[1], new_type, types[3]], declaration.function_type.outputs.data)
        attr = declaration.attributes['gemmini.integer_readout']
        declaration.attributes['gemmini.integer_readout'] = DictionaryAttr({**attr.data, 'scratch_bytes': IntegerAttr(count, i64)})
        declaration.attributes['gemmini.paired_readout'] = StringAttr(json.dumps(proof, sort_keys=True))
    return len(pending)
