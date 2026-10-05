"""Source/weight-bound unary requant fusion, preserving direct convolution.

Only immutable zero-bias captured channels are currently specialized here.
The general scalar proof supports nonzero bias, but this bundle refuses it until
its descriptor ABI explicitly binds the synthesized integer preload table.
Exact equivalence is the default. An explicitly requested one-step policy can
admit a zero-bias readout after proving its complete accumulator-domain error.
"""
import argparse
from dataclasses import asdict,replace
import hashlib,json,shutil,subprocess
from pathlib import Path
import numpy as np
from xdsl.dialects import func,tensor
from xdsl.dialects.linalg.ops import TransposeOp
from xdsl.dialects.builtin import ArrayAttr,DictionaryAttr,IntegerAttr,StringAttr,TensorType,UnitAttr,i8,i32,i64
from xdsl.ir import Region
from .frontend.parse import parse_module
from .contraction_patterns import match_integer_gemm
from .captured_requant import inspect_chain
from .golden_requant import synthesize_bias, prove_scale_bound, synthesize_store_scale
from .golden_contraction_upstream import choose_shape
from .golden_gemm import GoldenGemm
from .conv_schedule import select_kernel
from .golden_device_compile import compile_module
from .direct_conv_binding import match as match_conv,_transpose,serialize,emit_c_adapter
from .no_fsm_audit import audit_elf
from .exact_integer_readout import derive as derive_readout, emit_readout, fixedpoint_candidate
from .captured_residual_bundle import compile_adapter


def dense_adapter(s,symbol,kernel):
    def checks(name,shape):
        return ' || '.join([f'!{name}->aligned',f'{name}->offset<0']+[f'{name}->sizes[{i}]!={size} || {name}->strides[{i}]!={shape[1] if i==0 else 1}' for i,size in enumerate(shape)])
    return '''#include <stdint.h>
#ifndef GEMMINI_DIRECT_CONV_ABI
#define GEMMINI_DIRECT_CONV_ABI
typedef struct {void *allocated,*aligned; intptr_t offset,sizes[2],strides[2];} memref2;
typedef struct {void *allocated,*aligned; intptr_t offset,sizes[4],strides[4];} memref4;
#endif
'''+f'''extern void {kernel}(int8_t*,int8_t*,int8_t*);
void _mlir_ciface_{symbol}(memref2 *r,memref2 *a,memref2 *b,memref2 *c) {{
 if ({checks('a',[s.m,s.k])} || {checks('b',[s.k,s.n])} || {checks('c',[s.m,s.n])}) __builtin_trap();
 {kernel}((int8_t*)a->aligned+a->offset,(int8_t*)b->aligned+b->offset,(int8_t*)c->aligned+c->offset);*r=*c;
}}
'''


def scalar_oracle(s,kernel,direct):
    scale=s.scale.hex()+'f';low=0 if s.relu else -128
    if direct:
        loops=f'''for(int y=0;y<{s.oh};y++) for(int x=0;x<{s.ow};x++) for(int n=0;n<{s.cout};n++) {{
 uint32_t acc=0;
 for(int ky=0;ky<3;ky++) for(int kx=0;kx<3;kx++) for(int ci=0;ci<{s.cin};ci++) {{
 int iy=y*{s.stride}+ky-{0 if s.explicit_halo else 1},ix=x*{s.stride}+kx-{0 if s.explicit_halo else 1};
 if(iy>=0 && iy<{s.h+(2 if s.explicit_halo else 0)} && ix>=0 && ix<{s.w+(2 if s.explicit_halo else 0)})
 acc+=(uint32_t)((int32_t)a[(iy*{s.w+(2 if s.explicit_halo else 0)}+ix)*{s.cin}+ci]*(int32_t)b[((ky*3+kx)*{s.cin}+ci)*{s.cout}+n]); }}
 int index=(y*{s.ow}+x)*{s.cout}+n;'''
    else:
        loops=f'''for(int m=0;m<{s.m};m++) for(int n=0;n<{s.n};n++) {{
 uint32_t acc=0;for(int k=0;k<{s.k};k++)acc+=(uint32_t)((int32_t)a[m*{s.k}+k]*(int32_t)b[k*{s.n}+n]);
 int index=m*{s.n}+n;'''
    if s.output_dtype == 'i32':
        return f'''#include <stdint.h>\nvoid {kernel}(int8_t*a,int8_t*b,int32_t*c) {{{loops} c[index]=(int32_t)acc; }} }}\n'''
    return f'''#include <math.h>
void {kernel}(int8_t*a,int8_t*b,int8_t*c) {{
{loops}
 float value=nearbyintf((float)(int32_t)acc*{scale});
 if(value<{low})value={low};if(value>127)value=127;c[index]=(int8_t)value;
 }}
}}
'''


def _replay_layouts(value,layouts):
    operations=[]
    for original in layouts:
        ty=TensorType(i8,original.results[0].type.get_shape())
        if original.name=='tensor.collapse_shape':
            op=tensor.CollapseShapeOp(operands=[value],result_types=[ty],properties={'reassociation':original.properties['reassociation']})
        elif original.name=='tensor.expand_shape':
            op=tensor.ExpandShapeOp(value,[],original.properties['reassociation'],list(ty.get_shape()),ty)
        else:
            empty=tensor.EmptyOp([],ty);operations.append(empty)
            op=TransposeOp(value,empty.tensor,original.permutation,ty)
        operations.append(op);value=op.results[0]
    return operations,value


def rewrite_path(op,chain,symbol,direct,*,numeric_contract=None,integer_readout=None,source_sha=None,virtual_input=None):
    dims=chain['dimensions'];operations=[]
    if direct:
        if direct.orientation!='spatial_first':raise ValueError('fused captured direct path needs spatial-first contraction')
        ops,activation=_transpose(virtual_input if virtual_input is not None else direct.input,[0,2,3,1]);operations+=ops
        s=direct.shape;wt=TensorType(i8,[3,3,s.cin,s.cout])
        reassoc=ArrayAttr([ArrayAttr([IntegerAttr(i,i64) for i in (0,1,2)]),ArrayAttr([IntegerAttr(3,i64)])])
        weight=tensor.ExpandShapeOp(direct.weight,[],reassoc,list(wt.get_shape()),wt);operations.append(weight)
        inputs=[activation,weight.result]
    else:inputs=list(op.operands[:2])
    if integer_readout is not None:
        scratch=tensor.EmptyOp([],TensorType(i32,[dims.m,dims.n]));operations.append(scratch);inputs.append(scratch.tensor)
    ct=TensorType(i8,[dims.m,dims.n]);empty=tensor.EmptyOp([],ct);operations.append(empty)
    call=func.CallOp(symbol,[*inputs,empty.tensor],[ct]);operations.append(call)
    views,result=_replay_layouts(call.results[0],chain['layouts']);operations+=views
    if result.type!=chain['quantize'].results[0].type:raise ValueError('replayed output layout differs')
    module=op
    while module.parent_op() is not None:module=module.parent_op()
    op.parent.insert_ops_before(operations,op)
    chain['quantize'].results[0].replace_all_uses_with(result)
    for old in reversed(chain['operations']):old.parent.erase_op(old)
    op.parent.erase_op(op)
    attrs=ArrayAttr([DictionaryAttr({'bufferization.access':StringAttr(x)}) for x in (['read','read','write','write'] if integer_readout is not None else ['read','read','write'])])
    declaration=func.FuncOp(symbol,([x.type for x in inputs]+[ct],[ct]),Region(),visibility='private',arg_attrs=attrs)
    declaration.attributes['llvm.emit_c_interface']=UnitAttr();module.body.block.add_op(declaration)
    if numeric_contract is not None:
        declaration.attributes['merlin.numeric_contract']=DictionaryAttr({
            'unit':StringAttr('quantized_output_lsb'),
            'max_abs_error':IntegerAttr(numeric_contract['max_output_lsb_error'],i64),
            'selected_policy_limit':IntegerAttr(numeric_contract['selected_policy_limit'],i64),
            'source_region':StringAttr(numeric_contract['source_region']),
            'proof_domain':StringAttr('complete signed-int8 contraction accumulator interval'),
        })
    if integer_readout is not None:
        declaration.attributes['gemmini.integer_readout']=DictionaryAttr({'proof_sha256':StringAttr(hashlib.sha256(json.dumps(integer_readout,sort_keys=True).encode()).hexdigest()),'source_sha256':StringAttr(source_sha),'scratch_bytes':IntegerAttr(dims.m*dims.n*4,i64),'scratch_ownership':StringAttr('caller_owned_unique')})
    return declaration


def integer_adapter(schedule,symbol,kernel,direct,proof):
    adapter=emit_c_adapter(schedule,symbol,kernel) if direct else dense_adapter(schedule,symbol,kernel).replace('int8_t*c)','int32_t*c)').replace('int8_t*,int8_t*,int8_t*','int8_t*,int8_t*,int32_t*')
    adapter=adapter.replace('memref2 *c)', 'memref2 *scratch,memref2 *c)')
    output='(int32_t*)c->aligned+c->offset' if direct else '(int8_t*)c->aligned+c->offset'
    adapter=adapter.replace(output,'(int32_t*)scratch->aligned+scratch->offset')
    m,n=(schedule.oh*schedule.ow,schedule.cout) if direct else (schedule.m,schedule.n)
    check=f'if(!scratch->aligned || scratch->offset<0 || scratch->sizes[0]!={m} || scratch->sizes[1]!={n} || scratch->strides[0]!={n} || scratch->strides[1]!=1 || ((uintptr_t)((int32_t*)scratch->aligned+scratch->offset)&3))__builtin_trap();'
    adapter=adapter.replace(' if (', ' '+check+'\n if (',1)
    readout=symbol+'_readout'
    adapter=adapter.replace('*r=*c;',f'{readout}((const int32_t*)scratch->aligned+scratch->offset,(int8_t*)c->aligned+c->offset,{m*n});*r=*c;')
    return emit_readout(proof,readout,fixedpoint=True)+adapter


def build(capture:Path,llvm_bin:Path,output:Path,*,flat_spatial=False,max_output_lsb=0,exact_integer_readout=False,virtual_padding=False,banked_prefetch=False,grouped_b=False,separate_b_bank=False,full_k_banked_regions=(),resident_input_regions=(),resident_input_options=None,resident_input_policy=None,dense_input_policy=None,resident_stripes=False):
    from .golden_resident_conv import ResidentConvOptions
    resident_input_options=dict(resident_input_options or {})
    if set(resident_input_options)-set(resident_input_regions):
        raise ValueError('resident options require explicitly selected source regions')
    if any(not isinstance(o,ResidentConvOptions) for o in resident_input_options.values()):
        raise ValueError('resident options require typed ResidentConvOptions')
    if resident_input_policy not in (None, 'compact_channel_planes'):
        raise ValueError('unknown resident compiler policy')
    if dense_input_policy not in (None, 'banked_command_cost', 'resident_a_command_cost', 'transfer_command_cost'):
        raise ValueError('unknown dense compiler policy')
    if dense_input_policy is not None and full_k_banked_regions:
        raise ValueError('dense compiler policy cannot mix with source selections')
    if resident_input_policy is not None:
        if resident_input_regions or resident_input_options:
            raise ValueError('resident compiler policy cannot mix with source selections')
        if not virtual_padding or not flat_spatial:
            raise ValueError('resident compiler policy requires proved virtual padding and spatial layout')
    if virtual_padding and not flat_spatial:raise ValueError('virtual padding requires flat spatial scheduling')
    if type(resident_stripes) is not bool or (resident_stripes and (not virtual_padding or not flat_spatial)):
        raise ValueError('resident stripe policy requires boolean selection and proved virtual padding/spatial scheduling')
    from merlin.runtime.captured_constants import verify_capture_constant
    if type(max_output_lsb) is not int or max_output_lsb not in (0,1):
        raise ValueError('select an explicit zero- or one-step local output error policy')
    output.mkdir(parents=True,exist_ok=False)
    source=(capture/'model.mlir').read_text();module=parse_module(source)
    source_sha=hashlib.sha256(source.encode()).hexdigest();pins=json.loads((capture/'capture_receipt.json').read_text())['artifacts']
    if source_sha!=pins['model.mlir']['sha256']:raise ValueError('capture source hash changed')
    objects=[];declarations=[];native=[];routes=[];refused=[]
    for op in list(module.walk()):
        dims=match_integer_gemm(op)
        if dims is None:continue
        rid=getattr(op.attributes.get('prov.region_id'),'data','')
        try:chain=inspect_chain(op)
        except ValueError as e:refused.append(dict(region=rid,reason=str(e)));continue
        constant=verify_capture_constant(manifest_path=capture/'weights.safetensors.manifest.json',manifest_sha256=pins['weights.safetensors.manifest.json']['sha256'],safetensors_path=capture/'weights.safetensors',safetensors_sha256=pins['weights.safetensors']['sha256'],entry_argument_index=chain['bias'].index,source_shape=[dims.n],source_dtype='f32',max_payload_bytes=dims.n*4)
        biases=np.frombuffer(constant.logical_payload,dtype='<f4');bound=dims.k*16384
        proof=synthesize_bias(chain['scales'],biases,chain['reciprocal'],max(-(1<<31),-bound),min((1<<31)-1,bound),chain['relu'])
        error=0;readout=None
        if proof['accepted_channels']!=dims.n:
            if np.any(biases!=0):
                refused.append(dict(region=rid,reason='float transition proof refused',accepted_channels=proof['accepted_channels']));continue
            solved=synthesize_store_scale([*chain['scales'],chain['reciprocal']],max(-(1<<31),-bound),min((1<<31)-1,bound),chain['relu'])
            if solved['exact']:
                proof=dict(solved,source_bias='immutable all-zero channel vector; f32 addition preserves quantized output',channels=dims.n,accepted_channels=dims.n,integer_bias=[0]*dims.n)
            else:
                if exact_integer_readout:
                    readout=derive_readout([*chain['scales'],chain['reciprocal']],max(-(1<<31),-bound),min((1<<31)-1,bound),chain['relu'])
                    if fixedpoint_candidate(readout) is None:
                        raise ValueError('exact integer readout lacks a proved fixed-point estimate')
                    proof=dict(readout,scale=1.0,source_bias='immutable all-zero channel vector',channels=dims.n,accepted_channels=dims.n,integer_bias=[0]*dims.n,store_scale_refusal=solved)
                elif max_output_lsb==0:
                    refused.append(dict(region=rid,reason='no positive finite f32 store scale preserves all transitions',proof=solved));continue
                else:
                    bounded=prove_scale_bound([*chain['scales'],chain['reciprocal']],max(-(1<<31),-bound),min((1<<31)-1,bound),chain['relu'])
                    error=bounded['max_output_lsb_error']
                    if error>max_output_lsb:
                        refused.append(dict(region=rid,reason='local readout error exceeds selected policy',proof=bounded));continue
                    proof=dict(bounded,source_bias='immutable all-zero channel vector; f32 addition preserves quantized output',channels=dims.n,accepted_channels=dims.n,integer_bias=[0]*dims.n)
        if np.any(biases!=0) or any(x!=0 for x in proof['integer_bias']):
            refused.append(dict(region=rid,reason='nonzero bias requires explicit integer table ABI'));continue
        try:direct=match_conv(op)
        except ValueError:direct=None
        virtual_input=None;pad_proof=None;pad_refusal=None
        if direct and virtual_padding:
            from .virtual_padding import strip_zero_pad1
            try:virtual_input,pad_proof=strip_zero_pad1(direct.input,direct.shape)
            except ValueError as failure:pad_refusal=str(failure)
        base=replace(direct.shape,explicit_halo=False) if virtual_input is not None else (direct.shape if direct else choose_shape(dims))
        schedule=replace(base,output_dtype='i32' if readout else 'i8',scale=proof['scale'],relu=False if readout else chain['relu'])
        if not direct:schedule=replace(schedule,wide_store=True)
        symbol=f'gemmini_{"exact" if error==0 else "bounded"}_requant_{len(routes)}';kernel=symbol+'_kernel';work=output/symbol
        bias_index=chain['bias'].index
        numeric_contract=dict(max_output_lsb_error=error,selected_policy_limit=max_output_lsb,source_region=rid)
        declaration=rewrite_path(op,chain,symbol,direct,numeric_contract=numeric_contract,integer_readout=readout,source_sha=source_sha,virtual_input=virtual_input);declarations.append(declaration)
        if pad_proof is not None:declaration.attributes['gemmini.virtual_padding']=StringAttr(json.dumps(pad_proof,sort_keys=True))
        if direct:
            policy_options=None;policy_refusal=None;generator=None
            if rid in resident_input_regions:
                from .golden_resident_conv import GoldenResidentConv
                if virtual_input is None:raise ValueError('resident input requires source-proven virtual padding')
                options=resident_input_options.get(rid,ResidentConvOptions())
                generator=GoldenResidentConv(schedule,**asdict(options));schedule_kind='resident_input_channel_planes'
            elif resident_input_policy is not None:
                if virtual_input is None:
                    policy_refusal='source-proven virtual padding unavailable'
                else:
                    from .golden_resident_conv import choose_compact_resident
                    try:
                        generator,policy_options=choose_compact_resident(schedule)
                    except ValueError as failure:
                        policy_refusal=str(failure)
                    else:
                        schedule_kind='resident_input_channel_planes'
            if generator is None:
                generator,schedule_kind=select_kernel(schedule,flat_spatial=flat_spatial,virtual_padding=virtual_input is not None,
                    resident_stripes=resident_stripes and virtual_input is not None)
            schedule=generator.conv
        else:
            from .dense_schedule import select_kernel as select_dense,choose_banked_by_command_cost,choose_resident_a_by_command_cost,choose_transfer_by_command_cost
            dense_decision=None
            if dense_input_policy is not None:
                control,_=select_dense(schedule,banked_prefetch=banked_prefetch,grouped_b=grouped_b,separate_b_bank=separate_b_bank)
                choose={'banked_command_cost':choose_banked_by_command_cost,'resident_a_command_cost':choose_resident_a_by_command_cost,'transfer_command_cost':choose_transfer_by_command_cost}[dense_input_policy]
                _,dense_decision=choose(control.shape)
            generator,schedule_kind=select_dense(schedule,banked_prefetch=banked_prefetch,grouped_b=grouped_b,separate_b_bank=separate_b_bank,full_k_banked=rid in full_k_banked_regions,banked_command_policy=dense_input_policy=='banked_command_cost',resident_a_command_policy=dense_input_policy=='resident_a_command_cost',transfer_command_policy=dense_input_policy=='transfer_command_cost');schedule=generator.shape
        device=generator.build();device.body.block.first_op.properties['sym_name']=StringAttr(kernel)
        compilation=compile_module(device,llvm_bin,work)
        adapter=integer_adapter(schedule,symbol,kernel,bool(direct),readout) if readout else (emit_c_adapter(schedule,symbol,kernel) if direct else dense_adapter(schedule,symbol,kernel))
        (work/'adapter.c').write_text(adapter)
        adapter_compilation=compile_adapter(work/'adapter.c',work/'adapter.o',llvm_bin)
        objects.extend([work/'kernel.o',work/'adapter.o']);native.append(adapter+scalar_oracle(schedule,kernel,bool(direct)))
        routes.append(dict(region=rid,symbol=symbol,kernel=kernel,direct_conv=bool(direct),schedule_kind=schedule_kind,schedule=asdict(schedule),virtual_padding_proof=pad_proof,virtual_padding_refusal=pad_refusal,bias_argument=bias_index,bias_payload_sha256=constant.payload_sha256,numeric_contract=numeric_contract,proof=proof,integer_readout=readout,adapter_compilation=adapter_compilation,compilation=compilation))
        if rid in resident_input_options:
            routes[-1]['resident_options']=asdict(resident_input_options[rid])
        if direct and resident_stripes:
            routes[-1]['resident_stripe_policy_decision']=getattr(generator,'resident_stripe_decision',
                dict(applied=False,refusal='previous explicit resident schedule retained' if virtual_input is not None else 'source-proven virtual padding unavailable'))
        if direct and resident_input_policy is not None:
            routes[-1]['resident_policy']=resident_input_policy
            routes[-1]['resident_policy_refusal']=policy_refusal
            if policy_options is not None:
                routes[-1]['resident_options']=asdict(policy_options)
        if not direct and dense_input_policy is not None:
            routes[-1]['dense_policy']=dense_input_policy
            routes[-1]['dense_policy_decision']=dense_decision
    selected_regions={r['region'] for r in routes if 'full_k_banked_prefetch' in r['schedule_kind']}
    if selected_regions != set(full_k_banked_regions):raise ValueError('requested full-K banked source regions not all selected')
    selected_resident={r['region'] for r in routes if r['schedule_kind']=='resident_input_channel_planes'}
    if resident_input_policy is None and selected_resident != set(resident_input_regions):raise ValueError('requested resident-input source regions not all selected')
    if not routes:raise ValueError('no exactly provable captured epilogues')
    module.verify();printed=serialize(module,declarations);parse_module(printed).verify();(output/'rewritten.mlir').write_text(printed)
    (output/'native_oracle.c').write_text('\n'.join(native))
    linker=llvm_bin/'ld.lld'
    if not linker.is_file():linker=Path(shutil.which('ld.lld'))
    linked=output/'requant.o';subprocess.run([str(linker),'-r',*[str(p) for p in objects],'-o',str(linked)],check=True,capture_output=True)
    audit=audit_elf(linked.read_bytes())
    if audit['status']!='pass':raise ValueError('forbidden device instruction')
    result=dict(schema='gemmini_exact_captured_requant_bundle_v1' if max_output_lsb==0 else 'gemmini_bounded_captured_requant_bundle_v1',selected_max_output_lsb=max_output_lsb,source_sha256=source_sha,weights_sha256=pins['weights.safetensors']['sha256'],manifest_sha256=pins['weights.safetensors.manifest.json']['sha256'],rewritten_sha256=hashlib.sha256(printed.encode()).hexdigest(),object_sha256=hashlib.sha256(linked.read_bytes()).hexdigest(),routes=routes,refused=refused,nofsm_audit=audit,scope='Specializes only hash-bound captured zero-bias parameters. Runtime input images remain variable. Other weights/bias blobs require recompilation. Explicit local error limits do not establish full-model quality; original goldens must be retained and checked.')
    (output/'requant.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('capture',type=Path);p.add_argument('--llvm-bin',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--banked-prefetch',action='store_true');p.add_argument('--grouped-b',action='store_true');p.add_argument('--dense-input-policy',choices=('banked_command_cost','resident_a_command_cost','transfer_command_cost'));p.add_argument('--full-k-banked-region',action='append',default=[]);p.add_argument('--resident-input-policy',choices=('compact_channel_planes',));p.add_argument('--resident-stripes',action='store_true');p.add_argument('--resident-input-region',action='append',default=[]);p.add_argument('--separate-b-bank',action='store_true');p.add_argument('--virtual-padding',action='store_true');p.add_argument('--flat-spatial',action='store_true');p.add_argument('--exact-integer-readout',action='store_true');p.add_argument('--max-output-lsb',type=int,choices=(0,1),default=0,help='Explicit local error limit; a separate full-model quality gate is required');a=p.parse_args()
    result=build(a.capture,a.llvm_bin,a.output,flat_spatial=a.flat_spatial,max_output_lsb=a.max_output_lsb,exact_integer_readout=a.exact_integer_readout,virtual_padding=a.virtual_padding,banked_prefetch=a.banked_prefetch,grouped_b=a.grouped_b,separate_b_bank=a.separate_b_bank,full_k_banked_regions=a.full_k_banked_region,resident_input_regions=a.resident_input_region,resident_input_policy=a.resident_input_policy,dense_input_policy=a.dense_input_policy,resident_stripes=a.resident_stripes);print(json.dumps(dict(routes=len(result['routes']),direct=sum(x['direct_conv'] for x in result['routes']),refused=len(result['refused']),object_sha256=result['object_sha256']),indent=2))

if __name__=='__main__':main()
