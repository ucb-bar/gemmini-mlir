"""Source-bound optional two-store adapter; no sampled numeric eligibility."""
from dataclasses import replace
from merlin.llvmlower.enclosed_readout import synthesize, emit_pair_scan
from .golden_flat_conv import GoldenFlatConv
from .readout_store_plan import PairedReadoutPlan
from .direct_conv_binding import emit_c_adapter


def choose(generator, proof):
    if type(generator) is not GoldenFlatConv:
        return generator, None, dict(applied=False, refusal='selected producer has no paired store implementation')
    result = synthesize(proof['source_scales'], proof['accumulator_min'], proof['accumulator_max'], relu=proof['output_min']==0)
    if not result['accepted']:
        return generator, None, dict(applied=False, refusal=result)
    p=result['certificate']; plan=PairedReadoutPlan(tuple(p['source_scales']),tuple(p['store_scales']),p['lo'],p['hi'],p['relu'])
    candidate=GoldenFlatConv(generator.conv,wide_a=generator.wide_a,
        separate_b_bank=generator.separate_b_bank,band_rows=generator.band_rows,
        virtual_padding=generator.virtual_padding,pingpong_b=generator.pingpong_b,
        loop_spatial=generator.loop_spatial,store_plan=plan)
    return candidate, plan, dict(applied=True, proof=p, synthesis_radius=result['radius'],
        storage='fresh caller-owned byte scratch and byte output, guarded disjointness, stable through exact decoder')


def adapter(schedule,symbol,kernel,plan):
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
    return emit_pair_scan(plan.certificate(),decoder)+text


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
