from dataclasses import replace
import pytest
from merlin.llvmlower.source_attention_frontier import SourceAttentionFrontierPlan
from merlin.llvmlower.radix_product_groups import plan_radix_product_groups
from mlir_oot.attention_frontier_bridge import GroupedProductImport,emit_descriptor_bridge

PLAN=SourceAttentionFrontierPlan(2,3,4,8,3,2,.125,-87.,1.44,(-.079,-.224,.303,.0001),8388608.,1065353216.,127.,1e-5,-127,127)

def imports():
    result=[]
    for index,shape in enumerate(((3,8,4),(3,4,3),(3,4,2))):
        proof=plan_radix_product_groups(radix_bits=7,digits=3,reduction_length=shape[2])
        for degree,group in enumerate(proof.groups):
            result.append(GroupedProductImport(shape,degree,group.pairs,group.accumulator_bound,f'kernel_{index}_{degree}','a'*64,'b'*64))
    return result

def emit(items,**kw):
    return emit_descriptor_bridge(PLAN,items,writer_symbol='writer',executor_symbol='executor',fallback_symbol='source',source_binding_sha256='c'*64,consumer_binding_sha256='d'*64,workspace_bytes=4096,**kw)

def test_all_degrees_bind_bit_exponents():
    text=emit(imports())
    assert 'degree==4){kernel_0_4' in text
    assert '_mlir_ciface_source(i0, i1' in text
    assert 'D4*out,D1*workspace' in text

@pytest.mark.parametrize('kind',['missing','duplicate','pairs','bounds','symbol','sha'])
def test_invalid_imports_refuse(kind):
    items=imports()
    if kind=='missing': items.pop()
    elif kind=='duplicate': items.append(items[0])
    elif kind=='pairs': items[0]=replace(items[0],pairs=((1,1),))
    elif kind=='bounds': items[0]=replace(items[0],accumulator_absolute_bound=0)
    elif kind=='symbol': items[1]=replace(items[1],symbol=items[0].symbol)
    else: items[0]=replace(items[0],object_sha256='unknown')
    with pytest.raises(ValueError):emit(items)

@pytest.mark.parametrize('alignment',[True,7,12,2**32])
def test_alignment_refuses(alignment):
    with pytest.raises(ValueError):emit(imports(),workspace_alignment=alignment)

def test_native_descriptor_success_and_fallback(tmp_path):
    import subprocess
    from pathlib import Path
    source=emit(imports())
    definitions='\n'.join(f'void {i.symbol}(const int8_t*a,const int8_t*b,int32_t*c){{*c={i.degree+1};}}' for i in imports())
    args=', '.join(f'D4 *i{i}' for i in range(11))
    source+='\nstatic int calls, fallback_calls, refuse;\n'+definitions
    source+='''
int executor(const merlin_attention_view *v,merlin_attention_view*out,void*w,size_t n,merlin_attention_product product,void*ctx){
 if(refuse)return 0;
 int32_t z=0; int8_t a=0,b=0;
 if(!product(ctx,&a,&b,&z,3,8,4,4)||z!=5)return 0;
 if(v[0].offset!=2 || v[0].strides[3]!=2)return 0;
 ((uint16_t*)out->data)[out->offset]=123;calls++;return 1;
}
'''
    source+=f'void _mlir_ciface_source({args}, D4*out){{((uint16_t*)out->aligned)[out->offset]=456;fallback_calls++;}}\n'
    source+='''
int main(void){
 _Alignas(64) unsigned char storage[4096]; uint16_t data[16]={0},result[16]={0};
 D4 in={data,data,2,{1,1,1,2},{4,4,4,2}},out={result,result,1,{1,1,1,1},{1,1,1,1}};
 D1 workspace={storage,storage,0,{4096},{1}};
 writer(&in,&in,&in,&in,&in,&in,&in,&in,&in,&in,&in,&out,&workspace);
 if(calls!=1||fallback_calls||result[1]!=123||out.allocated!=result||out.offset!=1)return 1;
 refuse=1;
 writer(&in,&in,&in,&in,&in,&in,&in,&in,&in,&in,&in,&out,&workspace);
 if(fallback_calls!=1||result[1]!=456||out.aligned!=result)return 2;
 refuse=0;workspace.size[0]=4095;
 writer(&in,&in,&in,&in,&in,&in,&in,&in,&in,&in,&in,&out,&workspace);
 if(fallback_calls!=2||calls!=1)return 3;
 return 0;
}
'''
    c=tmp_path/'bridge.c';c.write_text(source)
    import merlin.llvmlower.source_attention_frontier as module
    headers=Path(module.__file__).resolve().parents[3]/'merlin/runtime/c'
    subprocess.run(['cc','-std=c11','-O2','-I',str(headers),str(c),'-o',str(tmp_path/'bridge')],check=True)
    subprocess.run([str(tmp_path/'bridge')],check=True)
