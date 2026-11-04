"""General scalar-region code generation and integer accelerator recognition."""
from math import prod
import re
import math
import struct
import json
from pathlib import Path
from scalar_float import emit_float_extremum

_SETTINGS_PATH = Path(__file__).resolve().parent / 'settings.json'
_LAYOUT_TILE_ROWS = json.loads(_SETTINGS_PATH.read_text()).get('layout_tile_rows',1) if _SETTINGS_PATH.exists() else 1

_CT={'i1':'unsigned char','i8':'signed char','i16':'short','i32':'int','i64':'long long','index':'long long','f32':'float','f64':'double'}

def _ct(t):
    if t not in _CT: raise ValueError('unsupported scalar type '+t)
    return _CT[t]

def _number(v):
    if isinstance(v,(int,float)):return v
    st=str(v).strip();raw=st.split(':')[0].strip()
    if raw in ('true','false'):return int(raw=='true')
    if raw.lower().startswith('0x') and ('f32' in st or 'f64' in st):
        bits=int(raw,16);return struct.unpack('>f' if 'f32' in st else '>d',bits.to_bytes(4 if 'f32' in st else 8,'big'))[0]
    try:return int(raw,0)
    except ValueError:return float(raw)

def _literal(v):
    v=_number(v)
    if isinstance(v,float):
        if math.isnan(v):return '__builtin_nanf("")'
        if math.isinf(v):return ('-' if v<0 else '')+'__builtin_inff()'
    return repr(v)

def _maps(raw):
    result=[]
    for m in raw:
        if isinstance(m,(list,tuple)): result.append(list(m));continue
        q=m.split('->',1)[-1].strip()
        if q.endswith('>'):q=q[:-1].strip()
        if not(q.startswith('(') and q.endswith(')')):raise ValueError('unsupported affine map '+m)
        result.append([x.strip() for x in q[1:-1].split(',') if x.strip()])
    return result

def _axis(x):
    if isinstance(x,int): return x
    m=re.fullmatch(r'd(\d+)',x)
    return int(m.group(1)) if m else None

def _index(mapping,shape):
    terms=[]
    for x,n in zip(mapping,shape):
        a=_axis(x)
        if a is None:
            e=str(x)
            if not re.fullmatch(r'[d0-9+* ()\-]+',e):raise ValueError('unsupported affine index '+e)
            e=re.sub(r'\bd(\d+)\b',lambda m:'v_d'+m.group(1),e)
        else:e='v_d'+str(a)
        terms.append((e,n))
    value='0'
    for e,n in terms:value=f'(({value})*{n}+({e}))'
    return value

def _expr(body):
    env={f'arg:{i}':('arg',i) for i in range(len(body['arguments']))}
    for i,c in enumerate(body.get('captures',[])):
        if 'const_value' in c:env[f'capture:{i}']=('const',c['const_value'])
        elif 'tensor' in c:env[f'capture:{i}']=('tensor',c['tensor'])
        else:raise ValueError('nonconstant unresolved scalar capture')
    for i,o in enumerate(body['operations']):
        args=[env[x] for x in o['operands']]
        if o['op']=='arith.constant': e=('const',_number(o['attributes']['value']))
        else:e=(o['op'],*args)
        env[f'op:{i}:0']=e
    return env[body['yields'][0]]

def _narrow(e):
    """Recognize exactly saturating signed i32-to-i8 (optionally ReLU)."""
    if e[0]!='arith.trunci':return None
    e=e[1];lo=-128;hi=127;relu=0
    if e[0]=='arith.minsi' and e[2]==('const',127):
        e=e[1]
    else:return None
    if e[0]=='arith.maxsi' and e[2] in [('const',-128),('const',0)]:
        relu=int(e[2]==('const',0));e=e[1]
    else:return None
    return (e[1],relu) if e[0]=='arg' else None

def _scalar_lines(body,arguments,prefix,names):
    env={f'arg:{i}':x for i,x in enumerate(arguments)};lines=[];value_shapes={}
    types={f'arg:{i}':t for i,t in enumerate(body['arguments'])}
    def unsigned(t):return {'i1':'unsigned char','i8':'unsigned char','i16':'unsigned short','i32':'unsigned int','i64':'unsigned long long','index':'unsigned long long'}[t]
    def signed_value(v,t):
        return f'(-((int)({v})&1))' if t=='i1' else f'(({_ct(t)})({v}))'
    def unsigned_value(v,t):
        return f'(({unsigned(t)})({v}))'
    for i,c in enumerate(body.get('captures',[])):
        types[f'capture:{i}']=c['dtype']
        if 'const_value' in c:env[f'capture:{i}']=f'(({_ct(c["dtype"])})({_literal(c["const_value"])}))'
        elif 'tensor' in c:
            env[f'capture:{i}']=names[c['tensor']];value_shapes[f'capture:{i}']=c.get('shape',[])
        else:raise ValueError('unsupported scalar capture')
    binops={'arith.addi':'+','arith.addf':'+','arith.subi':'-','arith.subf':'-','arith.muli':'*','arith.mulf':'*','arith.divf':'/','arith.divsi':'/','arith.remsi':'%','arith.xori':'^','arith.ori':'|','arith.andi':'&','arith.shli':'<<','arith.shrsi':'>>','arith.shrui':'>>','arith.divui':'/','arith.remui':'%'}
    casts={'arith.extsi','arith.extui','arith.trunci','arith.sitofp','arith.uitofp','arith.fptosi','arith.fptoui','arith.extf','arith.truncf','arith.index_cast'}
    for i,o in enumerate(body['operations']):
        if o.get('regions') or len(o['results'])!=1:raise ValueError('unsupported scalar nested regions/multiple results')
        op=o['op'];a=[env[x] for x in o['operands']];t=o['results'][0];at=o['attributes']
        if op=='arith.constant':e=_literal(at['value'])
        elif op in ('arith.addi','arith.subi','arith.muli','arith.shli'):
            # Widen narrow unsigned operands before C integer promotions can
            # introduce signed overflow. The final cast truncates modulo 2^w;
            # unsigned long long also defines i64 overflow modulo 2^64.
            u=[f'((unsigned long long){unsigned_value(v,types[r])})' for r,v in zip(o['operands'],a)]
            e=f'({u[0]} {binops[op]} {u[1]})'
        elif op in ('arith.shrui','arith.divui','arith.remui'):
            u=[f'((unsigned long long){unsigned_value(v,types[r])})' for r,v in zip(o['operands'],a)]
            e=f'({u[0]} {binops[op]} {u[1]})'
        elif op in ('arith.divsi','arith.remsi','arith.shrsi'):
            v=[signed_value(v,types[r]) for r,v in zip(o['operands'],a)]
            e=f'({v[0]} {binops[op]} {v[1]})'
        elif op in binops:e=f'({a[0]} {binops[op]} {a[1]})'
        elif op in ('arith.extui','arith.uitofp'):
            e=f'(({_ct(t)})({unsigned_value(a[0],types[o["operands"][0]])}))'
        elif op in ('arith.extsi','arith.sitofp'):
            e=f'(({_ct(t)})({signed_value(a[0],types[o["operands"][0]])}))'
        elif op=='arith.fptoui':e=f'(({unsigned(t)})({a[0]}))'
        elif op in casts:e=f'(({_ct(t)})({a[0]}))'
        elif op=='arith.negf':e=f'(-{a[0]})'
        elif op in ('arith.maximumf','arith.minimumf','arith.maxnumf','arith.minnumf'):
            e=emit_float_extremum(op,t,a[0],a[1])
        elif op in ['arith.maxsi','arith.maxui']:
            v=[unsigned_value(v,types[r]) if op=='arith.maxui' else signed_value(v,types[r]) if op=='arith.maxsi' else v for r,v in zip(o['operands'],a)]
            e=f'({v[0]} > {v[1]} ? {a[0]} : {a[1]})'
        elif op in ['arith.minsi','arith.minui']:
            v=[unsigned_value(v,types[r]) if op=='arith.minui' else signed_value(v,types[r]) if op=='arith.minsi' else v for r,v in zip(o['operands'],a)]
            e=f'({v[0]} < {v[1]} ? {a[0]} : {a[1]})'
        elif op=='arith.select':
            if t in ('f32','f64'):e=f'{"gm_scalar_select" if t=="f32" else "gm_scalar_select_double"}({a[0]},{a[1]},{a[2]})'
            else:e=f'({a[0]} ? {a[1]} : {a[2]})'
        elif op in ['arith.cmpi','arith.cmpf']:
            p=int(str(at['predicate']).split(':')[0].strip())
            if op=='arith.cmpi': cm={0:'==',1:'!=',2:'<',3:'<=',4:'>',5:'>=',6:'<',7:'<=',8:'>',9:'>='}
            else:cm={1:'==',2:'>',3:'>=',4:'<',5:'<=',6:'!=',9:'==',10:'>',11:'>=',12:'<',13:'<=',14:'!='}
            if p in cm:
                aa=a
                if op=='arith.cmpi' and p>=6:aa=[unsigned_value(v,types[r]) for r,v in zip(o['operands'],a)]
                elif op=='arith.cmpi' and 2<=p<=5:aa=[signed_value(v,types[r]) for r,v in zip(o['operands'],a)]
                e=f'({aa[0]} {cm[p]} {aa[1]})'
                if op=='arith.cmpf' and p==6:e=f'(({a[0]}=={a[0]}) & ({a[1]}=={a[1]}) & {e})'
                if op=='arith.cmpf' and p>=9:e=f'(({a[0]}!={a[0]}) | ({a[1]}!={a[1]}) | {e})'
            elif op=='arith.cmpf' and p in [0,15]:e=str(int(p==15))
            elif op=='arith.cmpf' and p in [7,8]:e=f'(({a[0]}=={a[0]}) & ({a[1]}=={a[1]}))' if p==7 else f'(({a[0]}!={a[0]}) | ({a[1]}!={a[1]}))'
            else:raise ValueError('unsupported comparison')
        elif op=='tensor.extract':
            srcshape=value_shapes.get(o['operands'][0])
            if srcshape is None or len(srcshape)!=len(a)-1:raise ValueError('tensor.extract requires ranked captured tensor')
            idx='0'
            for index,size in zip(a[1:],srcshape):idx=f'(({idx})*{size}+({index}))'
            e=f'{a[0]}[{idx}]'
        elif op=='linalg.index':e='v_d'+str(int(str(at['dim']).split(':')[0].strip()))
        elif op=='math.roundeven':e=f'gm_scalar_roundeven({a[0]})'
        elif op=='math.exp':e=f'gm_scalar_exp({a[0]})'
        elif op=='math.tanh':e=f'gm_scalar_tanh({a[0]})'
        elif op=='math.sin':e=f'gm_scalar_sincos({a[0]},0)'
        elif op=='math.cos':e=f'gm_scalar_sincos({a[0]},1)'
        elif op=='math.rsqrt':e=f'(1.0f/gm_scalar_sqrt({a[0]}))'
        elif op=='math.sqrt':e=f'gm_scalar_sqrt({a[0]})'
        elif op=='math.absf':e=f'{"__builtin_fabsf" if t=="f32" else "__builtin_fabs"}({a[0]})'
        else:raise ValueError('unsupported scalar opcode '+op)
        if t=='i1':e=f'((unsigned int)({e}) & 1u)'
        types[f'op:{i}:0']=t
        v=f'{prefix}_{i}';lines.append(f'{_ct(t)} {v} = ({_ct(t)})({e});');env[f'op:{i}:0']=v
    return lines,[env[x] for x in body['yields']]

_SELECT=r'''#ifndef GM_SCALAR_SELECT_DEFINED
#define GM_SCALAR_SELECT_DEFINED
GM_INLINE float gm_scalar_select(int c,float a,float b) { union {float f;unsigned int u;} x,y,z;x.f=a;y.f=b;unsigned int mask=0u-(unsigned int)(c!=0);z.u=(x.u&mask)|(y.u&~mask);return z.f; }
GM_INLINE double gm_scalar_select_double(int c,double a,double b) { union {double f;unsigned long long u;} x,y,z;x.f=a;y.f=b;unsigned long long mask=0ull-(unsigned long long)(c!=0);z.u=(x.u&mask)|(y.u&~mask);return z.f; }
#endif'''

_ROUND=_SELECT+r'''
#ifndef GM_SCALAR_ROUNDEVEN_DEFINED
#define GM_SCALAR_ROUNDEVEN_DEFINED
GM_INLINE float gm_scalar_roundeven(float x) {
  int small=(x<8388608.0f)&(x>-8388608.0f);
  float safe=gm_scalar_select(small,x,0.0f);
  int q=(int)safe;float d=safe-(float)q;int odd=q&1;
  int up=(d>0.5f)|((d==0.5f)&odd);
  int down=(d < -0.5f)|((d==-0.5f)&odd);
  q+=up-down;
  float rounded=gm_scalar_select(q==0,x*0.0f,(float)q);
  return gm_scalar_select(small,rounded,x);
}
#endif'''

def _emit_contiguous_movement(x,y,count,in_bytes,out_bytes,scale='1.0f',relu=0):
    """Use full 16-column DMA rows, with exact byte offsets for the tail."""
    rows,tail=divmod(count,16);bulk=rows*16;lines=[]
    if rows:
        lines.append(f'gm_movement({x},{y},{rows},16,{in_bytes},{out_bytes},{scale},{relu});')
    if tail:
        xp=f'((const unsigned char*)({x})+{bulk*in_bytes})'
        yp=f'((unsigned char*)({y})+{bulk*out_bytes})'
        lines.append(f'gm_movement({xp},{yp},1,{tail},{in_bytes},{out_bytes},{scale},{relu});')
    return lines


def _emit_contiguous_residual(x,r,y,count,out_bytes,lhs_scale='1.0f',rhs_scale='1.0f',relu=0):
    """Tile equal-shape byte inputs without changing accumulator readout."""
    rows,tail=divmod(count,16);bulk=rows*16;lines=[]
    if rows:
        lines.append(f'gm_residual({x},{r},{y},{rows},16,{out_bytes},{lhs_scale},{rhs_scale},{relu});')
    if tail:
        xp=f'((const signed char*)({x})+{bulk})'
        rp=f'((const signed char*)({r})+{bulk})'
        yp=f'((unsigned char*)({y})+{bulk*out_bytes})'
        lines.append(f'gm_residual({xp},{rp},{yp},1,{tail},{out_bytes},{lhs_scale},{rhs_scale},{relu});')
    return lines


def emit_vector(cmd,cb,names,unique):
    unique=unique() if callable(unique) else unique
    a=cmd.setdefault('attributes',{})
    a['lower_lane']=a['lowering_lane']='device'
    def host_lane(reason):
        a['lower_lane']=a['lowering_lane']='host';a['host_reason']=reason
    allow_device=a.get('prov',{}).get('placement')!='host_required'
    if allow_device:
        from device_layout import emit_layout
        copied = emit_layout(cmd,cb,names,unique,tile_rows=_LAYOUT_TILE_ROWS)
        if copied is not None:
            return copied
    o=cmd['operands'];ts=cb['tensors'];dst=o['dst'];out=ts[dst];dt=out['dtype'];shape=out['shape']
    names={k:f'(({_ct(ts[k]["dtype"])}*)({v}))' for k,v in names.items() if k in ts}
    Y=names[dst]
    if a.get('op')=='insert_slice':
        X=names[o['src0']];D=names[o['src1']];sizes=a['sizes'];offsets=a['offsets'];strides=a['strides'];count=prod(shape)
        host_lane('source tensor layout insertion requires indexed host data movement')
        lines=[f'for(unsigned long vx=0;vx<{count};++vx){Y}[vx]={D}[vx];']
        for k,size in enumerate(sizes):lines.append(f'for(int vs{k}=0;vs{k}<{size};++vs{k}){{')
        srcidx='0';dstidx='0'
        for k,(size,dim,off,step) in enumerate(zip(sizes,shape,offsets,strides)):
            srcidx=f'(({srcidx})*{size}+vs{k})';dstidx=f'(({dstidx})*{dim}+{off}+vs{k}*{step})'
        lines.append(f'{Y}[{dstidx}]={X}[{srcidx}];');lines+=['}']*len(sizes)
        return [],lines
    if a.get('op')=='concat':
        axis=a['axis'];axis=axis+len(shape) if axis<0 else axis;inner=prod(shape[axis+1:]);outer=prod(shape[:axis]);offset=0;lines=[]
        host_lane('source tensor concatenation requires indexed host data movement')
        srckeys=sorted((k for k in o if k.startswith('src') and k[3:].isdigit()),key=lambda k:int(k[3:]))
        for key in srckeys:
            name=o[key];width=ts[name]['shape'][axis];X=names[name]
            lines.append(f'for(unsigned long vb=0;vb<{outer};++vb)for(unsigned long vi=0;vi<{width*inner};++vi){Y}[vb*{shape[axis]*inner}+{offset*inner}+vi]={X}[vb*{width*inner}+vi];');offset+=width
        if offset!=shape[axis]:raise ValueError('concat axis shape mismatch')
        return [],lines
    if a.get('op')=='row_major_copy':
        src=o.get('src0',o.get('src'))
        if ts[src]['dtype']!=dt or prod(ts[src]['shape'])!=prod(shape):raise ValueError('row-major view changes storage')
        count=prod(shape)
        if allow_device and dt in ['i8','i32']:
            b=1 if dt=='i8' else 4
            return [],_emit_contiguous_movement(names[src],Y,count,b,b)
        host_lane('layout copy of dtype unsupported by integer accelerator movement')
        return [],[f'for(unsigned long vc=0;vc<{count};++vc) {Y}[vc]={names[src]}[vc];']
    srcs=[o[k] for k in sorted((k for k in o if k.startswith('src') and k[3:].isdigit()),key=lambda k:int(k[3:]))]
    if not srcs: srcs=[o[k] for k in ('lhs','rhs','src') if k in o]
    records=a.get('inputs') or [{'tensor':x} for x in srcs]
    input_specs=[ts[r['tensor']] if 'tensor' in r else r for r in records]
    input_names=[names[r['tensor']] if 'tensor' in r else None for r in records]
    body=a.get('scalar_body');red=cmd['opcode']=='VREDUCE';n=prod(shape)
    if not body:
        if red:
            body={'arguments':[ts[srcs[0]]['dtype'],dt],'captures':[],'operations':[{'op':'arith.addf' if dt.startswith('f') else 'arith.addi','operands':['arg:0','arg:1'],'results':[dt],'attributes':{}}],'yields':['op:0:0']}
        else:
            op={'add':'arith.add','mul':'arith.mul'}.get(a.get('combine'))
            if not op:raise ValueError('VECTOR_MAP requires scalar body or supported combine')
            body={'arguments':[dt,dt,dt],'captures':[],'operations':[{'op':op+('f' if dt.startswith('f') else 'i'),'operands':['arg:0','arg:1'],'results':[dt],'attributes':{}}],'yields':['op:0:0']}
    rawmaps=a.get('indexing_maps')
    if rawmaps:maps=_maps(rawmaps)
    elif red:maps=[list(range(len(ts[srcs[0]]['shape']))),list(range(len(shape)))];
    else:maps=[list(range(len(shape))) for _ in range(len(srcs)+1)]
    rank=max([int(k)+1 for m in maps for x in m for k in (re.findall(r'd(\d+)',str(x)) if not isinstance(x,int) else [str(x)])]+[0]);dims=[None]*rank
    extent_pairs=list(zip(maps,[x.get('shape',[]) for x in input_specs]+[shape]))
    for m,s in [extent_pairs[-1]]+extent_pairs[:-1]:
        for x,d in zip(m,s):
            k=_axis(x)
            if k is not None:
                if dims[k] is None:dims[k]=d
    if any(x is None for x in dims):raise ValueError('cannot infer loop domain')
    ident=list(range(rank));identity=lambda m:m==ident or m==[f'd{i}' for i in ident]
    expr=_expr(body)
    if not red and expr[0]=='arg' and expr[1]<len(records) and 'value' in records[expr[1]]:
        value=records[expr[1]]['value']
        if allow_device:
            from device_fill import emit_fill
            filled = emit_fill(dt,value,n,Y,'vfill_'+str(unique))
            if filled is not None:
                return filled
        host_lane('source scalar constant tensor initialization')
        return [],[f'for(unsigned long vf=0;vf<{n};++vf) {Y}[vf]=({_ct(dt)})({_literal(value)});']
    if allow_device and len(records)==len(srcs) and not red and len(srcs)==1 and identity(maps[0]) and identity(maps[-1]):
        if expr==('arith.extsi',('arg',0)) and ts[srcs[0]]['dtype']=='i8' and dt=='i32':
            return [],_emit_contiguous_movement(names[srcs[0]],Y,n,1,4)
        if expr in [('arith.maxsi',('arg',0),('const',0)),('arith.maxsi',('const',0),('arg',0))] and ts[srcs[0]]['dtype']=='i8' and dt=='i8':
            return [],_emit_contiguous_movement(names[srcs[0]],Y,n,1,1,relu=1)
        narrow=_narrow(expr)
        if narrow and narrow[0]==0 and ts[srcs[0]]['dtype']=='i32' and dt=='i8':
            return [],_emit_contiguous_movement(names[srcs[0]],Y,n,4,1,relu=narrow[1])
        if expr==('arg',0) and ts[srcs[0]]['dtype']==dt and dt in ['i8','i32']:
            b=1 if dt=='i8' else 4
            return [],_emit_contiguous_movement(names[srcs[0]],Y,n,b,b)
    if allow_device and len(records)==len(srcs) and not red and len(srcs)==2 and expr in [('arith.addi',('arg',0),('arg',1)),('arith.addi',('arg',1),('arg',0))]:
        if dt=='i32' and all(ts[x]['dtype']=='i32' for x in srcs) and identity(maps[0]) and identity(maps[-1]):
            M=prod(shape[:-1]);N=shape[-1]
            if identity(maps[1]):return [],[f'gm_add_i32_flat({names[srcs[0]]},{names[srcs[1]]},{Y},{n});']
            if maps[1] in [[rank-1],[f'd{rank-1}']]:return [],[f'gm_bias_add({names[srcs[0]]},{names[srcs[1]]},{Y},{M},{N});']
        if all(ts[x]['dtype']=='i8' for x in srcs) and dt=='i32' and all(identity(m) for m in maps):
            return [],_emit_contiguous_residual(names[srcs[0]],names[srcs[1]],Y,n,4)
    if allow_device and red and len(srcs)==1 and expr in [('arith.addi',('arg',0),('arg',1)),('arith.addi',('arg',1),('arg',0))] and dt=='i32':
        ss=ts[srcs[0]];K=ss['shape'][-1];M=prod(ss['shape'][:-1]);X=names[srcs[0]]
        if a.get('reduction_dims',[len(ss['shape'])-1])==[len(ss['shape'])-1] and shape==ss['shape'][:-1]:
            ini=a.get('initializers',[{'value':0}])[0]
            if ini.get('value')!=0:raise ValueError('integer accelerator reduction requires zero fill')
            z='vr'+str(unique)
            if ss['dtype']=='i8':
                return [f'static signed char {z}_ones[{K}] __attribute__((aligned(64)));'],[f'for(int vk=0;vk<{K};++vk){z}_ones[vk]=1;',f'gm_matmul({X},{z}_ones,{Y},{M},{K},1,4,1.0f,0,0);']
            if ss['dtype']=='i32':
                return [f'static int {z}_gather[{M}] __attribute__((aligned(64)));'],[f'for(int vm=0;vm<{M};++vm){Y}[vm]=0;',f'for(int vk=0;vk<{K};++vk) {{',f'for(int vm=0;vm<{M};++vm){z}_gather[vm]={X}[vm*{K}+vk];',f'gm_bias_add({Y},{z}_gather,{Y},1,{M});','}']
    # Scalar lanes are reserved for operations the integer accelerator cannot express.
    ops={x['op'] for x in body['operations']};unsupported_integer=bool(ops & {'arith.xori','arith.ori','arith.andi','arith.divsi','arith.remsi','arith.shli','arith.shrsi','arith.shrui','arith.divui','arith.remui','linalg.index'}) or dt in ('i64','index') or any(t in ('i64','index') for t in body['arguments'])
    floats=any(x.startswith('f') for x in body['arguments']) or dt.startswith('f')
    if not red and expr==('arg',0) and len(srcs)==1 and ts[srcs[0]]['dtype']==dt:
        unsupported_integer=True
    if not floats and not unsupported_integer and allow_device:
        raise ValueError('integer scalar region requires accelerator lowering: '+str(expr))
    if a.get('prov',{}).get('placement','').startswith('accelerator'):
        raise ValueError('required accelerator placement cannot use scalar lane')
    host_lane('source requires host placement' if not allow_device else 'target integer PE and accumulator lack floating-point scalar arithmetic' if floats else 'target instruction set lacks source scalar bitwise, index, wide-integer or integer division operation')
    lines=['{'];prefix='v'+str(unique)
    if red:
        init=a.get('initializers',[{'value':a.get('init_value',0)}])[0]
        if isinstance(init,dict) and 'tensor' in init:
            lines.append(f'for (unsigned long vi=0;vi<{n};++vi) {Y}[vi]={names[init["tensor"]]}[vi];')
            init=None
        elif isinstance(init,dict):init=init.get('value')
        if init is not None: lines.append(f'for (unsigned long vi=0;vi<{n};++vi) {Y}[vi]=({_ct(dt)})({_literal(init)});')
    for i,d in enumerate(dims):lines.append(f'for (int v_d{i}=0;v_d{i}<{d};++v_d{i}) {{')
    outidx=_index(maps[-1],shape)
    args=[f'{name}[{_index(m,spec.get("shape",[]))}]' if name is not None else f'(({_ct(spec["dtype"])})({_literal(rec["value"])}))' for rec,spec,name,m in zip(records,input_specs,input_names,maps)]
    args += [f'{Y}[{outidx}]']
    sl,ys=_scalar_lines(body,args,prefix,names);lines+=sl
    if len(ys)!=1:raise ValueError('vector region requires single output')
    lines.append(f'{Y}[{outidx}]={ys[0]};');lines+=['}']*rank;lines.append('}')
    helpers=[_SELECT] if floats or any(t in ('f32','f64') for op in body['operations'] for t in op['results']) else []
    if 'math.roundeven' in ops:helpers.append(_ROUND)
    if ops & {'math.exp','math.rsqrt','math.sqrt','math.tanh','math.sin','math.cos'}:helpers.append(_MATH)
    return helpers,lines

_MATH=_SELECT+r'''
#ifndef GM_SCALAR_MATH_DEFINED
#define GM_SCALAR_MATH_DEFINED
GM_INLINE float gm_scalar_sqrt(float x) { float y; __asm__("fsqrt.s %0,%1" : "=f"(y) : "f"(x)); return y; }
GM_INLINE float gm_scalar_exp(float x) {
  int in_range=(x<=88.722839f)&(x>=-103.972084f);
  float safe=gm_scalar_select(in_range,x,0.0f);
  float q=safe*1.4426950408889634f;
  int k=(int)(q+gm_scalar_select(q>=0,0.5f,-0.5f));
  float r=(safe-k*0.693145751953125f)-k*1.428606765330187e-6f;
  float p=1.0f+r*(1.0f+r*(0.5f+r*(0.1666666666666667f+r*(0.0416666666666667f+r*(0.0083333333333333f+r*(0.0013888888888889f+r*(0.0001984126984127f+r*0.0000248015873016f)))))));
  int low=k < -126,high=k>127;
  union { unsigned int u; float f; } scale;
  scale.u=(unsigned int)(k+127+24*low-high)<<23;
  float factor=gm_scalar_select(low,0x1p-24f,gm_scalar_select(high,2.0f,1.0f));
  float result=(p*scale.f)*factor;
  result=gm_scalar_select(x < -103.972084f,0.0f,result);
  result=gm_scalar_select(x>88.722839f,__builtin_inff(),result);
  return gm_scalar_select(x!=x,x,result);
}
GM_INLINE float gm_scalar_sincos(float x,int cosine) {
  union {float f;unsigned int u;} bits;bits.f=x;
  int sign=(int)(bits.u>>31), exponent=(int)((bits.u>>23)&255)-127;
  union {float f;unsigned int u;} absolute;absolute.u=bits.u&0x7fffffffu;
  int reduce=exponent>=-1;
  int safe_exponent=exponent*reduce-(1-reduce);
  /* floor((2/pi)*2^192), little-endian. A 24-bit significand
     times this constant leaves at least 65 fraction bits at FLT_MAX. */
  const unsigned int invpi[6]={0x3c439041u,0xdb629599u,0xf534ddc0u,0xfc2757d1u,0x4e441529u,0xa2f9836eu};
  unsigned int product[7];unsigned long long carry=0;
  unsigned int mantissa=(bits.u&0x7fffffu)|0x800000u;
  for(int j=0;j<6;++j){unsigned long long v=(unsigned long long)mantissa*invpi[j]+carry;product[j]=(unsigned int)v;carry=v>>32;}
  product[6]=(unsigned int)carry;
  int shift=215-safe_exponent;
  int k=(int)((product[shift>>5]>>(shift&31))&1);
  k|=(int)(((product[(shift+1)>>5]>>((shift+1)&31))&1)<<1);
  unsigned long long fraction=0;
  for(int j=1;j<65;++j){int pos=shift-j;fraction=(fraction<<1)|((product[pos>>5]>>(pos&31))&1);}
  double f=(double)fraction*0x1p-64;
  int round_up=f>=0.5;f-=(double)round_up;k=(k+round_up)&3;
  double r=gm_scalar_select_double(reduce,f*1.57079632679489661923,(double)absolute.f);
  double z=r*r;
  double sn=r+r*z*(-1.66666666666666324348e-1+z*(8.33333333332248946124e-3+z*(-1.98412698298579493134e-4+z*(2.75573137070700676789e-6+z*(-2.50507602534068634195e-8+z*1.58969099521155010221e-10)))));
  double cs=1.0-0.5*z+z*z*(4.16666666666666019037e-2+z*(-1.38888888888741095749e-3+z*(2.48015872894767294178e-5+z*(-2.75573143513906633035e-7+z*(2.08757232129817482790e-9+z*(-1.13596475577881948265e-11))))));
  /* Small inputs bypass reduction and therefore use quadrant zero. */
  k*=reduce;
  int quadrant=(k+cosine)&3;
  double wave=gm_scalar_select_double(quadrant==0,sn,gm_scalar_select_double(quadrant==1,cs,gm_scalar_select_double(quadrant==2,-sn,-cs)));
  union {float f;unsigned int u;} answer;answer.f=(float)wave;
  answer.u^=((unsigned int)sign & (unsigned int)(cosine==0))<<31;
  int infinity=(bits.u&0x7fffffffu)==0x7f800000u;
  float result=gm_scalar_select(infinity,__builtin_nanf(""),answer.f);
  return gm_scalar_select(x!=x,x,result);
}
GM_INLINE float gm_scalar_tanh(float x) {
  union {float f;unsigned int u;} absolute;absolute.f=x;absolute.u&=0x7fffffffu;
  float v=absolute.f;
  float bounded=gm_scalar_select(v<=10.0f,v,10.0f);
  float small_x=gm_scalar_select(v<0.01f,x,0.0f),z=small_x*small_x;
  float polynomial=small_x*(1.0f+z*(-0.3333333333333333f+z*(0.1333333333333333f+z*(-0.0539682539682540f))));
  float e=gm_scalar_exp(2.0f*bounded),t=(e-1.0f)/(e+1.0f);
  float signed_t=gm_scalar_select(x<0,-t,t);
  float saturated=gm_scalar_select(x<0,-1.0f,1.0f);
  float answer=gm_scalar_select(v<0.01f,polynomial,signed_t);
  answer=gm_scalar_select(v>10.0f,saturated,answer);
  return gm_scalar_select(x!=x,x,answer);
}
#endif'''
