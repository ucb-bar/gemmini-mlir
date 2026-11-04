"""Exact materialized tensor layout copies through accelerator DMA.

No scalar tensor values are read or written by generated host instructions.
Affine maps are checked as linear integer expressions over the static domain;
all source indices are proved in bounds before selecting the device path.
"""
from __future__ import annotations
import ast
import re
from math import prod


def _linear(expr, rank):
    if isinstance(expr, int):
        if not 0 <= expr < rank:
            raise ValueError('affine dimension outside domain')
        return 0, [int(i==expr) for i in range(rank)]
    def visit(n):
        if isinstance(n,ast.Constant) and type(n.value) is int:
            return n.value,[0]*rank
        if isinstance(n,ast.Name) and re.fullmatch(r'd\d+',n.id):
            axis=int(n.id[1:])
            if axis>=rank:raise ValueError('affine dimension outside domain')
            return 0,[int(i==axis) for i in range(rank)]
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.USub,ast.UAdd)):
            c,v=visit(n.operand);sgn=-1 if isinstance(n.op,ast.USub) else 1
            return c*sgn,[x*sgn for x in v]
        if isinstance(n,ast.BinOp):
            c,a=visit(n.left);d,b=visit(n.right)
            if isinstance(n.op,(ast.Add,ast.Sub)):
                s=-1 if isinstance(n.op,ast.Sub) else 1
                return c+s*d,[x+s*y for x,y in zip(a,b)]
            if isinstance(n.op,ast.Mult):
                if not any(a):return c*d,[c*y for y in b]
                if not any(b):return c*d,[d*x for x in a]
        raise ValueError('nonlinear/unsupported affine copy map')
    return visit(ast.parse(str(expr),mode='eval').body)


def _map(raw):
    if isinstance(raw,(list,tuple)):return list(raw)
    text=raw.split('->',1)[1].strip()
    if text.endswith('>'):text=text[:-1].strip()
    if not(text.startswith('(') and text.endswith(')')):raise ValueError('unsupported affine copy map')
    return [s.strip() for s in text[1:-1].split(',') if s.strip()]


def _flatten(mapping, shape, rank):
    if len(mapping)!=len(shape):raise ValueError('affine map rank mismatch')
    constant,coeff=0,[0]*rank
    for expr,size in zip(mapping,shape):
        c,v=_linear(expr,rank)
        constant=constant*size+c
        coeff=[a*size+b for a,b in zip(coeff,v)]
    return constant,coeff


def _expr(base, coefficients, names):
    terms=[str(base)] if base else []
    terms += [f'((gm_u64){name}*{coefficient})' for name,coefficient in zip(names,coefficients) if coefficient]
    return '+'.join(terms) or '0'


def _bulk(src,dst,count,width,tag):
    """Exact linear copy with bounded 16x16 accumulator tiles."""
    full=count//16;tail=count%16
    body=[f'gm_config_ld({16*width},0,{int(width==1)});',f'gm_config_st({16*width},1.0f,0);']
    if full:
        body += [f'for(int {tag}_row=0;{tag}_row<{full};{tag}_row+=16) {{',
                 f'int mr=gm_min16({full}-{tag}_row);',
                 f'gm_mvin((const void*)((gm_u64)({src})+(gm_u64){tag}_row*{16*width}),0x80000000u,mr,16,0);',
                 f'gm_mvout((void*)((gm_u64)({dst})+(gm_u64){tag}_row*{16*width}),{hex(0x80000000 if width==1 else 0xa0000000)}u,mr,16);','}']
    if tail:
        body += [f'gm_mvin((const void*)((gm_u64)({src})+{full*16*width}),0x80000000u,1,{tail},0);',
                 f'gm_mvout((void*)((gm_u64)({dst})+{full*16*width}),{hex(0x80000000 if width==1 else 0xa0000000)}u,1,{tail});']
    body.append('gm_fence();')
    return body


def _affine_copy(src,dst,dims,source,dest,width,tag,tile_rows=1):
    sc,sv=source;dc,dv=dest
    # Prefer a contiguous vector, otherwise independent DMA row strides move
    # up to sixteen affine coordinates with a single load/store pair.
    contiguous=[i for i,n in enumerate(dims) if sv[i]==dv[i]==1 and n>1]
    if contiguous:axis=max(contiguous,key=lambda i:dims[i]);columns=True
    else:
        candidates=[i for i,n in enumerate(dims) if dv[i]>0 and sv[i]>=0 and n>1]
        axis=max(candidates,key=lambda i:dims[i]) if candidates else None
        columns=False
    row_axis=None
    if columns and tile_rows>1:
        candidates=[i for i,n in enumerate(dims) if i!=axis and n>1 and sv[i]>0 and dv[i]>0]
        if candidates:row_axis=max(candidates,key=lambda i:dims[i])
    stride_axis=row_axis if columns else axis
    variables=[f'{tag}_d{i}' for i in range(len(dims))]
    outer=[i for i in range(len(dims)) if i not in (axis,row_axis)]
    body=[f'gm_config_ld({0 if stride_axis is None else sv[stride_axis]*width},0,{int(width==1)});',
          f'gm_config_st({0 if stride_axis is None else dv[stride_axis]*width},1.0f,0);']
    for i in outer:body.append(f'for(int {variables[i]}=0;{variables[i]}<{dims[i]};++{variables[i]}) {{')
    if row_axis is not None:
        body += [f'for(int {variables[row_axis]}=0;{variables[row_axis]}<{dims[row_axis]};{variables[row_axis]}+={tile_rows}) {{',
                 f'int nr=gm_min16({dims[row_axis]}-{variables[row_axis]});']
    if axis is not None:
        body += [f'for(int {variables[axis]}=0;{variables[axis]}<{dims[axis]};{variables[axis]}+=16) {{',
                 f'int nc=gm_min16({dims[axis]}-{variables[axis]});']
    source_offset=_expr(sc,sv,variables)
    dest_offset=_expr(dc,dv,variables)
    rows='nr' if row_axis is not None else ('1' if columns or axis is None else 'nc')
    cols='nc' if columns else '1'
    body += [f'gm_mvin((const void*)((gm_u64)({src})+({source_offset})*{width}),0x80000000u,{rows},{cols},0);',
             f'gm_mvout((void*)((gm_u64)({dst})+({dest_offset})*{width}),{hex(0x80000000 if width==1 else 0xa0000000)}u,{rows},{cols});']
    body += ['}']*(len(outer)+int(axis is not None)+int(row_axis is not None))
    body.append('gm_fence();')
    return body


def emit_layout(cmd, cb, names, unique, tile_rows=1):
    """Return exact device-copy (globals, body), or None for another operation."""
    if tile_rows not in (1,16):raise ValueError('layout tile_rows must be 1 or 16')
    if cmd.get('opcode')!='VECTOR_MAP':return None
    attrs=cmd.get('attributes',{})
    if attrs.get('prov',{}).get('placement')=='host_required':return None
    op=cmd['operands'];ts=cb['tensors'];y=op['dst'];dtype=ts[y]['dtype']
    # f32 is transported as raw bits over the full i32 accumulator path; no
    # floating point computation or conversion is introduced by this copy.
    # Public FIR: LoadController.scala108 selects has_acc_bitwidth when
    # shrink=0; Scratchpad.scala444-456 sends that stream to vsm_1. Its
    # VectorScalarMultiplier.scala94 directly wires each i32 input to output
    # (FIR159006-159021), bypassing the floating-point i8 load scaler.
    if dtype not in ('i8','i32','f32'):return None
    width=1 if dtype=='i8' else 4
    x=op.get('src0',op.get('src'))
    if x is None or ts[x]['dtype']!=dtype:return None
    ident=unique() if callable(unique) else unique
    tag='dma_layout_'+re.sub(r'[^A-Za-z0-9_]','_',str(ident))
    glob=[];body=[];src=names[x];dst=names[y]
    kind=attrs.get('op')
    if kind=='row_major_copy':
        if prod(ts[x]['shape'])!=prod(ts[y]['shape']):return None
        if x==y:return [],[]
        return [],_bulk(src,dst,prod(ts[y]['shape']),width,tag)
    if kind=='concat':
        shape=ts[y]['shape'];rank=len(shape)
        axis=int(attrs['axis']);axis=axis+rank if axis<0 else axis
        if not 0<=axis<rank:return None
        keys=sorted((k for k in op if k.startswith('src') and k[3:].isdigit()),key=lambda k:int(k[3:]))
        offset=0
        for index,key in enumerate(keys):
            name=op[key];spec=ts[name];sizes=spec['shape']
            if spec['dtype']!=dtype or len(sizes)!=rank or any(s<=0 for s in sizes):return None
            if any(s!=n for i,(s,n) in enumerate(zip(sizes,shape)) if i!=axis):return None
            if offset+sizes[axis]>shape[axis]:return None
            source=names[name]
            if name==y and len(keys)>1:return None
            identity=[f'd{i}' for i in range(rank)]
            destmap=[f'd{i}+{offset}' if i==axis else f'd{i}' for i in range(rank)]
            body += _affine_copy(source,dst,sizes,_flatten(identity,sizes,rank),_flatten(destmap,shape,rank),width,tag+f'_part{index}',tile_rows)
            offset+=sizes[axis]
        if offset!=shape[axis]:return None
        return glob,body
    if kind=='insert_slice':
        base=op.get('src1')
        if base is None or ts[base]['dtype']!=dtype or ts[base]['shape']!=ts[y]['shape']:return None
        sizes,offsets,strides=attrs['sizes'],attrs['offsets'],attrs['strides']
        shape=ts[y]['shape'];rank=len(shape)
        if len(sizes)!=rank or len(offsets)!=rank or len(strides)!=rank or list(sizes)!=ts[x]['shape']:return None
        if any(s<=0 or step<=0 or off<0 or off+(s-1)*step>=n for s,off,step,n in zip(sizes,offsets,strides,shape)):return None
        if x in (y,base):
            snapshot=tag+'_source'
            glob.append(f'static unsigned char {snapshot}[{prod(sizes)*width}] __attribute__((aligned(64)));')
            body += _bulk(src,snapshot,prod(sizes),width,tag+'_save')
            src=snapshot
        if base!=y:body += _bulk(names[base],dst,prod(shape),width,tag+'_base')
        identity=[f'd{i}' for i in range(rank)]
        destmap=[f'{off}+d{i}*{step}' for i,(off,step) in enumerate(zip(offsets,strides))]
        body += _affine_copy(src,dst,sizes,_flatten(identity,sizes,rank),_flatten(destmap,shape,rank),width,tag,tile_rows)
        return glob,body
    scalar=attrs.get('scalar_body',{})
    records=attrs.get('inputs',[])
    if scalar.get('operations') or scalar.get('captures') or scalar.get('yields')!=['arg:0'] or not records or records[0].get('tensor')!=x:return None
    if any(it!='parallel' for it in attrs.get('iterator_types',[])):return None
    maps=attrs.get('indexing_maps')
    if maps is None or len(maps)!=len(records)+1:return None
    try:
        sm,dm=_map(maps[0]),_map(maps[-1]);rank=len(attrs.get('iterator_types',[]))
        dims=[None]*rank
        for expr,size in zip(dm,ts[y]['shape']):
            c,v=_linear(expr,rank)
            axes=[i for i,a in enumerate(v) if a]
            if not axes:
                if c!=0 or size!=1:return None
            elif len(axes)==1 and c==0 and v[axes[0]]==1:
                axis=axes[0]
                if dims[axis] is not None:return None
                dims[axis]=size
            else:return None
        if any(n is None or n<=0 for n in dims):return None
        for expr,size in zip(sm,ts[x]['shape']):
            c,v=_linear(expr,rank)
            low=c+sum(min(0,a*(n-1)) for a,n in zip(v,dims))
            high=c+sum(max(0,a*(n-1)) for a,n in zip(v,dims))
            if low<0 or high>=size:return None
        source,dest=_flatten(sm,ts[x]['shape'],rank),_flatten(dm,ts[y]['shape'],rank)
        if any(v<0 for v in source[1]+dest[1]):return None
    except (ValueError,SyntaxError,IndexError,TypeError):return None
    if x==y:
        if source==dest:return [],[]
        snapshot=tag+'_source'
        glob.append(f'static unsigned char {snapshot}[{prod(ts[x]["shape"])*width}] __attribute__((aligned(64)));')
        body += _bulk(src,snapshot,prod(ts[x]['shape']),width,tag+'_save')
        src=snapshot
    if tile_rows>1 and source==dest and source[0]==0:
        # The checked output map is bijective over the whole output tensor,
        # so equal flattened maps visit the same complete linear storage.
        body += _bulk(src,dst,prod(ts[y]['shape']),width,tag)
    else:
        body += _affine_copy(src,dst,dims,source,dest,width,tag,tile_rows)
    return glob,body
