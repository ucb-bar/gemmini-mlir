"""Optional exact source im2col composition into the shared NHWC convolution."""
from math import prod

class WindowProductError(ValueError):pass


def window_geometry(source_shape,pieces,rhs_shape,output_shape,dtype='i8'):
    """Prove the original NCHW coordinate maps in concatenated reduction order."""
    from merlin.compile.scheduling.window_transfer import Window2D
    source_shape=tuple(source_shape);rhs_shape=tuple(rhs_shape);output_shape=tuple(output_shape)
    if dtype!='i8' or len(source_shape)!=4 or len(rhs_shape)!=2 or not pieces:
        raise WindowProductError('window product needs static i8 NCHW source and matrix RHS')
    if any(type(x) is not int or x<=0 for s in (source_shape,rhs_shape,output_shape) for x in s):
        raise WindowProductError('window product requires positive static geometry')
    b,ci,h,w=source_shape
    offsets=[tuple(p['offsets']) for p in pieces];sizes=[tuple(p['sizes']) for p in pieces];strides=[tuple(p['strides']) for p in pieces]
    if (any(len(v)!=4 or any(type(x) is not int for x in v) for rows in (offsets,sizes,strides) for v in rows)
            or any(x<0 for row in offsets for x in row) or any(x<=0 for rows in (sizes,strides) for row in rows for x in row)):
        raise WindowProductError('window maps must have four static integer axes with positive extents and strides')
    oh,ow=sizes[0][2:];st=strides[0]
    if (any(s!=(b,ci,oh,ow) for s in sizes) or any(s!=st for s in strides)
            or st[:2]!=(1,1) or any(type(v) is not int or v<=0 for v in st)):
        raise WindowProductError('window branches need equal full batch/channel extents and spatial strides')
    hs=sorted(set(o[2] for o in offsets));ws=sorted(set(o[3] for o in offsets))
    if (hs[0]!=0 or ws[0]!=0 or offsets!=[(0,0,y,x) for y in hs for x in ws]):
        raise WindowProductError('concat reduction is not ordered KH/KW/CI Cartesian windows')
    dh=hs[1]-hs[0] if len(hs)>1 else 1;dw=ws[1]-ws[0] if len(ws)>1 else 1
    if hs!=list(range(0,len(hs)*dh,dh)) or ws!=list(range(0,len(ws)*dw,dw)):
        raise WindowProductError('window dilation must be a positive affine lattice')
    win=Window2D((b,h,w,ci),(len(hs),len(ws)),st[2:],(dh,dw),(0,0,0,0))
    if tuple(win.output_spatial)!=(oh,ow) or rhs_shape[0]!=len(pieces)*ci or output_shape!=(b*oh*ow,rhs_shape[1]):
        raise WindowProductError('window/contraction dimensions differ from original source')
    for off,size,stride in zip(offsets,sizes,strides,strict=True):
        if any(o<0 or o+(n-1)*s>=d for o,n,s,d in zip(off,size,stride,source_shape,strict=True)):
            raise WindowProductError('source window exceeds its original tensor')
    return dict(schema='submission.source_window_product.v1',source_shape=list(source_shape),rhs_shape=list(rhs_shape),output_shape=list(output_shape),pieces=[dict(offsets=list(o),sizes=list(s),strides=list(t)) for o,s,t in zip(offsets,sizes,strides,strict=True)],nhwc_shape=[b,h,w,ci],kernel=[len(hs),len(ws),ci,rhs_shape[1]],stride=list(st[2:]),dilation=[dh,dw],padding=[0,0,0,0],source_coordinate_law='lhs[b*OH*OW+y*OW+x,(kh*KW+kw)*CI+c] = original[b,c,y*SH+kh*DH,x*SW+kw*DW]',reduction_order='KH then KW then CI; original ordered signed i8 products and modular i32 accumulation',materialized_window_bytes=0,removed_window_bytes=b*oh*ow*len(pieces)*ci,transpose_bytes=b*h*w*ci)


def _branch(value,consumer):
    from .source_groups import _spec,_integer_slice
    from .source_slice_dense import _reshape
    members=[];current=value;next_op=consumer
    def consume(op):
        if (op.parent is not consumer.parent or len(list(current.uses))!=1
                or next(iter(current.uses)).operation is not next_op):raise WindowProductError('window preparation has a live observer')
        members.append(op)
    while getattr(current.owner,'name','') in ('tensor.collapse_shape','tensor.expand_shape'):
        op=current.owner
        if not _reshape(op):raise WindowProductError('window reshape changes dense element order')
        consume(op);next_op=op;current=op.operands[0]
    op=current.owner
    if getattr(op,'name','')!='linalg.transpose' or tuple(op.permutation.get_values())!=(0,2,3,1):raise WindowProductError('window branch needs exact NCHW to NHWC transpose')
    before,after=_spec(op.operands[0]),_spec(current)
    if (before['dtype']!='i8' or after['dtype']!='i8' or len(before['shape'])!=4
            or after['shape']!=[before['shape'][i] for i in (0,2,3,1)]):
        raise WindowProductError('window transpose source/result tensor axes differ')
    consume(op);next_op=op;current=op.operands[0];leaf=_spec(current)['shape'];chain=[]
    while getattr(current.owner,'name','')=='tensor.extract_slice':
        op=current.owner;row=_integer_slice(op)
        if row is None:raise WindowProductError('window slice lacks exact static source proof')
        consume(op);source,attrs,_=row;chain.append(attrs);next_op=op;current=source
    if not chain or len(leaf)!=4 or _spec(value)!=dict(shape=[leaf[0]*leaf[2]*leaf[3],leaf[1]],dtype='i8'):
        raise WindowProductError('window piece flattening differs from NHWC row/channel order')
    off=[0]*4;stride=[1]*4
    for row in chain:
        off=[a+b*c for a,b,c in zip(row['offsets'],off,row['strides'],strict=True)]
        stride=[a*b for a,b in zip(stride,row['strides'],strict=True)]
    return current,members,dict(offsets=off,sizes=leaf,strides=stride)


def compose_window_product(plan):
    from .source_groups import _spec,_root_inputs,_dependencies,_Standalone
    from merlin.compile.analysis.contraction_shapes import zero_initialised
    lhs,rhs,init=_root_inputs(plan.root);concat=lhs.owner
    if getattr(concat,'name','')!='tensor.concat' or concat.dim.value.data!=1:return plan
    try:
        if not zero_initialised(plan.root):raise WindowProductError('window product does not support runtime matrix initialization')
        if (len(list(lhs.uses))!=1 or any(k not in ('epilogue','output_dtype','acc_scale','acc_scale_by_column') for k in plan.attributes)):
            raise WindowProductError('window product has observers or unsupported initialized numeric stages')
        branches=[_branch(v,concat) for v in concat.operands];source=branches[0][0]
        if any(row[0] is not source for row in branches):raise WindowProductError('window pieces do not share their original source')
        proof=window_geometry(_spec(source)['shape'],[row[2] for row in branches],_spec(rhs)['shape'],_spec(plan.output)['shape'])
        members=list(plan.operations)+[concat]+[op for row in branches for op in row[1]]
        owned=set(members)
        if any(v is not plan.output and any(u.operation not in owned for u in v.uses) for op in members for v in op.results):
            raise WindowProductError('window composition would discard a live intermediate')
        operations,inputs=_dependencies(plan.root,members)
        return _Standalone(plan.root,operations,inputs,plan.output,'SOURCE_WINDOW_PRODUCT',{'ifm':source,'rhs':rhs},dict(window_proof=proof,convolution_attributes=dict(plan.attributes)),dict(plan.diagnostic,source_window_product=proof),constants=plan.constants)
    except (WindowProductError,ValueError):return plan


def lower_source_window_product(ctx,operands,attributes):
    from .transpose import lower_transpose
    from .conv import lower_conv
    if set(operands)!={'ifm','rhs','dst'} or set(attributes)!={'window_proof','convolution_attributes'}:raise WindowProductError('window product contract changed')
    source,rhs,dst=(operands[k] for k in ('ifm','rhs','dst'));p=attributes['window_proof'];attrs=dict(attributes['convolution_attributes'])
    proof=window_geometry(ctx.spec(source)['shape'],p['pieces'],ctx.spec(rhs)['shape'],ctx.spec(dst)['shape'],ctx.spec(source)['dtype'])
    if proof!=p or source==dst or rhs==dst or ctx.spec(rhs)['dtype']!='i8' or ctx.spec(dst)['dtype']!=attrs.get('output_dtype','i32'):
        raise WindowProductError('original window shapes, types or coordinate proof changed')
    if any(k not in ('epilogue','output_dtype','acc_scale','acc_scale_by_column') for k in attrs):raise WindowProductError('window product cannot change source initializer or arithmetic')
    aliases=ctx.cb.get('params',{}).get('tensor_aliases',{});buffers=getattr(ctx,'buffers',{})
    if dst in aliases or source in aliases or rhs in aliases or (dst in buffers and any(v in buffers and buffers[v] is buffers[dst] for v in (source,rhs))):
        raise WindowProductError('window product requires materialized sources and a distinct destination')
    alias=f'_source_window_nhwc_{len(ctx.tensors)}';ctx.alloc(alias,proof['nhwc_shape'],'i8')
    transpose=lower_transpose(ctx,{'src':source,'dst':alias},{'permutation':[0,2,3,1],'output_dtype':'i8','transpose_schedule':'baseline'})
    conv=lower_conv(ctx,alias,rhs,dst,dict(attrs,layout='nhwc',kernel=proof['kernel'],stride=proof['stride'],dilation=proof['dilation'],padding=proof['padding']))
    receipt=dict(proof=proof,transpose=transpose,convolution=conv)
    ctx.metadata.setdefault('source_window_products',[]).append(receipt);return receipt
