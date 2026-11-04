"""Exact scalar compare/select bounds for complete-i32 narrowing readout.

Normalization is used only by the existing i8-output proof. Floating compares
require proved finite f32 operands; signed-zero selection differences cannot
change its final integer bits. This helper grants no placement/fanout ownership.
"""
import math


class SelectReadoutError(ValueError):
    pass


def _constant(expr):
    if expr.kind=='constant':return expr.value
    if expr.kind=='arith.sitofp' and len(expr.args)==1 and expr.args[0].kind=='constant':
        import numpy as np
        return float(np.float32(expr.args[0].value))
    return None


def _interval(expr):
    """Conservative finite binary32 domain, rooted only at signed i32 inputs."""
    c=_constant(expr)
    if c is not None:
        if isinstance(c,(int,float)) and math.isfinite(c):return float(c),float(c)
        raise SelectReadoutError('select bound is nonfinite')
    if expr.kind=='accumulator' and expr.dtype=='i32':return -2147483648.,2147483647.
    if expr.kind=='arith.sitofp' and expr.dtype=='f32' and len(expr.args)==1:
        a,b=_interval(expr.args[0]);return min(a,-0.),max(b,2147483648. if b==2147483647. else b)
    if expr.kind in ('arith.maximumf','arith.minimumf','arith.maxsi','arith.minsi'):
        a,b=_interval(expr.args[0]);c,d=_interval(expr.args[1]);fn=max if expr.kind in ('arith.maximumf','arith.maxsi') else min
        return fn(a,c),fn(b,d)
    if expr.kind=='math.roundeven':
        a,b=_interval(expr.args[0]);return float(round(a)),float(round(b))
    if expr.kind in ('arith.mulf','arith.divf','arith.addf') and expr.dtype=='f32':
        a,b=_interval(expr.args[0]);c,d=_interval(expr.args[1])
        if expr.kind=='arith.mulf':values=[a*c,a*d,b*c,b*d]
        elif expr.kind=='arith.addf':values=[a+c,b+d]
        else:
            if c<=0<=d:raise SelectReadoutError('select domain division may be nonfinite')
            values=[a/c,a/d,b/c,b/d]
        import numpy as np
        with np.errstate(all='ignore'):values=[float(np.float32(v)) for v in values]
        if not all(math.isfinite(v)for v in values):raise SelectReadoutError('select domain may overflow binary32')
        return min(values),max(values)
    raise SelectReadoutError('select operand has no proved finite signed-i32 domain')


def trace_select(op, scalar, used, Expr, safe_scalar):
    """Return canonical min/max Expr after proving actual predicate/branch SSA."""
    safe_scalar(op)
    if len(op.operands)!=3:raise SelectReadoutError('three select operands required')
    condition,yes,no=op.operands;cmp=condition.owner
    if getattr(cmp,'name','')not in ('arith.cmpf','arith.cmpi') or cmp.parent is not op.parent:
        raise SelectReadoutError('select must use a local direct comparison')
    if len(cmp.operands)!=2 or len(cmp.results)!=1:raise SelectReadoutError('binary comparison required')
    if any(not k.startswith('prov.')for k in cmp.attributes):raise SelectReadoutError('unproved compare attributes')
    for k,v in cmp.properties.items():
        if k=='predicate':continue
        if k=='fastmath' and str(v)=='#arith.fastmath<none>':continue
        raise SelectReadoutError('unproved compare property '+k)
    a,b=cmp.operands
    forward=yes is a and no is b;reverse=yes is b and no is a
    if not(forward or reverse):raise SelectReadoutError('select branches must be exactly the compared operands')
    predicate=cmp.properties['predicate'].value.data
    if cmp.name=='arith.cmpf':
        # Ordered and unordered relational forms agree on this proved finite
        # domain. Equal/unequal/unordered-test predicates do not imply bounds.
        if predicate in (2,3,9,10):kind='arith.maximumf'
        elif predicate in (4,5,11,12):kind='arith.minimumf'
        else:raise SelectReadoutError('comparison predicate is not a relational bound')
        if str(a.type)!='f32' or str(b.type)!='f32' or str(op.results[0].type)!='f32':raise SelectReadoutError('binary32 select required')
    else:
        if predicate in (4,5):kind='arith.maxsi'
        elif predicate in (2,3):kind='arith.minsi'
        else:raise SelectReadoutError('only signed relational integer comparisons admitted')
        if str(a.type)!='i32' or str(b.type)!='i32' or str(op.results[0].type)!='i32':raise SelectReadoutError('signed i32 select required')
    if reverse:kind={'arith.maximumf':'arith.minimumf','arith.minimumf':'arith.maximumf','arith.maxsi':'arith.minsi','arith.minsi':'arith.maxsi'}[kind]
    left,right=scalar(a),scalar(b)
    _interval(left);_interval(right)
    # Fold exact scalar casts such as public sitofp(i64 zero) only after its
    # actual signed cast has been traced and the finite domain established.
    def fold(e):
        c=_constant(e)
        return Expr('constant',value=c,dtype=str(op.results[0].type)) if c is not None else e
    used.add(cmp)
    return Expr(kind,(fold(left),fold(right)),dtype=str(op.results[0].type))


def strip_relu(expr):
    """Remove only max(x,0), optionally immediately inside signed sitofp."""
    if expr.kind=='arith.maximumf' and len(expr.args)==2:
        a,b=expr.args
        if _constant(a)==0:return b,True
        if _constant(b)==0:return a,True
    if expr.kind=='arith.sitofp' and len(expr.args)==1:
        inner=expr.args[0]
        if inner.kind=='arith.maxsi' and len(inner.args)==2:
            a,b=inner.args;x=b if _constant(a)==0 else a if _constant(b)==0 else None
            if x is not None:return type(expr)('arith.sitofp',(x,),dtype='f32'),True
    return expr,False
