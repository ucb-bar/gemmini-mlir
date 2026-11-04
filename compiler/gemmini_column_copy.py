"""Exact affine column-copy traversal; ordinary gather remains fallback.

Convolution/scale/layout preparation retain their existing owners. The only
replacement is an integer address traversal for an already checked bijection.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ColumnCopyRegion:
    first: int
    outer: int
    inner: int
    source: int
    outer_stride: int
    inner_stride: int


@dataclass(frozen=True)
class ColumnCopyPlan:
    permutation: tuple[int, ...]
    regions: tuple[ColumnCopyRegion, ...]

    def verify(self):
        p = self.permutation
        if (not p or len(p) >= 2**31 or any(type(x) is not int for x in p)
                or sorted(p) != list(range(len(p)))):
            raise ValueError('column copy requires a complete bounded bijection')
        used = 0
        for r in self.regions:
            if (type(r) is not ColumnCopyRegion or any(type(x) is not int for x in
                    (r.first,r.outer,r.inner,r.source,r.outer_stride,r.inner_stride))
                    or r.first != used or min(r.outer,r.inner) <= 0):
                raise ValueError('affine copy regions must partition destination columns')
            for q in range(r.outer):
                for j in range(r.inner):
                    c = r.first+q*r.inner+j
                    value = r.source+q*r.outer_stride+j*r.inner_stride
                    if c >= len(p) or not 0 <= value < len(p) or value != p[c]:
                        raise ValueError('affine copy differs from bounded source permutation')
            used += r.outer*r.inner
        if used != len(p):
            raise ValueError('affine copy does not cover every original column')


def plan_column_copy(permutation, *, max_regions=16):
    """Prove constant-stride runs, coalescing exact translated equal-length runs.

    This is an artifact-expansion bound, not measured profitability selection.
    Irregular maps exceeding the bound retain the ordinary indexed gather.
    """
    p = tuple(permutation)
    if (not p or len(p) >= 2**31 or any(type(x) is not int for x in p)
            or sorted(p) != list(range(len(p)))):
        raise ValueError('column copy requires a complete bounded bijection')
    if type(max_regions) is not int or max_regions <= 0:
        raise ValueError('column copy needs a positive explicit region bound')
    runs=[]; first=0
    while first < len(p):
        step=p[first+1]-p[first] if first+1 < len(p) else 0
        end=min(first+2,len(p))
        while end < len(p) and p[end]-p[end-1] == step:
            end+=1
        runs.append(ColumnCopyRegion(first,1,end-first,p[first],0,step))
        first=end
    regions=[]; i=0
    while i < len(runs):
        r=runs[i]; end=i+1
        stride=runs[end].source-r.source if end < len(runs) else 0
        while end < len(runs):
            t=runs[end]
            if (t.inner != r.inner or t.inner_stride != r.inner_stride
                    or t.source != r.source+(end-i)*stride):
                break
            end+=1
        regions.append(ColumnCopyRegion(r.first,end-i,r.inner,r.source,stride,r.inner_stride))
        i=end
    if len(regions) > max_regions:
        return None
    result=ColumnCopyPlan(p,tuple(regions));result.verify()
    return result


def copy_columns(host, source, destination, *, rows, permutation, dtype,
                 generic_copy, max_regions=16):
    """Use shared StoredTensor copy_traversal with exact constant-stride indices.

    Caller owns disjoint source/destination storage, fences and original dtype.
    No tensor arithmetic, source scale changes or command scheduling occur here.
    """
    if type(rows) is not int or not 0 < rows < 2**63:
        raise ValueError('column copy requires positive bounded static rows')
    if dtype not in ('i8','i32'):
        return generic_copy(host,source,destination,rows=rows,permutation=permutation,dtype=dtype)
    plan=plan_column_copy(permutation,max_regions=max_regions)
    item_bytes={'i8':1,'i32':4}[dtype]
    if rows*len(permutation)*item_bytes >= 2**63:
        raise ValueError('column copy exceeds signed byte address bounds')
    if plan is None:
        return generic_copy(host,source,destination,rows=rows,permutation=permutation,dtype=dtype)
    from gemmini_host import StoredTensor
    fn=host.fb; columns=len(plan.permutation)
    for r in plan.regions:
        def coordinates(indices,r=r):
            row,q,j=indices
            return row,fn.add_i(fn.const(r.first),fn.add_i(fn.mul_i(q,fn.const(r.inner)),j))
        def read(indices,r=r):
            row,q,j=indices
            col=fn.add_i(fn.const(r.source),fn.add_i(fn.mul_i(q,fn.const(r.outer_stride)),fn.mul_i(j,fn.const(r.inner_stride))))
            offset=fn.add_i(fn.mul_i(row,fn.const(columns)),col)
            return host._read(source,offset,dtype)
        logical=StoredTensor(host,(rows,columns),dtype,None,
            copy_traversal=((rows,r.outer,r.inner),coordinates,read))
        host._copy(logical,host._stored((rows,columns),dtype,destination))
    return dict(affine_regions=len(plan.regions),index_table_loads=0)
