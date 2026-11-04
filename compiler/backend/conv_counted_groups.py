"""Bounded affine bridge from checked private row groups to counted schedule IR.

The group API deliberately accepts concrete row numbers. This adapter proves
its affine extension over every bounded row/column before exposing a runtime
row induction variable. It grants no new storage or completion semantics.
"""
from dataclasses import dataclass
from merlin.sched.ir import Call, Const, Kernel, Loop, Var, check_structure
from merlin.compile.scheduling.llvm_program import emit_index_program


@dataclass(frozen=True)
class CountedGroupRows:
    group: object
    active_columns: int
    row_counts: tuple[int,...]
    program: Kernel

    def __post_init__(self):
        g=self.group;p=g.panel;g.__post_init__()
        if (type(self.active_columns)is not int or not 0<self.active_columns<=p.column_blocks
                or type(self.row_counts)is not tuple or not 0<len(self.row_counts)<=g.row_blocks
                or any(type(r)is not int or not 0<r<=p.tile for r in self.row_counts)
                or any(r!=p.tile for r in self.row_counts[:-1])):
            raise ValueError('counted group requires checked full rows and at most one original tail')
        # Bounded concrete group methods are the mapping authority, including
        # peeled N phases. No dynamic group index is passed to those methods.
        for m in range(len(self.row_counts)):
            if g.lhs_row(m)!=m*p.tile:
                raise ValueError('private LHS mapping has no admitted affine extension')
            for n in range(self.active_columns):
                if g.accumulator_row(m,n,active_column_blocks=self.active_columns)!=(m*self.active_columns+n)*p.tile:
                    raise ValueError('private accumulator mapping has no admitted affine extension')
        if self.program!=_program(self.active_columns,self.row_counts,p.tile) or check_structure(self.program):
            raise ValueError('counted row schedule changed its checked original iteration domain')

    def emit(self,builder,consume):
        self.__post_init__()
        def lower(tag,fields):
            if tag!='private_group_row':raise ValueError('unknown counted row mapping instruction')
            consume(fields)
        return emit_index_program(self.program,builder,lower)


def _program(active,rows,tile):
    def effect(m,count):
        return Call('private_group_row',(
            ('row_offset',m*tile),('lhs_row',m*tile),
            ('accumulator_base',m*(active*tile)),('rows',Const(count))))
    full=len(rows)-(rows[-1]!=tile)
    body=() if not full else (Loop('private_m',full,(effect(Var('private_m'),tile),)),)
    if full<len(rows):body+=(effect(Const(full),rows[-1]),)
    return Kernel('convolution_private_group_rows',(),body)


def plan_counted_group_rows(group,active_columns,row_counts):
    group.__post_init__()
    rows=tuple(row_counts)
    return CountedGroupRows(group,active_columns,rows,_program(active_columns,rows,group.panel.tile))
