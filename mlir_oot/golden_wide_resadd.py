"""Exact wide-integer residual using resident signed-i8 diagonal chunks.

ABI A,B,C,coefficient_tables. Tables contain diagonal 127, p%127, q%127
in that order, each 16x16 i8. Inputs/output are contiguous Mx64 i8, M%16=0.
The binder proves the final scale against every source operand pair separately.
No load rounding, intermediate readout, CPU duplication or FSM commands occur.
"""
import math
from dataclasses import dataclass
from xdsl.dialects import llvm
from xdsl.dialects.builtin import ModuleOp,StringAttr,IntegerAttr,DictionaryAttr,i64
from .codegen.builder import FnBuilder,PTR
from .golden_gemm import GoldenGemm,Shape
from .tables import isa,rtl_facts as F


def tables(p,q):
    return bytes((v if i==j else 0) for v in (127,p%127,q%127) for i in range(16) for j in range(16))


def _build_serial(m,p,q,scale,*,relu=True):
    if type(m) is not int or m<=0 or m%16:raise ValueError('positive M divisible by16 required')
    if any(type(x) is not int or not 1<=x<=32767 for x in (p,q)):raise ValueError('coefficients must be integers1..32767')
    if not math.isfinite(scale) or scale<=0:raise ValueError('positive finite readout scale required')
    e=GoldenGemm(Shape(m,64,16,bn=4));e.fb=FnBuilder([PTR]*4)
    a,b,c,coeff=e.fb.entry.args
    e._rocc('fence',{});e._rocc('flush',{})
    e._rocc('config_ex',{'dataflow':isa.WEIGHT_STATIONARY})
    for lid in (0,1):e._rocc('config_ld',{'stride':64,'load_id':lid})
    e._rocc('config_ld',{'stride':16,'load_id':2})
    e._rocc('config_st',{'stride':64,'acc_act':isa.RELU if relu else isa.NO_ACTIVATION,'acc_scale':scale})
    for index in range(3):
        ptr=e._ptr(coeff,e.fb.const(0),1,e.fb.const(index*256))
        e._rocc('mvin',{'local':8192+index*16,'rows':16,'cols':16,'load_id':2},ptr)
    def panel(row):
        for ptr,base,lid in ((a,0,0),(b,4096,1)):
            e._rocc('mvin',{'local':base,'rows':16,'cols':64,'load_id':lid},e._ptr(ptr,row,64,e.fb.const(0)))
        first=True
        for value,base,remainder_tile in ((p,0,1),(q,4096,2)):
            full,remainder=divmod(value,127)
            def chunk(weight,load_weight,initialize):
                for d in range(4):
                    e._rocc('preload',{'bd':weight if load_weight and d==0 else isa.GARBAGE_ADDR,
                        'c':isa.acc_addr(d*16,accumulate=not initialize),'bd_cols':16,'bd_rows':16,'c_cols':16,'c_rows':16})
                    e._rocc('compute',{'a':base+d*16,'a_cols':16,'a_rows':16,'accumulate':not (load_weight and d==0)})
            if full:
                chunk(8192,True,first);first=False
                if full>1:e.fb.for_loop(1,full,1,lambda unused:chunk(8192,False,False))
            if remainder:
                chunk(8192+remainder_tile*16,True,first);first=False
        e._rocc('mvout',{'local':isa.acc_addr(0),'rows':16,'cols':64},e._ptr(c,row,64,e.fb.const(0)))
    e.fb.for_loop(0,m,16,panel)
    e._rocc('fence',{})
    fn=llvm.FuncOp('gemmini_golden_wide_resadd',llvm.LLVMFunctionType([PTR]*4),linkage=llvm.LinkageAttr('external'),body=e.fb.finish())
    module=ModuleOp([fn]);module.attributes['gemmini.dim']=IntegerAttr(F.DIM,i64)
    module.attributes['gemmini.golden_wide_resadd']=StringAttr(f'M{m}:N64:p{p}:q{q}:scale{scale}:relu{int(relu)}')
    module.verify();return module


@dataclass(frozen=True)
class ResidualPrefetchPlan:
    """Two input slots and resident weights, each in a separate scratchpad bank.

    The ABI is the existing contiguous four-tile-wide signed-i8 residual. Each
    input slot holds both operands; coefficients are three diagonal tiles.
    Accumulators either retain the serial mapping or alternate two proved banks.
    Each output retains the serial schedule's dependency order.
    No source storage aliasing permission is introduced: the caller must retain
    the same tensor-value input/output contract as the serial kernel.
    """

    m: int
    p: int
    q: int
    banked_accumulators: bool = False

    def __post_init__(self):
        if type(self.m) is not int or self.m <= 0 or self.m % F.DIM:
            raise ValueError('positive M divisible by mesh dimension required')
        if any(type(v) is not int or not 1 <= v <= 32767 for v in (self.p, self.q)):
            raise ValueError('coefficients must be integers1..32767')
        if F.OPERAND_DTYPE != 'i8' or F.ACCUMULATOR_DTYPE != 'i32':
            raise ValueError('residual prefetch requires signed-i8 operands and i32 accumulator')
        if F.SPAD_BANKS < 3 or F.SPAD_BANK_ROWS * F.SPAD_BANKS != F.SPAD_ROWS:
            raise ValueError('residual prefetch requires three proved scratchpad banks')
        if 2 * self.panel_rows > F.SPAD_BANK_ROWS:
            raise ValueError('two residual operands do not fit one scratchpad bank')
        if 3 * F.DIM > F.SPAD_BANK_ROWS or self.panel_rows > F.ACC_ROWS:
            raise ValueError('residual coefficient or accumulator footprint exceeds capacity')
        if type(self.banked_accumulators) is not bool:
            raise ValueError('banked_accumulators must be boolean')
        if self.banked_accumulators and (F.ACC_BANKS < 2 or
                F.ACC_BANK_ROWS * F.ACC_BANKS != F.ACC_ROWS or
                self.panel_rows > F.ACC_BANK_ROWS):
            raise ValueError('residual accumulator slots require two proved banks with panel capacity')
        if 128 * (self.p + self.q) > (1 << 31) - 1:
            raise ValueError('residual accumulation may overflow signed i32')

    @property
    def columns(self):
        return 4 * F.DIM

    @property
    def panel_rows(self):
        return self.columns

    @property
    def weight_base(self):
        return F.SPAD_BANK_ROWS

    def operand_base(self, slot, operand):
        if slot not in (0, 1) or operand not in (0, 1):
            raise ValueError('residual slot and operand must be zero or one')
        return slot * 2 * F.SPAD_BANK_ROWS + operand * self.panel_rows

    def accumulator_base(self, slot):
        if slot not in (0, 1):
            raise ValueError('residual slot must be zero or one')
        return slot * F.ACC_BANK_ROWS if self.banked_accumulators else 0

    def attributes(self):
        return DictionaryAttr({key: IntegerAttr(value, i64) for key, value in {
            'panel_rows': self.panel_rows,
            'operand_slot_rows': 2 * self.panel_rows,
            'slot0_base': self.operand_base(0, 0),
            'slot1_base': self.operand_base(1, 0),
            'weight_base': self.weight_base,
            'weight_rows': 3 * F.DIM,
            'accumulator_rows': self.panel_rows,
            'banked_accumulators': int(self.banked_accumulators),
            'accumulator_slot1_base': self.accumulator_base(1),
            'panel_count': self.m // F.DIM,
            'chunks_per_panel': (self.p + 126) // 127 + (self.q + 126) // 127,
        }.items()})


def _build_prefetch(m,p,q,scale,*,relu=True,banked_accumulators=False,correction_symbol=None):
    plan = ResidualPrefetchPlan(m, p, q, banked_accumulators)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('positive finite readout scale required')
    if correction_symbol is not None and (not isinstance(correction_symbol,str) or
            not correction_symbol.isascii() or not correction_symbol.isidentifier()):
        raise ValueError('explicit correction function identifier required')
    e = GoldenGemm(Shape(m, plan.columns, F.DIM, bn=4))
    e.fb = FnBuilder([PTR] * 4)
    a, b, c, coeff = e.fb.entry.args
    e._rocc('fence', {}); e._rocc('flush', {})
    e._rocc('config_ex', {'dataflow': isa.WEIGHT_STATIONARY})
    for lid in (0, 1):
        e._rocc('config_ld', {'stride': plan.columns, 'load_id': lid})
    e._rocc('config_ld', {'stride': F.DIM, 'load_id': 2})
    e._rocc('config_st', {'stride': plan.columns,
        'acc_act': isa.RELU if relu else isa.NO_ACTIVATION, 'acc_scale': scale})
    for index in range(3):
        ptr = e._ptr(coeff, e.fb.const(0), 1, e.fb.const(index * F.DIM * F.DIM))
        e._rocc('mvin', {'local': plan.weight_base + index * F.DIM,
            'rows': F.DIM, 'cols': F.DIM, 'load_id': 2}, ptr)

    def load_panel(row, slot):
        for operand, ptr in enumerate((a, b)):
            e._rocc('mvin', {'local': plan.operand_base(slot, operand),
                'rows': F.DIM, 'cols': plan.columns, 'load_id': operand},
                e._ptr(ptr, row, plan.columns, e.fb.const(0)))

    def compute_panel(row, slot):
        acc_base = plan.accumulator_base(slot)
        first = True
        for operand, value, remainder_tile in ((0, p, 1), (1, q, 2)):
            full, remainder = divmod(value, 127)
            base = plan.operand_base(slot, operand)
            def chunk(weight, load_weight, initialize):
                for d in range(plan.columns // F.DIM):
                    e._rocc('preload', {
                        'bd': weight if load_weight and d == 0 else isa.GARBAGE_ADDR,
                        'c': isa.acc_addr(acc_base + d * F.DIM, accumulate=not initialize),
                        'bd_cols': F.DIM, 'bd_rows': F.DIM,
                        'c_cols': F.DIM, 'c_rows': F.DIM})
                    e._rocc('compute', {'a': base + d * F.DIM,
                        'a_cols': F.DIM, 'a_rows': F.DIM,
                        'accumulate': not (load_weight and d == 0)})
            if full:
                chunk(plan.weight_base, True, first); first = False
                if full > 1:
                    e.fb.for_loop(1, full, 1,
                        lambda unused: chunk(plan.weight_base, False, False))
            if remainder:
                chunk(plan.weight_base + remainder_tile * F.DIM, True, first)
                first = False
        e._rocc('mvout', {'local': isa.acc_addr(acc_base),
            'rows': F.DIM, 'cols': plan.columns},
            e._ptr(c, row, plan.columns, e.fb.const(0)))

    # Static bank slots avoid a dynamic address mux. The final one or two panels
    # are drained separately, so no DMA can read beyond the source tensor.
    if correction_symbol is None:
        load_panel(e.fb.const(0), 0)
        paired_rows = ((m // F.DIM - 1) // 2) * 2 * F.DIM
        def pair(row):
            next_row = e.fb.add_i(row, e.fb.const(F.DIM))
            load_panel(next_row, 1)
            compute_panel(row, 0)
            load_panel(e.fb.add_i(row, e.fb.const(2 * F.DIM)), 0)
            compute_panel(next_row, 1)
        e.fb.for_loop(0, paired_rows, 2 * F.DIM, pair)
        tail = e.fb.const(paired_rows)
        if m - paired_rows == 2 * F.DIM:
            last = e.fb.const(paired_rows + F.DIM)
            load_panel(last, 1)
            compute_panel(tail, 0)
            compute_panel(last, 1)
        else:
            compute_panel(tail, 0)
    else:
        def correct(row):
            pointers=[e._ptr(ptr,row,plan.columns,e.fb.const(0)) for ptr in (a,b,c)]
            e.fb.add(llvm.CallOp(correction_symbol,*pointers,e.fb.const(F.DIM*plan.columns)))
        # FENCE closes the previous output. The current device panel writes a
        # disjoint interval while the CPU corrects only that completed panel.
        load_panel(e.fb.const(0),0)
        compute_panel(e.fb.const(0),0)
        e._rocc('fence',{})
        stop=F.DIM+((m//F.DIM-1)//2)*2*F.DIM
        def pair(row):
            load_panel(row,1);compute_panel(row,1)
            correct(e.fb.sub_i(row,e.fb.const(F.DIM)))
            e._rocc('fence',{})
            next_row=e.fb.add_i(row,e.fb.const(F.DIM))
            load_panel(next_row,0);compute_panel(next_row,0)
            correct(row)
            e._rocc('fence',{})
        e.fb.for_loop(F.DIM,stop,2*F.DIM,pair,retain_loop=True)
        if (m//F.DIM-1)%2:
            last=e.fb.const(m-F.DIM)
            load_panel(last,1);compute_panel(last,1)
            correct(e.fb.const(m-2*F.DIM))
            e._rocc('fence',{})
        correct(e.fb.const(m-F.DIM))
    e._rocc('fence', {})
    fn = llvm.FuncOp('gemmini_golden_wide_resadd', llvm.LLVMFunctionType([PTR] * 4),
        linkage=llvm.LinkageAttr('external'), body=e.fb.finish())
    fn.attributes['gemmini.residual_m_prefetch'] = plan.attributes()
    functions=[fn]
    if correction_symbol is not None:
        functions.insert(0,llvm.FuncOp(correction_symbol,llvm.LLVMFunctionType([PTR]*3+[i64]),
            linkage=llvm.LinkageAttr('external')))
        fn.attributes['gemmini.residual_stream_correction']=StringAttr(correction_symbol)
    module = ModuleOp(functions)
    module.attributes['gemmini.dim'] = IntegerAttr(F.DIM, i64)
    module.attributes['gemmini.golden_wide_resadd'] = StringAttr(
        f'M{m}:N{plan.columns}:p{p}:q{q}:scale{scale}:relu{int(relu)}')
    module.attributes['gemmini.residual_m_prefetch'] = plan.attributes()
    module.verify()
    return module


def build(m,p,q,scale,*,relu=True,prefetch_m=False,banked_accumulators=False):
    """Select the explicit banked schedule; one-panel inputs keep serial emission."""
    if type(prefetch_m) is not bool:
        raise ValueError('prefetch_m must be boolean')
    if type(banked_accumulators) is not bool:
        raise ValueError('banked_accumulators must be boolean')
    if banked_accumulators and not prefetch_m:
        raise ValueError('banked accumulators require residual M prefetch')
    if not prefetch_m or m == F.DIM:
        return _build_serial(m,p,q,scale,relu=relu)
    return _build_prefetch(m,p,q,scale,relu=relu,banked_accumulators=banked_accumulators)
