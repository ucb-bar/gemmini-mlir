"""Exact bounded retained-command verification when the optional SMT tool is absent.

This checks every dynamic projected tile command, not sampled tensor values.
Pure staging/compute/readout phases and the selected ordered accumulator law
make row provenance and increasing-K accumulator states sufficient for every
integer source element. Bounds refuse oversized work instead of skipping it.
"""
from dataclasses import dataclass,replace
from hashlib import sha256
from pathlib import Path
from merlin.compile.scheduling.block_schedule import Load,Preload,Compute,Store,InitializeAccumulator
from merlin.compile.scheduling.compact_dense import DenseBiasSource
from merlin.compile.scheduling.dense_panel_pipeline import complete_dense_retained_reduction
from merlin.compile.scheduling.loop_schedule import propose_loop_transforms
from merlin.sched.ir import Call,Loop,instances,evaluate,check_structure
from merlin.sched.ir.integer_domain import check_integer_domain

DRAIN='merlin_dense_panel_drain'
MAX_COMMANDS=2000000
MAX_OUTPUT_ROW_TILES=1000000


def verify_retained_commands(kernel,p,r,s,a,law,*,static_budget=128,allow_disjoint_mixed=False):
    """Exact source/residency/completion proof over finite original domains."""
    law.__post_init__();a.__post_init__()
    if (not a.synchronous_completion or not a.exact_accumulator_continuation
            or (p.tile_m,p.tile_k,p.tile_n)!=(16,16,16)
            or (p.lhs_bytes,p.rhs_bytes,p.accumulator_bytes)!=(1,1,4)):
        raise ValueError('finite retained verifier requires exact i8/i8/i32 primitive adapter')
    if (law.hardware_identity!=a.configuration_identity or
            law.numerical_policy_identity!=a.numerical_policy_identity):
        raise ValueError('finite retained verifier execution law differs from adapter')
    expected=[('lhs',(p.m,p.k),'i8','read'),('rhs',(p.k,p.n),'i8','read')]
    if p.bias:expected.append(('bias',(p.n,),'i32','read'))
    expected.append(('output',(p.m,p.n),'i32','write'))
    if [(arg.name,arg.shape,arg.dtype,arg.access) for arg in kernel.args]!=expected:
        raise ValueError('finite retained verifier source ABI differs')
    if check_structure(kernel):raise ValueError('invalid finite retained kernel')
    check_integer_domain(kernel,a.index_bits)
    def counts(body):
        static=dynamic=0
        for node in body:
            if type(node) is Loop:
                st,dy=counts(node.body);static+=st;dynamic+=dy*node.extent
            else:static+=1;dynamic+=1
        return static,dynamic
    static,dynamic=counts(kernel.body)
    if static>static_budget or dynamic>MAX_COMMANDS or p.m*p.tiles[2]>MAX_OUTPUT_ROW_TILES:
        raise ValueError('finite retained verification exceeds exact command or output domain bound')
    # Original physical windows must be disjoint, within declared backing
    # capacity, and within the complete resource budget.
    views=(s.lhs,s.rhs,s.accumulator)
    for i,left in enumerate(views):
        if left is None or left.base<0 or left.capacity<=0:raise ValueError('invalid finite retained window')
        for right in views[i+1:]:
            if left.backing==right.backing and left.base*left.unit_bytes<(right.base+right.capacity)*right.unit_bytes and right.base*right.unit_bytes<(left.base+left.capacity)*left.unit_bytes:
                raise ValueError('finite retained physical windows alias')
    if (s.lhs.capacity*s.lhs.unit_bytes+s.rhs.capacity*s.rhs.unit_bytes>r.operand_bytes or
            s.accumulator.capacity*s.accumulator.unit_bytes>r.accumulator_bytes):
        raise ValueError('finite retained assigned capacity exceeds original resources')
    memory={};acc={};outputs=set();products=set();initializers=set();pending=None;phase=None;epoch_counts={};drains=0
    epoch_writes=set();epoch_reads=set()
    totals={'lhs_loads':0,'rhs_loads':0,'bias_initializations':0,'compute_pairs':0,'output_tiles':0}
    def require(test,message):
        if not test:raise ValueError('finite retained source check: '+message)
    def touch(kind):
        nonlocal phase
        if allow_disjoint_mixed and kind in ('load','compute') and phase in (None,'load','compute','mixed'):
            phase=kind if phase in (None,kind) else 'mixed'
        else:
            require(phase in (None,kind),'mixed engine ownership within an undrained epoch')
            phase=kind
        epoch_counts[kind]=epoch_counts.get(kind,0)+1
    def window(view,row,rows):
        require(view.base<=row and 0<rows and row+rows<=view.base+view.capacity,'physical row outside assigned window')
    for call,env in instances(kernel):
        if call.instr==DRAIN:
            require(pending is None,'unpaired descriptor at completion')
            require(epoch_counts.get('compute',0)<=law.max_issued_compute_pairs,'compute epoch exceeds admitted issued bound')
            require(epoch_counts.get('load',0)<=law.max_issued_loads,'load epoch exceeds admitted issued bound')
            require(epoch_counts.get('store',0)<=law.max_issued_readouts,'store epoch exceeds admitted issued bound')
            require(epoch_counts.get('init',0)<=getattr(law,'max_issued_initializations',0),'initializer epoch exceeds admitted issued bound')
            phase=None;epoch_counts={};epoch_reads.clear();epoch_writes.clear();drains+=1;continue
        projected=replace(call,args=tuple((key,evaluate(value,env)) for key,value in call.args))
        op=a.project(projected)
        encoded=a.encode(op)
        require(encoded.instr==projected.instr and tuple((key,evaluate(value,{})) for key,value in encoded.args)==projected.args,'actual instruction codec round trip differs')
        require(pending is None or type(op) is Compute,'descriptor and compute are not adjacent')
        if type(op) is Load:
            touch('load');left=op.role=='lhs';role='lhs' if left else 'weight'
            require(left or op.role in ('rhs','weight'),'unknown source operand')
            rows_limit,cols_limit=(p.m,p.k) if left else (p.k,p.n)
            require(op.step==1 and 0<op.rows<=16 and 0<op.cols<=16,'unsupported transfer geometry')
            require(op.dram_row==op.block[0]*16 and op.dram_col==op.block[1]*16,'load source block coordinates differ')
            require(0<=op.dram_row<rows_limit and op.rows==min(16,rows_limit-op.dram_row) and 0<=op.dram_col<cols_limit and op.cols==min(16,cols_limit-op.dram_col),'load source bounds or tails differ')
            window(s.lhs if left else s.rhs,op.row,op.rows)
            for i in range(op.rows):
                require(op.row+i not in epoch_reads and op.row+i not in epoch_writes,'load overwrites a pending reader or another pending write')
                epoch_writes.add(op.row+i)
                memory[op.row+i]=(role,op.dram_row+i,op.dram_col,op.cols)
            totals['lhs_loads' if left else 'rhs_loads']+=1
        elif type(op) is InitializeAccumulator:
            touch('init');require(p.bias,'initializer absent from original source')
            mr,nc=op.block[0]*16,op.block[1]*16
            require(0<=mr<p.m and 0<=nc<p.n and op.dram_col==nc and op.rows==min(16,p.m-mr) and op.cols==min(16,p.n-nc),'initializer source coordinates or tails differ')
            require(op.source_row_stride==0 and op.source_col_stride==1,'initializer is not original bias broadcast')
            require((mr,nc) not in initializers,'duplicate original initializer')
            initializers.add((mr,nc))
            window(s.accumulator,op.row,op.rows)
            for i in range(op.rows):acc[op.row+i]=(mr+i,nc,op.cols,0)
            totals['bias_initializations']+=1
        elif type(op) is Preload:
            require(phase in ((None,'load','compute','mixed') if allow_disjoint_mixed else (None,'compute')),'preload crosses an undrained ownership boundary')
            pending=op
        elif type(op) is Compute:
            touch('compute');d=pending;require(d is not None and op.fresh_weights,'compute lacks fresh adjacent descriptor')
            mr,kk=op.input_block[0]*16,op.input_block[1]*16
            wk,nc=d.weight_block[0]*16,d.weight_block[1]*16
            require(kk==wk and 0<=mr<p.m and 0<=kk<p.k and 0<=nc<p.n,'product source coordinates differ')
            require(op.rows==d.rows==min(16,p.m-mr) and op.cols==d.weight_rows==min(16,p.k-kk) and d.cols==min(16,p.n-nc),'product tails differ')
            require((mr,kk,nc) not in products,'duplicate original product tile')
            products.add((mr,kk,nc))
            require(d.accumulate==int(p.bias or kk!=0),'accumulator initialization/continuation differs')
            window(s.lhs,op.input_row,op.rows);window(s.rhs,d.weight_row,d.weight_rows);window(s.accumulator,d.accumulator_row,d.rows)
            read_rows=set(range(op.input_row,op.input_row+op.rows))|set(range(d.weight_row,d.weight_row+d.weight_rows))
            require(not read_rows.intersection(epoch_writes),'compute reads an operand whose load has not completed')
            epoch_reads.update(read_rows)
            for t in range(d.weight_rows):require(memory.get(d.weight_row+t)==('weight',kk+t,nc,d.cols),'RHS latest loaded source differs')
            for i in range(op.rows):
                require(memory.get(op.input_row+i)==('lhs',mr+i,kk,op.cols),'LHS latest loaded source differs')
                row=d.accumulator_row+i
                if d.accumulate:require(acc.get(row)==(mr+i,nc,d.cols,kk),'ACC provenance or increasing K differs')
                acc[row]=(mr+i,nc,d.cols,kk+op.cols)
            pending=None;totals['compute_pairs']+=1
        elif type(op) is Store:
            touch('store');mr,nc=op.dram_row,op.dram_col
            require(0<=mr<p.m and mr%16==0 and 0<=nc<p.n and nc%16==0 and op.rows==min(16,p.m-mr) and op.cols==min(16,p.n-nc),'output source coordinates or tails differ')
            window(s.accumulator,op.accumulator_row,op.rows)
            for i in range(op.rows):
                require(acc.get(op.accumulator_row+i)==(mr+i,nc,op.cols,p.k),'store does not observe its complete original product')
                require((mr+i,nc) not in outputs,'duplicate output observer')
                outputs.add((mr+i,nc))
            totals['output_tiles']+=1
        else:raise ValueError('finite retained verifier encountered unsupported operation')
    require(pending is None and phase is None,'kernel ends without complete final epoch')
    require(len(outputs)==p.m*p.tiles[2],'original output domain is incomplete')
    require(len(products)==p.tiles[0]*p.tiles[1]*p.tiles[2],'original product domain is incomplete')
    require(len(initializers)==(p.tiles[0]*p.tiles[2] if p.bias else 0),'original initializer domain is incomplete')
    return dict(schema='authored_dense_finite_retained.v1',verification='exact exhaustive projected tile commands; no tensor sampling',
        kernel_sha256=kernel.digest(),dynamic_commands=dynamic,static_commands=static,
        dynamic_epoch_drains=drains,output_row_tiles_verified=len(outputs),
        numerical_policy_identity=a.numerical_policy_identity,ordered_accumulator_commits=law.qualification_identity,
        missing_dependency='z3',disjoint_mixed_epochs_checked=allow_disjoint_mixed,bounds=dict(max_commands=MAX_COMMANDS,max_output_row_tiles=MAX_OUTPUT_ROW_TILES),**totals)


@dataclass(frozen=True)
class FiniteRetainedResult:
    kernel:object
    realization:dict
    checked_identity:str
    verifier_identity:str

    def revalidate(self):
        if self.kernel.digest()!=self.checked_identity or sha256(Path(__file__).read_bytes()).hexdigest()!=self.verifier_identity:
            raise ValueError('finite retained checked program or verifier changed')
    def emit_llvm(self,builder,lower_call):
        from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter,emit_index_program
        self.revalidate();result=emit_index_program(self.kernel,FnBuilderIndexAdapter(builder),lower_call);self.revalidate();return result
    def materialize_scf(self):
        from merlin.compile.scheduling.scf_commands import materialize_scf_commands
        self.revalidate();return materialize_scf_commands(self.kernel)


def bounded_retained_schedule(raw,transforms,p,block,r,s,a,law,plan,source):
    """Construct the identical shared retained realization and prove every call."""
    from merlin.compile.scheduling.schedule_ir import read_dense_schedule_plan,check_dense_schedule_commands
    decision=read_dense_schedule_plan(plan,p,r,s,a,source_identity=source,dtypes=('i8','i8','i32'),static_budget=128,
        bias_source=DenseBiasSource(p.n,'i32') if p.bias else None)
    if decision.block!=block or decision.pipeline_kind!='retained_full_k':raise ValueError('bounded retained decision differs')
    kernel=complete_dense_retained_reduction(propose_loop_transforms(raw,transforms),p,block,r,s,a)
    report=verify_retained_commands(kernel,p,r,s,a,law)
    check_dense_schedule_commands(decision,report['static_commands'],128)
    report['schedule_ir']={'sha256':decision.plan_sha256,'text':decision.plan_text,'source_identity':source,
        'pipeline_kind':'retained_full_k','prefetch':False,'decisions_consumed':['block_tiles','loop_transforms','completion']}
    return FiniteRetainedResult(kernel,report,kernel.digest(),sha256(Path(__file__).read_bytes()).hexdigest())


def bounded_streaming_schedule(p,r,s,a,law,plan,source,*,name='dense_streaming'):
    """Reconstruct original shared streaming kernel and check every epoch.

    Only the selected combined segmented law or the original two-window law
    may admit disjoint future loads. No RAW/WAR scoreboard assumption replaces
    a completion: every operand reader must use previously completed loads.
    """
    from merlin.compile.scheduling.schedule_ir import read_dense_schedule_plan,check_dense_schedule_commands
    from merlin.compile.scheduling.dense_segmented_pipeline import DenseSegmentedAdmission,build_dense_segmented_pipeline
    from merlin.compile.scheduling.dense_panel_pipeline import DensePanelOverlapAdmission,build_dense_panel_pipeline
    law.__post_init__()
    bias_source=DenseBiasSource(p.n,'i32') if p.bias else None
    decision=read_dense_schedule_plan(plan,p,r,s,a,source_identity=source,dtypes=('i8','i8','i32'),
        static_budget=128,bias_source=bias_source)
    if decision.loop_transforms or decision.pipeline_issue_order!='compute_then_prefetch':
        raise ValueError('bounded streaming verifier requires the original unmodified issue schedule')
    if decision.pipeline_kind=='retained_lhs_segmented_k':
        if type(law) is not DenseSegmentedAdmission:raise ValueError('finite segmented verifier requires selected combined execution law')
        builder=build_dense_segmented_pipeline;checked_law=law.reduction
    elif decision.pipeline_kind=='two_window':
        if type(law) is not DensePanelOverlapAdmission:raise ValueError('finite two-window verifier requires selected overlap law')
        builder=build_dense_panel_pipeline;checked_law=law
    else:raise ValueError('bounded streaming verifier requires an original streaming plan')
    kernel=builder(p,decision.block,r,decision.storage,a,name=name,dtypes=('i8','i8','i32'),
        prefetch=decision.pipeline_prefetch,outer_order=decision.outer_order,
        window_offsets=decision.pipeline_window_offsets,issue_order=decision.pipeline_issue_order,bias_source=bias_source)
    report=verify_retained_commands(kernel,p,r,decision.storage,a,checked_law,allow_disjoint_mixed=True)
    check_dense_schedule_commands(decision,report['static_commands'],128)
    report['schema']='authored_dense_finite_streaming.v1'
    report['schedule_ir']={'sha256':decision.plan_sha256,'text':decision.plan_text,'source_identity':source,
        'pipeline_kind':decision.pipeline_kind,'prefetch':decision.pipeline_prefetch,
        'pipeline_issue_order':decision.pipeline_issue_order,
        'decisions_consumed':['block_tiles','outer_order','row_bases','panel_pipeline','pipeline_window_offsets','pipeline_issue_order']}
    return FiniteRetainedResult(kernel,report,kernel.digest(),sha256(Path(__file__).read_bytes()).hexdigest())
