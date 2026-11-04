"""Checked full-K shared-input sibling products, behind an explicit caller choice.

The shared scheduler checks all actual member effects and private windows. The
terminal consumer uses the common ISA encoder and synchronizes every ownership
transition. This is compiler-side evidence, never independent certification.
"""
from __future__ import annotations
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import json

from merlin.sched.ir import Call, Ptr, as_expr
from merlin.compile.scheduling.block_schedule import Load, Store
from merlin.compile.scheduling.compact_dense import DenseLoopStorage, DenseMemoryView
from merlin.compile.scheduling.compact_dense_group import schedule_compact_dense_group
from merlin.compile.scheduling.compact_dense_group_check import DenseLHSReuseLaw
from merlin.transforms.dense_planning import DenseProblem, DenseResources, dense_blocks
from . import isa
from .dense import _adapter
from .dense_readout import store_tile, _pack


def sibling_groups(cb):
    """Find ordered independent raw sibling outputs by exact CB value identity.

    Each member records its original COMMIT position so the caller can consume
    it once. Unknown compute/control operations terminate a proposal.
    """
    residents, products, runs = {}, {}, []
    run = []
    def finish():
        nonlocal run
        if len(run) > 1:
            runs.append(tuple(run))
        run = []
    for index, command in enumerate(cb.get('commands', [])):
        code, a, attrs = command['opcode'], command.get('operands', {}), command.get('attributes', {})
        if code == 'RES_PACK':
            residents[a['dst']] = residents.get(a['src'], a['src'])
        elif code in ('MATMUL', 'MATMUL_RESIDENT'):
            products[a['dst']] = (a['lhs'], residents.get(a['rhs'], a['rhs']), attrs)
        elif code == 'EVICT':
            pass
        elif code == 'COMMIT' and a['src'] in products:
            lhs, rhs, product_attrs = products[a['src']]
            merged = {**product_attrs, **attrs}
            if merged.get('epilogue', []) or merged.get('output_dtype', 'i32') != 'i32' or merged.get('bias') or merged.get('accumulator_init'):
                finish()
                continue
            row = {'lhs': lhs, 'rhs': rhs, 'dst': a['dst'], 'attributes': merged, 'command_index': index}
            if run and (lhs != run[0]['lhs'] or any(x['dst'] in (lhs, rhs) or row['dst'] in (x['lhs'],x['rhs'],x['dst']) for x in run) or len(run) == 8):
                finish()
            run.append(row)
        else:
            finish()
    finish()
    return tuple(runs)


def checked_schedule(specs, *, retention_law=None, equal_operand_partitions=False):
    """Return a frozen checked group, or None when its full-K panels do not fit."""
    specs = tuple(tuple(map(int, spec)) for spec in specs)
    if any(len(spec) != 3 or min(spec) <= 0 for spec in specs):
        return None
    dim = isa.DIM
    members = len(specs)
    if not 2 <= members <= 8:
        return None
    m, k = specs[0][:2]
    if any((a,b) != (m,k) for a,b,_ in specs):
        return None
    adapter = _adapter()
    evidence_path = Path(__file__).with_name('dense_ordering.json')
    evidence_bytes = evidence_path.read_bytes()
    evidence = json.loads(evidence_bytes)
    if evidence['completion_order']['execute']['order'] != 'in_order':
        return None
    identity = 'authored-shared-lhs-synchronous-private-windows:' + sha256(evidence_bytes + Path(__file__).read_bytes() + Path(isa.__file__).read_bytes() + Path(__file__).with_name('dense.py').read_bytes()).hexdigest()
    adapter = replace(adapter, qualification_identity=identity)
    if retention_law is None:
        # This adapter's emitted members write only their checked disjoint RHS
        # and accumulator windows. Full fences close every issue before reuse;
        # logical ABI buffers are valid/disjoint and input pointers are read-only.
        # Suppressing repeated reads of that already-read immutable valid input
        # has no fault or volatile observation in this ordinary memory ABI.
        retention_law = DenseLHSReuseLaw(adapter.configuration_identity, adapter.numerical_policy_identity,
            identity, identity, lhs_unchanged_by_member_operations=True,
            independent_member_interleaving=True, suppressed_loads_no_faults=True,
            suppressed_loads_no_extra_observations=True)
    kt = (k + dim - 1) // dim
    # Reserve the actual bounded shared activation panel, leaving the rest for
    # private RHS panels. An arbitrary half-SPAD reservation needlessly denies
    # the same four-column retained block used by independent decode members.
    one_panel = kt * dim
    max_m = min(4,(m+dim-1)//dim,(isa.SPAD_ROWS-members*one_panel)//one_panel)
    if max_m < 1:
        return None
    lhs_rows = max_m * one_panel
    rhs_rows = ((isa.SPAD_ROWS-lhs_rows)//members//dim)*dim
    acc_rows = (isa.ACC_ROWS//members//dim)*dim
    if one_panel > min(lhs_rows,rhs_rows) or acc_rows < dim:
        return None
    if equal_operand_partitions:
        # Isolated placement alternative: one power-of-two row partition per
        # live operand, leaving unused partitions unallocated. This is an
        # address layout, not an assumption about bank decoding or ordering.
        # The same library admission/effect checks below prove all footprints.
        partition_count = 1 << members.bit_length()
        partition_rows = (isa.SPAD_ROWS // partition_count // dim) * dim
        if one_panel > partition_rows:
            return None
        lhs_rows = rhs_rows = partition_rows
    shared = DenseMemoryView('shared_lhs','spad',dim,0,lhs_rows)
    storages = tuple(DenseLoopStorage(shared,
        DenseMemoryView(f'rhs{i}','spad',dim,lhs_rows+i*rhs_rows,rhs_rows),
        DenseMemoryView(f'acc{i}','acc',dim*4,i*acc_rows,acc_rows)) for i in range(members))
    problems = tuple(DenseProblem(a,b,n,dim,dim,dim,1,1,4) for a,b,n in specs)
    resources = DenseResources(isa.SPAD_ROWS*dim,isa.ACC_ROWS*dim*4,
        max_m_blocks=4,max_n_blocks=4,max_k_blocks=kt,partial_output_tiles=True,partial_output_blocks=True)
    # Use the same bounded block domain as retained dense. Capacity bounds,
    # not capsule identities, choose how many M rows share each loaded RHS and
    # how many N tiles share one completion epoch. Keep K completely resident.
    domains = tuple(tuple(b for b in dense_blocks(p, resources,
        k_limit=lambda bm,bn,s=storage,p=p: s.k_limit(p,resources,bm,bn))
        if b.k == kt and storage.admits(p,b,resources))
        for p,storage in zip(problems,storages))
    common_m = set.intersection(*(set(b.m for b in domain) for domain in domains))
    if not common_m:
        return None
    bm = max(common_m)
    blocks = tuple(min((b for b in domain if b.m == bm),
                      key=lambda b: (b.loads,-b.m*b.n,b.operand_bytes)) for domain in domains)
    def encode_member(member, operation):
        call = adapter.encode(operation)
        args = call.args
        if isinstance(operation, Load):
            name, stride = ('lhs', k) if operation.role == 'lhs' else (f'rhs{member}', specs[member][2])
            args += (('external_pointer', Ptr(name, as_expr(operation.dram_row) * stride + operation.dram_col)),)
        elif isinstance(operation, Store):
            args += (('external_pointer', Ptr(f'output{member}', (as_expr(operation.dram_row) * specs[member][2] + operation.dram_col) * 4)),)
        return Call(('shared_common_' if member is None else f'shared_member{member}_') + call.instr, args)
    def project_member(call):
        if call.instr.startswith('shared_common_'):
            member, mnemonic = None, call.instr[len('shared_common_'):]
        elif call.instr.startswith('shared_member'):
            value, mnemonic = call.instr[len('shared_member'):].split('_',1)
            member = int(value)
        else:
            raise ValueError('unknown shared product instruction owner')
        return member, adapter.project(Call(mnemonic, call.args))
    return schedule_compact_dense_group(problems,resources,storages,adapter,
        encode_member=encode_member,project_member=project_member,retention_law=retention_law,
        name='shared_products',dtypes=('i8','i8','i32'),static_budget=256,
        external_buffers_disjoint=True,verification_timeout_ms=5000,max_members=8,
        block_tiles=tuple((b.m,b.k,b.n) for b in blocks))



def _epoch_index_kernel(kernel):
    """Freeze pure staging/execute epochs of the checked original operation tree.

    Every load epoch independently proves pairwise disjoint actual DMA row
    writes. A full drain closes it before readers. Pure execute epochs keep all
    original descriptor/compute pairs, operand contents and issue order, and
    drain before any subsequent DMA overwrite or readout. Their accumulator
    continuation relies only on selected in-order execute completion. No local
    RAW/WAR scoreboard or load completion order is assumed.
    """
    import z3
    from merlin.sched.ir import Loop, Const, Var, Sum, Scaled, Select
    queries = 0
    def owner(call):
        if call.instr.startswith('shared_common_'):
            return None
        return int(call.instr[len('shared_member'):].split('_',1)[0])
    def calls(item):
        if isinstance(item, Loop):
            return tuple(c for child in item.body for c in calls(child))
        return (item,)
    def phase(call):
        if 'gemmini_mvin_' in call.instr:
            return 'load'
        if call.instr.endswith(('gemmini_preload','gemmini_compute_preloaded')):
            return 'execute'
        if call.instr.endswith('gemmini_mvout'):
            return 'store'
        raise ValueError('unknown group effect '+call.instr)
    def integer(item):
        if isinstance(item, Loop):
            return replace(item, body=tuple(integer(child) for child in item.body))
        return replace(item,args=tuple((k,v) for k,v in item.args if not isinstance(v,Ptr)))
    def expression(expr, env):
        if isinstance(expr,Const):
            return z3.IntVal(expr.value)
        if isinstance(expr,Var):
            return env[expr.name]
        if isinstance(expr,Sum):
            return sum(expression(term,env) for term in expr.terms)
        if isinstance(expr,Scaled):
            return expression(expr.expr,env)*expr.factor
        if isinstance(expr,Select):
            return z3.If(env[expr.var]==expr.equals,expression(expr.then,env),expression(expr.other,env))
        raise ValueError('unknown group address expression')
    def disjoint_staging(item, outer):
        nonlocal queries
        sites = []
        def walk(node, local=()):
            if isinstance(node,Loop):
                for child in node.body:
                    walk(child,(*local,(node.var,node.extent)))
            else:
                sites.append((dict(node.args),local))
        walk(item)
        shared = {name:z3.Int('outer:'+name) for name,_ in outer}
        ambient = [z3.And(shared[name]>=0,shared[name]<extent) for name,extent in outer]
        for i,(left,lscopes) in enumerate(sites):
            for j,(right,rscopes) in enumerate(sites[i:],i):
                le = {**shared,**{name:z3.Int('left:'+name) for name,_ in lscopes}}
                re = {**shared,**{name:z3.Int('right:'+name) for name,_ in rscopes}}
                solver = z3.Solver()
                solver.set(timeout=2000)
                solver.add(*ambient)
                solver.add(*(z3.And(le[name]>=0,le[name]<extent) for name,extent in lscopes))
                solver.add(*(z3.And(re[name]>=0,re[name]<extent) for name,extent in rscopes))
                if i == j:
                    solver.add(z3.Or(*(le[name]!=re[name] for name,_ in lscopes)))
                la,lr = expression(left['addr'],le),expression(left['rows'],le)
                ra,rr = expression(right['addr'],re),expression(right['rows'],re)
                solver.add(la < ra+rr,ra < la+lr)
                queries += 1
                if solver.check() != z3.unsat:
                    raise ValueError('shared staging epoch has overlapping or unproved DMA writes')
    def epochs(body, outer=(), configured=None):
        result = []
        for item in body:
            actual = calls(item)
            owners = {owner(call) for call in actual}
            selected = configured
            if len(owners)==1 and None not in owners:
                member = next(iter(owners))
                if member != selected:
                    result.append(Call(f'shared_config_rhs_{member}',()))
                    selected = member
            phases = {phase(call) for call in actual}
            if phases == {'load'}:
                disjoint_staging(item,outer)
                result.extend((integer(item),Call('shared_epoch_drain',())))
            elif phases == {'execute'}:
                result.extend((integer(item),Call('shared_epoch_drain',())))
            elif phases == {'store'}:
                # The common exact readout encoder retains its own completion.
                result.append(integer(item))
            elif isinstance(item,Loop):
                result.append(replace(item,body=epochs(item.body,(*outer,(item.var,item.extent)),selected)))
            else:
                raise ValueError('unclassified shared product ownership boundary')
        return tuple(result)
    emitted = replace(kernel,body=epochs(kernel.body))
    def original_data(body):
        return tuple(replace(item,body=original_data(item.body)) if isinstance(item,Loop) else item
                     for item in body if isinstance(item,Loop) or
                     not item.instr.startswith(('shared_epoch_drain','shared_config_rhs_')))
    if original_data(emitted.body) != tuple(integer(item) for item in kernel.body):
        raise ValueError('shared completion mapping changed original data calls or counted loops')
    return emitted, {'load_noninterference_queries':queries,
        'serial_projection_preserved':True,
        'completion':'disjoint pure staging and in-order pure execute; full drains between epochs',
        'original_pointer_kernel_sha256':kernel.digest(), 'epoch_index_kernel_sha256':emitted.digest()}


def lower_shared_products(ctx, members, *, retention_law=None, single_panel_only=False,
                          equal_operand_partitions=False):
    """Emit one checked sibling group; return False without emission if ineligible.

    ``single_panel_only`` limits this alternative to complete output panels,
    using the actual checked library choice, before any context mutation.
    ``equal_operand_partitions`` isolates operand placement in equal power-of-two
    row windows; it preserves the common full-K scheduler and ISA consumer.
    ``members`` are sibling_groups rows. Caller must skip their original COMMITs
    only after this returns a receipt. This alternative requires explicit A/B
    selection by the caller and preserves compact fallback for every refusal.
    """
    if len(members) < 2 or any(row['lhs'] != members[0]['lhs'] for row in members):
        return False
    from .resident_rhs import _proof_dependency_available
    if not _proof_dependency_available():return None
    destinations = [row['dst'] for row in members]
    inputs = {row[key] for row in members for key in ('lhs','rhs')}
    if len(set(destinations)) != len(destinations) or inputs.intersection(destinations):
        return False
    aliases = ctx.cb.get('params', {}).get('tensor_aliases', {})
    if any(name in aliases for name in inputs.union(destinations)):
        return False
    specs = []
    for row in members:
        attrs = row.get('attributes', {})
        if attrs.get('epilogue', []) or attrs.get('output_dtype', 'i32') != 'i32' or attrs.get('bias') or attrs.get('accumulator_init'):
            return False
        if attrs.get('relu') or attrs.get('acc_scale_by_column') is not None or float(attrs.get('acc_scale', 1.0)) != 1.0:
            return False
        lhs,rhs,dst = (ctx.spec(row[k]) for k in ('lhs','rhs','dst'))
        if lhs['dtype'] != rhs['dtype'] or lhs['dtype'] != 'i8' or dst['dtype'] != 'i32':
            return False
        a,b,c = lhs['shape'],rhs['shape'],dst['shape']
        if len(a) != 2 or len(b) != 2 or a[1] != b[0] or c != [a[0],b[1]]:
            return False
        specs.append((a[0],a[1],b[1]))
    try:
        schedule = checked_schedule(tuple(specs),retention_law=retention_law,
                                    equal_operand_partitions=equal_operand_partitions)
    except ValueError:
        # A library admission or bounded verification refusal leaves the
        # original compact member lowering available without any emitted work.
        return False
    if schedule is None:
        return False
    schedule.revalidate()
    if single_panel_only and any(
        (m+isa.DIM-1)//isa.DIM > block.m or (n+isa.DIM-1)//isa.DIM > block.n
        for (m,k,n),block in zip(specs,schedule.choice.blocks,strict=True)
    ):
        return False
    try:
        index_kernel, epoch_evidence = _epoch_index_kernel(schedule.kernel)
    except ValueError:
        return False
    from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter, emit_index_program
    fb = ctx.fb
    ctx.fence()
    ctx.emit(*isa.config_ex(dataflow=1))
    ctx.emit(*isa.config_ld(specs[0][1],channel=0))
    columns = {}
    def lower(instruction, arguments):
        if instruction == 'shared_epoch_drain':
            ctx.fence()
            return
        if instruction.startswith('shared_config_rhs_'):
            member = int(instruction.removeprefix('shared_config_rhs_'))
            ctx.current_source_index = members[member].get('attributes',{}).get('merlin.source_op_index')
            ctx.emit(*isa.config_ld(specs[member][2],channel=1))
            return
        if instruction.startswith('shared_common_'):
            member, mnemonic = None, instruction[len('shared_common_'):]
        else:
            value,mnemonic = instruction[len('shared_member'):].split('_',1)
            member = int(value)
        a = {key: bridge.value(value) for key,value in arguments.items()}
        row = members[0 if member is None else member]
        attrs = row.get('attributes', {})
        ctx.current_source_index = attrs.get('merlin.source_op_index')
        m,k,n = specs[0 if member is None else member]
        if mnemonic.startswith('gemmini_mvin_'):
            left = mnemonic == 'gemmini_mvin_lhs'
            stride = k if left else n
            offset = fb.add_i(fb.mul_i(a['dram_row'],fb.const(stride)),a['dram_col'])
            ctx.emit(isa.FUNCT['mvin'] if left else isa.FUNCT['mvin2'],ctx.ptr(row['lhs'] if left else row['rhs'],offset),_pack(fb,a['addr'],a['rows'],a['cols']))
        elif mnemonic == 'gemmini_preload':
            columns[member] = a['cols']
            acc = fb.or_i(fb.or_i(a['acc'],fb.const(1<<31)),fb.shl_i(a['accumulate'],30))
            ctx.emit(isa.FUNCT['preload'],_pack(fb,a['weight'],a['depth'],a['cols']),_pack(fb,acc,a['rows'],a['cols']))
        elif mnemonic == 'gemmini_compute_preloaded':
            ctx.emit(isa.FUNCT['compute_preloaded'],_pack(fb,a['lhs'],a['rows'],a['depth']),_pack(fb,isa.GARBAGE,a['depth'],columns[member]))
        elif mnemonic == 'gemmini_mvout':
            store_tile(ctx,row['dst'],a['acc'],a['rows'],a['cols'],n,a['dram_row'],a['dram_col'],attrs)
        else:
            raise ValueError('unsupported checked shared product operation '+mnemonic)
    bridge = FnBuilderIndexAdapter(fb)
    emitted = emit_index_program(index_kernel,bridge,lower)
    emitted['checked_pointer_kernel_sha256'] = schedule.kernel.digest()
    schedule.revalidate()
    ctx.fence()
    return {**schedule.realization,'operand_placement':('equal_power_of_two_partitions' if equal_operand_partitions else 'bounded_lhs_private_rhs'), 'emission':emitted,'completion_epochs':epoch_evidence,'kernel_sha256':schedule.kernel.digest(),
            'scope':'checked shared full-K effects; independent execution and A/B cycles required'}
