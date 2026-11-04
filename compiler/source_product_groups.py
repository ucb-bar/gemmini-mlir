"""Conservative semantic discovery of adjacent independent raw dense products.

No instruction is emitted and the input graph is never mutated. Source task
identity remains per member; consumers must preserve that identity even when
sharing a physical input load. Every source transport and non-dense operation
ends a discovery window. RES_PACK/EVICT only manage handles in this command ABI.
"""
from __future__ import annotations

from copy import deepcopy
from math import isfinite
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProductCandidate:
    command_index: int
    commit_index: int
    lhs: str
    rhs: str
    output: str
    accumulator: str
    m: int
    k: int
    n: int
    attrs: dict[str, Any]
    source_task_index: int | None
    source_op_index: int | None
    region_id: str | None
    enclosing_scope: tuple

    def as_sibling_product(self):
        from dense_siblings import SiblingProduct
        return SiblingProduct(self.lhs, self.rhs, self.output, self.m, self.k,
                              self.n, deepcopy(self.attrs), (), True, False, None)

    def as_rhs_product(self):
        from dense_rhs_epochs import RHSProduct
        return RHSProduct(self.lhs, self.rhs, self.output, self.m, self.k,
                          self.n, deepcopy(self.attrs), (), True, False, None)


@dataclass(frozen=True)
class ProductGroup:
    members: tuple[ProductCandidate, ...]
    # These original commands still run through the incumbent handle bookkeeping.
    administrative_indices: tuple[int, ...]
    requires_per_member_attribution: bool = True

    @property
    def products(self):
        return tuple(member.as_sibling_product() for member in self.members)

    @property
    def commit_indices(self):
        return tuple(member.commit_index for member in self.members)

    @property
    def command_indices(self):
        return tuple(member.command_index for member in self.members)

    @property
    def first_commit_index(self):
        return self.members[0].commit_index


@dataclass(frozen=True)
class DiscoveryRefusal:
    command_index: int
    reason: str


@dataclass(frozen=True)
class ProductDiscovery:
    groups: tuple[ProductGroup, ...]
    refusals: tuple[DiscoveryRefusal, ...]


def _attrs(cb, index):
    attrs = deepcopy(cb['commands'][index].get('attributes') or {})
    attrs.update(deepcopy(cb.get('params', {}).get('command_source_attribution', {}).get(str(index), {})))
    return attrs


def _positive_scalar(value):
    if type(value) not in (int, float):
        return False
    try:
        return isfinite(value) and value > 0
    except OverflowError:
        return False


def _scope(cb, attrs):
    provenance = attrs.get('source_provenance') or {}
    return (provenance.get('entry', cb.get('params', {}).get('source_entry')),
            attrs.get('source_enclosing_region'), attrs.get('source_control_region'),
            attrs.get('source_function'))


def _discover_dense_groups(cb: dict, *, max_members: int = 8,
                           allow_readout: bool = False,
                           shared_operand: str = 'lhs') -> ProductDiscovery:
    """Find deterministic bounded groups; report why original commits remain single.

    Physical operand names are resolved through the live resident-handle map,
    never through shape/value equality. Different component provenance labels
    are retained per member. Distinct enclosing control regions, source task
    gaps, effects, source transports, transposition, aliasing, bias or readout
    transformations prevent grouping. With ``allow_readout=True``, scalar
    positive acc_scale and optional following ReLU are source candidates; the
    schedule helper still must prove their complete numerical readout contract.
    Declared ABI buffers remain unchanged.
    """
    if not 2 <= max_members <= 8:
        raise ValueError('max_members must be between two and eight')
    commands, tensors = cb.get('commands', []), cb.get('tensors', {})
    transport_positions = {row['before_command_index'] for row in
                           cb.get('params', {}).get('source_transport_tasks', ())}
    residents, accumulators = {}, {}
    windows: list[list[ProductCandidate]] = [[]]
    refusals: list[DiscoveryRefusal] = []
    admin = []
    epoch = 0
    def barrier():
        nonlocal epoch
        epoch += 1
        if windows[-1]:
            windows.append([])
    for index, command in enumerate(commands):
        if index in transport_positions:
            barrier()
        opcode, operands = command['opcode'], command.get('operands') or {}
        attrs = _attrs(cb, index)
        if opcode == 'RES_PACK':
            source = residents.get(operands.get('src'), operands.get('src'))
            if attrs.get('layout', 'packed_rhs') != 'packed_rhs' or source not in tensors:
                barrier()
                residents.pop(operands.get('dst'), None)
            else:
                residents[operands['dst']] = source
                admin.append(index)
        elif opcode == 'EVICT':
            residents.pop(operands.get('handle'), None)
            admin.append(index)
        elif opcode in ('MATMUL', 'MATMUL_RESIDENT'):
            accumulators[operands['dst']] = (index, operands.get('lhs'),
                residents.get(operands.get('rhs'), operands.get('rhs')), attrs, epoch)
        elif opcode == 'COMMIT':
            origin = accumulators.pop(operands.get('src'), None)
            reason = None
            if origin is None:
                reason = 'commit lacks one live original product'
            else:
                product_index, lhs, rhs, product_attrs, product_epoch = origin
                options = {**product_attrs, **attrs}
                stages = options.get('epilogue', [])
                scalar = options.get('acc_scale', 1.0)
                scaled_readout = (allow_readout
                    and stages in (['acc_scale'], ['acc_scale', 'relu'])
                    and _positive_scalar(scalar)
                    and 'acc_scale_by_column' not in options
                    and options.get('output_dtype') == 'i8')
                if product_epoch != epoch:
                    reason = 'effect or source transport separates product and commit'
                elif lhs not in tensors or rhs not in tensors or operands.get('dst') not in tensors:
                    reason = 'unresolved physical tensor operand'
                elif options.get('rhs_transposed', False):
                    reason = 'transposed RHS'
                elif (operands.get('bias') or options.get('bias') or (stages and not scaled_readout)
                      or any(key.startswith('source_epilogue_') for key in options)):
                    reason = 'bias or transformed readout'
                elif any(options.get(key) not in (None, False, 0, 'zero') for key in
                         ('seed', 'initializer', 'accumulate', 'initial_value')):
                    reason = 'nonzero or unknown accumulator seed'
                else:
                    a, w, out = tensors[lhs], tensors[rhs], tensors[operands['dst']]
                    if ((a.get('dtype'), w.get('dtype')) != ('i8', 'i8')
                            or not ((out.get('dtype') == 'i32' and options.get('output_dtype', 'i32') == 'i32' and not stages)
                                    or (out.get('dtype') == 'i8' and scaled_readout))):
                        reason = ('requires signed i8 inputs and raw i32 or admitted scalar-scaled i8 output'
                                  if allow_readout else 'requires signed i8 inputs and raw i32 output')
                    elif (len(a['shape']) != 2 or len(w['shape']) != 2 or len(out['shape']) != 2
                          or a['shape'][1] != w['shape'][0]
                          or out['shape'] != [a['shape'][0], w['shape'][1]]
                          or any(type(d) is not int or d <= 0 for d in a['shape'] + w['shape'])):
                        reason = 'requires positive matching rank-two dense shapes'
                    elif operands['dst'] in (lhs, rhs) or lhs == rhs:
                        reason = 'physical input/output aliases'
                    elif _scope(cb, product_attrs) != _scope(cb, attrs):
                        reason = 'product and commit cross enclosing control scope'
                    elif (product_attrs.get('source_task_index') != attrs.get('source_task_index')
                          or product_attrs.get('prov.region_id') != attrs.get('prov.region_id')
                          or product_attrs.get('source_primary_op_index') != attrs.get('source_primary_op_index')):
                        reason = 'product and commit source identity differs'
                    elif (attrs.get('source_task_index') is not None
                          and type(attrs.get('source_primary_op_index')) is not int):
                        reason = 'attributed product lacks exact primary source operation'
                    elif shared_operand == 'rhs' and not _immutable_rhs_evidence(cb, rhs)['proven']:
                        reason = _immutable_rhs_evidence(cb, rhs)['reason']
                    else:
                        windows[-1].append(ProductCandidate(product_index, index, lhs, rhs,
                            operands['dst'], operands['src'], a['shape'][0], a['shape'][1],
                            w['shape'][1], deepcopy(options), attrs.get('source_task_index'),
                            attrs.get('source_primary_op_index'), attrs.get('prov.region_id'),
                            _scope(cb, options)))
            if reason:
                refusals.append(DiscoveryRefusal(index, reason))
                barrier()
        else:
            barrier()
    groups = []
    for window in windows:
        pending = []
        def finish():
            if len(pending) >= 2:
                start, end = pending[0].command_index, pending[-1].commit_index
                groups.append(ProductGroup(tuple(pending), tuple(i for i in admin if start <= i <= end)))
            elif pending:
                refusals.append(DiscoveryRefusal(pending[0].commit_index, 'no adjacent independent compatible sibling'))
            pending.clear()
        for candidate in window:
            compatible = True
            if pending:
                first, last = pending[0], pending[-1]
                outputs = {p.output for p in pending}
                if shared_operand == 'lhs':
                    identity_matches = ((candidate.lhs, candidate.m, candidate.k, candidate.enclosing_scope) ==
                                        (first.lhs, first.m, first.k, first.enclosing_scope)
                                        and candidate.rhs not in {p.rhs for p in pending})
                else:
                    identity_matches = ((candidate.rhs, candidate.k, candidate.n, candidate.enclosing_scope) ==
                                        (first.rhs, first.k, first.n, first.enclosing_scope)
                                        and candidate.lhs not in {p.lhs for p in pending})
                compatible = (identity_matches
                              and candidate.output not in outputs
                              and candidate.output not in {p.lhs for p in pending} | {p.rhs for p in pending}
                              and candidate.lhs not in outputs and candidate.rhs not in outputs
                              and len(pending) < max_members)
                tasks = (last.source_task_index, candidate.source_task_index)
                if tasks != (None, None):
                    compatible = compatible and all(type(t) is int for t in tasks) and tasks[1] in (tasks[0], tasks[0] + 1)
            if not compatible:
                finish()
            pending.append(candidate)
        finish()
    return ProductDiscovery(tuple(groups), tuple(sorted(refusals, key=lambda row: row.command_index)))



def discover_product_groups(cb: dict, *, max_members: int = 8,
                            allow_readout: bool = False) -> ProductDiscovery:
    """Discover shared-LHS candidates; existing admission behavior is unchanged."""
    return _discover_dense_groups(cb, max_members=max_members,
                                  allow_readout=allow_readout, shared_operand='lhs')


def _coalesce_product_tasks(cb: dict, accepted_groups, *, allow_readout: bool = False,
                            max_members: int = 8, shared_operand: str = 'lhs'):
    """Coarsen actual admitted sibling schedules into exact common task owners.

    Call only after the scheduler has admitted each supplied original group.
    Returns a new CB, discovery refreshed against it, and an original-to-current
    task map. No source operation/region identity or physical tensor is changed.
    A task containing other work cannot be partially merged. Cached host LLVM
    cannot be retagged by this semantic graph transformation.
    """
    from source_attribution import task_definitions
    transformed = deepcopy(cb)
    accepted_groups = tuple(accepted_groups)
    def discover(graph):
        if shared_operand == 'rhs':
            return discover_rhs_product_groups(graph, max_members=max_members)
        return discover_product_groups(graph, allow_readout=allow_readout, max_members=max_members)
    current = discover(cb)
    def signature(group):
        return tuple((m.command_index, m.commit_index, m.lhs, m.rhs, m.output,
                      m.source_task_index, m.source_op_index, m.region_id)
                     for m in group.members)
    available = {signature(group): group for group in current.groups}
    for group in accepted_groups:
        group_type = RHSProductGroup if shared_operand == 'rhs' else ProductGroup
        if (not isinstance(group, group_type) or signature(group) not in available
                or available[signature(group)] != group):
            raise ValueError('accepted product group is stale or was not semantically admitted')
    if not accepted_groups or all(all(m.source_task_index is None for m in g.members)
                                  for g in accepted_groups):
        return transformed, discover(transformed), {
            'task_map': {}, 'merged_original_tasks': [],
            'reason': 'no attributed source tasks need coalescing'}
    definitions = task_definitions(cb)
    merged_sets, occupied = [], set()
    for group in accepted_groups:
        tasks = {m.source_task_index for m in group.members}
        if any(type(task) is not int for task in tasks):
            raise ValueError('cannot mix attributed and unattributed product members')
        tasks = sorted(tasks)
        if tasks != list(range(tasks[0], tasks[-1] + 1)):
            raise ValueError('only adjacent original tasks can be coalesced')
        if occupied.intersection(tasks):
            raise ValueError('accepted groups overlap original source task owners')
        occupied.update(tasks)
        member_commands = {i for m in group.members for i in (m.command_index, m.commit_index)}
        handles = {cb['commands'][m.command_index]['operands']['rhs'] for m in group.members}
        rhs_buffers = {m.rhs for m in group.members}
        allowed_commands = set(member_commands)
        for index, command in enumerate(cb['commands']):
            operands = command.get('operands') or {}
            if (command['opcode'] == 'RES_PACK' and operands.get('dst') in handles
                    and operands.get('src') in rhs_buffers):
                allowed_commands.add(index)
            elif command['opcode'] == 'EVICT' and operands.get('handle') in handles:
                allowed_commands.add(index)
        actual_commands = set()
        for task in tasks:
            if not 0 <= task < len(definitions):
                raise ValueError('accepted product source task does not exist')
            row = definitions[task]
            if 'host' in row['kind'] or 'transport' in row['kind']:
                raise ValueError('host or transport source tasks cannot be coalesced')
            actual_commands.update(row.get('command_indices', ()))
        if not member_commands <= actual_commands or not actual_commands <= allowed_commands:
            raise ValueError('original source task contains work outside admitted product members')
        first, last = min(actual_commands), max(actual_commands)
        if any(first < row['before_command_index'] <= last for row in
               cb.get('params', {}).get('source_transport_tasks', ())):
            raise ValueError('source transport crosses the admitted product group')
        # Command metadata and explicit task records must agree before editing.
        for index in actual_commands:
            if _attrs(cb, index).get('source_task_index') not in tasks:
                raise ValueError('task definitions disagree with actual command owners')
        if len(tasks) > 1:
            merged_sets.append(tuple(tasks))
    if merged_sets and cb.get('params', {}).get('_host_llvm'):
        raise ValueError('cached host LLVM requires actual emitter task regrouping before coalescing')
    merge_roots = {task: group[0] for group in merged_sets for task in group}
    task_map, roots, fresh_definitions = {}, {}, []
    for row in definitions:
        old = row['task_index']
        root = merge_roots.get(old, old)
        if root not in roots:
            roots[root] = len(fresh_definitions)
            fresh = deepcopy(row)
            fresh['task_index'] = roots[root]
            # Ranges are an emission product and must be measured again.
            fresh.pop('instruction_start', None)
            fresh.pop('instruction_end', None)
            fresh_definitions.append(fresh)
        task_map[old] = roots[root]
        if old == root:
            continue
        target = fresh_definitions[roots[root]]
        target['kind'] = 'source_product_group'
        for key in ('source_op_indices', 'command_indices', 'reads', 'writes', 'source_region_ids'):
            if key in target or key in row:
                # Missing access lists stay absent, allowing the exact source
                # access derivation to recompute the merged boundary at emit.
                if key in ('reads', 'writes') and not all(key in definitions[t] for t in
                        next(group for group in merged_sets if root == group[0])):
                    target.pop(key, None)
                else:
                    target[key] = sorted(set(target.get(key, ())) | set(row.get(key, ())))
        target.pop('symbol', None)
    params = transformed.setdefault('params', {})
    params['source_task_definitions'] = fresh_definitions
    params.pop('global_program_plan', None)
    ownership = {row['task_index']: row['source_op_indices'] for row in fresh_definitions}
    def update_scope(attrs):
        old = attrs.get('source_task_index')
        if old is not None:
            if old not in task_map:
                raise ValueError('task reference has no original task definition')
            attrs['source_task_index'] = task_map[old]
            attrs['source_op_indices'] = list(ownership[task_map[old]])
    for command in transformed.get('commands', ()):
        update_scope(command.get('attributes') or {})
    for attrs in params.get('command_source_attribution', {}).values():
        update_scope(attrs)
    for transport in params.get('source_transport_tasks', ()):
        update_scope(transport)
    if 'host_source_tasks' in params:
        params['host_source_tasks'] = [task_map[t] if type(t) is int else t
                                       for t in params['host_source_tasks']]
    # No original source ownership can disappear or acquire another owner.
    old_ops = sorted(i for row in definitions for i in row['source_op_indices'])
    new_ops = sorted(i for row in fresh_definitions for i in row['source_op_indices'])
    if old_ops != new_ops or len(set(new_ops)) != len(new_ops):
        raise ValueError('coalescing changes exact source operation ownership')
    refreshed = discover(transformed)
    return transformed, refreshed, {'task_map': task_map,
        'merged_original_tasks': [list(tasks) for tasks in merged_sets],
        'source_operation_coverage': new_ops,
        'scope': 'actual admitted independent product schedule',
        'shared_operand': shared_operand,
        'source_primary_operations_and_regions': 'preserved on every original command'}


@dataclass(frozen=True)
class RHSProductGroup:
    """Source candidates only: no RHS retention/scheduling law is asserted."""
    members: tuple[ProductCandidate, ...]
    administrative_indices: tuple[int, ...]
    immutable_rhs: dict[str, Any]
    requires_per_member_attribution: bool = True

    @property
    def products(self):
        return tuple(member.as_rhs_product() for member in self.members)

    @property
    def first_commit_index(self):
        return self.members[0].commit_index

    @property
    def shared_rhs(self):
        return self.members[0].rhs

    @property
    def command_indices(self):
        return tuple(member.command_index for member in self.members)

    @property
    def commit_indices(self):
        return tuple(member.commit_index for member in self.members)


@dataclass(frozen=True)
class RHSProductDiscovery:
    groups: tuple[RHSProductGroup, ...]
    refusals: tuple[DiscoveryRefusal, ...]


def _immutable_rhs_evidence(cb, rhs):
    """Prove one ordinary read-only entry tensor; never infer SSA from values."""
    def refuse(reason):
        return {'proven': False, 'tensor': rhs, 'reason': reason}
    spec = cb.get('tensors', {}).get(rhs, {})
    if spec.get('role') not in ('input', 'weight'):
        return refuse('immutable RHS requires a declared entry input or weight')
    if (spec.get('encoding') is not None or spec.get('layout') not in (None, 'row_major')
            or any(spec.get(key) is not None for key in ('alias_of', 'base_tensor'))
            or spec.get('volatile', False)):
        return refuse('immutable RHS requires ordinary unencoded nonaliased tensor layout')
    arguments = cb.get('kernel_abi', {}).get('args', ())
    accesses = [row.get('access') for row in arguments if row.get('tensor') == rhs]
    if accesses != ['read']:
        return refuse('immutable RHS lacks one exact read-only ABI argument')
    params = cb.get('params', {})
    if 'tensor_source_map' in params:
        origin = params['tensor_source_map'].get(rhs)
        if (not isinstance(origin, dict) or set(origin) != {'arg_index'}
                or type(origin['arg_index']) is not int or origin['arg_index'] < 0):
            return refuse('initial immutable RHS route requires exact entry SSA argument identity')
        identity = deepcopy(origin)
    else:
        # Interface tensor declarations directly define immutable typed SSA
        # entry buffers. The canonical physical name is that declaration's name.
        identity = {'interface_tensor': rhs}
    for command in cb.get('commands', ()):
        if (command.get('operands') or {}).get('dst') == rhs:
            return refuse('a command writes the proposed immutable RHS')
    for row in params.get('source_transport_tasks', ()):
        if row.get('fill', {}).get('dst') == rhs or any(
                copy.get('dst') == rhs for copy in row.get('copies', ())):
            return refuse('a source transport writes the proposed immutable RHS')
    for row in params.get('source_task_definitions', ()):
        if rhs in row.get('writes', ()):
            return refuse('a source task declares a write to the proposed immutable RHS')
    return {'proven': True, 'tensor': rhs, 'source_identity': identity,
            'dtype': spec.get('dtype'), 'shape': deepcopy(spec.get('shape')),
            'layout': 'ordinary row-major', 'abi_access': 'read',
            'write_command_indices': [], 'write_source_transports': [],
            'qualification': 'immutable source candidate; no physical retention law'}


def discover_rhs_product_groups(cb: dict, *, max_members: int = 8) -> RHSProductDiscovery:
    """Find adjacent independent raw products sharing the same immutable RHS.

    Members have distinct physical LHS/output tensors, identical RHS identity
    and K/N, and may have different M. This deliberately requires an ordinary
    read-only entry SSA tensor as RHS. Computed/prepared RHS values need a future
    explicit immutable-view proof; matching shapes or provenance never suffice.
    Effects, source transports, enclosing-control changes and source-task gaps
    retain the same hard barriers as LHS discovery. No instruction is emitted.
    """
    found = _discover_dense_groups(cb, max_members=max_members,
                                  allow_readout=False, shared_operand='rhs')
    groups = tuple(RHSProductGroup(group.members, group.administrative_indices,
                   _immutable_rhs_evidence(cb, group.members[0].rhs))
                   for group in found.groups)
    return RHSProductDiscovery(groups, found.refusals)


def product_group_evidence(cb: dict, groups, *, original_cb: dict | None = None,
                           coalescing_report: dict | None = None) -> dict:
    """Explain exact source membership, ordered ABI and retained effect barriers.

    With ``original_cb``, preservation checks compare real before/after graphs.
    Without it, before/after invariants remain explicitly unknown. This reports
    source legality only, and grants no scheduling or hardware residency law.
    """
    groups = tuple(groups)
    commands = cb.get('commands', ())
    abi = cb.get('kernel_abi', {})
    transports = cb.get('params', {}).get('source_transport_tasks', ())
    records = []
    for group in groups:
        members = []
        for member in group.members:
            consumers = [index for index, command in enumerate(commands)
                         if member.output in [value for key, value in
                             (command.get('operands') or {}).items() if key != 'dst']]
            members.append({'command_index': member.command_index,
                'commit_index': member.commit_index, 'lhs': member.lhs,
                'rhs': member.rhs, 'output': member.output,
                'shape': {'m': member.m, 'k': member.k, 'n': member.n},
                'source_task_index': member.source_task_index,
                'source_primary_op_index': member.source_op_index,
                'source_region_id': member.region_id,
                'original_attrs': deepcopy(member.attrs),
                'output_consumer_command_indices': consumers})
        start, end = min(group.command_indices), max(group.commit_indices)
        effects = [{'command_index': index, 'opcode': commands[index]['opcode']}
                   for index in range(start, end + 1)
                   if commands[index]['opcode'] not in
                   ('RES_PACK', 'EVICT', 'MATMUL', 'MATMUL_RESIDENT', 'COMMIT')]
        crossing_transports = [deepcopy(row) for row in transports
                              if start < row['before_command_index'] <= end]
        records.append({'shared_operand': 'rhs' if isinstance(group, RHSProductGroup) else 'lhs',
            'members': members, 'effect_barriers_inside_group': effects,
            'source_transports_inside_group': crossing_transports,
            'administrative_indices': list(group.administrative_indices),
            'immutable_rhs': deepcopy(group.immutable_rhs) if isinstance(group, RHSProductGroup) else None,
            'outputs_retained': [member.output for member in group.members]})
    preserved = {'ordered_abi': None, 'source_value_bindings': None,
                 'command_effects': None, 'source_transports': None,
                 'primary_operations_and_regions': None}
    if original_cb is not None:
        def semantics(graph):
            return [(row['opcode'], row.get('operands') or {},
                     {key: value for key, value in _attrs(graph, index).items()
                      if key not in ('source_task_index', 'source_op_indices')})
                    for index, row in enumerate(graph.get('commands', ()))]
        def transport_effects(graph):
            return [{key: value for key, value in row.items() if key != 'source_task_index'}
                    for row in graph.get('params', {}).get('source_transport_tasks', ())]
        def provenance(graph):
            return [(attrs.get('source_primary_op_index'), attrs.get('prov.region_id'))
                    for attrs in (_attrs(graph, index) for index in range(len(graph.get('commands', ()))))]
        preserved = {
            'ordered_abi': original_cb.get('kernel_abi') == cb.get('kernel_abi'),
            'source_value_bindings': original_cb.get('params', {}).get('tensor_source_map') == cb.get('params', {}).get('tensor_source_map'),
            'command_effects': semantics(original_cb) == semantics(cb),
            'source_transports': transport_effects(original_cb) == transport_effects(cb),
            'primary_operations_and_regions': provenance(original_cb) == provenance(cb)}
    return {'schema': 'source_product_group_evidence_v1',
        'source_sha256': cb.get('params', {}).get('source_sha256'),
        'ordered_abi_arguments': deepcopy(abi.get('args')),
        'ordered_declared_outputs': deepcopy(abi.get('outputs')),
        'groups': records, 'preservation': preserved,
        'coalescing': deepcopy(coalescing_report),
        'qualification': 'source semantics only; scheduler admission and target lifetime proof required'}


def coalesce_product_tasks(cb: dict, accepted_groups, *, allow_readout: bool = False,
                           max_members: int = 8) -> tuple[dict, ProductDiscovery, dict]:
    """Coalesce scheduler-admitted shared-LHS groups, preserving existing ABI."""
    return _coalesce_product_tasks(cb, accepted_groups, allow_readout=allow_readout,
                                   max_members=max_members, shared_operand='lhs')


def coalesce_rhs_product_tasks(cb: dict, accepted_groups, *, max_members: int = 8
                               ) -> tuple[dict, RHSProductDiscovery, dict]:
    """Coalesce admitted raw RHS schedules with exact unioned source ownership.

    Scheduler admission must precede this transformation. Returned members use
    the actual merged task owner; each member's original primary operation and
    region remain attached. No command, source value, tensor or ABI is removed.
    """
    return _coalesce_product_tasks(cb, accepted_groups, max_members=max_members,
                                   shared_operand='rhs')
