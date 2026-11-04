"""Optional weight-resident full-K M-panel lookahead with two LHS slots."""
import ast
import hashlib
import json
from functools import lru_cache
from pathlib import Path

from merlin.compile.scheduling.compact_dense import DenseLoopStorage, DenseMemoryView
from merlin.compile.scheduling.dense_lhs_panel_pipeline import DenseLhsPanelAdmission, schedule_dense_lhs_panels
from merlin.transforms.dense_planning import DenseProblem, DenseResources, dense_blocks
from . import isa


def _lhs_binding(source, block):
    # Reuse the established immutable common-emitter configuration binding;
    # its MVIN2 checks alone do not admit future LHS traffic. Check the actual
    # selected MVIN edges separately below, before constructing this product.
    from .dense_rhs_pipeline import _binding
    original = _binding(source, block)
    path = Path(__file__).with_name('conv_ordering.json')
    record = json.loads(path.read_text())
    ordering = record['ordering']
    cross = {(r['older'], r['younger'], r['space']): r for r in ordering['cross_queue']}
    selected = {}
    for reader in ('PRELOAD.weight_spad', 'COMPUTE_PRELOADED.a_spad'):
        for older, younger in ((reader, 'MVIN.spad_addr'), ('MVIN.spad_addr', reader)):
            row = cross.get((older, younger, 'scratchpad'), {})
            if row.get('overlap') != 'completion' or row.get('disjoint') not in ('issue', 'none', 'completion'):
                raise ValueError('LHS panel lacks selected MVIN/execute operand dependency law')
            selected[older + '->' + younger] = row
    accesses = ordering['accesses']
    if (accesses['MVIN.spad_addr']['funct'] != 2 or accesses['MVIN.spad_addr']['queue'] != 'load'
        or accesses['COMPUTE_PRELOADED.a_spad']['queue'] != 'execute'
        or accesses['PRELOAD.weight_spad']['queue'] != 'execute'
        or ordering['same_queue']['load']['by_class'].get('MVIN') != 'issue'):
        raise ValueError('LHS panel selected MVIN queue or finite lane progress changed')
    # Exact ordinary MVIN/compute semantics are source-bound to the common
    # emitter's one untransposed execute setup and existing typed tile encoder.
    tree = ast.parse(Path(__file__).with_name('dense.py').read_text())
    setups = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and node.func.attr == 'config_ex']
    if (len(setups) != 1 or setups[0].args or len(setups[0].keywords) != 1
        or setups[0].keywords[0].arg != 'dataflow'
        or not isinstance(setups[0].keywords[0].value, ast.Constant)
        or setups[0].keywords[0].value.value != 1):
        raise ValueError('LHS panel common emitter no longer has its one ordinary execute setup')
    if (ordering['extents']['MVIN.spad_addr']['status'] != 'derived'
        or ordering['extents']['COMPUTE_PRELOADED.a_spad']['status'] != 'derived'):
        raise ValueError('LHS panel lacks selected actual MVIN/compute extent facts')
    finite = record['finite_entries']
    if (finite['load']['status'] != 'derived' or finite['execute']['status'] != 'derived'
        or finite['load']['value'] < 1 or finite['execute']['value'] < 2
        or finite['store']['status'] != 'derived' or finite['store']['value'] < 1):
        raise ValueError('LHS panel selected finite queue capacity is unavailable')
    own = Path(__file__).resolve()
    pins = (*original.binding_sources, (str(own), hashlib.sha256(own.read_bytes()).hexdigest()))
    return DenseLhsPanelAdmission(original.hardware_identity, original.numerical_policy_identity,
        original.qualification_identity + ':selected-MVIN-lhs-edges', source,
        block.m * block.k * block.n, max(block.m * block.k, block.n * block.k),
        finite['execute']['value'] // 2, finite['load']['value'], pins), selected


@lru_cache(maxsize=256)
def lhs_panel_schedule(m, k, n, bias=False, column_scales=False, *, m_tiles=4, n_tiles=4,
                       unroll_compute_n=False):
    """Return a checked actual product or refusal for explicit M/N widths.

    Scope is uniform readout and no bias/seed, with full K in retained RHS and
    both LHS slots. Capacities come from selected real SPAD/ACC facts. No hook,
    empirical selector, array STAY or automatic timing is installed here.
    Explicit unroll_compute_n=True inlines pairs in bounded compute N loops;
    complete transformed-source and epoch checks remain mandatory.
    """
    if type(m_tiles) is not int or m_tiles not in (4, 8):
        raise ValueError('LHS panel M width must be exactly 4 or 8 tiles')
    if type(n_tiles) is not int or not 1 <= n_tiles <= 4:
        raise ValueError('LHS panel N width must be exactly 1..4 tiles')
    if type(unroll_compute_n) is not bool:
        raise ValueError('LHS panel compute N unroll must be an explicit boolean')
    if bias or column_scales:
        return None, 'LHS lookahead currently requires no seed and uniform readout'
    dim = isa.DIM
    p = DenseProblem(m, k, n, dim, dim, dim, 1, 1, 4, bias=False)
    if p.tiles[0] <= m_tiles or p.tiles[2] < n_tiles:
        return None, 'LHS lookahead requires multiple M panels and the requested original N width'
    slot_rows = m_tiles * p.tiles[1] * dim
    rhs_rows = n_tiles * p.tiles[1] * dim
    if 2 * slot_rows + rhs_rows > isa.SPAD_ROWS or m_tiles * n_tiles * dim > isa.ACC_ROWS:
        return None, 'two full-K LHS slots and retained RHS panel exceed selected SPAD/ACC rows'
    s = DenseLoopStorage(DenseMemoryView('lhs', 'spad', dim, 0, 2 * slot_rows),
        DenseMemoryView('rhs', 'spad', dim, 2 * slot_rows, rhs_rows),
        DenseMemoryView('acc', 'acc', 4 * dim, 0, isa.ACC_ROWS))
    r = DenseResources(isa.SPAD_ROWS * dim, isa.ACC_ROWS * 4 * dim,
        max_m_blocks=m_tiles, max_n_blocks=n_tiles, max_k_blocks=p.tiles[1],
        partial_output_tiles=True, partial_output_blocks=True)
    choices = [b for b in dense_blocks(p, r) if (b.m, b.k, b.n) == (m_tiles, p.tiles[1], n_tiles)
               and s.admits(p, b, r)]
    if len(choices) != 1:
        return None, 'LHS lookahead original block or full-K private assignment is not admitted'
    block = choices[0]
    source = hashlib.sha256(json.dumps([m, k, n, False, 'i8-i8-i32'], sort_keys=True).encode()).hexdigest()
    admission, edges = _lhs_binding(source, block)
    from .dense import _adapter
    result = schedule_dense_lhs_panels(p, block, r, s, _adapter(), admission,
        name='dense_lhs_panels', source_identity=source, external_buffers_disjoint=True,
        unroll_compute_n=unroll_compute_n)
    result.realization['lhs_panel_storage'] = dict(lhs_slot_rows=slot_rows, lhs_slots=2,
        retained_rhs_rows=rhs_rows, operand_rows=isa.SPAD_ROWS, accumulator_rows=isa.ACC_ROWS,
        explicit_m_tiles=m_tiles, explicit_n_tiles=n_tiles)
    result.realization['selected_lhs_ordering'] = edges
    result.realization['finite_queues'] = dict(load_entries=admission.pending_loads,
        execute_pairs=admission.pending_compute_pairs, store_commands=1,
        progress='selected ordered lane issue and real ready/valid backpressure',
        overlap='may serialize: selected MVIN-before-compute disjoint edge waits for completion')
    return result, None


def lhs_panel_pass_family():
    """Bounded actual-product alternatives for explicit offline exploration."""
    from merlin.transforms.pass_exploration import PassFamily
    from merlin.transforms.pipeline_policy import RegisteredPass
    from merlin.transforms.pipeline_state import PassSpec
    def bind(options):
        def action(session):
            product, refusal = lhs_panel_schedule(*session.module, **options)
            if product is not None:
                product.revalidate()
            return {'checked_lhs_panels': (product, refusal)}
        return RegisteredPass(PassSpec('lhs_panels', 0, produces=('checked_lhs_panels',),
            mutates_ir=False, ir_level='lhs-full-k-source-v1'), action)
    return PassFamily('lhs_panels', bind, (('m_tiles', (4, 8)), ('n_tiles', (1, 2, 3, 4))))
