"""Explicit bounded full-K RHS burst trials; no automatic timing or selector."""
from functools import lru_cache
from merlin.compile.scheduling.compact_dense import DenseLoopStorage, DenseMemoryView
from merlin.compile.scheduling.dense_rhs_bursts import schedule_dense_rhs_bursts
from merlin.transforms.dense_planning import DenseProblem, DenseResources, dense_blocks
from . import isa
from .dense_rhs_pipeline import rhs_panel_schedule


@lru_cache(maxsize=256)
def rhs_burst_schedule(m, k, n, bias=False, column_scales=False, *, n_tiles=3, k_chunk=2):
    """Return checked product/refusal for an explicit N=2/3, K-chunk=1/2/4.

    Uses the selected immutable CONFIG_LD law, full-K storage and existing full
    drains. Scope and hardware legality remain those of rhs_panel_schedule.
    This callable installs no default and makes no native performance claim.
    """
    if type(n_tiles) is not int or n_tiles not in (2, 3):
        raise ValueError('RHS burst N width must be exactly 2 or 3 tiles')
    if type(k_chunk) is not int or k_chunk not in (1, 2, 4):
        raise ValueError('RHS burst K chunk must be exactly 1, 2 or 4 tiles')
    original, refusal = rhs_panel_schedule(m, k, n, bias, column_scales, n_tiles=n_tiles)
    if original is None:
        return None, refusal
    original.revalidate()
    from .dense import _adapter
    dim = isa.DIM
    problem = DenseProblem(m, k, n, dim, dim, dim, 1, 1, 4, bias=False)
    lhs_rows = problem.tiles[1] * dim
    storage = DenseLoopStorage(DenseMemoryView('lhs', 'spad', dim, 0, lhs_rows),
        DenseMemoryView('rhs', 'spad', dim, lhs_rows, isa.SPAD_ROWS - lhs_rows),
        DenseMemoryView('acc', 'acc', 4 * dim, 0, isa.ACC_ROWS))
    resources = DenseResources(isa.SPAD_ROWS * dim, isa.ACC_ROWS * 4 * dim,
        max_m_blocks=1, max_n_blocks=n_tiles, max_k_blocks=problem.tiles[1],
        partial_output_tiles=True, partial_output_blocks=True)
    choices = [b for b in dense_blocks(problem, resources)
        if (b.m, b.k, b.n) == (1, problem.tiles[1], n_tiles)
        and 2 * b.k * b.n * dim <= storage.rhs.capacity
        and storage.admits(problem, b, resources)]
    if len(choices) != 1:
        return None, 'RHS burst no longer has the original legal full-K storage geometry'
    result = schedule_dense_rhs_bursts(original, problem, choices[0], resources,
        storage, _adapter(), k_chunk=k_chunk)
    result.realization['rhs_panel_storage'] = dict(original.realization['rhs_panel_storage'])
    result.realization['rhs_burst_options'] = dict(n_tiles=n_tiles, k_chunk=k_chunk)
    return result, None


def rhs_burst_pass_family():
    """Finite real-product options for explicit offline exploration only."""
    from merlin.transforms.pass_exploration import PassFamily
    from merlin.transforms.pipeline_policy import RegisteredPass
    from merlin.transforms.pipeline_state import PassSpec
    def bind(options):
        def action(session):
            result, refusal = rhs_burst_schedule(*session.module, **options)
            if result is not None:
                result.revalidate()
            return {'checked_rhs_bursts': (result, refusal)}
        return RegisteredPass(PassSpec('rhs_bursts', 0, produces=('checked_rhs_bursts',),
            mutates_ir=False, ir_level='rhs-full-k-source-v1'), action)
    return PassFamily('rhs_bursts', bind, (('n_tiles', (2, 3)), ('k_chunk', (1, 2, 4))))
