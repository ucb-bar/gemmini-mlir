"""Optional shared frontier issue transform; all source/epoch proofs follow."""
import os
from pathlib import Path
from merlin.compile.scheduling.dense_segmented_pipeline import interleave_dense_segmented_frontier
ORDERS=('current','prefetch_rhs_between_columns','prefetch_rhs_between_k_tiles')

def resolve_issue_order(value=None):
    if value is None:
        from merlin.compile.tuning_knobs import knob
        from . import isa
        default=knob('DENSE_EPOCH_ISSUE_ORDER',default='current',values=ORDERS,root=Path(isa.__file__).resolve().parents[1])
        value=os.environ.get('DENSE_EPOCH_ISSUE_ORDER',default)
    if value not in ORDERS:raise ValueError('DENSE_EPOCH_ISSUE_ORDER requires a registered frontier order')
    return value

def reorder_resident_frontier(kernel,adapter,issue_order='current'):
    from .dense_lhs_retention import _resident_startup
    issue_order=resolve_issue_order(issue_order)
    startup=_resident_startup(kernel,adapter)
    if issue_order=='current':return kernel,dict(issue_order=issue_order,source_kernel_sha256=kernel.digest(),actual_kernel_sha256=kernel.digest(),pass_order=[])
    actual,report=interleave_dense_segmented_frontier(kernel,adapter,issue_order=issue_order)
    if _resident_startup(actual,adapter)!=startup:raise ValueError('frontier changed immutable resident LHS startup')
    return actual,{**report,'pass_order':['shared_segmented_source','interleave_dense_segmented_frontier','exhaustive_finite_source','readout_batching','panel_epochs','consumption_verification']}

def issue_order_family(compile_checked):
    """Bounded real checked-lowering family; no static performance ranking."""
    from merlin.transforms.pass_exploration import PassFamily
    from merlin.transforms.pipeline_policy import RegisteredPass
    from merlin.transforms.pipeline_state import PassSpec
    def bind(options):
        def action(session):
            result=compile_checked(session.module,issue_order=options['issue_order'])
            result.revalidate()
            session.module=result.kernel
            return {'program':result}
        return RegisteredPass(PassSpec('resident_frontier',1,produces=('program',),ir_level='dense-source'),action)
    return PassFamily('resident_frontier',bind,(('issue_order',ORDERS),))
