"""Optional shared segmented boundary-RHS residency; no target law added."""
from pathlib import Path
import os

CHOICES=('off','boundary')


def resolve_rhs_residency(selection=None):
    from merlin.compile.tuning_knobs import knob
    configured=knob('DENSE_SEGMENT_RHS_RESIDENCY',default='off',values=CHOICES,
        root=Path(__file__).resolve().parents[1])
    selection=selection if selection is not None else os.environ.get('DENSE_SEGMENT_RHS_RESIDENCY',configured)
    if selection not in CHOICES:
        raise ValueError('DENSE_SEGMENT_RHS_RESIDENCY must be off or boundary')
    return selection


def select_boundary_rhs(original,problem,block,resources,storage,adapter,*,timeout_ms):
    from dataclasses import replace
    from time import monotonic, process_time
    from merlin.compile.scheduling.dense_segmented_pipeline import retain_boundary_rhs
    from .dense_residency_proof_budget import boundary_rhs_proof_budget, BOUNDARY_RHS_PROOF_POLICY
    selected_timeout=boundary_rhs_proof_budget(timeout_ms)
    started,started_cpu=monotonic(),process_time()
    try:
        result=retain_boundary_rhs(original,problem,block,resources,storage,adapter,
            static_budget=512,timeout_ms=selected_timeout)
    except (ValueError,TimeoutError) as error:
        # A refused optional transformation always preserves the admitted
        # incumbent, including its exact source and command identity.
        result=replace(original,realization={**original.realization,
            'boundary_rhs':dict(option='boundary',applied=False,refusal=str(error),
                                actual_kernel_sha256=original.kernel.digest())})
    result.realization['boundary_rhs']['proof_resources']=dict(
        policy=BOUNDARY_RHS_PROOF_POLICY,incumbent_timeout_ms=timeout_ms,
        selected_timeout_ms=selected_timeout,wall_seconds=monotonic()-started,
        process_cpu_seconds=process_time()-started_cpu,
        checks='unchanged complete source, continuation, storage and epoch checks')
    result.revalidate()
    return result
