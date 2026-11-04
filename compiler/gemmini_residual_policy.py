"""Frozen source/resource region for checked loaded-sum residual epochs.

The fixed two-complete-group guard predates the fresh held-out measurement.
Compilation performs shared legality checks and resource predicates only; no
case names, runtime tuning or cycle estimates select a schedule. Empirical
support covers two original dense-product/residual geometries, with and without
final ReLU. It does not establish performance outside that panel or RTL timing.
"""
from __future__ import annotations

from merlin.transforms.pipeline_policy import PolicyFacts, ResourcePredicate
from merlin.transforms.pipeline_state import AnalysisStamp, PipelineContext


SELECTED_IDENTITIES = {
    'facts_sha256': 'b8ff28edc096392cec356f9fececdf17ba70cd3e4b02e6c5931aedc43bb117af',
    'ordering_sha256': 'd6908ce444d5a0d500660dc5a5bc8560f7d2c26a0c8406edbc4c7b76dd513ea6',
    'target_contract_sha256': '4876bf36060bb624526511a2ec45315f344008120fa38b8faa666e61b3c5ab4e',
}

# Evidence identities are receipts, never dispatch keys. The independent fresh
# held-out source passed owner qualification; training passed exact numerics but
# retained an owner host-compute veto in both arms. Only provisional whole-call
# model ranking is established; the compiler preserves the serial alternative.
PAIRED_EVIDENCE = (
    dict(role='training', source_sha256='daf66e4d4382051562e6ffcf563e24cb86c2a27fc77e93fc30dc26bb6680c55a',
         serial_request='4_1793141407193338189_dca4026bc3463011052d5064ae488dbb',
         epochs_request='6_1793141418060344210_2a5dee910b38c39f77d308a080cda1ed',
         serial_cycles=730411, epochs_cycles=709405, numeric='exact pass',
         owner_qualification='host_compute veto both arms'),
    dict(role='fresh-held-out', source_sha256='f4c777731aea3c55634627e6d318274ec1bc524fa0ddde8a8b06de02b37f7e92',
         serial_request='3_1793142282250416126_dabe3a05fd89af3250c68933ad1f55ed',
         epochs_request='4_1793142287660765662_f04304d479708d60a530ab1dff7bc7a5',
         serial_cycles=308459, epochs_cycles=192989, numeric='exact pass',
         owner_qualification='pass both arms'),
)


def select_residual_schedule(plan, admission):
    """Select an already admitted plan through current shared resource facts.

    The auto region retains the empirically checked unit-scaled i8 contract.
    Other numerically admitted contracts remain available through explicit
    epochs, and all existing scalar/alias/completion checks remain mandatory.
    """
    from gemmini_residual_schedule import digest
    plan.revalidate(admission['source'], plan.layout, plan.contract,
                    admission['project'], scalar_proof=admission['proof'])
    evidence = admission['selected_evidence']
    proof = admission['proof']
    numeric = proof['source_numeric']
    resources = dict(
        complete_capacity_groups=plan.grouped_extent // plan.live_tiles,
        selected_hardware=int(all(evidence.get(k) == v for k, v in SELECTED_IDENTITIES.items())),
        unit_scaled_i8=int(proof['output_dtype'] == 'i8'
            and numeric.get('lhs_scale_f32') == 1.0
            and numeric.get('rhs_scale_f32') == 1.0
            and numeric.get('bound_lsb') == 0),
    )
    facts = PolicyFacts(AnalysisStamp(plan.source_kernel_sha256,
        PipelineContext(resources=digest(evidence), numerical_policy=plan.contract.scalar_proof_sha256),
        'checked_loaded_sum'), resources=tuple(sorted(resources.items())))
    guards = (ResourcePredicate('complete_capacity_groups', 2),
              ResourcePredicate('selected_hardware', 1), ResourcePredicate('unit_scaled_i8', 1))
    selected = all(g.match(facts) is True for g in guards)
    return ('epochs' if selected else 'serial'), dict(
        policy='residual-complete-capacity-groups-v1',
        selected='epochs' if selected else 'serial', source_identity=plan.source_kernel_sha256,
        resources=resources, resource_identity=digest(evidence),
        predicates=[dict(resource=g.resource, at_least=g.at_least) for g in guards],
        evidence=PAIRED_EVIDENCE, empirical_qualification='provisional matched whole-call Spike model only',
        scope='frozen two-group guard; unit-scaled i8; no claim of optimum beyond two geometries')
