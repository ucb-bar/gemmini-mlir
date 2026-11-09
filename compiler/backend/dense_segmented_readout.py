"""Causal final-K stores after checked grouped-frontier and wide-load passes.

The actual selected kernel is transformed and emitted. Wide loads are decoded
with their original checked codecs; expansion must commute exactly with the
ready transformation. Existing segmented source/slot proofs and the stock
ready consumer then check the complete expanded effects. No wrapper is cast to
a different execution owner and no operand staging drain is removed.
"""
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

from merlin.compiler.selection.knobs import knob
from merlin.compiler.schedule.compact_dense import _digest
from merlin.compiler.schedule.dense_panel_pipeline import _verify_dense_epochs
from merlin.compiler.schedule.dense_ready_readout import (
    DenseReadyReadoutCheckingProjection, _ready_execution_bound,
    propose_dense_ready_readout,
)
from merlin.compiler.schedule.dense_segmented_pipeline import (
    DenseSegmentedPipelineResult, verify_dense_segmented_pipeline,
)
from merlin.compiler.schedule.panel_tile_loads import (
    PanelTileLoadResult, expand_panel_tile_loads,
)


def segmented_readout_selection():
    return knob('DENSE_SEGMENTED_READOUT', default='separate',
        values=('separate', 'ready'), root=Path(__file__).resolve().parents[1])


def _owners(result):
    """Recognize only the existing selected refinement chain, retaining codecs."""
    from .dense_rhs_transfers import DenseRhsTransferResult
    from .dense_segmented_frontier import DenseGroupedFrontierResult
    result.revalidate()
    bands = None
    owner = result
    if type(owner) is DenseRhsTransferResult:
        bands = owner.checked
        if type(bands) is not PanelTileLoadResult:
            raise ValueError('segmented ready needs the original checked wide-band owner')
        owner = bands.original
    if type(owner) is DenseGroupedFrontierResult:
        owner = owner.original
    if type(owner) is not DenseSegmentedPipelineResult:
        raise ValueError('segmented ready needs the original typed segmented source owner')
    return owner, bands


def _expand(kernel, bands):
    if bands is None:
        return kernel, None
    return expand_panel_tile_loads(kernel, project=bands.project,
        encode_band=bands.encode_band, max_static_nodes=bands.max_static_nodes)


def _proposal(original, problem, adapter):
    base, bands = _owners(original)
    expanded, before = _expand(original.kernel, bands)
    actual, changes = propose_dense_ready_readout(original.kernel, problem, adapter)
    checked, after = _expand(actual, bands)
    expected, logical_changes = propose_dense_ready_readout(expanded, problem, adapter)
    if checked != expected or checked.digest() != expected.digest():
        raise ValueError('segmented causal frontier and exact wide-band expansion do not commute')
    return base, actual, checked, expanded, changes, dict(
        original_actual_sha256=original.kernel.digest(),
        original_expansion_sha256=expanded.digest(),
        actual_sha256=actual.digest(), actual_expansion_sha256=checked.digest(),
        before=before, after=after, logical_changes=logical_changes,
        correspondence='identical final-K frontier before/after all actual load bands are decoded')


@dataclass(frozen=True)
class DenseSegmentedReadyResult:
    original: object
    kernel: object
    realization: dict
    problem: object
    storage: object
    adapter: object
    admission: object
    checked_identity: str
    checked_report_sha256: str
    owner_sha256: str

    def revalidate(self):
        _, actual, _, _, _, projection = _proposal(self.original, self.problem, self.adapter)
        if (actual != self.kernel
                or self.kernel.digest() != self.realization['kernel_sha256']
                or _digest((self.original.kernel.digest(), self.kernel.digest(),
                    asdict(self.problem), asdict(self.storage), repr(self.adapter),
                    asdict(self.admission), projection)) != self.checked_identity
                or _digest(self.realization['segmented_ready_proof']) != self.checked_report_sha256
                or sha256(Path(__file__).read_bytes()).hexdigest() != self.owner_sha256):
            raise ValueError('segmented ready source, actual effects, proof or adapter changed')

    def materialize_scf(self):
        from merlin.compiler.schedule.scf_commands import materialize_scf_commands
        self.revalidate()
        result = materialize_scf_commands(self.kernel)
        self.revalidate()
        return result

    def emit_llvm(self, builder, lower_call):
        from merlin.compiler.schedule.llvm_program import FnBuilderIndexAdapter, emit_index_program
        self.revalidate()
        bridge = FnBuilderIndexAdapter(builder)
        result = emit_index_program(self.kernel, bridge,
            lambda name, args: lower_call(name, {k: bridge.value(v) for k, v in args.items()}))
        self.revalidate()
        return result


def maybe_segmented_ready_readout(original, problem, block, resources, storage,
                                 adapter, *, eligible, selection=None):
    """Run AFTER grouped future issue and selected block64 load admission.

    Eligibility must come from the ordinary source readout check. The existing
    terminal encoder consumes ready_readout_selected and establishes one
    unchanged CONFIG_ST after its initial drain through its final call drain.
    """
    selection = segmented_readout_selection() if selection is None else selection
    if selection not in ('separate', 'ready'):
        raise ValueError('invalid segmented ready readout selection')
    if selection != 'ready' or not eligible:
        original.realization['segmented_ready_selection'] = dict(selection=selection,
            applied=False, uniform_source_eligible=bool(eligible))
        return original
    from .dense_ready import causal_readout_admission
    base, actual, checked, expanded, changes, projection = _proposal(original, problem, adapter)
    bound = _ready_execution_bound(base, problem, block, resources, storage)
    admission, binding = causal_readout_admission(block, adapter)
    admission.validate_for(adapter, bound)
    # Recheck the actual transformed input, including every grouped load band,
    # under its SAME original segmented execution/source/slot ownership law.
    before = verify_dense_segmented_pipeline(expanded, problem, block, resources,
        storage, adapter, base.admission, static_budget=128, timeout_ms=30000,
        output_readout=base.output_readout, bias_source=base.bias_source)
    proof = _verify_dense_epochs(checked, problem, resources, storage, adapter, bound,
        static_budget=128, timeout_ms=30000, output_readout=base.output_readout,
        bias_source=base.bias_source, ordered_accumulator_commits=True,
        source_phase_fission=True, readout_compute_admission=admission,
        readout_source_projection=DenseReadyReadoutCheckingProjection(expanded, problem))
    report = dict(original.realization, kernel_sha256=actual.digest(),
        ready_readout_selected=True, ready_readout_target_binding=binding,
        segmented_ready_selection=dict(selection=selection, applied=True,
            uniform_source_eligible=True),
        segmented_ready_transformations=changes, segmented_ready_projection=projection,
        segmented_ready_input_proof=before, segmented_ready_proof=proof,
        segmented_ready_admission=asdict(admission),
        segmented_ready_completion='original operand staging/slot-reuse and frontier-end drains retained; stores follow each complete final-K run')
    result = DenseSegmentedReadyResult(original, actual, report, problem, storage,
        adapter, admission, _digest((original.kernel.digest(), actual.digest(),
            asdict(problem), asdict(storage), repr(adapter), asdict(admission), projection)),
        _digest(proof), sha256(Path(__file__).read_bytes()).hexdigest())
    result.revalidate()
    return result
