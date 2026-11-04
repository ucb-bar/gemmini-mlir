"""Optional load epochs over the original complete-private horizontal effects.

Only padded, unsplit RHS and input writes are batched. Bias retains its original
completion, as do all readiness, group-last-use and invocation exit boundaries.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path

from merlin.compile.scheduling.horizontal_convolution import HorizontalConvolutionPanels, emit_horizontal_convolution_panels
from merlin.compile.scheduling.window_transfer import ConvolutionLoadIssueLaw, ConvolutionComputeEpochLaw
from merlin.compile.scheduling.affine_gather_runs import ConvolutionGatherRuns


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _library_identity():
    from merlin.compile.scheduling import horizontal_convolution, affine_gather_runs, window_transfer
    return tuple(_sha(Path(m.__file__).read_bytes()) for m in
                 (horizontal_convolution, affine_gather_runs, window_transfer))


@dataclass(frozen=True)
class HorizontalLoadAdmission:
    plan: HorizontalConvolutionPanels
    law: ConvolutionLoadIssueLaw
    selection_json: str
    facts_sha256: str
    ordering_sha256: str
    library_sha256: tuple
    rhs_commands: int
    input_commands_per_group: int
    checked_sha256: str

    def _binding(self):
        return _sha(repr((self.plan, self.law, self.selection_json, self.facts_sha256,
            self.ordering_sha256, self.library_sha256, self.rhs_commands,
            self.input_commands_per_group)).encode())

    def revalidate(self):
        if type(self.plan) is not HorizontalConvolutionPanels or type(self.law) is not ConvolutionLoadIssueLaw:
            raise ValueError('horizontal loads require exact shared source and load issue law')
        self.plan.__post_init__()
        self.law.__post_init__()
        if (self.checked_sha256 != self._binding() or self.library_sha256 != _library_identity()
                or self.law.source_identity != _sha(repr(self.plan).encode())
                or self.law.hardware_identity != self.ordering_sha256
                or not all((self.law.local_range_dependencies, self.law.ordered_load_configuration,
                    self.law.paired_descriptor_compute, self.law.finite_queue_backpressure,
                    self.law.complete_device_drain, self.law.immutable_mapped_disjoint_buffers))
                or self.law.load_issue_budget != max(self.rhs_commands,
                    self.input_commands_per_group + int(self.plan.source.bias))):
            raise ValueError('horizontal load checked source, law or library changed')

    def receipt(self):
        self.revalidate()
        return dict(selected=True, mechanism='original horizontal disjoint private load epochs',
            admission_sha256=self.checked_sha256, source_identity=self.law.source_identity,
            library_sha256=self.library_sha256, facts_sha256=self.facts_sha256,
            ordering_sha256=self.ordering_sha256, issue_law=asdict(self.law),
            rhs_issued_bound=self.rhs_commands, input_issued_bound=self.input_commands_per_group,
            first_bias_load_before_its_retained_drain=int(self.plan.source.bias),
            preserved_completions=['bias load', 'horizontal_rhs_ready', 'horizontal_group_ready',
                'compute readiness', 'horizontal_group_last_use', 'invocation entry and exit'],
            omitted_completions='only post-load_weight and padded unsplit post-load_horizontal_input fences',
            disjointness='RHS slot rows; LHS ky/chunk/mi row partition and exact affine gather fragments',
            target_execution_verified=False, timing=None)


def admit_horizontal_loads(plan, selection, *, facts_bytes, ordering_selection,
                           input_transport='padded', split_input_blocks=False):
    """Refuse missing selected premises before any builder mutation."""
    if type(plan) is not HorizontalConvolutionPanels or type(selection) is not dict:
        raise ValueError('horizontal load admission requires original typed plan and selection')
    if (input_transport != 'padded' or split_input_blocks is not False
            or selection.get('split_input_blocks', {}).get('selected')):
        return None
    try:
        from gemmini_conv import _epoch_schedule
        from merlin.compile.scheduling.convolution_exploration import ConvolutionExplorationSource
        plan.__post_init__()
        digest = _sha(json.dumps(ordering_selection['ordering'], sort_keys=True).encode())
        facts_sha = _sha(facts_bytes)
        program = _sha(repr(plan).encode())
        if (ordering_selection['base_facts']['sha256'] != facts_sha
                or ordering_selection['ordering_sha256'] != digest
                or selection.get('selected') is not True or selection['program_identity'] != program
                or selection['transfer_law']['facts_sha256'] != facts_sha
                or selection['transfer_law']['ordering_sha256'] != digest
                or selection['compute_epoch'].get('selected') is not True
                or selection['compute_epoch']['epoch_law']['source_identity'] != program
                or selection['compute_epoch']['epoch_law']['hardware_identity'] != digest
                or selection['private_storage']['operand_rows'] != [0, plan.operand_rows]
                or selection['private_storage']['accumulator_rows'] != [0, plan.accumulator_rows]):
            return None
        source = ConvolutionExplorationSource(plan.source, plan.column_intervals, selection['source_identity'])
        epoch = ConvolutionComputeEpochLaw(**selection['compute_epoch']['epoch_law'])
        epoch.require(selection['compute_epoch']['compute_pairs'])
        if plan.row_blocks > 1:
            array = selection['array_retention']
            if (array.get('selected') is not True or array['horizontal_program_identity'] != program
                    or array['array_law']['hardware_identity'] != digest):
                return None
        schedule = _epoch_schedule(source, selection['compute_epoch']['compute_pairs'])
        if schedule is None or schedule.issue_law is None:
            return None
        schedule.verify_current()
        original = schedule.issue_law
        if original.hardware_identity != digest:
            return None
        # Actual selected load class/routing and finite physical admission.
        record = ordering_selection['ordering']
        if record.get('status') != 'derived':
            return None
        config = next(x for x in record['funct_queue'] if x.get('funct') == 0)
        if (config.get('selector') != dict(operand='rs1', offset=0, width=2)
                or config.get('by_code', {}).get('1') != 'load'):
            return None
        for name, funct in [('MVIN.spad_addr',2), ('MVIN2.spad_addr',1)]:
            access = record['accesses'][name]
            if (access.get('funct') != funct or access.get('queue') != 'load'
                    or access.get('direction') != 'write' or record['extents'][name]['status'] != 'derived'):
                return None
        entries = json.loads(facts_bytes)['facts']['timing_structure']['command_classes']['classes']['load']['entries']
        if entries.get('status') != 'derived' or type(entries.get('value')) is not int or entries['value'] < 1:
            return None
        p, w, d = plan.source, plan.padded_window, plan.source.primitive
        # Shared gather proves fragments partition each mi row interval. Each
        # horizontal block maps chunk to a distinct d*row_blocks stride, so
        # ky/chunk/mi/row never aliases another pending destination cell.
        gather = ConvolutionGatherRuns(w.input_shape, (w.input_shape[0],*w.output_spatial), w.stride, w.dilation, w.kernel, d)
        rhs = plan.reduction_tiles * len(plan.column_intervals)
        inputs = w.kernel[0] * plan.row_blocks * gather.max_fragments(d)
        # The first bias transfer may share the pending input epoch until its
        # retained post-bias fence. Include it without suppressing that fence.
        law = replace(original, source_identity=program, load_issue_budget=max(rhs,inputs+int(p.bias)),
            qualification_identity=original.qualification_identity+';original-horizontal-disjoint-private-padded-load-epochs-v1')
        result = HorizontalLoadAdmission(plan,law,json.dumps(selection,sort_keys=True),facts_sha,
            digest,_library_identity(),rhs,inputs,'')
        result = replace(result,checked_sha256=result._binding())
        result.revalidate()
        return result
    except (KeyError,TypeError,ValueError,StopIteration):
        return None


def emit_horizontal_load_epochs(admission, builder, lower, complete, *, readout_admission=None):
    """Consume the SAME original effects, optionally through readout epochs.

    Parent lower accepts complete_after_effect; False suppresses only that
    original effect's post-load fence. Input transport must be padded/unsplit.
    All real completion callbacks are passed through unchanged.
    """
    if type(admission) is not HorizontalLoadAdmission or not callable(lower) or not callable(complete):
        raise ValueError('horizontal loads require checked admission and original callbacks')
    admission.revalidate()
    stage = 'rhs'
    rhs = 0
    inputs = 0
    def issue(kind, fields, *, complete_after_effect=True):
        nonlocal rhs, inputs
        if kind == 'load_weight':
            if stage != 'rhs' or rhs >= admission.rhs_commands:
                raise ValueError('horizontal RHS loads escaped their original private epoch')
            p, d = admission.plan, admission.plan.source.primitive
            slot, ni = divmod(rhs, len(p.column_intervals))
            ky, chunk = divmod(slot, p.horizontal_chunks)
            column, columns = p.column_intervals[ni]
            expected = dict(reduction_row=ky*p.horizontal_columns+chunk*d,
                output_column=column, local_row=p.rhs_base+(ni*p.reduction_tiles+slot)*d,
                rows=min(d,p.horizontal_columns-chunk*d), columns=columns)
            if fields != expected:
                raise ValueError('horizontal RHS changed original source coefficients or disjoint private rows')
            rhs += 1
        elif kind == 'load_horizontal_input':
            w = admission.plan.padded_window
            if (stage != 'inputs' or fields['columns'] != admission.plan.horizontal_columns
                    or fields['block_stride_rows'] != admission.plan.input_block_stride
                    or fields['row_stride_elements'] != w.stride[1]*w.input_shape[3]):
                raise ValueError('horizontal input load changed its exact private row partition')
            inputs += 1
            if inputs > admission.input_commands_per_group:
                raise ValueError('horizontal input loads exceed checked gather bound')
        elif kind == 'initialize_bias':
            if stage != 'inputs':
                raise ValueError('horizontal bias escaped input readiness epoch')
        elif stage != 'compute':
            raise ValueError('horizontal compute/readout preceded real load readiness')
        lower(kind,fields,complete_after_effect=(complete_after_effect and kind not in ('load_weight','load_horizontal_input')))
    def finish(kind, fields):
        nonlocal stage, inputs
        if kind == 'horizontal_rhs_ready':
            if stage != 'rhs' or rhs != admission.rhs_commands:
                raise ValueError('horizontal RHS epoch omitted original loads')
            complete(kind,fields)
            stage = 'inputs'
        elif kind == 'horizontal_group_ready':
            if stage != 'inputs' or not inputs:
                raise ValueError('horizontal group lacks original input loads')
            complete(kind,fields)
            stage = 'compute'
        elif kind == 'horizontal_group_last_use':
            if stage != 'compute':
                raise ValueError('horizontal group reuse preceded readiness')
            complete(kind,fields)
            stage, inputs = 'inputs', 0
        else:
            raise ValueError('horizontal load adapter received unknown lifetime boundary')
    if readout_admission is None:
        receipt = emit_horizontal_convolution_panels(admission.plan,builder,issue,finish)
    else:
        from gemmini_horizontal_readout import emit_horizontal_readout_epochs
        if readout_admission.plan != admission.plan:
            raise ValueError('horizontal load/readout adapters select different sources')
        receipt = emit_horizontal_readout_epochs(readout_admission,builder,issue,finish)
    if stage != 'inputs' or inputs:
        raise ValueError('horizontal load epoch retained private work at exit')
    admission.revalidate()
    return dict(receipt,horizontal_load_epoch=admission.receipt())
