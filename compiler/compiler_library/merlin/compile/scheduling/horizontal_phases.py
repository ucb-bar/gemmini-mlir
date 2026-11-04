"""Sequential complete-K horizontal panels over exact original column intervals.

A local column subview changes no external tensor pitch or numerical policy.
Consumers retain the original source/attrs and project local columns before
terminal lowering. Every phase finishes all original stores before private rows
can be reused. This module supplies no asynchronous or target execution law.
"""
from dataclasses import dataclass, field, replace
import hashlib
from math import prod

from .window_transfer import CountedConvolutionPlan
from .horizontal_convolution import (
    HorizontalConvolutionPanels, plan_horizontal_convolution_panels,
    emit_horizontal_convolution_panels,
)


def _intervals(source, intervals):
    if type(intervals) is not tuple or not intervals:
        raise ValueError('horizontal phases require explicit complete source column intervals')
    end = 0
    for item in intervals:
        if (type(item) is not tuple or len(item) != 2
                or any(type(x) is not int for x in item)
                or item[0] != end or not 0 < item[1] <= source.column_tile):
            raise ValueError('horizontal phases must preserve each original numeric interval intact')
        end += item[1]
    if end != source.output_channels:
        raise ValueError('horizontal phases must cover every original output column exactly once')


def _row_choice(bound, choices):
    if choices is None:
        return bound
    if (type(choices) is not tuple or not choices
            or any(type(x) is not int or not 0 < x < 1 << 63 for x in choices)
            or tuple(sorted(set(choices))) != choices):
        raise ValueError('horizontal phases need ordered distinct positive row choices')
    return max((x for x in choices if x <= bound), default=0)


@dataclass(frozen=True)
class HorizontalConvolutionPhase:
    original_column_start: int
    original_column_intervals: tuple[tuple[int, int], ...]
    inner: HorizontalConvolutionPanels

    @property
    def columns(self):
        return self.inner.source.output_channels

    @property
    def original_column_end(self):
        return self.original_column_start + self.columns

    def __post_init__(self):
        if type(self.inner) is not HorizontalConvolutionPanels:
            raise ValueError('horizontal phase needs the original checked inner panel plan')
        self.inner.__post_init__()
        if type(self.original_column_start) is not int or self.original_column_start < 0:
            raise ValueError('horizontal phase needs a nonnegative original column origin')
        expected = tuple((self.original_column_start + column, width)
                         for column, width in self.inner.column_intervals)
        if self.original_column_intervals != expected:
            raise ValueError('horizontal phase local view differs from its complete original intervals')

    def project(self, kind, fields):
        """Translate only original output columns; preserve all private operands."""
        if kind not in ('load_weight', 'initialize_bias', 'compute', 'store'):
            return fields
        column, width = fields.get('output_column'), fields.get('columns')
        if (type(column) is not int or type(width) is not int
                or (column, width) not in self.inner.column_intervals):
            raise ValueError('horizontal phase effect does not select an original atomic interval')
        return dict(fields, output_column=column + self.original_column_start)


@dataclass(frozen=True)
class HorizontalConvolutionPhases:
    original_source: CountedConvolutionPlan
    column_intervals: tuple[tuple[int, int], ...]
    max_row_blocks: int
    maximum_transfer_columns: int
    private_rows_owned: bool
    phases: tuple[HorizontalConvolutionPhase, ...]
    row_block_choices: tuple[int, ...] | None = None
    _identity: str = field(init=False, repr=False)

    def __post_init__(self):
        self._validate()
        object.__setattr__(self, '_identity', hashlib.sha256(repr(self).encode()).hexdigest())

    def _validate(self):
        source = self.original_source
        if type(source) is not CountedConvolutionPlan:
            raise ValueError('horizontal phases require the complete original source')
        source.__post_init__()
        _intervals(source, self.column_intervals)
        if (self.private_rows_owned is not True
                or any(type(x) is not int or not 0 < x < 1 << 63
                       for x in (self.max_row_blocks, self.maximum_transfer_columns))
                or type(self.phases) is not tuple or not self.phases):
            raise ValueError('horizontal phases require bounded resources and private ownership')
        _row_choice(self.max_row_blocks, self.row_block_choices)
        seen = []
        end = 0
        for phase in self.phases:
            if type(phase) is not HorizontalConvolutionPhase:
                raise ValueError('horizontal phases require checked complete local phase views')
            phase.__post_init__()
            if (phase.original_column_start != end
                    or phase.inner.source != replace(source, output_channels=phase.columns)
                    or phase.inner.maximum_transfer_columns != self.maximum_transfer_columns
                    or phase.inner.row_blocks > self.max_row_blocks
                    or self.row_block_choices is not None and phase.inner.row_blocks not in self.row_block_choices):
                raise ValueError('horizontal phase changed original source, ordering or resource scope')
            seen.extend(phase.original_column_intervals)
            end = phase.original_column_end
        if tuple(seen) != self.column_intervals or end != source.output_channels:
            raise ValueError('horizontal phases dropped, split, duplicated or reordered original columns')

    def revalidate(self):
        self._validate()
        if self._identity != hashlib.sha256(repr(self).encode()).hexdigest():
            raise ValueError('horizontal phases or original source changed after admission')
        return self

    @property
    def identity(self):
        self.revalidate()
        return self._identity

    def summary(self):
        self.revalidate()
        return dict(schema='merlin.horizontal_convolution_phases.v1',
            identity=self.identity, original_source_identity=self.original_source.identity,
            original_output_channels=self.original_source.output_channels,
            original_column_intervals=self.column_intervals,
            original_coverage=self.original_source.coverage,
            original_external_pitch_columns=self.original_source.output_channels,
            phases=tuple(dict(original_column_start=p.original_column_start,
                              original_column_end=p.original_column_end,
                              original_column_intervals=p.original_column_intervals,
                              inner=p.inner.summary()) for p in self.phases),
            completion='all phase stores and real phase-last-use drain before private reuse',
            numeric_intervals_split=False, target_execution=False, timing=None)


def plan_horizontal_convolution_phases(source, column_intervals, *, max_row_blocks,
                                      maximum_transfer_columns, private_rows_owned=False,
                                      row_block_choices=None):
    """Keep full-fit panels; otherwise pack atomic N intervals by row capacity.

    For overcapacity sources, first bound M by the rows needed for one complete
    K/N panel, then take the largest consecutive interval group fitting that M.
    Each view is admitted by the incumbent horizontal planner. No interval is
    split, no numerical state is inferred, and unavailable geometry returns None.
    Optional row_block_choices restricts M to caller-declared supported unrolls;
    it supplies no physical array-retention or asynchronous execution law.
    """
    if type(source) is not CountedConvolutionPlan or private_rows_owned is not True:
        raise ValueError('horizontal phases require original source and explicit private ownership')
    source.__post_init__()
    _intervals(source, column_intervals)
    if any(type(x) is not int or not 0 < x < 1 << 63
           for x in (max_row_blocks, maximum_transfer_columns)):
        raise ValueError('horizontal phases require positive bounded transfer/unroll limits')
    requested = _row_choice(max_row_blocks, row_block_choices)
    if requested < 1:
        return None
    full = plan_horizontal_convolution_panels(source, max_row_blocks=requested,
        maximum_transfer_columns=maximum_transfer_columns, private_rows_owned=True,
        column_intervals=column_intervals)
    if full is not None:
        chosen = _row_choice(full.row_blocks, row_block_choices)
        if chosen < 1:
            return None
        if chosen != full.row_blocks:
            full = plan_horizontal_convolution_panels(source, max_row_blocks=chosen,
                maximum_transfer_columns=maximum_transfer_columns, private_rows_owned=True,
                column_intervals=column_intervals)
        phases = (HorizontalConvolutionPhase(0, column_intervals, full),)
    else:
        w, d = source.window, source.primitive
        if (w.dilation[1] != 1 or source.row_tile != d
                or w.kernel[1] * w.input_shape[-1] > maximum_transfer_columns):
            return None
        reduction_tiles = w.kernel[0] * ((w.kernel[1] * w.input_shape[-1] + d - 1) // d)
        rhs_panel_rows = reduction_tiles * d
        pixels = w.input_shape[0] * prod(w.output_spatial)
        m = _row_choice(min(requested, (pixels+d-1)//d,
                source.operand_rows//rhs_panel_rows-1, source.accumulator_rows//d), row_block_choices)
        if m < 1:
            return None
        intervals_per_phase = min(source.operand_rows//rhs_panel_rows-m,
                                  source.accumulator_rows//(m*d))
        if intervals_per_phase < 1:
            return None
        result = []
        for first in range(0, len(column_intervals), intervals_per_phase):
            originals = column_intervals[first:first+intervals_per_phase]
            start = originals[0][0]
            width = sum(n for _, n in originals)
            local = tuple((column-start, n) for column, n in originals)
            view = replace(source, output_channels=width)
            inner = plan_horizontal_convolution_panels(view, max_row_blocks=m,
                maximum_transfer_columns=maximum_transfer_columns, private_rows_owned=True,
                column_intervals=local)
            if inner is None:
                return None
            result.append(HorizontalConvolutionPhase(start, originals, inner))
        phases = tuple(result)
    return HorizontalConvolutionPhases(source, column_intervals, max_row_blocks,
                                      maximum_transfer_columns, True, phases, row_block_choices)


def emit_horizontal_convolution_phases(plan, builder, lower, complete):
    """Emit complete local views, projecting original columns before the terminal.

    The caller retains original external weight/output/bias strides and numeric
    attrs. Input preparation must already be ready and immutable across phases.
    Every complete callback implements a real completion point; none is elided.
    """
    if (type(plan) is not HorizontalConvolutionPhases or not callable(lower)
            or not callable(complete) or any(not callable(getattr(builder, n, None))
                                            for n in ('op', 'cmp', 'loop', 'branch'))):
        raise ValueError('horizontal phases need checked plan, control and real effect/completion callbacks')
    plan.revalidate()
    emissions = []
    for index, phase in enumerate(plan.phases):
        def project(kind, fields, phase=phase):
            return lower(kind, phase.project(kind, fields))
        emissions.append(emit_horizontal_convolution_panels(phase.inner, builder, project, complete))
        complete('horizontal_phase_last_use', dict(phase_index=index,
            output_column=phase.original_column_start, columns=phase.columns,
            local_row=0, local_rows=phase.inner.operand_rows))
        plan.revalidate()
    return dict(plan.summary(), phase_emissions=tuple(emissions))
