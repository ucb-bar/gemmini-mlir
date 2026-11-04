"""Original-window input preparation followed by checked linear-K raw tiles.

Packing is input transport only. The ordinary preparation emitter owns every
source byte/padding guard; this adapter preserves original reduction order,
ACC halo ownership, bias and the original sole narrowing spatial observation.
"""
from dataclasses import dataclass, asdict, replace
import hashlib
import json
from .window_transfer import CountedConvolutionPlan, Window2D
from .convolution_tile_epochs import emit_convolution_compute_epoch_tile


@dataclass(frozen=True)
class PackedRawConvolution:
    original: CountedConvolutionPlan
    workspace_bytes: int

    def __post_init__(self):
        if type(self.original) is not CountedConvolutionPlan:
            raise ValueError('packed raw tiles require the original counted convolution')
        self.original.__post_init__()
        if (type(self.workspace_bytes) is not int or self.workspace_bytes < self.required_bytes
                or self.workspace_bytes >= 1 << 63):
            raise ValueError('packed raw panel exceeds caller-owned byte capacity')
        self.packed.__post_init__()

    @property
    def reduction(self):
        p = self.original
        return p.window.kernel[0] * p.window.kernel[1] * p.window.input_shape[-1]

    @property
    def required_bytes(self):
        return self.original.row_tile * self.reduction

    @property
    def packed(self):
        p = self.original
        return replace(p, window=Window2D((1, 1, p.row_tile, self.reduction), (1, 1)),
                       channel_tile=min(p.primitive, self.reduction))

    @property
    def identity(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    def prepare(self, issue, *, output_row, rows):
        self.__post_init__()
        if type(rows) is not int or not 0 < rows <= self.original.row_tile:
            raise ValueError('packed raw tile exceeds original M capacity')
        issue('pack_original_input', dict(window=self.original.window,
            output_row=output_row, rows=rows, panel_bytes=rows*self.reduction,
            workspace_bytes=self.workspace_bytes, adapter_identity=self.identity))
        return PreparedRawPanel(self.identity, output_row, rows)

    def emit(self, builder, issue, complete, *, output_row, rows, **kwargs):
        panel = self.prepare(issue, output_row=output_row, rows=rows)
        return self.emit_prepared(builder, issue, complete, panel=panel, **kwargs)

    def emit_prepared(self, builder, issue, complete, *, panel, output_column,
             columns, accumulator_row, law, epoch_law, compute_pairs):
        self.__post_init__()
        if type(panel) is not PreparedRawPanel or panel.adapter_identity != self.identity:
            raise ValueError('prepared raw panel belongs to a different exact source or workspace')
        rows = panel.rows
        if type(rows) is not int or not 0 < rows <= self.original.row_tile:
            raise ValueError('prepared raw tile exceeds original M capacity')
        if epoch_law is None or law.source_identity != self.original.identity:
            raise ValueError('packed raw tiles require source-qualified compute epochs')
        epoch_law.require(compute_pairs)
        if epoch_law.source_identity != self.original.identity:
            raise ValueError('packed raw epochs have different original source')
        before = self.identity
        p = self.packed
        pairs = (self.reduction + p.channel_tile - 1) // p.channel_tile
        packed_law = replace(law, source_identity=p.identity,
            load_issue_budget=p.row_tile+1+int(p.bias))
        packed_epoch = replace(epoch_law, source_identity=p.identity,
            max_compute_pairs=pairs)

        def lower(kind, fields):
            if kind == 'load_input':
                fields = dict(fields, packed_input=True)
            issue(kind, fields)
        emit_convolution_compute_epoch_tile(p, builder, lower, complete,
            output_row=0, output_column=output_column, rows=rows, columns=columns,
            accumulator_row=accumulator_row, law=packed_law,
            epoch_law=packed_epoch, compute_pairs=pairs)
        if self.identity != before:
            raise ValueError('packed raw source or private resources changed during emission')
        return dict(schema='merlin.spatial_packed_raw.v1', identity=before,
            original_source=self.original.identity, packed_source=p.identity,
            panel_shape=[rows,self.reduction], workspace_bytes=self.workspace_bytes,
            original_reduction_order='ky,kx,ci; no arithmetic or readout during packing',
            completion='full original raw tile before workspace reuse; original spatial store drain')


@dataclass(frozen=True)
class PreparedRawPanel:
    """Original immutable source-row token; validity ends on the next packing.

    Only the structured emitter may retain/reuse it. This token does not grant
    asynchronous storage ownership or permit reordering a later preparation.
    """
    adapter_identity: str
    original_output_row: object
    rows: int
