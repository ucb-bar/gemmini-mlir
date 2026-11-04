"""Bounded device-only column readout from an exact i32 convolution tile.

The accumulator address selects rows, not a column within a row. A full-width
raw MVOUT followed by strided full-i32 MVIN3 subviews moves each contiguous
coefficient region to accumulator column zero. Only the final MVOUT narrows.
Caller-owned scratch is reused after a complete drain; no host tensor arithmetic
or whole-image intermediate is introduced.
"""
from dataclasses import dataclass, field

from merlin.llvmlower.integer_readout import partition_column_scales


def regional_staged_candidate(input_channels, kernel, output_channels, edge,
                              row_tile, hardware_identity=None):
    """Provisional resource guard; scratch and numerical legality stay separate.

    A full-width compute saves repeated column-zero contractions but the staged
    readout adds an exact store/refill plus drains. Require at least one array
    edge of reduction tiles and a full output column tile to amortize that
    fixed readout work. This is the provisional default geometry alternative, with separate
    storage/numeric admission and direct fallback. It is not a physical timing
    qualification. No source identity participates in selection.
    """
    import hashlib
    from merlin.transforms.pipeline_policy import PolicyFacts, ResourcePredicate
    from merlin.transforms.pipeline_state import AnalysisStamp, PipelineContext

    kernel = tuple(kernel)
    values = (input_channels, output_channels, edge, row_tile) + kernel
    if len(kernel) != 2 or any(type(v) is not int or v <= 0 for v in values):
        raise ValueError('regional column readout requires positive static geometry')
    if row_tile > edge:
        raise ValueError('regional column readout row tile exceeds selected array')
    reduction = kernel[0]*kernel[1]*input_channels
    resources = dict(array_edge=edge, reduction_elements=reduction,
        reduction_tiles=(reduction+edge-1)//edge,
        output_channels=output_channels, row_tile=row_tile,
        scratch_bytes=edge*row_tile*4)
    identity = hashlib.sha256(repr(tuple(sorted(resources.items()))).encode()).hexdigest()
    hardware = hardware_identity or 'array-edge-%d' % edge
    facts = PolicyFacts(AnalysisStamp(identity,
        PipelineContext(hardware=hardware, resources=hardware),
        'convolution-column-readout-geometry'),
        shapes=((kernel[0], kernel[1], input_channels, output_channels),),
        resources=tuple(sorted(resources.items())))
    guards = (ResourcePredicate(resource='reduction_tiles', at_least=edge),
              ResourcePredicate(resource='output_channels', at_least=edge))
    selected = all(g.match(facts) is True for g in guards)
    return dict(schema='authored.regional_staged_readout.v1',
        candidate='staged_columns' if selected else 'direct',
        geometry_selected=selected, resources=resources,
        predicates=[dict(resource='reduction_tiles', at_least=edge),
                    dict(resource='output_channels', at_least=edge)],
        qualification='provisional geometry alternative; separate storage/numeric admission',
        default='regional_staged')


@dataclass
class StagedColumnReadout:
    scratch: object
    scratch_bytes: int
    gains: tuple
    output_channels: int
    row_tile: int
    edge: int
    accumulator_row: int
    relu: bool = False
    _partitions: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        if self.scratch is None or type(self.scratch_bytes) is not int:
            raise ValueError('staged column readout requires explicit caller scratch')
        if any(type(v) is not int or v <= 0 for v in
                (self.output_channels, self.row_tile, self.edge)):
            raise ValueError('staged column readout requires positive typed extents')
        if self.row_tile > self.edge or self.scratch_bytes < self.required_bytes:
            raise ValueError('staged column scratch is smaller than the original tile')
        if len(self.gains) != self.output_channels or type(self.relu) is not bool:
            raise ValueError('staged column readout requires complete original coefficients')
        if type(self.accumulator_row) is not int or self.accumulator_row < 0:
            raise ValueError('staged readout requires reserved accumulator rows')

    @property
    def required_bytes(self):
        return self.edge * self.row_tile * 4

    def emit(self, enc, out, fields):
        first, columns = fields['output_column'], fields['columns']
        if type(first) is not int or type(columns) is not int or not (
                0 <= first < self.output_channels and
                0 < columns <= self.edge and first + columns <= self.output_channels):
            raise ValueError('staged readout requires a checked static original column interval')
        key = first, columns
        if key not in self._partitions:
            self._partitions[key] = partition_column_scales(
                self.gains[first:first+columns], columns, relu=self.relu,
                max_partitions=self.edge)
        # This store copies exact accumulator bits. Source bias is already in
        # those bits; source scaling and activation occur only at final readout.
        enc.store(self.scratch, fields['accumulator_row'], fields['rows'], columns,
                  self.edge*4, dtype='i32', scale=1.0, relu=False)
        enc.fence()
        for region in self._partitions[key]['regions']:
            offset, count = region['first_column'], region['columns']
            enc.load(enc.offset(self.scratch, offset*4), self.accumulator_row | (1 << 31),
                     fields['rows'], count, self.edge*4, stream=2, scale=1.0, shrunk=False)
            enc.fence()
            output_offset = enc.add(enc.mul(fields['output_row'], self.output_channels),
                                    first + offset)
            enc.store(enc.offset(out, output_offset), self.accumulator_row,
                      fields['rows'], count, self.output_channels, dtype='i8',
                      scale=region['scale'], relu=self.relu)
            enc.fence()

    def record(self):
        return dict(schema='authored.staged_column_readout.v1',
            scratch_bytes=self.required_bytes, scratch_row_bytes=self.edge*4,
            reserved_accumulator=[self.accumulator_row,self.accumulator_row+self.row_tile],
            source_coefficients=list(self.gains), relu=self.relu,
            transfer='raw i32 MVOUT; full-i32 strided MVIN3; one narrowing MVOUT',
            completion='full drain after each staged store, refill and final store',
            host_tensor_arithmetic=False)


# Private ABI scratch budget, below the selected 8 KiB frame preflight ceiling.
# This is a compiler workspace bound, not an accelerator capacity claim.
STAGED_EPOCH_SCRATCH_LIMIT = 4096


def staged_epoch_workspace(input_shape, weight_shape, *, kernel=None,
                           strides=(1, 1), dilation=(1, 1),
                           padding=(0, 0, 0, 0), facts_path=None, **_):
    from pathlib import Path
    from convolution import _resources
    from merlin.compile.scheduling.window_transfer import Window2D
    ws = tuple(weight_shape)
    kh, kw = tuple(kernel or ws)[:2]
    co = ws[-1]
    pad = tuple(padding)
    if len(pad) == 2:
        pad = pad + pad
    window = Window2D(tuple(input_shape), (kh, kw), tuple(strides), tuple(dilation), pad)
    path = str(facts_path or Path(__file__).parent/'data/facts.json')
    edge, _, acc_rows, _ = _resources(path)
    row_tile = min(edge, window.output_spatial[1])
    tile_bytes = edge*row_tile*4
    slots = min((co+edge-1)//edge, (acc_rows-row_tile)//row_tile,
                STAGED_EPOCH_SCRATCH_LIMIT//tile_bytes)
    if slots <= 0:
        raise ValueError('staged epochs need a bounded scratch tile and disjoint accumulator rows')
    return dict(shape=[row_tile, edge*slots], bytes=tile_bytes*slots,
                slot_count=slots, row_tile=row_tile, array_edge=edge,
                private_scratch_limit=STAGED_EPOCH_SCRATCH_LIMIT)


@dataclass
class StagedEpochColumnReadout(StagedColumnReadout):
    """Exact source stores grouped inside a checked shared readout epoch.

    Raw stores use distinct scratch slots. A real drain publishes the whole
    subgroup before refills; selected ACC RAW/WAR orders each refill/narrow pair.
    A real retirement drain closes every scratch subgroup before slot reuse.
    The original shared block/phase completion remains mandatory.
    """
    slot_count: int = 1
    _held: list = field(default_factory=list, init=False, repr=False)
    _bound: dict = field(default_factory=dict, init=False, repr=False)
    _groups: list = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        if (type(self.slot_count) is not int or self.slot_count <= 0
                or self.required_bytes*self.slot_count > self.scratch_bytes
                or self.required_bytes*self.slot_count > STAGED_EPOCH_SCRATCH_LIMIT):
            raise ValueError('staged epoch scratch slots exceed the explicit private workspace')

    def bind(self, plan, readout_law, ordering, ordering_path):
        import json, hashlib
        from pathlib import Path
        from merlin.compile.scheduling.window_transfer import ConvolutionReadoutEpochLaw
        selected = json.loads(Path(ordering_path).read_text())
        cross = {(r['older'],r['younger'],r['space']):r
                 for r in selected['ordering']['cross_queue']}
        edges = [('MVIN3.spad_addr','MVOUT.acc_addr','accumulator'),
                 ('MVOUT.acc_addr','MVIN3.spad_addr','accumulator')]
        if (type(readout_law) is not ConvolutionReadoutEpochLaw
                or not ordering.get('admitted')
                or selected['ordering_sha256'] != ordering.get('ordering_sha256')
                or hashlib.sha256(json.dumps(selected['ordering'],sort_keys=True).encode()).hexdigest()
                    != selected['ordering_sha256']
                or not all(cross.get(k,{}).get('overlap') == 'completion' for k in edges)
                or not all((readout_law.local_read_after_write,
                    readout_law.accumulator_result_visibility,
                    readout_law.ordered_configuration_capture,
                    readout_law.finite_queue_progress,
                    readout_law.complete_device_drain,
                    readout_law.exclusive_private_state))
                or plan.row_tile != self.row_tile or plan.column_tile != self.edge
                or plan.accumulator_rows > self.accumulator_row):
            raise ValueError('staged epochs require checked original readout ownership and selected refill/readout RAW/WAR')
        self._bound = dict(source_identity=readout_law.source_identity,
            plan_identity=plan.identity, ordering_sha256=ordering['ordering_sha256'],
            required_completion_edges=[list(k) for k in edges],
            pending_readouts=readout_law.pending_readouts,
            max_original_readouts=readout_law.max_issued_readouts,
            max_issued_narrow_readouts=self.slot_count*self.edge,
            issued_bound_is_not_queue_capacity=True)

    def emit(self, enc, out, fields):
        if not self._bound:
            raise ValueError('staged epochs were not bound to the actual shared readout owner')
        f = dict(fields)
        first, cols, acc, rows = (f[k] for k in
            ('output_column','columns','accumulator_row','rows'))
        if (any(type(v) is not int for v in (first,cols,acc,rows))
                or not 0 < rows <= self.row_tile or not 0 < cols <= self.edge
                or first < 0 or first+cols > self.output_channels
                or acc < 0 or acc+rows > self.accumulator_row):
            raise ValueError('staged epoch changed an original column/accumulator interval')
        if self._held:
            prev = self._held[-1]
            if (first != prev['output_column']+prev['columns']
                    or f['output_row'] != prev['output_row'] or rows != prev['rows']
                    or any(acc < h['accumulator_row']+h['rows']
                           and h['accumulator_row'] < acc+rows for h in self._held)):
                raise ValueError('staged epoch requires contiguous original columns and disjoint live ACC cells')
        slot = len(self._held)
        enc.store(enc.offset(self.scratch,slot*self.required_bytes),acc,rows,cols,
                  self.edge*4,dtype='i32',scale=1.0,relu=False)
        self._held.append(f)
        if len(self._held) == self.slot_count:
            self.flush(enc,out)

    def flush(self, enc, out):
        if not self._held:
            return
        # Publishes raw i32 DRAM writes before any load reads their bytes.
        enc.fence()
        for slot,f in enumerate(self._held):
            first,columns = f['output_column'],f['columns']
            key = first,columns
            if key not in self._partitions:
                self._partitions[key] = partition_column_scales(
                    self.gains[first:first+columns],columns,relu=self.relu,
                    max_partitions=self.edge)
            for region in self._partitions[key]['regions']:
                offset,count = region['first_column'],region['columns']
                enc.load(enc.offset(self.scratch,slot*self.required_bytes+offset*4),
                    self.accumulator_row | (1<<31),f['rows'],count,self.edge*4,
                    stream=2,scale=1.0,shrunk=False)
                output_offset = enc.add(enc.mul(f['output_row'],self.output_channels),first+offset)
                enc.store(enc.offset(out,output_offset),self.accumulator_row,
                    f['rows'],count,self.output_channels,dtype='i8',
                    scale=region['scale'],relu=self.relu)
        # Retires all reads of scratch and final stores before scratch reuse.
        enc.fence()
        self._groups.append(dict(original_columns=[(f['output_column'],f['columns']) for f in self._held],
            original_accumulators=[(f['accumulator_row'],f['rows']) for f in self._held],
            issued_raw_readouts=len(self._held),
            issued_narrow_readouts=sum(len(self._partitions[f['output_column'],f['columns']]['regions'])
                                       for f in self._held),
            source_row=str(self._held[0]['output_row']),raw_publication_drain=True,
            scratch_retirement_drain=True))
        self._held.clear()

    def complete(self, enc, out, kind, fields):
        if kind == 'store':
            self.flush(enc,out)
        elif self._held:
            raise ValueError('original convolution ownership ended before its staged readout block')
        enc.fence()

    def record(self):
        if self._held:
            raise ValueError('staged readout has an uncompleted original store frontier')
        return dict(schema='authored.staged_column_readout_epoch.v1',
            source_coefficients=list(self.gains),relu=self.relu,
            scratch_bytes=self.required_bytes*self.slot_count,
            scratch_slots=self.slot_count,private_scratch_limit=STAGED_EPOCH_SCRATCH_LIMIT,
            reserved_accumulator=[self.accumulator_row,self.accumulator_row+self.row_tile],
            source_bound_admission=self._bound,actual_static_subgroup_templates=self._groups,
            transfer='raw i32 MVOUT subgroup; real publication drain; full-i32 MVIN3 and exact scaled MVOUT pairs; real scratch-retirement drain',
            completion='original shared readout/phase drains retained; subgroup drains strengthen completion',
            host_tensor_arithmetic=False)
