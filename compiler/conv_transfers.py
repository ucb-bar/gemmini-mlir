"""Fact-bound wide convolution input rows with an exact shared ring consumer.

The shared ConvolutionResidentInputRows owner maps every original channel band
into disjoint local rows. Its consumer rewrites LHS addresses through that same
map; widening never overwrites the single streamed channel window. Original
rolling ready/last-use drains remain explicit. No target encoding is duplicated.
"""
import hashlib
import json
from pathlib import Path

from merlin.compile.scheduling.window_transfer import ConvolutionResidentInputRows
from merlin.compile.scheduling.tile_load_coalescing import TileLoadAddressLaw, TileLoadCoalescing


def regional_input_candidate(window, output_channels, edge, hardware_identity, *,
                             out_dtype=None, bias=None, gains=(), relu=None):
    """Frozen geometry region; physical address/storage legality stays separate."""
    from merlin.transforms.pipeline_policy import PolicyFacts, ResourcePredicate
    from merlin.transforms.pipeline_state import AnalysisStamp, PipelineContext

    resources = dict(kernel_points=window.kernel[0]*window.kernel[1],
        input_channels=window.input_shape[-1], output_channels=output_channels,
        array_edge=edge, output_width=window.output_spatial[1],
        primitive_pitch_tail=int(0 < window.input_shape[-1] < edge),
        raw_i32_source=int(out_dtype == 'i32' and bias is False
            and relu is False and bool(gains) and all(g == 1.0 for g in gains)))
    identity = hashlib.sha256(repr(tuple(sorted(resources.items()))).encode()).hexdigest()
    facts = PolicyFacts(AnalysisStamp(identity,
        PipelineContext(hardware=hardware_identity, resources=hardware_identity),
        'convolution-resident-input-geometry'),
        resources=tuple(sorted(resources.items())))
    predicates = (dict(resource='kernel_points', at_least=2),
        dict(resource='input_channels', at_least=edge),
        dict(resource='output_channels', at_least=edge))
    wide = all(ResourcePredicate(**g).match(facts) is True for g in predicates)
    # A primitive-pitched short-CI ring amortizes dispatch across sufficiently
    # long output rows. This is a provisional profitability region only: the
    # original address, capacity, ownership and consumer checks still run.
    tail_predicates = (dict(resource='kernel_points', at_least=2),
        dict(resource='primitive_pitch_tail', at_least=1),
        dict(resource='output_channels', at_least=edge),
        dict(resource='output_width', at_least=2*edge),
        dict(resource='raw_i32_source', at_least=1))
    tail = all(ResourcePredicate(**g).match(facts) is True for g in tail_predicates)
    return dict(schema='authored.regional_rolling_input.v1',
        geometry_selected=wide or tail, resources=resources,
        predicates=predicates if wide else tail_predicates,
        candidate='rolling_wide' if wide else 'rolling_narrow' if tail else 'stream',
        region='aligned_channels' if wide else 'primitive_pitch_tail' if tail else None,
        qualification=('provisional matched model evidence; independent physical timing UNKNOWN'
            if wide else 'provisional one large paired complete-model observation; cutoff neighbors timing UNKNOWN'),
        legality='separate original ring/RHS/ACC admission and checked consumer required')


def plan_input_transfer(window, *, mode, edge, operand_rows, channel_tile, retain_input,
                        facts_path, ordering_path=None, output_channels=None, output_rows=1):
    if type(output_rows) is not int or output_rows not in (1, 2, 4):
        raise ValueError('vertical input horizon must be 1, 2 or 4 output rows')
    if mode not in ('stream','rolling_narrow','rolling_wide','regional_rolling'):
        raise ValueError('CONV_INPUT_TRANSFER must be stream, rolling_narrow, rolling_wide or regional_rolling')
    receipt = dict(requested=mode, selected='stream')
    if mode == 'regional_rolling':
        if type(output_channels) is not int or output_channels <= 0:
            raise ValueError('regional rolling input requires original output channels')
        region = regional_input_candidate(window, output_channels, edge,
            hashlib.sha256(Path(facts_path).read_bytes()).hexdigest())
        receipt['regional'] = region
        mode = region['candidate']
    if mode == 'stream':
        return None, receipt
    if not retain_input or window.stride[1] != 1:
        receipt['reason'] = 'resident consumer requires input retention and unit horizontal stride'
        return None, receipt
    facts_file=Path(facts_path)
    ordering_file=Path(ordering_path) if ordering_path else Path(__file__).parent/'data/command_ordering.json'
    ordering=json.loads(ordering_file.read_text())['ordering']['extents']['MVIN.spad_addr']
    blocks=ordering['blocks']; stride=blocks['stride']
    if (ordering['status'] != 'derived' or blocks['block'] != edge or channel_tile != edge
            or stride['code'] != 1 or stride['state']['value'] != 0):
        receipt['reason']='selected MVIN block geometry does not bind the original channel tile'
        return None, receipt
    max_columns=((1 << blocks['cols']['width'])-1)//edge*edge
    dma=json.loads(facts_file.read_text())['facts']['timing_structure']['memory_clients']['clients']['dma_read']['max_request_bytes']
    wide_columns=min(max_columns,dma//edge*edge)
    if wide_columns < edge:
        receipt['reason']='selected DMA bound cannot contain one original channel tile'
        return None,receipt
    stage_channels=edge if mode == 'rolling_narrow' else wide_columns
    row_pitch=window.input_shape[2]+window.padding[1]+window.padding[3]
    slots=(window.kernel[0]-1)*window.dilation[0]+1+(output_rows-1)*window.stride[0]
    channel_blocks=(window.input_shape[-1]+edge-1)//edge
    input_end=slots*row_pitch*channel_blocks
    rhs_base=(input_end+channel_tile-1)//channel_tile*channel_tile
    if rhs_base+channel_tile>operand_rows or row_pitch >= (1 << stride['width']):
        receipt['reason']='exact ring and disjoint streamed RHS do not fit selected local rows'
        return None,receipt
    # Validate the same physical block map on a maximal emitted source row run.
    geometry=TileLoadCoalescing(min(edge,row_pitch),window.input_shape[-1],edge,
        stage_channels,0,row_pitch,operand_rows)
    identity=hashlib.sha256(ordering_file.read_bytes()).hexdigest()
    law=TileLoadAddressLaw(edge,min(edge,(1 << ordering['rows']['width'])-1),
        stage_channels,1,window.input_shape[-1],row_pitch,'selected-MVIN-block-map:'+identity)
    config=dict(base=0,row_pitch=row_pitch,slots=slots,private_first=0,
        private_rows=input_end,stage_rows=min(edge,row_pitch),halo_value=0 if any(window.padding) else None,
        stage_channels=stage_channels)
    receipt.update(selected=mode,input_rows=input_end,rhs_base=rhs_base,
        output_row_horizon=output_rows,actual_ring_slots=slots,
        channel_blocks=channel_blocks,stage_channels=stage_channels,
        max_DMA_request_bytes=dma,block_stride_rows=row_pitch,
        address_law_identity=law.identity,ordering_sha256=identity,
        mapping='slot*row_pitch*channel_blocks + (channel//edge)*row_pitch + padded_x; lane channel%edge',
        completion='unchanged shared rolling input-ready and source-row last-use full drains',
        proof='shared ConvolutionResidentInputRows exact source map and consumer; TileLoadCoalescing bounded injective bands')
    return dict(config=config,rhs_base=rhs_base),receipt


def bind_input_rows(plan, source_identity, transfer):
    if transfer is None:
        return None
    return ConvolutionResidentInputRows(plan.window,source_identity,plan.channel_tile,
                                        **transfer['config'])


def bind_readout_transfers(plan, *, requested, gains, out_dtype, relu,
                           staged, readout_law, facts_path, ordering_path=None):
    """Bind the exported original-store partition to selected native MVOUT.

    This changes only physical transfer width. The shared epoch consumer waits
    for every original producer, checks original column/ACC coordinates, and
    preserves its final ownership drain. In particular, a short M tile cannot
    claim the selected fixed 16-row block spacing, and raw i32/staged exits do
    not inherit the scaled-i8 block law.
    """
    from merlin.compile.scheduling.window_transfer import ConvolutionReadoutTransferLaw
    receipt = dict(requested=bool(requested), selected='original_stores')
    if not requested:
        return None, receipt
    if (out_dtype != 'i8' or staged is not None or len(set(gains)) != 1
            or readout_law is None):
        receipt['reason'] = 'requires an original uniform scaled-i8 readout epoch'
        return None, receipt
    facts_raw = Path(facts_path).read_bytes()
    ordering_file = Path(ordering_path) if ordering_path else Path(__file__).parent/'data/command_ordering.json'
    selected = json.loads(ordering_file.read_bytes())
    order = selected['ordering']
    hw = hashlib.sha256(facts_raw).hexdigest()
    if (selected['base_facts']['sha256'] != hw
            or hashlib.sha256(json.dumps(order, sort_keys=True).encode()).hexdigest()
                != selected['ordering_sha256']):
        receipt['reason'] = 'selected native readout map has stale facts or identity'
        return None, receipt
    ext = order['extents']['MVOUT.acc_addr']
    blocks = ext['blocks']
    # The encoder resets all pooling fields on every ordinary scaled store.
    # These are exactly the conditions under which the selected block map was
    # derived; no load mapping or compute retention premise substitutes for it.
    conditions = ext.get('conditions', ())
    block = blocks['block']
    if (ext.get('status') != 'derived' or block != plan.primitive
            or blocks['stride'] != dict(constant=block)
            or plan.row_tile != block or plan.column_tile != block
            or len(conditions) != 1
            or conditions[0].get('field') != 'ConfigMvoutRs1.pool_stride'
            or conditions[0].get('reset') != 0):
        receipt['reason'] = 'original ACC pitch or numerical mode does not match selected MVOUT blocks'
        return None, receipt
    dma = json.loads(facts_raw)['facts']['timing_structure']['memory_clients']['clients']['dma_write']['max_request_bytes']
    max_columns = min(((1 << blocks['cols']['width'])-1)//block*block,
                      dma//block*block)
    if max_columns < 2*block or plan.output_channels <= block:
        receipt['reason'] = 'no adjacent original output tiles fit the selected transfer bound'
        return None, receipt
    state = hashlib.sha256(repr(('original-f32-scale-rne-i8', float(gains[0]),
                                bool(relu))).encode()).hexdigest()
    law = ConvolutionReadoutTransferLaw(hw, readout_law.numerical_policy_identity,
        plan.identity, 'selected-native-MVOUT-block-map:'+selected['ordering_sha256'],
        block, block, max_columns, ((0,plan.output_channels,state),),
        scaled_i8=True, exact_block_column_map=True, original_numerical_state=True)
    receipt.update(selected='wide_original_stores', law_sha256=law.identity,
        ordering_sha256=selected['ordering_sha256'], block_rows=block,
        max_columns=max_columns, numerical_state_sha256=state,
        completion='unchanged checked epoch producers and final full ownership drain')
    return law, receipt
