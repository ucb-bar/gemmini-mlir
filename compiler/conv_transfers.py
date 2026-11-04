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


def regional_input_candidate(window, output_channels, edge, hardware_identity):
    """Frozen geometry region; physical address/storage legality stays separate."""
    from merlin.transforms.pipeline_policy import PolicyFacts, ResourcePredicate
    from merlin.transforms.pipeline_state import AnalysisStamp, PipelineContext

    resources = dict(kernel_points=window.kernel[0]*window.kernel[1],
        input_channels=window.input_shape[-1], output_channels=output_channels,
        array_edge=edge)
    identity = hashlib.sha256(repr(tuple(sorted(resources.items()))).encode()).hexdigest()
    facts = PolicyFacts(AnalysisStamp(identity,
        PipelineContext(hardware=hardware_identity, resources=hardware_identity),
        'convolution-resident-input-geometry'),
        resources=tuple(sorted(resources.items())))
    predicates = (dict(resource='kernel_points', at_least=2),
        dict(resource='input_channels', at_least=edge),
        dict(resource='output_channels', at_least=edge))
    selected = all(ResourcePredicate(**g).match(facts) is True for g in predicates)
    return dict(schema='authored.regional_rolling_input.v1',
        geometry_selected=selected, resources=resources, predicates=predicates,
        candidate='rolling_wide' if selected else 'stream',
        qualification='provisional matched model evidence; independent physical timing UNKNOWN')


def plan_input_transfer(window, *, mode, edge, operand_rows, channel_tile, retain_input,
                        facts_path, ordering_path=None, output_channels=None):
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
    slots=(window.kernel[0]-1)*window.dilation[0]+1
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
