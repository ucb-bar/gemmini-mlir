"""Optional bounded geometry region for checked convolution array groups.

The predicates delimit an experimental region around paired 4/8-array-channel
3x3 stride-one observations. The paired timing panel motivates the region, without certifying its interior.
The ordinary group planner, original owners and selected ordering law remain
mandatory. No source identity, model name, or benchmark name selects a path.
"""
import hashlib
import json


def select_array_retention(requested, window, output_channels, *, edge,
                           accumulator_rows, hardware_identity):
    """Return effective off/groups and a resource-relative region receipt.

    Explicit groups preserves its independent legality-only experimental path.
    regional admits at least two horizontal M tiles and capacity for two full
    output-channel accumulator tiles. The CI/CO range is the paired panel envelope,
    not a hardware legality threshold. Region neighbors still need paired cycles.
    """
    if requested not in ('off', 'groups', 'regional'):
        raise ValueError('CONV_ARRAY_RETENTION must be off, groups or regional')
    if any(type(v) is not int or v <= 0 for v in (edge, accumulator_rows, output_channels)):
        raise ValueError('array region requires positive original resource geometry')
    if requested != 'regional':
        return requested, dict(requested=requested, effective=requested,
            qualification='explicit choice; checked planner remains mandatory')
    from merlin.transforms.pipeline_policy import PolicyFacts, ResourcePredicate, ShapePredicate
    from merlin.transforms.pipeline_state import AnalysisStamp, PipelineContext
    co_tiles = (output_channels + edge - 1) // edge
    resources = dict(unit_stride=int(window.stride == (1,1)),
        unit_dilation=int(window.dilation == (1,1)),
        horizontal_m_tiles=(window.output_spatial[1] + edge - 1) // edge,
        full_output_m_capacity=accumulator_rows // (co_tiles * edge),
        edge=edge, accumulator_rows=accumulator_rows,
        input_channel_tiles=(window.input_shape[-1]+edge-1)//edge,
        output_channel_tiles=co_tiles)
    shapes=(tuple(window.input_shape), (*window.kernel,window.input_shape[-1],output_channels))
    identity=hashlib.sha256(json.dumps([shapes,resources],sort_keys=True).encode()).hexdigest()
    facts=PolicyFacts(AnalysisStamp(identity,
        PipelineContext(hardware=hardware_identity,resources=hardware_identity),
        'convolution-array-region'),shapes=shapes,resources=tuple(sorted(resources.items())))
    shape_guards=(dict(operand=1,axis=0,at_least=3,at_most=3),
        dict(operand=1,axis=1,at_least=3,at_most=3),
        dict(operand=1,axis=2,at_least=4*edge,at_most=8*edge),
        dict(operand=1,axis=3,at_least=4*edge,at_most=8*edge))
    resource_guards=tuple(dict(resource=k,at_least=v)for k,v in
        [('unit_stride',1),('unit_dilation',1),('horizontal_m_tiles',2),('full_output_m_capacity',2)])
    predicates=tuple(ShapePredicate(**g)for g in shape_guards)+tuple(ResourcePredicate(**g)for g in resource_guards)
    matches=tuple(g.match(facts)for g in predicates)
    effective='groups' if all(m is True for m in matches)else'off'
    return effective,dict(schema='submission.conv_array_region.v1',requested=requested,effective=effective,
        shapes=shapes,resources=resources,shape_predicates=shape_guards,
        resource_predicates=resource_guards,matches=matches,
        legality=dict(status='pending checked planner',
            necessary_capacity_predicates=resource_guards[-2:],
            note='capacity fit is necessary; selected WS, ownership, channel tile, readout and completion checks remain mandatory'),
        profitability=dict(status='provisional paired model envelope',
            evidence='paired full-model 4/8-array-channel 3x3 stride-one shapes',
            limitation='interior and neighbor timings UNKNOWN; no measured gSIM speedup'),
        qualification='optional bounded region; independent legality mandatory; neighbor profitability UNKNOWN')


def row_block_limit(value=4):
    """Explicit bounded 2-vs-4 experiment; capacity still chooses actual M."""
    if type(value) is not int or value not in (2,4):
        raise ValueError('array group experiment requires max_row_blocks 2 or 4')
    return value
