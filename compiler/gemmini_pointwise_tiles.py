"""Pure opt-in tiles for an admitted prepared grouped pointwise convolution.

Options propose source row tiles and shared-ring DMA width only. Existing
capacity, lifetime, ordering and terminal owners still consume the real plan.
No files, knobs, timing selection or physical queue reinterpretation live here.
"""
from dataclasses import dataclass, replace
import hashlib
import json
from math import lcm


@dataclass(frozen=True)
class PointwiseTileResources:
    """Selected stream-zero block addressing, derived from explicit inputs."""
    facts_sha256: str
    block: int
    max_columns: int
    max_rows: int
    max_block_stride: int
    max_request_bytes: int

    @classmethod
    def from_selected(cls, facts_bytes, ordering_selection):
        """Validate the same selected extent law used by the frozen variant."""
        digest = hashlib.sha256(facts_bytes).hexdigest()
        if ordering_selection['base_facts']['sha256'] != digest:
            raise ValueError('pointwise wide input selected facts differ')
        extent = ordering_selection['ordering']['extents']['MVIN.spad_addr']
        blocks = extent.get('blocks', {})
        cols, stride = blocks.get('cols', {}), blocks.get('stride', {})
        rows = extent.get('rows', {})
        if (extent.get('status') != 'derived'
                or type(blocks.get('block')) is not int
                or cols.get('operand') != 'rs2' or cols.get('offset') != 32
                or type(cols.get('width')) is not int or cols['width'] <= 0
                or rows.get('operand') != 'rs2' or rows.get('offset') != 48
                or type(rows.get('width')) is not int or rows['width'] <= 0
                or stride.get('code') != 1 or stride.get('operand') != 'rs1'
                or stride.get('offset') != 16
                or type(stride.get('width')) is not int or stride['width'] <= 0
                or stride.get('state', {}).get('value') != 0):
            raise ValueError('pointwise wide input needs selected stream0 block addressing')
        facts = json.loads(facts_bytes)
        request = facts['facts']['timing_structure']['memory_clients']['clients']['dma_read']['max_request_bytes']
        if type(request) is not int or request <= 0:
            raise ValueError('pointwise wide input needs a positive DMA request extent')
        return cls(digest, blocks['block'], (1 << cols['width']) - 1,
                   (1 << rows['width']) - 1, (1 << stride['width']) - 1, request)


def plan_pointwise_tiles(source, resources, selection, *, prepared_columns=False,
                         stage_channels=None, balanced_rows=False):
    """Return ``(source, input_ring, selection)`` without mutating any input.

    A None ring means the caller keeps its ordinary rolling preparation. Off,
    unsupported and unadmitted sources return before any resource access or
    rolling planning. Explicit invalid active options refuse rather than silently
    selecting a different schedule. Admission is legality, not timing evidence.
    """
    diagnostics = selection.get('pointwise_experiment', {})
    w = source.plan.window
    supported = (w.kernel == (1, 1) and w.stride in ((1, 1), (2, 2))
                 and w.dilation == (1, 1) and not any(w.padding))
    if (not prepared_columns or not diagnostics.get('admitted')
            or diagnostics.get('mode') not in ('resident', 'array') or not supported):
        return source, None, selection
    if type(balanced_rows) is not bool:
        raise ValueError('balanced_rows must be boolean')
    if stage_channels is None and not balanced_rows:
        return source, None, selection
    from gemmini_pointwise_policy import select_pointwise
    from gemmini_conv_policy import prepare_rolling_input
    original_source, original_selection = source, selection
    p = source.plan
    if stage_channels is not None:
        if (type(stage_channels) is not int or stage_channels <= 0
                or resources.block != p.primitive
                or stage_channels % resources.block
                or stage_channels > resources.max_request_bytes
                or stage_channels > resources.max_columns):
            raise ValueError('pointwise wide input exceeds selected DMA or encoded extent')
        # Small original K may use a channel tile narrower than the hardware
        # primitive. A wide staging limit must be whole blocks in BOTH layouts;
        # it grants no permission to pad K or reinterpret scratchpad lanes.
        staging_unit = lcm(resources.block, p.channel_tile)
        if stage_channels % staging_unit:
            return original_source, None, original_selection
    if balanced_rows:
        width = w.output_spatial[1]
        blocks = (width + p.primitive - 1) // p.primitive
        source = replace(source, plan=replace(p, row_tile=(width + blocks - 1) // blocks))
    # Rebind the exact shared capacity proposal to the actual retiled source.
    _, _, selection = select_pointwise(source, 'rolling_rhs', 8, selection,
                                       mode=diagnostics['mode'])
    if not selection['pointwise_experiment']['admitted']:
        return original_source, None, original_selection
    prepared, ring = prepare_rolling_input(source)
    if ring is None:
        return original_source, None, original_selection
    if stage_channels is not None:
        if (stage_channels % ring.channel_tile
                or ring.stage_rows > resources.max_rows
                or ring.row_pitch > resources.max_block_stride):
            return original_source, None, original_selection
        ring = replace(ring, stage_channels=stage_channels)
    return prepared, ring, selection
