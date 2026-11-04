"""Finite K-panel alternatives consumed by the original checked dense planner.

The frozen narrow policy owns applicability. This adapter changes only the
shared resource cap; it never ranks by case identity or emits instructions.
The default16 delegates the exact incumbent call and keeps its cached artifact.
"""
from __future__ import annotations

import os
from pathlib import Path
from merlin.compile.tuning_knobs import knob

K_TILE_LIMITS = (4,8,16,32,64)


def narrow_k_tile_setting():
    selected=knob('DENSE_NARROW_K_TILES',default=16,values=K_TILE_LIMITS,
        root=Path(__file__).resolve().parents[1])
    if 'DENSE_NARROW_K_TILES' in os.environ:
        raw=os.environ['DENSE_NARROW_K_TILES']
        if raw not in tuple(map(str,K_TILE_LIMITS)):
            raise ValueError('DENSE_NARROW_K_TILES requires one of 4,8,16,32,64')
        selected=int(raw)
    if type(selected) is not int or selected not in K_TILE_LIMITS:
        raise ValueError('DENSE_NARROW_K_TILES requires one of 4,8,16,32,64')
    return selected


def narrow_geometry_schedule(m,k,n,bias,column_scales,geometry,*,planner=None,k_tiles=None):
    from .dense_pipeline import _retained_schedule
    planner=planner or _retained_schedule
    selected=narrow_k_tile_setting() if k_tiles is None else k_tiles
    if type(selected) is not int or selected not in K_TILE_LIMITS:
        raise ValueError('narrow K panel limit must be one of 4,8,16,32,64')
    arguments=(m,k,n,bias,column_scales,True,geometry,'half','interleaved','lhs')
    # None preserves the established planner's exact cache key and emission.
    result,refusal=(planner(*arguments) if selected==16 else
        planner(*arguments,segment_k_tiles=selected))
    if result is not None:
        result.revalidate()
        block=result.realization['panel_geometry']['selected_block_tiles']
        if block[1] > selected:
            raise ValueError('checked segmented planner did not consume selected K limit')
    return result,refusal,selected
