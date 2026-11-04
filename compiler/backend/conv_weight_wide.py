"""Selected resident convolution weight transfers with exact tile-band custody.

This adapter changes only the complete RHS preparation region, between the
original phase ownership drains. It uses the library rectangle planner and
independently expands every actual wide transfer against the original source
and resident-consumer rows. It grants no new completion or arithmetic law.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

from merlin.compile.scheduling.tile_load_coalescing import (
    TileLoadCoalescing, TileLoadAddressLaw, TileLoadMapping, expand_tile_load_mapping,
)
from merlin.sched.ir import Const
from .dense_wide_loads import selected_addressing_evidence


def _digest(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


@dataclass(frozen=True)
class ConvolutionWeightTransfers:
    source_identity: str
    panel: object
    pitch: int
    phases: tuple
    addressing_json: str
    identity: str

    def revalidate(self):
        self.panel.__post_init__()
        if self.identity != _digest(self._record()):
            raise ValueError('resident weight transfer owner changed after admission')
        for widths,depth,actual in self.phases:
            self._check(widths,depth,actual)

    def _record(self):
        from dataclasses import asdict
        return dict(source=self.source_identity,panel=asdict(self.panel),pitch=self.pitch,
                    phases=self.phases,addressing=json.loads(self.addressing_json))

    def _check(self,widths,depth,actual):
        """Every expanded ACTUAL band equals an original affine source tile.

        These are offsets from the current (tap,chunk,phase) origin. Both
        programs add exactly reduction_row*pitch+phase_column on DRAM and
        rhs_base+slot*active*16 locally. Prove relative maps once, then check
        all original consumer slot bounds through the library resident owner.
        """
        p=self.panel;active=len(widths)
        if (not self.source_identity or p.tile!=16 or not 0<active<=p.column_blocks
                or not 0<depth<=16 or any(not 0<c<=16 for c in widths)
                or any(c!=16 for c in widths[:-1]) or sum(widths)>self.pitch):
            raise ValueError('weight bands leave original source rectangle')
        evidence=json.loads(self.addressing_json)
        law=TileLoadAddressLaw(16,16,64,1,self.pitch,16,evidence['identity'])
        projected=[]
        for column,columns,local in actual:
            mapping=TileLoadMapping('weight','spad',Const(column),Const(local),Const(depth),Const(columns),law)
            for band in expand_tile_load_mapping(mapping):
                projected.append((band.source_byte.value,band.local_row.value,
                                  band.rows.value,band.columns.value))
        expected=[(n*16,n*16,depth,c)for n,c in enumerate(widths)]
        if projected!=expected:
            raise ValueError('actual wide bands differ from original weight source/consumer map')
        for slot in range(p.reduction_tiles):
            tap,chunk=divmod(slot,p.channel_tiles)
            for n,c in enumerate(widths):
                row=p.rhs_row(tap,chunk,n,active_column_blocks=active)
                if row!=p.rhs_base+(slot*active+n)*16 or row+depth>p.rhs_base+p.rhs_rows:
                    raise ValueError('actual wide band exceeds original resident consumer rows')

    def transfers(self,widths,depth):
        self.revalidate()
        for w,d,actual in self.phases:
            if w==tuple(widths)and d==depth:return actual
        raise ValueError('emission requested an unadmitted original weight phase')

    def record(self):
        self.revalidate()
        return dict(schema='selected_convolution_weight_bands.v1',owner_sha256=self.identity,
            source_sha256=self.source_identity,addressing=json.loads(self.addressing_json),
            phase_templates=[dict(widths=w,depth=d,actual=a)for w,d,a in self.phases],
            original_commands=sum(len(w)for w,d,a in self.phases),
            actual_commands=sum(len(a)for w,d,a in self.phases),
            completion='unchanged RHS-ready and RHS-last-use phase drains; no compute before RHS-ready',
            semantics='all actual expanded bands equal original source tiles and every resident consumer row',
            complete_call_cycles='UNKNOWN',target_execution='not_run')


def prepare_weight_transfers(source,group,pitch):
    """Read-only admission, before target/context mutation or LLVM emission."""
    group.__post_init__();p=source.plan;panel=group.panel
    if p.primitive!=16 or panel.tile!=16 or type(pitch)is not int or pitch<p.output_channels:
        raise ValueError('wide weights require original DIM16 i8 physical row pitch')
    facts=json.loads(Path(__file__).with_name('dense_load_facts.json').read_text())
    evidence=selected_addressing_evidence(facts)
    full,tail=divmod(p.output_channels,16)
    phases,last=divmod(full,panel.column_blocks)
    widths=[]
    if phases:widths.append((16,)*panel.column_blocks)
    remainder=(16,)*last+((tail,)if tail else ())
    if remainder:widths.append(remainder)
    ci=p.window.input_shape[-1];depths=(16,)if ci>=16 else ()
    if ci%16:depths+=((ci%16),)
    plans=[]
    for w in widths:
        for depth in depths:
            # Keep a partial N tile separate; never pad its source bytes.
            full_tiles=sum(c==16 for c in w)
            actual=[]
            if full_tiles:
                planner=TileLoadCoalescing(depth,full_tiles*16,16,64,0,16,len(w)*16)
                actual.extend(planner.commands())
            if w[-1]!=16:actual.append((full_tiles*16,w[-1],full_tiles*16))
            plans.append((w,depth,tuple(actual)))
    owner=ConvolutionWeightTransfers(source.identity,panel,pitch,tuple(plans),
        json.dumps(evidence,sort_keys=True),'')
    from dataclasses import replace
    owner=replace(owner,identity=_digest(owner._record()))
    owner.revalidate()
    return owner
