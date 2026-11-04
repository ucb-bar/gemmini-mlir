"""Checked four-channel-tile input preparation for retained NHWC convolution.

Raw gemmini.h lines427-438 explicitly prepare up to four A K tiles at once,
source +DIM bytes and local +DIM rows per tile. Source row pitch is independent
of that local block stride. Padding keeps primitive zero-source calls.
"""
from dataclasses import dataclass, replace
from functools import lru_cache
from hashlib import sha256

from merlin.compile.scheduling.tile_load_coalescing import (
    TileLoadAddressLaw, TileLoadMapping, propose_tile_load_coalescing,
)
from merlin.compile.scheduling.panel_tile_loads import expand_panel_tile_loads
from merlin.compile.scheduling.llvm_program import emit_index_program
from merlin.sched.ir import Const, Kernel, Loop, Var, call, concretize, instances
from .dense_rhs_transfers import REFERENCE_FILES
from . import isa


@dataclass(frozen=True)
class ConvInputTransfers:
    primitive: object
    kernel: object
    realization: dict

    def emit(self, builder, lower, fragment):
        if self.kernel.digest() != self.realization['actual_kernel_sha256']:
            raise ValueError('input preparation product changed after admission')
        # All runtime row counts 1..DIM have independently checked final bands.
        # Fragment source/local translations do not change the checked deltas.
        product = self.primitive if fragment['zero'] else self.kernel
        def effect(kind, fields):
            if kind != 'load_input':raise ValueError('non-input preparation effect')
            lower(kind, dict(fragment,
                first_element=0 if fragment['zero'] else builder.op(
                    'add', fragment['first_element'], fields['source_delta']),
                local_row=builder.op('add',fragment['local_row'],fields['local_delta']),
                columns=fields['columns']))
        return emit_index_program(product,builder,effect)


@lru_cache(maxsize=128)
def plan_spatial_input_transfers(ci, stride, lhs_base, rhs_base):
    if (any(type(x)is not int for x in (ci,stride,lhs_base,rhs_base))
            or ci < 64 or ci % 64 or stride <= 0 or lhs_base < 0
            or lhs_base + 64 > rhs_base):
        return None
    identity=sha256(repr(sorted(REFERENCE_FILES.items())).encode()).hexdigest()
    law=TileLoadAddressLaw(16,16,64,1,stride*ci,16,
        'selected-gemmini-software-A-block-mvin:'+identity)
    def original(rows):
        c=Var('channel_tile')
        return Kernel('conv_input_four_channel_tiles',(),(
            Loop('channel_tile',4,(call('load_input',source_delta=c*16,
                local_delta=c*16,rows=rows,columns=16),)),))
    def project(effect):
        if effect.instr!='load_input':return None
        f=dict(effect.args)
        return TileLoadMapping('original_nhwc_input','private_lhs',
            f['source_delta'],f['local_delta'],f['rows'],f['columns'],law)
    def encode(effect,columns):
        return replace(effect,args=tuple((k,Const(columns)if k=='columns'else v)for k,v in effect.args))
    def band(effect,mapping):
        return replace(effect,args=tuple((k,{'source_delta':mapping.source_byte,
            'local_delta':mapping.local_row,'rows':mapping.rows,'columns':mapping.columns}.get(k,v))
            for k,v in effect.args))
    checked=[]
    for rows in range(1,17):
        source=original(rows)
        actual,proposal=propose_tile_load_coalescing(source,project=project,
            encode=encode,max_tiles=4,coalesce_tail=True)
        expanded,projection=expand_panel_tile_loads(actual,project=project,encode_band=band)
        decoded=[(e.instr,concretize(e,env))for e,env in instances(expanded)]
        expected=[('load_input',dict(source_delta=i*16,local_delta=i*16,
            rows=rows,columns=16))for i in range(4)]
        if decoded!=expected:raise ValueError('actual input bands differ from NHWC/source private layout')
        # Check the actual terminal words independently of the symbolic mapper.
        # Translations and all row counts preserve every expanded source byte.
        words=[]
        for e,env in instances(actual):
            f=concretize(e,env);word=isa.mvin(4096+f['source_delta'],lhs_base+f['local_delta'],f['rows'],f['columns'])
            function,dram,matrix=word
            if function!=2:raise ValueError('input block uses wrong load channel')
            local=matrix&0xffffffff;columns=(matrix>>32)&0xffff;depth=(matrix>>48)&0xffff
            for n in range((columns+15)//16):
                words.append(('load_input',dict(source_delta=dram-4096+n*16,
                    local_delta=local-lhs_base+n*16,rows=depth,columns=min(16,columns-n*16))))
        if words!=expected:raise ValueError('encoded input words differ from final source bands')
        checked.append(dict(rows=rows,actual_sha256=actual.digest(),expanded_sha256=expanded.digest()))
    return ConvInputTransfers(original(16),actual,dict(
        source_kernel_sha256=original(16).digest(),actual_kernel_sha256=actual.digest(),
        reference_files=REFERENCE_FILES,proposal=proposal,projection=projection,
        all_runtime_rows_checked=checked,private_lhs_interval=[lhs_base,lhs_base+64],
        private_rhs_start=rhs_base,source_pitch_bytes=stride*ci,
        source_axes=['image','output_y','output_x','kernel_y','kernel_x','channel_group','channel_tile'],
        ordering='four disjoint input tiles prepared, original channel-order N consumers; existing full completion retained',
        padding='original primitive zero-source loads retained'))
