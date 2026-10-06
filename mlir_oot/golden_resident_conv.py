"""Primitive exact 3x3 convolution with one resident, virtually padded input.

A channel tile owns a complete padded spatial plane. CONFIG_LD block_stride
places wide input loads directly into this layout; shifted resident rows
supply all nine taps without repeated input DMA. The mesh row stride follows
the source spatial stride. Each output row fits one DIM tile; resource
admission and optional remaining-row weight placement are explicit.
"""
from dataclasses import asdict, dataclass
import json
from xdsl.dialects import llvm
from xdsl.dialects.builtin import StringAttr, IntegerAttr, i64
from .golden_gemm import GoldenGemm, Shape, _ceil_div, _groups
from .tables import rtl_facts as F, isa


@dataclass(frozen=True)
class ResidentConvOptions:
    rows_per_tile: int = 1
    loop_channels: bool = False
    prefetch_b: bool = False


def choose_compact_resident(conv, *, prefetch_b=False):
    """Derive an opt-in schedule from spatial span and declared resources.

    This is a target compiler policy, independent of source provenance or model
    identity. The largest legal adjacent-row group reduces mesh padding and
    ordinary channel loops bound code size. Resource refusal leaves selection
    of an alternative schedule to the caller; this is no universal speed claim.
    """
    conv.validate()
    if conv.w + 2 > F.DIM:
        raise ValueError('resident width plus halo exceeds DIM')
    rows = min(conv.h, 1 + (F.DIM - conv.w) // (conv.w + 2))
    if type(prefetch_b) is not bool:
        raise ValueError('resident weight prefetch selection must be boolean')
    options = ResidentConvOptions(rows_per_tile=rows, loop_channels=not prefetch_b,
        prefetch_b=prefetch_b)
    return GoldenResidentConv(conv, **asdict(options)), options


class GoldenResidentConv(GoldenGemm):
    def __init__(self, s, *, rows_per_tile=1, loop_channels=False, prefetch_b=False,
            weight_base=None, source_stride=False, row_residue=False):
        s.validate()
        if type(loop_channels) is not bool:
            raise ValueError('resident channel-loop selection must be boolean')
        self.loop_channels = loop_channels
        if type(prefetch_b) is not bool:
            raise ValueError('resident weight prefetch selection must be boolean')
        if prefetch_b and loop_channels:
            raise ValueError('resident weight prefetch needs static reduction commands')
        self.prefetch_b = prefetch_b
        if type(source_stride) is not bool:
            raise ValueError('source stride residency selection must be boolean')
        self.source_stride = source_stride
        if type(row_residue) is not bool or (row_residue and (not source_stride or s.stride != 2)):
            raise ValueError('resident row residue layout requires source stride2')
        self.row_residue = row_residue
        self.conv = s
        if not source_stride and (s.stride != 1 or s.w + 2 > F.DIM):
            raise ValueError('compact resident convolution needs stride1 and width+halo<=DIM')
        if s.explicit_halo or s.ow > F.DIM or s.cin % F.DIM:
            raise ValueError('resident convolution needs unpadded input, output width<=DIM and aligned Cin')
        self.input_pitch = _ceil_div(s.w+2,s.stride)*s.stride if row_residue else s.w+2
        self.residue_rows = _ceil_div(s.h+2,s.stride) if row_residue else s.h+2
        self.plane = self.residue_rows*self.input_pitch*(s.stride if row_residue else 1)
        self.output_pitch = self.input_pitch//s.stride if row_residue else self.input_pitch
        if self.plane >= 1 << 16:
            raise ValueError('resident input plane stride exceeds ISA field')
        if (type(rows_per_tile) is not int or not 1 <= rows_per_tile <= s.oh or
                (rows_per_tile-1)*self.output_pitch+s.ow > F.DIM):
            raise ValueError('resident row group must fit its padded spatial span in DIM')
        self.rows_per_tile = rows_per_tile
        self.row_tiles = tuple((y, min(rows_per_tile, s.oh-y),
            (min(rows_per_tile, s.oh-y)-1)*self.output_pitch+s.ow)
            for y in range(0, s.oh, rows_per_tile))
        if weight_base is not None and (type(weight_base) is not int or
                weight_base < 0 or weight_base % F.DIM):
            raise ValueError('resident weight base must be a nonnegative DIM-aligned row')
        if prefetch_b and weight_base is not None:
            raise ValueError('explicit resident weight placement does not admit bank lookahead')
        self.bbase = 2*F.SPAD_BANK_ROWS if weight_base is None else weight_base
        self.explicit_weight_base = weight_base
        self.bases = (self.bbase, 3*F.SPAD_BANK_ROWS) if prefetch_b else (self.bbase,)
        if _ceil_div(s.cin,F.DIM)*self.plane > self.bbase:
            raise ValueError('resident input overlaps weight banks')
        if self.bbase+s.bn*F.DIM > F.SPAD_ROWS or len(self.row_tiles)*s.bn*F.DIM > F.ACC_ROWS:
            raise ValueError('resident convolution output/weight panel exceeds resources')
        if prefetch_b and (s.bn*F.DIM > F.SPAD_BANK_ROWS or
                self.bases[-1]+s.bn*F.DIM > F.SPAD_ROWS):
            raise ValueError('resident weight slots overlap or exceed scratchpad')
        for ki in range(s.cin//F.DIM):
            for y,_,span in self.row_tiles:
                for kh in range(3):
                    for kw in range(3):
                        start=ki*self.plane+self.source_offset(y,kh,kw)
                        stop=start+(span-1)*s.stride+1
                        end=(ki+1)*self.plane
                        if self.row_residue:
                            end=ki*self.plane+(kh%s.stride+1)*self.residue_rows*self.input_pitch
                        if stop > end:
                            raise ValueError('strided resident compute crosses its input plane')
        self.conv=s
        super().__init__(Shape(s.oh*s.ow,s.cout,s.cin,bm=len(self.row_tiles),bn=s.bn,
            output_dtype=s.output_dtype,scale=s.scale,relu=s.relu,wide_store=True,reuse_b=True))

    def input_row(self, padded_y):
        """Bijection from allocated source rows into stride-residue slabs."""
        if self.row_residue:
            return (padded_y%self.conv.stride)*self.residue_rows+padded_y//self.conv.stride
        return padded_y

    def source_offset(self, output_y, kh, kw):
        return self.input_row(output_y*self.conv.stride+kh)*self.input_pitch+kw

    def build(self):
        s=self.conv;pw=self.input_pitch
        self._emit_config()
        if s.stride != 1:
            from .ir.gemmini_dialect import ConfigExOp
            configs=[op for op in self.fb.entry.ops if isinstance(op,ConfigExOp)]
            assert len(configs)==1
            configs[0].attributes['a_stride']=IntegerAttr(s.stride,i64)
        self._rocc('config_ld',{'stride':s.cin,'block_stride':self.plane,'load_id':0})
        zero=self.fb.add(llvm.IntToPtrOp(self.fb.const(0))).results[0]
        # These loads partition the scratch input exactly: no overwrite races.
        for ci in range(0,s.cin,4*F.DIM):
            cols=min(4*F.DIM,s.cin-ci);base=(ci//F.DIM)*self.plane
            padded_rows=self.residue_rows*s.stride if self.row_residue else s.h+2
            for y in range(padded_rows):
                row=base+self.input_row(y)*pw
                if not 1<=y<=s.h:
                    for x in range(0,pw,F.DIM):
                        self._rocc('mvin',{'local':row+x,'rows':min(F.DIM,pw-x),'cols':cols,'load_id':0},zero)
                else:
                    self._rocc('mvin',{'local':row,'rows':1,'cols':cols,'load_id':0},zero)
                    for x in range(0,s.w,F.DIM):
                        ptr=self._ptr(self.a,self.fb.const((y-1)*s.w+x),s.cin,self.fb.const(ci))
                        self._rocc('mvin',{'local':row+x+1,'rows':min(F.DIM,s.w-x),'cols':cols,'load_id':0},ptr)
                    self._rocc('mvin',{'local':row+s.w+1,'rows':pw-s.w-1,'cols':cols,'load_id':0},zero)
        def channel(n0,nr):
            for kh in range(3):
                for kw in range(3):
                    def reduction(ki, first, static_ci=None):
                        krow = self.fb.const((kh*3+kw)*s.cin+static_ci) if static_ci is not None else self.fb.add_i(
                            self.fb.const((kh*3+kw)*s.cin),self.fb.mul_i(ki,self.fb.const(F.DIM)))
                        for d in range(0,len(nr),4):
                            ptr=self._ptr(self.b,krow,s.cout,self._tile(n0,d))
                            self._rocc('mvin',{'local':self.bbase+d*F.DIM,'rows':F.DIM,'cols':sum(nr[d:d+4]),'load_id':1},ptr)
                        for d,cols in enumerate(nr):
                            for tile,(y,count,span) in enumerate(self.row_tiles):
                                self._rocc('preload',{'bd':self.bbase+d*F.DIM if tile==0 else isa.GARBAGE_ADDR,
                                    'c':isa.acc_addr((tile*s.bn+d)*F.DIM,accumulate=not first),
                                    'bd_cols':cols,'bd_rows':F.DIM,'c_cols':cols,'c_rows':span})
                                offset=self.source_offset(y,kh,kw)
                                attrs={'a_cols':F.DIM,'a_rows':span,'accumulate':tile!=0}
                                if static_ci is not None:
                                    attrs['a']=(static_ci//F.DIM)*self.plane+offset
                                    self._rocc('compute',attrs)
                                else:
                                    address=self.fb.add_i(self.fb.mul_i(ki,self.fb.const(self.plane)),self.fb.const(offset))
                                    attrs.update(a_max=(s.cin//F.DIM-1)*self.plane+offset,
                                        a_reserved_rows=(s.cin//F.DIM)*self.plane)
                                    self._rocc('compute',attrs,address)
                    if self.loop_channels:
                        # The first update initializes C; every later channel/tap
                        # accumulates in the original increasing HWIO order.
                        first=kh==0 and kw==0
                        if first:reduction(self.fb.const(0),True)
                        self.fb.for_loop(int(first),s.cin//F.DIM,1,
                            lambda ki:reduction(ki,False))
                    else:
                        for ci in range(0,s.cin,F.DIM):
                            reduction(None,kh==0 and kw==0 and ci==0,ci)
            # Padding lanes inside a grouped tile are never materialized. Each
            # valid row has its original NHWC destination and scratch row pitch.
            for tile,(y,count,span) in enumerate(self.row_tiles):
                for row in range(count):
                    for d in range(0,len(nr),4 if s.output_dtype=='i8' else 1):
                        step=4 if s.output_dtype=='i8' else 1
                        ptr=self._ptr(self.c,self.fb.const((y+row)*s.ow),s.cout,self._tile(n0,d),4 if s.output_dtype=='i32' else 1)
                        self._rocc('mvout',{'local':isa.acc_addr((tile*s.bn+d)*F.DIM+row*self.output_pitch,full_row=s.output_dtype=='i32'),
                            'rows':s.ow,'cols':sum(nr[d:d+step])},ptr)
        def prefetched_channel(n0,nr):
            kt=s.cin//F.DIM
            def load(index):
                base=self.bases[index%2]
                for d in range(0,len(nr),4):
                    ptr=self._ptr(self.b,self.fb.const(index*F.DIM),s.cout,self._tile(n0,d))
                    self._rocc('mvin',{'local':base+d*F.DIM,'rows':F.DIM,
                        'cols':sum(nr[d:d+4]),'load_id':1},ptr)
            load(0)
            for index in range(9*kt):
                # The next panel writes a disjoint bank while the current one
                # remains live. Source HWIO reduction order is unchanged.
                if index+1 < 9*kt:
                    load(index+1)
                kh,kw=divmod(index//kt,3);ki=index%kt
                for d,cols in enumerate(nr):
                    for tile,(y,count,span) in enumerate(self.row_tiles):
                        self._rocc('preload',{'bd':self.bases[index%2]+d*F.DIM if tile==0 else isa.GARBAGE_ADDR,
                            'c':isa.acc_addr((tile*s.bn+d)*F.DIM,accumulate=index!=0),
                            'bd_cols':cols,'bd_rows':F.DIM,'c_cols':cols,'c_rows':span})
                        self._rocc('compute',{'a':ki*self.plane+self.source_offset(y,kh,kw),
                            'a_cols':F.DIM,'a_rows':span,'accumulate':tile!=0})
            for tile,(y,count,span) in enumerate(self.row_tiles):
                for row in range(count):
                    for d in range(0,len(nr),4 if s.output_dtype=='i8' else 1):
                        step=4 if s.output_dtype=='i8' else 1
                        ptr=self._ptr(self.c,self.fb.const((y+row)*s.ow),s.cout,self._tile(n0,d),4 if s.output_dtype=='i32' else 1)
                        self._rocc('mvout',{'local':isa.acc_addr((tile*s.bn+d)*F.DIM+row*self.output_pitch,full_row=s.output_dtype=='i32'),
                            'rows':s.ow,'cols':sum(nr[d:d+step])},ptr)
        self._for_groups(_groups(s.cout,s.bn),prefetched_channel if self.prefetch_b else channel)
        module=self._finish('gemmini_golden_resident_conv')
        module.attributes['gemmini.resident_conv_shape']=StringAttr(json.dumps(asdict(s),sort_keys=True))
        module.attributes['gemmini.resident_conv_layout']=StringAttr('channel-tile-major padded spatial planes; disjoint input DMA partitions')
        if self.rows_per_tile != 1:
            module.attributes['gemmini.resident_conv_rows_per_tile']=StringAttr(str(self.rows_per_tile))
        if self.loop_channels:
            module.attributes['gemmini.resident_conv_loop_channels']=StringAttr('bounded ordinary CPU channel loop')
        if self.prefetch_b:
            module.attributes['gemmini.resident_conv_prefetch_b']=StringAttr('one panel lookahead; disjoint bank2/bank3 weight lifetimes')
        if self.explicit_weight_base is not None:
            module.attributes['gemmini.resident_conv_weight_base']=StringAttr(str(self.bbase))
        if self.source_stride:
            module.attributes['gemmini.resident_conv_source_stride']=StringAttr(str(s.stride))
        if self.row_residue:
            module.attributes['gemmini.resident_conv_row_residue']=StringAttr(json.dumps(dict(
                stride=s.stride,input_pitch=self.input_pitch,residue_rows=self.residue_rows,
                output_pitch=self.output_pitch,allocated_plane_rows=self.plane),sort_keys=True))
        return module


def command_counts(s, *, rows_per_tile=1):
    """Exact primitive counts and transferred activation bytes for this schedule."""
    g=GoldenResidentConv(s, rows_per_tile=rows_per_tile)
    kt,nt=_ceil_div(s.cin,F.DIM),_ceil_div(s.cout,F.DIM)
    bloads=sum(_ceil_div(min(s.bn,nt-d),4) for d in range(0,nt,s.bn))
    compute=9*kt*nt*len(g.row_tiles)
    return dict(mvin_a=(2*_ceil_div(s.w+2,F.DIM)+s.h*(2+_ceil_div(s.w,F.DIM)))*_ceil_div(s.cin,4*F.DIM),
        mvin_b=9*kt*bloads,compute=compute,preload=compute,
        mvout=s.oh*(nt if s.output_dtype=='i32' else bloads),
        padded_array_issue_cycles=compute*F.DIM,
        activation_dram_bytes=s.h*s.w*s.cin,
        resident_input_rows=kt*g.plane,input_plane_stride=g.plane,
        accumulator_rows=len(g.row_tiles)*s.bn*F.DIM,weight_base_row=g.bbase,
        host_im2col_bytes=0)
