"""Invocation-local reuse of one exact uniform convolution store configuration.

This adapter filters actual terminal commands from the common readout helper.
No completion is omitted. The entire private convolution has one store pitch,
one uniform scale, one activation and no pooling; its compute/load commands do
not write store configuration. Initial state is unknown: establish it after the
entry drain, and never reuse it across operations or invocations.
"""
from dataclasses import asdict

from merlin.compile.scheduling.configuration import ConfigurationEffects

from . import isa


class UniformReadoutContext:
    def __init__(self,ctx,prepared,dst):
        p=prepared.source.plan;attrs=prepared.attributes
        dtype=attrs.get('output_dtype',ctx.spec(dst)['dtype'])
        scales=attrs.get('acc_scale_by_column')
        if (dtype not in ('i8','i32')or any('pool'in str(s).lower()for s in attrs.get('epilogue',()))
                or scales is not None and(len(scales)!=p.output_channels or len(set(map(float,scales)))!=1)):
            raise ValueError('configuration reuse needs a uniform private convolution readout')
        scale=scales[0]if scales else attrs.get('acc_scale',1.0)
        relu=bool(attrs.get('relu',False)or'relu'in attrs.get('epilogue',()))
        if dtype=='i32'and(float(scale)!=1.0 or scales is not None or relu):
            raise ValueError('full-width readout cannot change its numeric contract')
        self.ctx=ctx
        self.command=isa.config_st(prepared.output_pitch*(4 if dtype=='i32'else 1),
            scale=scale,activation=int(relu))
        if self.command[1]&~15 or not prepared.epoch.ordered_configuration_capture:
            raise ValueError('configuration reuse requires nonpooling ordered store state')
        self.effects=ConfigurationEffects(writes=(('store.rs1',self.command[1]),
            ('store.rs2',self.command[2])),ordered=True,pure_write=True)
        self.source=prepared.source.identity
        self.omitted_sites=0;self.store_sites=0
        # Caller has just drained the previous invocation. This exact setter
        # precedes every later MVOUT, including all counted-loop iterations.
        ctx.emit(*self.command)

    def __getattr__(self,name):return getattr(self.ctx,name)

    def emit(self,*command):
        if len(command)==3 and all(type(v)is int for v in command)and tuple(command)==self.command:
            self.omitted_sites+=1
            return
        # Fail closed if the common helper ever selects another store setting.
        # Such a change invalidates this invocation-wide uniform-state proof.
        if (len(command)==3 and command[0]==isa.FUNCT['config']
                and type(command[1])is int and command[1]&3==2):
            raise ValueError('readout helper changed the selected store configuration')
        if command[0]==isa.FUNCT['mvout']:self.store_sites+=1
        self.ctx.emit(*command)

    def receipt(self):
        return dict(mechanism='uniform_convolution_readout_configuration_reuse',
            source_sha256=self.source,configuration=list(self.command),
            effects=asdict(self.effects),
            hardware_identity='b8ff28edc096392cec356f9fececdf17ba70cd3e4b02e6c5931aedc43bb117af',
            state_fields='literal StoreController stride/scale/activation; pooling fields all zero',
            state_lifetime='entry full drain through private convolution final full drain',
            initial_state='unknown; actual literal setter emitted after entry drain',
            preservation='only load/execute configuration between readouts; no pooling or column-varying scale',
            completion='all original common-helper before-config and after-store fences retained',
            quantization='original store_tile unchanged; exact matching literal configuration only',
            omitted_static_configuration_sites=self.omitted_sites,retained_static_store_sites=self.store_sites,
            counts='static emission sites, not runtime counts or timing evidence',timing='UNKNOWN')
