"""Block-local terminal configuration cleanup using the shared schedule pass.

Integration: call retain_instruction(builder, funct, rs1, rs2) in FunctionBuilder
insn BEFORE materializing operands, and return immediately when it is false.
Only this hook may omit an instruction. It must cover every terminal .insn.
Unknown inline assembly/fences are detected in the intervening emitted lines.
The optional across_fence mode recognizes only the encoder's plain CPU fence.

The qualification is deliberately narrow: normal full EX, LD[0..2], and nonpool
ST commands write persistent controller configuration in their ordered command
queues. LOAD/STORE/PRELOAD/COMPUTE consume those registers without changing them.
No claim is made for pooling setup, partial EX writes, normalization, flush,
unknown commands, cross-block propagation, source-owner changes, or dynamic
configuration operands. Those retain their commands and forget known state.
"""
from pathlib import Path

from merlin.compile.scheduling.configuration import (
    ConfigurationAdapter, ConfigurationEffects, eliminate_redundant_configuration,
)
from merlin.sched.ir import Call, Const, Kernel


def _project(call):
    """Decode actual packed static words; no candidate sets/assumes are trusted."""
    if call.instr=='cpu_fence':
        # Ordering effect retained in the program, without a controller write.
        return ConfigurationEffects(ordered=True)
    args={key:value.value for key,value in call.args}
    funct=args['funct']
    if funct in (1,2,3,4,5,6,14):
        return ConfigurationEffects(ordered=True)
    if funct!=0 or args['static_mask']!=3:
        return None
    rs1=args['rs1'];rs2=args['rs2'];kind=rs1&3
    if kind==0:
        # set_only_strides is a partial write: retain without propagating it.
        if rs1&(1<<7): return None
        bank='ex'
    elif kind==1:
        state=(rs1>>3)&3
        if state>2: return None
        bank=f'ld{state}'
    elif kind==2:
        # Pool configuration can carry additional controller state; excluded.
        if rs1&~15: return None
        bank='st'
    else:
        return None
    return ConfigurationEffects(writes=((bank+'.rs1',rs1),(bank+'.rs2',rs2)),
                                ordered=True,pure_write=True)


_ADAPTER=ConfigurationAdapter(_project,
    'gemmini.explicit-packed-controller-configuration.v1',
    'gemmini.ordered-full-ex-ld-nonpool-st.persistent-registers.v1')

_FENCE_ADAPTER=ConfigurationAdapter(_project,_ADAPTER.configuration_identity,
    _ADAPTER.qualification_identity+'.cpu-fence-no-rocc-write.v1')


def _plain_cpu_fence(line):
    """Recognize only FunctionBuilder.fence's zero-operand asm, with attributes.

    The selected facts interfaces.funct_decode_table.custom_opcode is 0x7b;
    CONFIG register writes enter that RoCC dispatch path. FunctionBuilder.fence
    emits the ordinary RISC-V `fence`, not a custom instruction or reset. Thus
    this fence orders completion without overwriting persistent configuration.
    This reasoning does not admit FLUSH (custom funct7) or arbitrary asm.
    """
    line=line.strip()
    prefix='"llvm.inline_asm"() <{asm_string = "fence", constraints = "~{memory}", has_side_effects}>'
    return line.startswith(prefix) and line.endswith(': () -> ()')


class TerminalConfigurationCSE:
    """Bounded straight-block adapter; returned decisions reach real emission."""
    def __init__(self,builder,policy=None,max_calls=256):
        if policy is None:
            from merlin.compile.tuning_knobs import knob
            policy=knob('configuration_cse',default='off',values=('off','block','across_fence'),
                        root=Path(__file__).resolve().parent)
        if policy not in ('off','block','across_fence'): raise ValueError('invalid configuration_cse policy')
        if type(max_calls) is not int or max_calls<1: raise ValueError('positive max_calls required')
        self.policy=policy;self.max_calls=max_calls
        self.adapter=_FENCE_ADAPTER if policy=='across_fence' else _ADAPTER
        self.calls=[];self.owner=None;self.cursor=len(builder.lines)
        self.pending=None;self.ordinal=0
        self.receipt={'schema':'gemmini.terminal_configuration_cse.v1','policy':policy,
                      'source_commands':0,'retained_commands':0,'removed_configurations':0,
                      'state_resets':0,'adapter':self.adapter.record(),'last_rewrite':None}
        if policy=='across_fence': self.receipt['preserved_cpu_fences']=0

    def _reset(self):
        self.calls=[];self.receipt['state_resets']+=1

    def retain(self,builder,funct,rs1,rs2):
        self.receipt['source_commands']+=1
        if self.policy=='off':
            self.receipt['retained_commands']+=1
            return True
        # Recognize only the single asm emitted by the preceding approved hook.
        # Across-fence mode keeps a recognized fence as a retained effect in
        # the checked stream. It never removes or moves the real emitted fence.
        for line in builder.lines[self.cursor:]:
            if 'llvm.inline_asm' in line:
                if self.pending is not None and self.pending in line:
                    self.pending=None
                elif self.policy=='across_fence' and _plain_cpu_fence(line):
                    if len(self.calls)>=self.max_calls: self._reset()
                    self.ordinal+=1
                    self.calls.append(Call('cpu_fence',(('ordinal',Const(self.ordinal)),)))
                    self.receipt['preserved_cpu_fences']+=1
                else:
                    self._reset()
        self.pending=None;self.cursor=len(builder.lines)
        owner=(builder.current_block,builder.task_index,builder.source_op_index,builder.region_id)
        if owner!=self.owner or len(self.calls)>=self.max_calls:
            self._reset();self.owner=owner
        self.ordinal+=1
        static_mask=int(isinstance(rs1,int))|int(isinstance(rs2,int))<<1
        args=[('funct',Const(funct)),('static_mask',Const(static_mask)),('ordinal',Const(self.ordinal))]
        if static_mask&1: args.append(('rs1',Const(rs1&0xffffffffffffffff)))
        if static_mask&2: args.append(('rs2',Const(rs2&0xffffffffffffffff)))
        call=Call('terminal',tuple(args));self.calls.append(call)
        result=eliminate_redundant_configuration(
            Kernel('terminal_block',(),tuple(self.calls)),self.adapter,
            enabled=True,max_calls=self.max_calls,max_fields=16)
        retain=call in result.kernel.body
        self.receipt['last_rewrite']=result.report
        if retain:
            self.receipt['retained_commands']+=1
            self.pending=f'.insn r 0x7b, 3, {funct}, x0'
        else:
            self.receipt['removed_configurations']+=1
        return retain


def retain_instruction(builder,funct,rs1,rs2):
    """Minimal pre-materialization FunctionBuilder.insn hook; false means omit."""
    state=getattr(builder,'_terminal_configuration_cse',None)
    if state is None:
        state=TerminalConfigurationCSE(builder)
        builder._terminal_configuration_cse=state
        builder.configuration_cse_receipt=state.receipt
    return state.retain(builder,funct,rs1,rs2)
