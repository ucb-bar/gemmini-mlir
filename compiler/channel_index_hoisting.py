"""Optional counted-convolution address LICM, before target code generation.

Only operations emitted by ConvolutionAddressBuilder._cached are candidates.
No tensor access, configuration, inline assembly, fence, allocation, loop bound
or branch is moved. Existing source range simplification remains authoritative.
"""
from dataclasses import dataclass, field
from pathlib import Path
import re
from conv_addressing import ConvolutionAddressBuilder
from merlin.compile.tuning_knobs import knob

VALUES=('off','loop_invariants')
DECLARATION=dict(values=list(VALUES), mechanism='dominating counted-loop preheader reuse of total pure convolution address expressions', note='Default off. Preserves exact target command order and source arithmetic; no performance claim without paired timing.')


def selected():
    return knob('channel_index_hoisting',default='off',values=VALUES,root=Path(__file__).parent)


@dataclass
class _Scope:
    marker: str
    block: str
    available: set
    lines: list=field(default_factory=list)
    cache: dict=field(default_factory=dict)


class HoistedConvolutionAddressBuilder(ConvolutionAddressBuilder):
    """Lift expressions only when every SSA operand dominates a preheader.

    Availability consists of arguments/current-block definitions before the
    outermost loop, enclosing loop indices, and earlier explicitly lifted pure
    definitions. Unknown SSA producers conservatively prevent motion. Each
    lifted definition is inserted before the original loop's entry branch.
    The counted CFG, induction variables and all effects remain untouched.
    """
    def __init__(self,builder):
        super().__init__(builder)
        self.scopes=[]
        self.loop_indices=[]
        self._serial=0
        self.stats=dict(moved_expressions=0,moved_lines=0,reused_expressions=0)

    def _preheader_values(self):
        values=set(getattr(self.builder,'args',{}).values())
        values.update(self.loop_indices)
        # Definitions in the current real basic block dominate its terminator.
        # Do not assume definitions in earlier branch blocks dominate this one.
        for line in reversed(self.builder.lines):
            if line.lstrip().startswith('^'):break
            match=re.match(r'\s*(%[A-Za-z0-9_.$-]+)\s*=',line)
            if match:values.add(match.group(1))
        for scope in self.scopes:values.update(scope.available)
        return values

    @staticmethod
    def _total(key):
        if key[0] in ('cmp','select','ptr'):return True
        if key[0]!='op':return False
        _,op,a,b,ty=key
        if op in ('add','sub','mul','and','or','xor'):return True
        if op in ('udiv','urem'):return isinstance(b,int) and 0<b<2**64
        if op in ('shl','lshr'):return isinstance(b,int) and 0<=b<int(ty[1:])
        return False

    def _cached(self,key,create):
        if not self.scopes or not self._total(key):return super()._cached(key,create)
        dependencies={x for x in key if isinstance(x,str) and x.startswith('%')}
        destination=next((s for s in self.scopes if dependencies<=s.available),None)
        if destination is None:return super()._cached(key,create)
        # Keep CSE under the same source owner as the original builder.
        ownership=(getattr(self.builder,'task_index',None),getattr(self.builder,'source_op_index',None),getattr(self.builder,'region_id',None))
        cachekey=(ownership,key)
        if cachekey in destination.cache:
            self.stats['reused_expressions']+=1
            return destination.cache[cachekey]
        start=len(self.builder.lines);block=self.builder.current_block
        simplification=getattr(self.builder,'address_simplification',None)
        try:
            # Existing per-block CSE may refer to a bound constant emitted
            # after our insertion marker. Fresh constants avoid forward uses.
            if simplification is not None:self.builder.address_simplification='off'
            self.builder.current_block=destination.block
            value=create()
        finally:
            self.builder.current_block=block
            if simplification is not None:self.builder.address_simplification=simplification
        emitted=self.builder.lines[start:]
        del self.builder.lines[start:]
        # create is an existing pure arithmetic/address encoder, not a target
        # callback. Enforce the boundary before committing any moved text.
        allowed=('llvm.mlir.constant','llvm.add','llvm.sub','llvm.mul','llvm.and','llvm.or','llvm.xor','llvm.udiv','llvm.urem','llvm.shl','llvm.lshr','llvm.icmp','llvm.select','llvm.getelementptr')
        if any(line.split(' = ',1)[-1].split(' ',1)[0].split('(',1)[0].strip(chr(34)) not in allowed for line in emitted):
            raise ValueError('non-pure operation in convolution address hoisting')
        destination.lines.extend(emitted);destination.cache[cachekey]=value
        # Every emitted value now dominates this loop and its descendants.
        added={m.group(1) for line in emitted if (m:=re.match(r'\s*(%[A-Za-z0-9_.$-]+)\s*=',line))}
        if isinstance(value,str):added.add(value)
        at=self.scopes.index(destination)
        for scope in self.scopes[at:]:scope.available.update(added)
        self.stats['moved_expressions']+=bool(emitted)
        self.stats['moved_lines']+=len(emitted)
        return value

    def loop(self,start,stop,step,callback):
        if all(isinstance(x,int) for x in (start,stop,step)):
            if step<=0:raise ValueError('convolution coordinate loop requires positive step')
            if stop<=start:return
            if start+step>=stop:callback(start);return
        # Non-textual interpreters retain precisely the existing finite effects.
        if not hasattr(self.builder,'lines') or not hasattr(self.builder,'types'):
            return super().loop(start,stop,step,callback)
        self._serial+=1;marker=f'    // channel_index_preheader_{id(self)}_{self._serial}'
        scope=_Scope(marker,self.builder.current_block,self._preheader_values())
        self.builder.lines.append(marker);self.scopes.append(scope)
        def body(index):
            # The induction value is available before every strictly nested
            # loop, but never before its own defining loop.
            if all(isinstance(x,int) for x in (start,stop,step)) and 0<=start<stop<2**63:
                self.ranges[index]=(start,stop-1)
            self.loop_indices.append(index)
            try:callback(index)
            finally:self.loop_indices.pop()
        try:self.builder.loop(start,stop,step,body)
        finally:
            self.scopes.pop()
            where=self.builder.lines.index(marker)
            self.builder.lines[where:where+1]=scope.lines


def wrap(builder,*,mode=None):
    mode=selected() if mode is None else mode
    if mode not in VALUES:raise ValueError('unknown channel index hoisting mode')
    if mode=='off':return ConvolutionAddressBuilder(builder)
    # Concrete projection executes original operations with integer indices;
    # it needs no textual SSA availability transformation.
    if not hasattr(builder,'lines') or not hasattr(builder,'types'):
        return ConvolutionAddressBuilder(builder)
    return HoistedConvolutionAddressBuilder(builder)
