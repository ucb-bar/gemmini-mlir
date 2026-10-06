"""Draft explicit OOT RV64GC scalar-FMA instruction implementation."""
from dataclasses import dataclass

@dataclass(frozen=True)
class RV64GCScalarFmaCapability:
    isa:str='rv64gc'
    abi:str='lp64d'
    ieee_f32_fma:bool=True
    ieee_gradual_underflow:bool=True

    def emit(self,packet,temporary):
        if(self.isa,self.abi,self.ieee_f32_fma,self.ieee_gradual_underflow)!=('rv64gc','lp64d',True,True):raise ValueError('explicit RV64GC IEEE scalarFMA capability required')
        n=len(packet.calls)
        if n not in (2,4):raise ValueError('supported resource packets have2or4lanes')
        lhs=[c.operands[0]for c in packet.calls];rhs=[c.operands[1]for c in packet.calls];add=[c.operands[2]for c in packet.calls]
        ty='{ '+', '.join(['float']*n)+' }';outputs=['=&f']*n
        if packet.mode=='constant_rhs':
            params=[*(f'float {v}'for v in lhs),f'i32 {packet.constant_words[0]}',*(f'float {v}'for v in add)]
            asm=f'fmv.w.x ft0,${2*n}'+''.join(f'\\0Afmadd.s ${j},${j+n},ft0,${j}'for j in range(n));constraints=[*outputs,*['f']*n,'r',*(str(j)for j in range(n)),'~{ft0}']
        elif packet.mode=='constant_addend':
            params=[*(f'float {v}'for v in rhs),f'i32 {packet.constant_words[0]}',*(f'float {v}'for v in lhs)]
            asm=f'fmv.w.x ft0,${2*n}'+''.join(f'\\0Afmadd.s ${j},${j},${j+n},ft0'for j in range(n));constraints=[*outputs,*['f']*n,'r',*(str(j)for j in range(n)),'~{ft0}']
        elif packet.mode=='constant_lhs_addend':
            params=[*(f'i32 {v}'for v in packet.constant_words),*(f'float {v}'for v in rhs)]
            asm=f'fmv.w.x ft0,${n}\\0Afmv.w.x ft1,${n+1}'+''.join(f'\\0Afmadd.s ${j},ft0,${j},ft1'for j in range(n));constraints=[*outputs,'r','r',*(str(j)for j in range(n)),'~{ft0}','~{ft1}']
        else:raise ValueError('unsupported source-proven constantFMA mode')
        lines=[f'{temporary} = call {ty} asm "{asm}", "'+','.join(constraints)+'"('+', '.join(params)+')']
        lines.extend(f'  {c.result} = extractvalue {ty} {temporary}, {j}'for j,c in enumerate(packet.calls))
        return '\n'.join(lines)
