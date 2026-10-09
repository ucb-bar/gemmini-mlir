"""Draft opt-in typed source FMA packet analysis; contains no CPU ISA emission."""
from dataclasses import dataclass
import hashlib
from merlin.llvmlower.late_quant_rne import _tokens,_functions,_statement_boundary,_identity
from merlin.llvmlower.constant_float_clamp import _finite_f32,_ssa_key,_has_environment_scope

@dataclass(frozen=True)
class FmaCall:
    result:str
    operands:tuple[str,str,str]
    start:int
    end:int

@dataclass(frozen=True)
class ConstantFmaPacket:
    calls:tuple[FmaCall,...]
    mode:str
    constant_words:tuple[int,...]
    source_function:str
    source_block:str
    source_sha256:str


def _call(tokens,i):
    t=[v.text for v in tokens[i:i+15]]
    if len(t)!=15 or not t[0].startswith('%') or t[1:4]!=['=','call','float']:
        return None
    if (_identity(t[4])!='llvm.fma.f32' or not t[4].startswith('@')
        or t[5]!='(' or t[6]!=t[9] or t[6]!=t[12] or t[6]!='float'
        or t[8]!=',' or t[11]!=',' or t[14]!=')'
        or not _statement_boundary(tokens,i+15)):
        return None
    return FmaCall(t[0],(t[7],t[10],t[13]),tokens[i].start,tokens[i+14].end)


def analyze(text,*,width=4,ordinary_nontrapping=False,exception_flags_unobserved=False):
    """Inspect exact consecutive independent scalar FMAs; source stays unchanged.

    The source LLVM module must be verified at the normal compile seam. All
    dynamic operands must be previously defined in this block or typed float
    formal parameters; no other instruction, memory operation, label or call is
    crossed. Finite nonzero constants use the existing integer/RNE decoder.
    Unknown instruction flags, call attributes and metadata refuse the group.
    """
    if width not in (2,4):raise ValueError('supported explicit widths are2and4')
    if not ordinary_nontrapping or not exception_flags_unobserved:return ()
    tokens=_tokens(text)
    if _has_environment_scope(tokens):return ()
    for i,token in enumerate(tokens):
        identity=_identity(token.text)
        words=''.join(c if c.isalnum()or c in '_.'else' 'for c in identity).split()
        if 'frm'in words:return ()
        if identity in ('unsafe-fp-math','no-nans-fp-math','no-infs-fp-math','no-signed-zeros-fp-math')and i+2<len(tokens)and tokens[i+1].text=='='and _identity(tokens[i+2].text)=='true':return ()
        if identity.startswith('denormal-fp-math')and i+2<len(tokens)and tokens[i+1].text=='='and _identity(tokens[i+2].text)!='ieee,ieee':return ()
    digest=hashlib.sha256(text.encode()).hexdigest();packets=[]
    for body in _functions(tokens):
        if not body:continue
        first=next(i for i,v in enumerate(tokens)if v.start==body[0].start)
        define=max(i for i,v in enumerate(tokens[:first])if v.text=='define')
        header=tokens[define:first];symbol=next(i for i,v in enumerate(header)if v.text.startswith('@'))
        name=_identity(header[symbol].text)
        # Only explicit float formal operands are accepted; other types/attrs
        # are not guessed. Most tensor-body operands are SSA loads in the block.
        arguments=set();j=symbol+2;depth=1
        while j<len(header) and depth:
            depth+=(header[j].text=='(')-(header[j].text==')')
            if j+1<len(header) and header[j].text=='float' and header[j+1].text.startswith('%'):
                arguments.add(_ssa_key(header[j+1].text))
            j+=1
        defined=set(arguments);block='entry';i=0
        while i<len(body):
            if i+1<len(body) and body[i+1].text==':':
                block=body[i].text;defined=set(arguments);i+=2;continue
            group=tuple(_call(body,i+15*k) for k in range(width))
            if all(group):
                results={_ssa_key(c.result)for c in group}
                values=tuple(tuple(c.operands[k]for c in group)for k in range(3))
                variable=tuple(all(v.startswith('%')and _ssa_key(v)in defined and _ssa_key(v)not in results for v in vs)for vs in values)
                const=[]
                for vs in values:
                    decoded=tuple(_finite_f32(v)for v in vs)
                    const.append(decoded[0][0]if all(decoded)and len({v[0]for v in decoded})==1 else None)
                mode='';words=()
                if const[1]is not None and variable[0]and variable[2]:mode='constant_rhs';words=(const[1],)
                elif const[0]is not None and const[2]is not None and variable[1]:mode='constant_lhs_addend';words=(const[0],const[2])
                elif const[2]is not None and variable[0]and variable[1]:mode='constant_addend';words=(const[2],)
                if mode:
                    packets.append(ConstantFmaPacket(group,mode,words,name,block,digest));defined.update(results);i+=15*width;continue
            if i+1<len(body)and body[i].text.startswith('%')and body[i+1].text=='=':
                defined.add(_ssa_key(body[i].text))
            i+=1
    return tuple(packets)


def rewrite(text,*,emitter=None,width=4,ordinary_nontrapping=False,exception_flags_unobserved=False,temporary_prefix='constant.fma.packet'):
    report=dict(schema='explicit_constant_fma_packet_v1',source_sha256=hashlib.sha256(text.encode()).hexdigest(),width=width,packets=[],default_unchanged=emitter is None)
    if emitter is None:return text,report
    if not temporary_prefix or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.$'for c in temporary_prefix)or temporary_prefix[0].isdigit():raise ValueError('invalid generatedSSA prefix')
    packets=analyze(text,width=width,ordinary_nontrapping=ordinary_nontrapping,exception_flags_unobserved=exception_flags_unobserved)
    reserved={_ssa_key(t.text)for t in _tokens(text)if t.text.startswith('%')};edits=[];counter=0
    for packet in packets:
        while _ssa_key('%'+temporary_prefix+'.'+str(counter))in reserved:counter+=1
        temp='%'+temporary_prefix+'.'+str(counter);reserved.add(_ssa_key(temp));counter+=1
        replacement=emitter(packet,temp)
        if not isinstance(replacement,str)or not replacement:raise ValueError('emitter must return LLVM instructions for exact originalresults')
        edits.append((packet.calls[0].start,packet.calls[-1].end,replacement))
        report['packets'].append(dict(source_function=packet.source_function,source_block=packet.source_block,results=[c.result for c in packet.calls],operands=[c.operands for c in packet.calls],mode=packet.mode,constant_words=packet.constant_words,source_span=[packet.calls[0].start,packet.calls[-1].end],source_sha256=packet.source_sha256))
    for start,end,replacement in reversed(edits):text=text[:start]+replacement+text[end:]
    report['selected_sha256']=hashlib.sha256(text.encode()).hexdigest();return text,report
