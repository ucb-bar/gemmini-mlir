"""Gemmini terminal command encoding and LLVM address/control construction.

Field positions follow the selected register_bundle_layouts RTL facts. Only
explicit commands are supported; hardware loop-descriptor commands are refused.
Payload arithmetic belongs to the accelerator, while this builder emits host
address arithmetic and counted control flow.
"""
from contextlib import contextmanager
import struct
import json
from pathlib import Path

DIM = 16
ACC = 1 << 31
ACCUMULATE = 1 << 30
FULL = 1 << 29
GARBAGE = 0xffffffff
CONFIG, LOAD2, LOAD, STORE, COMPUTE_FLIP, COMPUTE_STAY, PRELOAD, FLUSH, LOAD3 = 0,1,2,3,4,5,6,7,14

def f32_bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]

def _literal_period(values):
    """Length of the shortest exact repeating prefix, including a ragged tail."""
    borders=[0]*len(values)
    for i in range(1,len(values)):
        j=borders[i-1]
        while j and values[i]!=values[j]: j=borders[j-1]
        if values[i]==values[j]: j+=1
        borders[i]=j
    return len(values)-borders[-1]

def pack_matrix(addr, cols=16, rows=16):
    """Pack a static local matrix descriptor, without truncating bad fields."""
    for value, width in ((addr,32),(cols,16),(rows,16)):
        if not isinstance(value,int) or not 0 <= value < 1 << width:
            raise ValueError('matrix descriptor field out of range')
    return addr | (cols << 32) | (rows << 48)

def encode_config_ex(dataflow=1, activation=0, shift=0, scale=1.0, c_stride=1,
                     a_stride=1, a_transpose=False, b_transpose=False,
                     set_only_strides=False, scale_bits=None):
    bits = f32_bits(scale) if scale_bits is None else scale_bits
    return (bits << 32) | (a_stride << 16) | (int(b_transpose)<<9) | (int(a_transpose)<<8) | (int(set_only_strides)<<7) | (activation<<3) | (dataflow<<2), (c_stride<<48) | shift

def encode_config_ld(stride, scale=1.0, shrink=False, block_stride=16,
                     pixel_repeats=1, id=0, scale_bits=None):
    bits = f32_bits(scale) if scale_bits is None else scale_bits
    return (bits<<32) | (block_stride<<16) | (pixel_repeats<<8) | (id<<3) | (int(shrink)<<2) | 1, stride

def encode_config_st(stride, activation=0, scale=1.0, scale_bits=None, **pool):
    bits = f32_bits(scale) if scale_bits is None else scale_bits
    fields={'pool_stride':4,'pool_size':6,'upad':8,'lpad':10,'pool_out_dim':24,'porows':32,'pocols':40,'orows':48,'ocols':56}
    unknown=set(pool)-set(fields)
    if unknown: raise ValueError(f'unknown pooling fields: {unknown}')
    return 2 | (activation<<2) | sum(pool.get(k,0)<<s for k,s in fields.items()), (bits<<32) | stride

class FunctionBuilder:
    """Textual LLVM dialect builder with symbolic i64 address expressions.

    SSA values are strings including their percent prefix. Optional local CSE
    folds integer identities and reuses pure expressions within one block and
    source owner. Every target command has a memory clobber and side effects.
    Counted loops preserve compact code size.
    """
    def __init__(self, arg_names=(), symbol='gemmini_kernel'):
        self.symbol=symbol
        self.args={name: '%'+name for name in arg_names}
        self.lines=[]
        self.globals=[]
        self.serial=0
        self.label_serial=0
        self.current_block='entry'
        self.types={v:'!llvm.ptr' for v in self.args.values()}
        self.task_index=None
        self.source_op_index=None
        self.region_id=None
        from merlin.compile.tuning_knobs import knob
        self.address_simplification=knob('address_simplification',default='off',
            values=('off','local_cse'),root=Path(__file__).resolve().parent)
        self.literal_dispatch_policy=knob('literal_dispatch_policy',default='off',
            values=('off','periodic'),root=Path(__file__).resolve().parent)
        self.literal_dispatch_receipts=[]
        self._pure_values={}
        self._integer_literals={}
    def temp(self):
        self.serial+=1
        return f'%v{self.serial}'
    @contextmanager
    def task_scope(self,task_index=None,source_op_index=None,region_id=None):
        old=(self.task_index,self.source_op_index,self.region_id)
        self.task_index,self.source_op_index,self.region_id=task_index,source_op_index,region_id
        try: yield self
        finally: self.task_index,self.source_op_index,self.region_id=old
    def metadata(self):
        attrs=[]
        if self.task_index is not None: attrs.append(f'merlin.global_task = {int(self.task_index)} : i64')
        if self.source_op_index is not None: attrs.append(f'merlin.source_op_index = {int(self.source_op_index)} : i64')
        if self.region_id is not None: attrs.append('prov.region_id = '+json.dumps(str(self.region_id)))
        return ', '.join(attrs)
    def emit(self, text):
        attrs=self.metadata()
        if attrs:
            # Branch attribute dictionaries follow successors, while typed
            # operations place their dictionaries immediately before types.
            existing=text.find(' {')
            if existing>=0:
                text=text[:existing+2]+attrs+', '+text[existing+2:]
            elif text.startswith(('llvm.br ','llvm.cond_br ')):
                text+=' {'+attrs+'}'
            else:
                where=text.rfind(' : ')
                if where<0: raise ValueError('cannot attach source provenance')
                text=text[:where]+' {'+attrs+'}'+text[where:]
        self.lines.append('    '+text)
    def result(self,text,ty='i64'):
        # Exact expression CSE is local to a real CFG block and one source
        # owner. No memory observation, allocation or device effect enters it.
        opcode=text.split(' ',1)[0].split('(',1)[0].strip('"')
        pure=opcode in {
            'llvm.mlir.constant','llvm.add','llvm.sub','llvm.mul','llvm.shl',
            'llvm.lshr','llvm.and','llvm.or','llvm.udiv','llvm.urem','llvm.icmp',
            'llvm.select','llvm.getelementptr','llvm.ptrtoint','llvm.inttoptr',
            'llvm.zext','llvm.sext','llvm.trunc','llvm.bitcast','llvm.mlir.addressof'}
        key=(self.current_block,self.task_index,self.source_op_index,
             self.region_id,ty,text)
        enabled=self.address_simplification=='local_cse' and pure
        if enabled and key in self._pure_values: return self._pure_values[key]
        v=self.temp(); self.emit(f'{v} = {text}'); self.types[v]=ty
        if enabled: self._pure_values[key]=v
        return v
    def const(self,value,ty='i64'):
        if ty.startswith('i'):
            bits=int(ty[1:]); value=int(value)
            if value >= 1 << (bits-1): value-=1<<bits
        result=self.result(f'"llvm.mlir.constant"() <{{value = {value} : {ty}}}> : () -> {ty}',ty)
        if ty.startswith('i'): self._integer_literals[result]=(int(value),ty)
        return result
    def value(self,x,ty='i64'): return self.const(x,ty) if isinstance(x,(int,bool)) else x
    def binary(self,op,a,b,ty='i64'):
        if self.address_simplification=='local_cse':
            def literal(x):
                if isinstance(x,int): return x
                value=self._integer_literals.get(x)
                return value[0] if value is not None and value[1]==ty else None
            av,bv=literal(a),literal(b)
            if op in ('add','or'):
                if av==0:return b
                if bv==0:return a
            if op=='sub' and bv==0:return a
            if op=='mul':
                if av==1:return b
                if bv==1:return a
                if av==0 or bv==0:return 0
            if op=='and':
                if av==0 or bv==0:return 0
                if av==-1:return b
                if bv==-1:return a
            if op in ('shl','lshr') and bv==0:return a
            if op=='udiv' and bv==1:return a
        if isinstance(a,int) and isinstance(b,int):
            f={'add':lambda:a+b,'sub':lambda:a-b,'mul':lambda:a*b,'shl':lambda:a<<b,'lshr':lambda:(a&((1<<64)-1))>>b,'and':lambda:a&b,'or':lambda:a|b,'udiv':lambda:a//b,'urem':lambda:a%b}
            if op in f:return f[op]()
        return self.result(f'llvm.{op} {self.value(a,ty)}, {self.value(b,ty)} : {ty}',ty)
    def add(self,a,b,ty='i64'): return self.binary('add',a,b,ty)
    def sub(self,a,b,ty='i64'): return self.binary('sub',a,b,ty)
    def mul(self,a,b,ty='i64'): return self.binary('mul',a,b,ty)
    def shl(self,a,b,ty='i64'): return self.binary('shl',a,b,ty)
    def lshr(self,a,b,ty='i64'): return self.binary('lshr',a,b,ty)
    def and_(self,a,b,ty='i64'): return self.binary('and',a,b,ty)
    def or_(self,a,b,ty='i64'): return self.binary('or',a,b,ty)
    def udiv(self,a,b,ty='i64'): return self.binary('udiv',a,b,ty)
    def urem(self,a,b,ty='i64'): return self.binary('urem',a,b,ty)
    def icmp(self,pred,a,b,ty='i64'):
        return self.result(f'llvm.icmp "{pred}" {self.value(a,ty)}, {self.value(b,ty)} : {ty}','i1')
    def cmp(self,pred,a,b,ty='i64'): return self.icmp(pred,a,b,ty)
    def op(self,name,a,b,typ='i64'): return self.binary(name,a,b,typ)
    def select(self,c,a,b,ty='i64'):
        return self.result(f'llvm.select {self.value(c,"i1")}, {self.value(a,ty)}, {self.value(b,ty)} : i1, {ty}',ty)
    def umin(self,a,b): return min(a,b) if isinstance(a,int) and isinstance(b,int) else self.select(self.icmp('ult',a,b),a,b)
    def ptr_offset(self,ptr,byte_offset):
        if byte_offset==0:return ptr
        return self.result(f'llvm.getelementptr {ptr}[{self.value(byte_offset)}] : (!llvm.ptr, i64) -> !llvm.ptr, i8','!llvm.ptr')
    def ptrtoint(self,ptr): return self.result(f'llvm.ptrtoint {ptr} : !llvm.ptr to i64')
    def inttoptr(self,value): return self.result(f'llvm.inttoptr {self.value(value)} : i64 to !llvm.ptr','!llvm.ptr')
    def global_buffer(self,count,ty='i8',alignment=64):
        if not isinstance(count,int) or count<=0: raise ValueError('static positive buffer size required')
        widths={'i8':1,'i16':2,'i32':4,'i64':8,'f32':4,'f64':8}
        if ty not in widths: raise ValueError('unsupported global buffer element type')
        size=count*widths[ty]
        name=f'private_buffer_{len(self.globals)}'
        self.globals.append(f'  llvm.mlir.global internal @{name}(dense<0> : tensor<{size}xi8>) {{alignment = {alignment} : i64}} : !llvm.array<{size} x i8>')
        return self.result(f'llvm.mlir.addressof @{name} : !llvm.ptr','!llvm.ptr')
    def constant_i32_table(self,values):
        if not values: raise ValueError('constant table must not be empty')
        name=f'constant_table_{len(self.globals)}'
        payload=b''.join(struct.pack('<I',int(x)&0xffffffff) for x in values)
        literal=''.join('\\'+format(byte,'02X') for byte in payload)
        self.globals.append(f'  llvm.mlir.global private constant @{name}("{literal}") {{alignment = 64 : i64}} : !llvm.array<{len(payload)} x i8>')
        return self.result(f'llvm.mlir.addressof @{name} : !llvm.ptr','!llvm.ptr')
    def constant_i32_lookup(self,values,index):
        """Select immutable 32-bit command bits with a balanced CFG, as i64.

        This is command construction, not a tensor lookup: no memory is read.
        The caller must establish 0 <= index < len(values). Exactly one literal
        reaches the merge at runtime. Source/task ownership follows this scope.
        """
        if not values: raise ValueError('constant lookup must not be empty')
        values=tuple(int(value)&0xffffffff for value in values)
        if isinstance(index,int):
            if not 0<=index<len(values): raise ValueError('constant lookup index out of range')
            return self.value(values[index])
        if len(set(values))==1: return self.value(values[0])
        self.label_serial+=1; identity=self.label_serial
        merge=f'lookup_merge{identity}'
        def tree(start,stop):
            if stop-start==1 or len(set(values[start:stop]))==1:
                value=self.value(values[start])
                self.emit(f'llvm.br ^{merge}({value} : i64)')
                return
            middle=(start+stop)//2
            self.label_serial+=1; n=self.label_serial
            left=f'lookup_left{n}'; right=f'lookup_right{n}'
            cond=self.icmp('ult',index,middle)
            self.emit(f'llvm.cond_br {cond}, ^{left}, ^{right}')
            self.lines.append(f'  ^{left}:'); self.current_block=left
            tree(start,middle)
            self.lines.append(f'  ^{right}:'); self.current_block=right
            tree(middle,stop)
        tree(0,len(values))
        result=self.temp(); self.types[result]='i64'
        self.lines.append(f'  ^{merge}({result}: i64):'); self.current_block=merge
        return result
    def constant_i32_dispatch(self,values,index,callback):
        """Execute callback once with the selected literal uint32 command bits.

        The caller establishes 0 <= index < len(values). A balanced branch tree
        exposes literals directly to each callback; no table load or result phi
        obscures the command's static operands. Uniform subtrees share a leaf.
        Callback emissions retain the current source/task ownership and must
        fall through normally, as for ``branch`` callbacks.

        With the opt-in periodic policy, exact repeated uint32 bits use their
        shortest prefix and an index remainder (also for a partial last period).
        This relies on the same in-bounds precondition: dynamic invalid indices
        have no defined selection, while static invalid indices raise ValueError.
        """
        values=tuple(int(value)&0xffffffff for value in values)
        if not values: raise ValueError('constant dispatch must not be empty')
        if isinstance(index,int):
            if not 0<=index<len(values): raise ValueError('constant dispatch index out of range')
            callback(values[index])
            return
        count=len(values)
        reduction='none'
        if self.literal_dispatch_policy=='periodic':
            period=_literal_period(values)
            if period<count:
                values=values[:period]
                if period>1:
                    if period&(period-1)==0:
                        index=self.and_(index,period-1); reduction='and'
                    else:
                        index=self.urem(index,period); reduction='urem'
        self.literal_dispatch_receipts.append({
            'policy':self.literal_dispatch_policy,'input_values':count,
            'dispatch_values':len(values),'index_reduction':reduction,
            'global_task':self.task_index,'source_op_index':self.source_op_index,
            'region_id':self.region_id})
        def tree(start,stop):
            if stop-start==1 or len(set(values[start:stop]))==1:
                callback(values[start])
                return
            middle=(start+stop)//2
            self.branch(self.icmp('ult',index,middle),
                        lambda:tree(start,middle),lambda:tree(middle,stop))
        tree(0,len(values))
    def load(self,ptr,ty='i8'):
        return self.result(f'llvm.load {ptr} : !llvm.ptr -> {ty}',ty)
    def store(self,value,ptr,ty='i8'): self.emit(f'llvm.store {self.value(value,ty)}, {ptr} : {ty}, !llvm.ptr')
    def alloca(self,count,ty='i8',alignment=64):
        return self.result(f'llvm.alloca {self.value(count)} x {ty} {{alignment = {alignment} : i64}} : (i64) -> !llvm.ptr','!llvm.ptr')
    def cast(self,value,from_ty,to_ty,op='zext'):
        return self.result(f'llvm.{op} {self.value(value,from_ty)} : {from_ty} to {to_ty}',to_ty)
    @contextmanager
    def counted_loop(self,start,stop,step=1):
        if isinstance(step,int) and step<=0:raise ValueError('positive loop step required')
        self.label_serial+=1; n=self.label_serial
        hdr=f'loop{n}'; body=f'body{n}'; end=f'end{n}'
        lo=self.value(start); hi=self.value(stop); inc=self.value(step)
        self.emit(f'llvm.br ^{hdr}({lo} : i64)')
        i=self.temp(); self.types[i]='i64'
        self.lines.append(f'  ^{hdr}({i}: i64):'); self.current_block=hdr
        c=self.icmp('slt',i,hi)
        self.emit(f'llvm.cond_br {c}, ^{body}, ^{end}')
        self.lines.append(f'  ^{body}:');self.current_block=body
        yield i
        nxt=self.add(i,inc);self.emit(f'llvm.br ^{hdr}({nxt} : i64) {{loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}}')
        self.lines.append(f'  ^{end}:');self.current_block=end
    def branch(self,condition,then_callback,else_callback=None):
        self.label_serial+=1; n=self.label_serial
        yes=f'yes{n}'; no=f'no{n}'; end=f'merge{n}'
        self.emit(f'llvm.cond_br {self.value(condition,"i1")}, ^{yes}, ^{no}')
        self.lines.append(f'  ^{yes}:'); self.current_block=yes
        then_callback()
        self.emit(f'llvm.br ^{end}')
        self.lines.append(f'  ^{no}:'); self.current_block=no
        if else_callback is not None: else_callback()
        self.emit(f'llvm.br ^{end}')
        self.lines.append(f'  ^{end}:'); self.current_block=end
    def loop(self,start,stop,step,callback):
        with self.counted_loop(start,stop,step) as i: callback(i)
    def for_loop(self,start,stop,step,callback,**kwargs):
        with self.counted_loop(start,stop,step) as i: callback(i)
    def insn(self,funct,rs1=0,rs2=0):
        if funct not in (0,1,2,3,4,5,6,7,14,23): raise ValueError('unsupported/prohibited instruction role')
        from isa_configuration import retain_instruction
        if not retain_instruction(self,funct,rs1,rs2): return
        a,b=self.value(rs1),self.value(rs2)
        self.emit(f'"llvm.inline_asm"({a}, {b}) <{{asm_string = ".insn r 0x7b, 3, {funct}, x0, $0, $1", constraints = "r,r,~{{memory}}", has_side_effects}}> : (i64, i64) -> ()')
    def flush(self,skip=0):
        self.insn(FLUSH,skip,0)
    def fence(self): self.emit('"llvm.inline_asm"() <{asm_string = "fence", constraints = "~{memory}", has_side_effects}> : () -> ()')
    def matrix(self,addr,cols=16,rows=16):
        return self.or_(self.or_(addr,self.shl(cols,32)),self.shl(rows,48))
    def config_ex(self,**kwargs): self.insn(CONFIG,*encode_config_ex(**kwargs))
    def config_ld(self,stride,scale=1.0,shrink=False,block_stride=16,pixel_repeats=1,id=0,scale_bits=None):
        bits=f32_bits(scale) if scale_bits is None else scale_bits
        rs1=self.or_(self.shl(bits,32), (block_stride<<16)|(pixel_repeats<<8)|(id<<3)|(int(shrink)<<2)|1)
        self.insn(CONFIG,rs1,stride)
    def config_st(self,stride,activation=0,scale=1.0,scale_bits=None,**pool):
        rs1,_=encode_config_st(0,activation,scale,0,**pool)
        bits=f32_bits(scale) if scale_bits is None else scale_bits
        self.insn(CONFIG,rs1,self.or_(self.shl(bits,32),stride))
    def mvin(self,ptr,addr,cols=16,rows=16,id=0):
        if id not in (0,1,2):raise ValueError('load state id must be 0,1,2')
        self.insn((LOAD,LOAD2,LOAD3)[id],self.ptrtoint(ptr) if self.types.get(ptr)=='!llvm.ptr' else ptr,self.matrix(addr,cols,rows))
    def mvout(self,ptr,addr,cols=16,rows=16):
        self.insn(STORE,self.ptrtoint(ptr) if self.types.get(ptr)=='!llvm.ptr' else ptr,self.matrix(addr,cols,rows))
    def preload(self,b_addr,c_addr,b_cols=16,b_rows=16,c_cols=16,c_rows=16):
        self.insn(PRELOAD,self.matrix(b_addr,b_cols,b_rows),self.matrix(c_addr,c_cols,c_rows))
    def compute(self,a_addr,bd_addr,a_cols=16,a_rows=16,bd_cols=16,bd_rows=16,accumulated=False):
        self.insn(COMPUTE_STAY if accumulated else COMPUTE_FLIP,self.matrix(a_addr,a_cols,a_rows),self.matrix(bd_addr,bd_cols,bd_rows))
    @property
    def instruction_count(self):
        """Emitted body operations; excludes labels, globals and final return."""
        return sum(not line.lstrip().startswith('^') for line in self.lines)
    def finish(self,verify=False,include_completion=True):
        args=', '.join(f'{v}: !llvm.ptr' for v in self.args.values())
        text='module {\n'+'\n'.join(self.globals)+'\n  llvm.func @'+self.symbol+'('+args+') {\n'+'\n'.join(self.lines)+'\n    llvm.return\n  }\n}\n'
        if include_completion:
            completion = ('  llvm.func @kernel_complete() {\n'
                          '    "llvm.inline_asm"() <{asm_string = "fence", constraints = "~{memory}", has_side_effects}> : () -> ()\n'
                          '    llvm.return\n  }\n')
            text=text[:-2]+completion+'}\n'
        if verify:
            from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context
            from xdsl.parser import Parser
            Parser(make_llvm_context(),text).parse_module().verify()
        return text
