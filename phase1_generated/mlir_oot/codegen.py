"""Typed target schedule -> LLVM SSA, control flow and raw admitted ISA operations."""
import json
from io import StringIO
from xdsl.printer import Printer
from xdsl.dialects import llvm
from xdsl.dialects.builtin import i8,i64,IntegerType,IntegerAttr,ModuleOp,UnitAttr,TensorType,DenseIntOrFPElementsAttr
from xdsl.ir import Block,Region,ParametrizedAttribute
from xdsl.irdl import irdl_attr_definition
from lowering import TEMPLATES,DIM,TileLoopOp,GatherOp,ResidentInputOp,ResidentSlabOp,TileOriginOp,IdentityTileOp,FenceOp,BUNDLES,pack,fbits

@irdl_attr_definition
class CommandLoopAttr(ParametrizedAttribute):
    """Standard LLVM metadata preserving source-bound command multiplicity.

    The installed LLVMAttrDefs.td defines these loop-annotation fields. Disable
    non-forced loop transforms and unrolling on command-bearing backedges;
    ordinary scalar instruction selection and address optimization still apply.
    """
    name='llvm.loop_annotation'
    def print_parameters(self,printer):
        printer.print_string('<disableNonforced = true, unroll = <disable = true>>')

class LLVMEmitter:
    def __init__(self,args,program):
        self.args=args;self.entry=Block(arg_types=[llvm.LLVMPointerType() for _ in args])
        self.block=self.entry;self.region=Region(self.entry);self.cache={};self.iv=None;self.ivs={}
        self.scratch=None;self.zero=None;self.identity=None;self.coordinate_cache={}
        self.conv_origin=None;self.resident_a_addr=None
        if any(isinstance(o,IdentityTileOp) for o in program.walk()):
            # A compiler-owned immutable constant needs no runtime preparation.
            # Every invocation only reads it through DMA. Its symbolic address
            # is resolved by the ordinary linker, never a baked DRAM address.
            self.identity=self.add(llvm.AddressOfOp('gemmini_identity_i8_'+str(DIM),llvm.LLVMPointerType()))
        if any(isinstance(o,GatherOp) and json.loads(o.recipe.data).get('packed') for o in program.walk()):
            self.scratch=self.add(llvm.AllocaOp(self.constant(DIM*DIM),i8,alignment=8))
        if any(isinstance(o,GatherOp) for o in program.walk()):
            # Invalid taps read an immutable DIM-byte zero row through DMA.
            # The row is compiler-owned padding, never a source tensor value.
            # Read-only lifetime spans every pending transfer and invocation;
            # linked device regions can share it without preparation stores.
            self.zero=self.add(llvm.AddressOfOp('gemmini_zero_i8_'+str(DIM),llvm.LLVMPointerType()))
    def add(self,op):
        self.block.add_op(op)
        return op.results[0] if op.results else None
    def constant(self,v,bits=64):
        key=(v,bits)
        if key not in self.cache:
            op=llvm.ConstantOp(IntegerAttr(v,IntegerType(bits)),IntegerType(bits))
            if self.entry.last_op is not None and isinstance(self.entry.last_op,(llvm.BrOp,llvm.CondBrOp)):
                self.entry.insert_op_before(op,self.entry.last_op)
            else:self.entry.add_op(op)
            self.cache[key]=op.result
        return self.cache[key]
    def binary(self,cls,a,b):return self.add(cls(a,b))
    def cmp(self,a,b,predicate):
        return self.add(llvm.ICmpOp(a,b,IntegerAttr(llvm.ICmpPredicateFlag(predicate).to_int(),i64)))
    def loop_branch(self,condition,body,args,exitblock):
        branch=llvm.CondBrOp(condition,body,args,exitblock,[])
        branch.attributes['loop_annotation']=CommandLoopAttr()
        self.add(branch)
    def gep(self,ptr,offset):
        # LLVM's embedded GEP indices are signed i32. Source-derived DRAM
        # offsets are byte counts and may exceed that range even though each
        # on-chip allocation remains one mesh tile. Preserve such offsets as
        # i64 SSA indices rather than truncating a large tensor address.
        if isinstance(offset,int) and not -(1<<31)<=offset<(1<<31):
            if not -(1<<63)<=offset<(1<<63):
                raise ValueError('DRAM byte offset exceeds the RV64 pointer domain')
            offset=self.constant(offset)
        return self.add(llvm.GEPOp.from_mixed_indices(ptr,[offset],i8))
    def operand(self,s):
        if not s.startswith('@'):return self.constant(int(s))
        fields=s[1:].split(':');idx=fields[0];offset=int(fields[1]);strides=[int(v) for v in fields[2:]]
        ptr=self.identity if idx=='i' else self.scratch if idx=='s' else self.entry.args[int(idx)]
        # Preserve pointer provenance until the RoCC integer-operand boundary.
        # GEP's element type expresses the byte stride without integer address
        # multiply/add chains. These are byte arrays, never loaded tensor values.
        if offset:ptr=self.gep(ptr,offset if -(1<<31)<=offset<(1<<31) else self.constant(offset))
        for axis,stride in zip(('m','k','n','batch','tap'),strides):
            if stride:
                if axis not in self.ivs:raise ValueError('Dynamic tile address outside '+axis+' loop')
                if stride<0:raise ValueError('Negative dynamic byte stride')
                pointee=i8 if stride==1 else llvm.LLVMArrayType(stride,i8)
                ptr=self.add(llvm.GEPOp.from_mixed_indices(ptr,[self.ivs[axis]],pointee))
        return self.add(llvm.PtrToIntOp(ptr))
    def minimum(self,a,b):
        return self.add(llvm.SelectOp(self.cmp(a,b,'ult'),a,b))
    def index_value(self,value):
        return self.constant(value) if isinstance(value,int) else value
    def index_add(self,a,b):
        if isinstance(a,int) and isinstance(b,int):return a+b
        if isinstance(a,int) and a==0:return b
        if isinstance(b,int) and b==0:return a
        return self.binary(llvm.AddOp,self.index_value(a),self.index_value(b))
    def index_mul(self,a,b):
        if isinstance(a,int) and isinstance(b,int):return a*b
        if isinstance(a,int) and a==0 or isinstance(b,int) and b==0:return 0
        if isinstance(a,int) and a==1:return b
        if isinstance(b,int) and b==1:return a
        return self.binary(llvm.MulOp,self.index_value(a),self.index_value(b))
    def strided_gep(self,ptr,index,stride):
        """Express byte address scaling as typed pointer indexing."""
        if isinstance(index,int):return self.gep(ptr,index*stride) if index else ptr
        pointee=i8 if stride==1 else llvm.LLVMArrayType(stride,i8)
        return self.add(llvm.GEPOp.from_mixed_indices(ptr,[index],pointee))
    def tile_origin(self,op):
        """Resolve one aligned NHWC tile base without integer pointer math.

        The base may precede the source allocation for padding. GEPs carry no
        inbounds promise, and border gathers select the immutable zero row
        before DMA. The source pointer is consumed only for proven valid taps.
        """
        x=json.loads(op.recipe.data)
        bn=self.ivs['batch'] if x['batch_dynamic'] else x['batch']
        oy=self.ivs['oy'] if x['y_dynamic'] else x['oy']
        ox=self.ivs['ox'] if x['x_dynamic'] else x['ox']
        ptr=self.entry.args[x['arg']]
        ptr=self.strided_gep(ptr,bn,x['h']*x['w']*x['ci'])
        ptr=self.strided_gep(ptr,oy,x['sh']*x['w']*x['ci'])
        ptr=self.strided_gep(ptr,ox,x['sw']*x['ci'])
        ptr=self.gep(ptr,-(x['pt']*x['w']+x['pl'])*x['ci'])
        self.conv_origin={'ptr':ptr,'batch':bn,'oy':oy,'ox':ox,'arg':x['arg']}
        spatial=self.index_add(self.index_mul(bn,x['oh']),oy)
        self.ivs['m']=self.index_value(self.index_add(self.index_mul(spatial,x['ow']),ox))
    def gather_instruction(self,name,a,b,owner):
        asm=llvm.InlineAsmOp(TEMPLATES[name],'r,r,~{memory}',[a,b],has_side_effects=True)
        asm.attributes.update({k:v for k,v in owner.attributes.items() if k in ('merlin.source_op_index','prov.region_id')})
        self.add(asm)
    def gather_config(self,stride,owner):
        config=1|pack('ConfigMvinRs1',scale=fbits(1),stride=DIM)
        self.gather_instruction('CONFIG_CMD',self.constant(config),stride,owner)
    def gather_descriptor(self,row,count,depth):
        fields=BUNDLES['MvinRs2']['fields']
        val=self.constant(pack('MvinRs2',local_addr=0,num_rows=0,num_cols=depth))
        for name,value in (('local_addr',row),('num_rows',count)):
            shift=fields[name]['offset']
            if shift:value=self.binary(llvm.ShlOp,value,self.constant(shift))
            val=self.binary(llvm.OrOp,val,value)
        return val
    def resident_input(self,op):
        """Fill a proved NHWC canvas with bounded row and segment loops.

        The outer IV visits every batch/padded-y row once; each segment cursor
        covers exactly [0,length), with a positive DIM-bounded final transfer.
        Source DMA is reachable only in the valid-y block and stays inside one
        source row. Padding repeats the private immutable DIM-byte zero row at
        stride zero. Descriptor addresses remain below the disjoint RHS base.
        Code size has at most four LOAD sites regardless of batch/height/width.
        Malformed geometry or a changed allocation proof is refused; the
        scheduler retains its general direct-DMA path for nonresident shapes.
        """
        x=json.loads(op.recipe.data);width=x['canvas_w'];height=x['canvas_h']
        positive=('batch','h','w','ci','canvas_h','canvas_w','rows','b_base')
        if (any(type(x[k]) is not int or x[k]<=0 for k in positive)
            or any(type(x[k]) is not int or x[k]<0 for k in ('pt','pl','pb','pr'))
            or width!=x['w']+x['pl']+x['pr']
            or height!=x['h']+x['pt']+x['pb']
            or x['batch']*width*height!=x['rows']
            or x['rows']>x['b_base'] or x['ci']>DIM
            or x['batch']*x['h']*x['w']*x['ci']>=1<<63):
            raise ValueError('Resident input geometry or row proof changed')
        # Dynamic descriptors retain the same admitted field bounds as the
        # statically packed schedule. OR-ing dynamic values cannot mask overflow.
        pack('MvinRs2',local_addr=x['rows']-1,num_rows=DIM,num_cols=x['ci'])
        if self.zero is None:raise ValueError('Resident padding requires immutable zero storage')
        zero_pointer=self.add(llvm.PtrToIntOp(self.zero))
        row_body=Block(arg_types=[i64]);row_exit=Block()
        self.region.add_block(row_body);self.region.add_block(row_exit)
        self.add(llvm.BrOp(row_body,self.constant(0)));self.block=row_body
        row=row_body.args[0]
        bn=self.binary(llvm.UDivOp,row,self.constant(height))
        sy=self.binary(llvm.SubOp,self.binary(llvm.URemOp,row,self.constant(height)),self.constant(x['pt']))
        local=self.binary(llvm.MulOp,row,self.constant(width))
        valid_body=Block();padding_body=Block();row_join=Block()
        for block in (valid_body,padding_body,row_join):self.region.add_block(block)
        self.add(llvm.CondBrOp(self.cmp(sy,self.constant(x['h']),'ult'),valid_body,[],padding_body,[]))
        def segment(start,length,source=None):
            if not length:return
            self.gather_config(self.constant(x['ci'] if source is not None else 0),op)
            body=Block(arg_types=[i64]);exitblock=Block()
            self.region.add_block(body);self.region.add_block(exitblock)
            self.add(llvm.BrOp(body,self.constant(0)));self.block=body
            offset=body.args[0]
            count=self.minimum(self.constant(DIM),self.binary(llvm.SubOp,self.constant(length),offset))
            addr=self.index_add(self.index_add(local,start),offset)
            pointer=(self.add(llvm.PtrToIntOp(self.strided_gep(source,offset,x['ci'])))
                     if source is not None else zero_pointer)
            self.gather_instruction('LOAD_CMD',pointer,self.gather_descriptor(self.index_value(addr),count,x['ci']),op)
            nxt=self.binary(llvm.AddOp,offset,count)
            self.loop_branch(self.cmp(nxt,self.constant(length),'ult'),body,[nxt],exitblock)
            self.block=exitblock
        self.block=valid_body
        source_row=self.index_mul(self.index_add(self.index_mul(bn,x['h']),sy),x['w']*x['ci'])
        source=self.gep(self.entry.args[x['arg']],self.index_value(source_row))
        segment(0,x['pl'])
        segment(x['pl'],x['w'],source)
        segment(x['pl']+x['w'],x['pr'])
        self.add(llvm.BrOp(row_join))
        self.block=padding_body;segment(0,width);self.add(llvm.BrOp(row_join))
        self.block=row_join
        nxt=self.binary(llvm.AddOp,row,self.constant(1))
        self.loop_branch(self.cmp(nxt,self.constant(x['batch']*height),'ult'),row_body,[nxt],row_exit)
        self.block=row_exit
    def resident_gather(self,op,x):
        origin=self.conv_origin
        if origin is None or origin['arg']!=x['arg'] or x.get('dynamic_channel'):
            raise ValueError('Resident input needs its source origin and one channel slice')
        canvas=x['resident_input'];tap=self.ivs['tap'] if x.get('dynamic_tap') else x['r']//x['ci']
        ky=self.index_value(tap//x['kw']) if isinstance(tap,int) else self.binary(llvm.UDivOp,tap,self.constant(x['kw']))
        kx=self.index_value(tap%x['kw']) if isinstance(tap,int) else self.binary(llvm.URemOp,tap,self.constant(x['kw']))
        spatial_y=self.index_add(self.index_mul(origin['oy'],x['sh']),self.index_mul(ky,x['dh']))
        spatial_x=self.index_add(self.index_mul(origin['ox'],x['sw']),self.index_mul(kx,x['dw']))
        batch_y=self.index_add(self.index_mul(origin['batch'],canvas['canvas_h']),spatial_y)
        row=self.index_add(self.index_mul(batch_y,canvas['canvas_w']),spatial_x)
        if self.resident_a_addr is not None:raise ValueError('Resident input compute was not consumed')
        self.resident_a_addr=self.index_value(row)
    def resident_slab(self,op):
        """Refresh one complete padded kernel-y slab for the current output row.

        A source row is selected only when its signed coordinate is in range.
        Invalid source rows and left/right padding read the immutable zero row
        at stride zero. A LOAD has at most DIM rows and at most ci<=DIM bytes
        per source row; no host scalar tensor value is read.
        """
        x=json.loads(op.recipe.data);width=x['canvas_w']
        if x['rows']!=x['kh']*width or self.zero is None:
            raise ValueError('Resident slab proof or zero storage changed')
        bn=self.ivs['batch'] if x['batch_dynamic'] else x['batch']
        oy=self.ivs['oy'] if x['y_dynamic'] else x['oy']
        zero_pointer=self.add(llvm.PtrToIntOp(self.zero))
        for ky in range(x['kh']):
            sy=self.index_add(self.index_mul(oy,x['sh']),ky*x['dh']-x['pt'])
            valid_y=self.cmp(self.index_value(sy),self.constant(x['h']),'ult')
            segments=((0,x['pl'],False),(x['pl'],x['w'],True),
                      (x['pl']+x['w'],x['pr'],False))
            for start,length,source_segment in segments:
                if not length:continue
                if source_segment:
                    src_row=self.index_mul(self.index_add(self.index_mul(bn,x['h']),sy),x['w']*x['ci'])
                    source=self.gep(self.entry.args[x['arg']],self.index_value(src_row))
                    stride=self.add(llvm.SelectOp(valid_y,self.constant(x['ci']),self.constant(0)))
                else:stride=self.constant(0)
                self.gather_config(stride,op)
                for offset in range(0,length,DIM):
                    count=min(DIM,length-offset)
                    addr=ky*width+start+offset
                    descriptor=pack('MvinRs2',local_addr=addr,num_rows=count,num_cols=x['ci'])
                    if source_segment:
                        source_chunk=self.gep(source,offset*x['ci']) if offset else source
                        chosen=self.add(llvm.SelectOp(valid_y,source_chunk,self.zero))
                        chunk=self.add(llvm.PtrToIntOp(chosen))
                    else:chunk=zero_pointer
                    self.gather_instruction('LOAD_CMD',chunk,self.constant(descriptor),op)
    def slab_gather(self,op,x):
        origin=self.conv_origin
        if origin is None or origin['arg']!=x['arg'] or x.get('dynamic_channel'):
            raise ValueError('Resident slab needs its source origin and one channel slice')
        tap=self.ivs['tap'] if x.get('dynamic_tap') else x['r']//x['ci']
        ky=self.index_value(tap//x['kw']) if isinstance(tap,int) else self.binary(llvm.UDivOp,tap,self.constant(x['kw']))
        kx=self.index_value(tap%x['kw']) if isinstance(tap,int) else self.binary(llvm.URemOp,tap,self.constant(x['kw']))
        row=self.index_add(self.index_mul(ky,x['resident_slab']['canvas_w']),
                           self.index_add(self.index_mul(origin['ox'],x['sw']),self.index_mul(kx,x['dw'])))
        if self.resident_a_addr is not None:raise ValueError('Resident slab compute was not consumed')
        self.resident_a_addr=self.index_value(row)
    def gather(self,op):
        """Coalesce horizontal source rows into one strided device DMA.

        A group ends at the output row, tensor validity boundary or tile end.
        Every group writes consecutive scratchpad rows; padding uses stride zero
        from the initialized DIM-byte buffer. No source tensor value is read by
        scalar code. All descriptor positions come from the RTL field bundle.
        Runtime tap and channel loops preserve the flattened ky,kx,channel
        reduction order while keeping the gather body independent of K length.
        Each channel slice must stay within one input pixel's channel range;
        lowering owns that invariant and the DIM-bounded row/depth verifier.
        """
        x=json.loads(op.recipe.data)
        if x.get('horizontal_fusion'):
            self.gather_horizontal(op,x)
            return
        if x.get('resident_slab'):
            self.slab_gather(op,x)
            return
        if x.get('resident_input'):
            self.resident_gather(op,x)
            return
        if x.get('aligned_origin'):
            self.gather_aligned(op,x)
            return
        if x.get('packed'):
            if x.get('dynamic_tap') or x.get('dynamic_channel'):
                raise ValueError('Runtime tap/channel gather requires direct DMA')
            self.gather_packed(op,x)
            return
        tap=x['r'];ky=tap//(x['kw']*x['ci']);kx=tap//x['ci']%x['kw'];ch=tap%x['ci']
        pitch=x['ci']*x['sw']
        if not (x['dynamic'] or x.get('dynamic_tap') or x.get('dynamic_channel')):
            addresses=[]
            for row in range(x['rows']):
                mi=x['i']+row;bn,spatial=divmod(mi,x['oh']*x['ow']);oy,ox=divmod(spatial,x['ow'])
                sy=oy*x['sh']+ky*x['dh']-x['pt'];sx=ox*x['sw']+kx*x['dw']-x['pl']
                addresses.append(((bn*x['h']+sy)*x['w']+sx)*x['ci']+ch if 0<=sy<x['h'] and 0<=sx<x['w'] else None)
            row=0
            while row<len(addresses):
                offset=addresses[row];count=1
                while row+count<len(addresses) and addresses[row+count]==(None if offset is None else offset+count*pitch):count+=1
                src=self.zero if offset is None else self.gep(self.entry.args[x['arg']],offset)
                self.gather_config(self.constant(0 if offset is None else pitch),op)
                descriptor=pack('MvinRs2',local_addr=row,num_rows=count,num_cols=x['depth'])
                self.gather_dma(src,descriptor,op)
                row+=count
            return
        # Runtime grouping stays bounded by the current DIM-row tile. The outer
        # spatial induction variable is unaffected by the nested DMA cursor.
        body=Block(arg_types=[i64]);exitblock=Block();self.region.add_block(body);self.region.add_block(exitblock)
        self.add(llvm.BrOp(body,self.constant(0)));self.block=body
        row=body.args[0];base=self.ivs['m'] if x['dynamic'] else self.constant(x['i'])
        if x.get('dynamic_tap'):
            tap=self.ivs['tap']
            ky=self.binary(llvm.UDivOp,tap,self.constant(x['kw']))
            kx=self.binary(llvm.URemOp,tap,self.constant(x['kw']))
            yshift=self.binary(llvm.SubOp,self.binary(llvm.MulOp,ky,self.constant(x['dh'])),self.constant(x['pt']))
            xshift=self.binary(llvm.SubOp,self.binary(llvm.MulOp,kx,self.constant(x['dw'])),self.constant(x['pl']))
        else:
            yshift=self.constant(ky*x['dh']-x['pt'])
            xshift=self.constant(kx*x['dw']-x['pl'])
        channel=self.constant(ch)
        if x.get('dynamic_channel'):
            channel=self.binary(llvm.AddOp,channel,self.ivs['k'])
        mi=self.binary(llvm.AddOp,base,row)
        bn=self.binary(llvm.UDivOp,mi,self.constant(x['oh']*x['ow']))
        spatial=self.binary(llvm.URemOp,mi,self.constant(x['oh']*x['ow']))
        oy=self.binary(llvm.UDivOp,spatial,self.constant(x['ow']))
        ox=self.binary(llvm.URemOp,spatial,self.constant(x['ow']))
        sy=self.binary(llvm.AddOp,self.binary(llvm.MulOp,oy,self.constant(x['sh'])),yshift)
        sx=self.binary(llvm.AddOp,self.binary(llvm.MulOp,ox,self.constant(x['sw'])),xshift)
        syvalid=self.cmp(sy,self.constant(x['h']),'ult')
        valid=self.binary(llvm.AndOp,syvalid,self.cmp(sx,self.constant(x['w']),'ult'))
        rowmax=self.minimum(self.binary(llvm.SubOp,self.constant(x['rows']),row),self.binary(llvm.SubOp,self.constant(x['ow']),ox))
        right_count=self.binary(llvm.UDivOp,self.binary(llvm.AddOp,self.binary(llvm.SubOp,self.constant(x['w']),sx),self.constant(x['sw']-1)),self.constant(x['sw']))
        left_count=self.binary(llvm.UDivOp,self.binary(llvm.SubOp,self.constant(x['sw']-1),sx),self.constant(x['sw']))
        invalid_count=self.add(llvm.SelectOp(self.cmp(sx,self.constant(0),'slt'),self.minimum(rowmax,left_count),rowmax))
        invalid_count=self.add(llvm.SelectOp(syvalid,invalid_count,rowmax))
        count=self.add(llvm.SelectOp(valid,self.minimum(rowmax,right_count),invalid_count))
        linear=self.binary(llvm.AddOp,self.binary(llvm.MulOp,bn,self.constant(x['h'])),sy)
        linear=self.binary(llvm.AddOp,self.binary(llvm.MulOp,linear,self.constant(x['w'])),sx)
        offset=self.binary(llvm.AddOp,self.binary(llvm.MulOp,linear,self.constant(x['ci'])),channel)
        src=self.add(llvm.SelectOp(valid,self.gep(self.entry.args[x['arg']],offset),self.zero))
        stride=self.add(llvm.SelectOp(valid,self.constant(pitch),self.constant(0)))
        self.gather_config(stride,op)
        self.gather_instruction('LOAD_CMD',self.add(llvm.PtrToIntOp(src)),self.gather_descriptor(row,count,x['depth']),op)
        nxt=self.binary(llvm.AddOp,row,count)
        self.loop_branch(self.cmp(nxt,self.constant(x['rows']),'ult'),body,[nxt],exitblock)
        self.block=exitblock
    def gather_horizontal(self,op,x):
        """Move one complete contiguous NHWC kernel row to an A tile.

        Lowering calls this only for an interior output tile with sw=dw=1
        and kw*ci<=DIM. The source base is the first kernel pixel; each next
        output x starts ci bytes later. The configured LOAD stride is ci and
        the descriptor's width is kw*ci, so overlapping windows preserve the
        flattened reduction order without host tensor reads.
        """
        origin=self.conv_origin
        if origin is None or origin['arg']!=x['arg'] or x['sw']!=1 or x['depth']>DIM:
            raise ValueError('Horizontal gather lost its interior source proof')
        ptr=self.gep(origin['ptr'],x['ky']*x['dh']*x['w']*x['ci'])
        descriptor=pack('MvinRs2',local_addr=x['local_addr'],num_rows=x['rows'],num_cols=x['depth'])
        self.gather_dma(ptr,descriptor,op)
    def gather_aligned(self,op,x):
        """Gather a horizontal tile from a previously resolved spatial origin.

        A tile stays within one output row. Static tap bounds are expressed in
        logical output coordinates, so border validity never needs flattened
        source indexing. Runtime taps retain the same bounds through sy/sx
        predicates. Neither path reads source values in scalar code.
        """
        origin=self.conv_origin
        if origin is None or origin['arg']!=x['arg']:
            raise ValueError('Aligned gather requires matching spatial origin')
        ky=x['r']//(x['kw']*x['ci']);kx=x['r']//x['ci']%x['kw'];ch=x['r']%x['ci']
        ptr=origin['ptr']
        if x.get('dynamic_tap'):
            ky=self.binary(llvm.UDivOp,self.ivs['tap'],self.constant(x['kw']))
            kx=self.binary(llvm.URemOp,self.ivs['tap'],self.constant(x['kw']))
            ptr=self.strided_gep(ptr,ky,x['dh']*x['w']*x['ci'])
            ptr=self.strided_gep(ptr,kx,x['dw']*x['ci'])
            ptr=self.gep(ptr,ch) if ch else ptr
        else:ptr=self.gep(ptr,(ky*x['dh']*x['w']+kx*x['dw'])*x['ci']+ch)
        if x.get('dynamic_channel'):ptr=self.gep(ptr,self.ivs['k'])
        if x['interior']:
            self.gather_dma(ptr,x['tile_descriptor'],op)
            return
        if not x.get('dynamic_tap'):
            self.gather_aligned_static(op,x,ptr,ky,kx)
            return
        oy=origin['oy'];ox=origin['ox'];pitch=x['ci']*x['sw']
        sy=self.index_add(self.index_add(self.index_mul(oy,x['sh']),self.index_mul(ky,x['dh'])),-x['pt'])
        syvalid=self.cmp(self.index_value(sy),self.constant(x['h']),'ult')
        xshift=self.index_add(self.index_mul(kx,x['dw']),-x['pl'])
        body=Block(arg_types=[i64]);exitblock=Block();self.region.add_block(body);self.region.add_block(exitblock)
        self.add(llvm.BrOp(body,self.constant(0)));self.block=body
        row=body.args[0];xc=self.index_value(self.index_add(ox,row))
        rowmax=self.binary(llvm.SubOp,self.constant(x['rows']),row)
        sx=self.index_value(self.index_add(self.index_mul(xc,x['sw']),xshift))
        xvalid=self.cmp(sx,self.constant(x['w']),'ult')
        right_count=self.binary(llvm.UDivOp,self.binary(llvm.AddOp,self.binary(llvm.SubOp,self.constant(x['w']),sx),self.constant(x['sw']-1)),self.constant(x['sw']))
        left_count=self.binary(llvm.UDivOp,self.binary(llvm.SubOp,self.constant(x['sw']-1),sx),self.constant(x['sw']))
        before=self.cmp(sx,self.constant(0),'slt')
        valid=self.binary(llvm.AndOp,syvalid,xvalid)
        invalid_count=self.add(llvm.SelectOp(before,self.minimum(rowmax,left_count),rowmax))
        invalid_count=self.add(llvm.SelectOp(syvalid,invalid_count,rowmax))
        count=self.add(llvm.SelectOp(valid,self.minimum(rowmax,right_count),invalid_count))
        src=self.add(llvm.SelectOp(valid,self.strided_gep(ptr,row,pitch),self.zero))
        stride=self.add(llvm.SelectOp(valid,self.constant(pitch),self.constant(0)))
        self.gather_config(stride,op)
        self.gather_instruction('LOAD_CMD',self.add(llvm.PtrToIntOp(src)),self.gather_descriptor(row,count,x['depth']),op)
        nxt=self.binary(llvm.AddOp,row,count)
        self.loop_branch(self.cmp(nxt,self.constant(x['rows']),'ult'),body,[nxt],exitblock)
        self.block=exitblock
    def gather_aligned_static(self,op,x,ptr,ky,kx):
        """Partition one static tap into at most three forward DMA segments.

        Aligned spatial tiles stay inside one output row. A static tap's valid
        output coordinates form an interval [lo_x,hi_x), so intersecting that
        interval with [ox,ox+rows) gives left padding, valid source rows, and
        right padding. The two clamped endpoints are monotone and in [0,rows].
        Only positive segments issue DMA, in increasing scratchpad-row order.
        An invalid Y uses the initialized zero row for the complete tile.
        Source-pointer DMA is reached only for the proven valid interval;
        source values never pass through scalar loads or preparation stores.
        """
        origin=self.conv_origin;oy=origin['oy'];ox=origin['ox'];rows=x['rows']
        ceil=lambda a,b:-(-a//b)
        lo_y=max(0,min(x['oh'],ceil(x['pt']-ky*x['dh'],x['sh'])))
        hi_y=max(0,min(x['oh'],ceil(x['h']+x['pt']-ky*x['dh'],x['sh'])))
        lo_x=max(0,min(x['ow'],ceil(x['pl']-kx*x['dw'],x['sw'])))
        hi_x=max(0,min(x['ow'],ceil(x['w']+x['pl']-kx*x['dw'],x['sw'])))
        if isinstance(oy,int):yvalid=lo_y<=oy<hi_y
        elif lo_y==0 and hi_y==x['oh']:yvalid=True
        elif lo_y==hi_y:yvalid=False
        else:
            yvalid=self.binary(llvm.AndOp,self.cmp(oy,self.constant(lo_y),'uge'),self.cmp(oy,self.constant(hi_y),'ult'))
        def endpoint(bound):
            if isinstance(ox,int):return max(0,min(rows,bound-ox))
            if bound==0:return 0
            if bound==x['ow']:return rows
            delta=self.binary(llvm.SubOp,self.constant(bound),ox)
            clipped=self.minimum(self.constant(rows),delta)
            return self.add(llvm.SelectOp(self.cmp(ox,self.constant(bound),'ult'),clipped,self.constant(0)))
        if yvalid is False:left=right=rows
        else:
            left=endpoint(lo_x);right=endpoint(hi_x)
            if yvalid is not True:
                left=self.add(llvm.SelectOp(yvalid,self.index_value(left),self.constant(rows)))
                right=self.add(llvm.SelectOp(yvalid,self.index_value(right),self.constant(rows)))
        def difference(a,b):
            if isinstance(a,int) and isinstance(b,int):return a-b
            if isinstance(b,int) and b==0:return a
            return self.binary(llvm.SubOp,self.index_value(a),self.index_value(b))
        pitch=x['ci']*x['sw']
        def segment(row,count,valid):
            if isinstance(count,int) and count==0:return
            if not isinstance(count,int):
                body=Block();exitblock=Block()
                self.region.add_block(body);self.region.add_block(exitblock)
                self.add(llvm.CondBrOp(self.cmp(count,self.constant(0),'ne'),body,[],exitblock,[]))
                self.block=body
            src=self.strided_gep(ptr,row,pitch) if valid else self.zero
            self.gather_config(self.constant(pitch if valid else 0),op)
            descriptor=self.gather_descriptor(self.index_value(row),self.index_value(count),x['depth'])
            self.gather_instruction('LOAD_CMD',self.add(llvm.PtrToIntOp(src)),descriptor,op)
            if not isinstance(count,int):
                self.add(llvm.BrOp(exitblock));self.block=exitblock
        segment(0,left,False)
        segment(left,difference(right,left),True)
        segment(right,difference(rows,right),False)
    def gather_packed(self,op,x):
        """Copy-only NHWC -> bounded row-major tile preparation.

        Every consumed byte is written once per K slice. The fixed DIM*DIM
        allocation lives for one invocation; tails move only initialized cells.
        K remains the original flattened (ky,kx,channel) order. This transform
        generates no tensor arithmetic; PRELOAD/COMPUTE execute every product.
        """
        # The previous asynchronous DMA must finish before the CPU reuses
        # invocation-local packing storage. The memory clobber also prevents
        # stock LLVM from moving the copy stores across the completion fence.
        self.add(llvm.InlineAsmOp('fence','~{memory}',[],has_side_effects=True))
        for row in range(x['rows']):
            if x['dynamic']:
                base=self.ivs['m'];key=(base,row,x['oh'],x['ow'])
                if key not in self.coordinate_cache:
                    mi=self.binary(llvm.AddOp,base,self.constant(row))
                    bn=self.binary(llvm.UDivOp,mi,self.constant(x['oh']*x['ow']))
                    spatial=self.binary(llvm.URemOp,mi,self.constant(x['oh']*x['ow']))
                    oy=self.binary(llvm.UDivOp,spatial,self.constant(x['ow']))
                    ox=self.binary(llvm.URemOp,spatial,self.constant(x['ow']))
                    self.coordinate_cache[key]=(bn,oy,ox)
                bn,oy,ox=self.coordinate_cache[key]
            else:
                bn,spatial=divmod(x['i']+row,x['oh']*x['ow']);oy,ox=divmod(spatial,x['ow'])
            column=0
            while column<x['depth']:
                r=x['r']+column;ky=r//(x['kw']*x['ci']);kx=r//x['ci']%x['kw'];ch=r%x['ci']
                available=min(x['depth']-column,x['ci']-ch,8)
                width=1<<(available.bit_length()-1)
                if x['dynamic']:
                    sy=self.binary(llvm.AddOp,self.binary(llvm.MulOp,oy,self.constant(x['sh'])),self.constant(ky*x['dh']-x['pt']))
                    sx=self.binary(llvm.AddOp,self.binary(llvm.MulOp,ox,self.constant(x['sw'])),self.constant(kx*x['dw']-x['pl']))
                    valid=self.binary(llvm.AndOp,self.cmp(sy,self.constant(x['h']),'ult'),self.cmp(sx,self.constant(x['w']),'ult'))
                    linear=self.binary(llvm.AddOp,self.binary(llvm.MulOp,bn,self.constant(x['h'])),sy)
                    linear=self.binary(llvm.AddOp,self.binary(llvm.MulOp,linear,self.constant(x['w'])),sx)
                    offset=self.binary(llvm.AddOp,self.binary(llvm.MulOp,linear,self.constant(x['ci'])),self.constant(ch))
                    src=self.add(llvm.SelectOp(valid,self.gep(self.entry.args[x['arg']],offset),self.zero))
                else:
                    sy=oy*x['sh']+ky*x['dh']-x['pt'];sx=ox*x['sw']+kx*x['dw']-x['pl']
                    offset=((bn*x['h']+sy)*x['w']+sx)*x['ci']+ch
                    src=self.gep(self.entry.args[x['arg']],offset) if 0<=sy<x['h'] and 0<=sx<x['w'] else self.zero
                value=self.add(llvm.LoadOp(src,IntegerType(width*8),alignment=1))
                self.add(llvm.StoreOp(value,self.gep(self.scratch,row*DIM+column),alignment=1))
                column+=width
        self.gather_dma(self.scratch,x['tile_descriptor'],op)
    def gather_dma(self,src,descriptor,owner):
        asm=llvm.InlineAsmOp(TEMPLATES['LOAD_CMD'],'r,r,~{memory}',[self.add(llvm.PtrToIntOp(src)),self.constant(descriptor)],has_side_effects=True)
        asm.attributes.update({k:v for k,v in owner.attributes.items() if k in ('merlin.source_op_index','prov.region_id')})
        self.add(asm)
    def emit_ops(self,ops):
        for op in ops:
            if isinstance(op,TileLoopOp):
                old_origin=self.conv_origin;old_m=self.ivs.get('m')
                body=Block(arg_types=[i64]);exitblock=Block();self.region.add_block(body);self.region.add_block(exitblock)
                self.add(llvm.BrOp(body,self.constant(op.lower.value.data)))
                self.block=body;old_iv=self.iv;self.iv=body.args[0];axis=op.axis.data;old_axis=self.ivs.get(axis);self.ivs[axis]=self.iv
                self.emit_ops(op.body.block.ops)
                nxt=self.binary(llvm.AddOp,self.ivs[axis],self.constant(op.step.value.data))
                # Compare the current IV to the final valid start. Adding step
                # to a final IV near INT64_MAX may wrap, even though every
                # executed coordinate and source byte extent is legal. The
                # wrapped next value is never consumed on the exit edge.
                last=op.lower.value.data+((op.upper.value.data-1-op.lower.value.data)//op.step.value.data)*op.step.value.data
                condition=self.cmp(self.ivs[axis],self.constant(last),'ult')
                self.loop_branch(condition,body,[nxt],exitblock)
                self.iv=old_iv;self.block=exitblock
                if old_axis is None:self.ivs.pop(axis)
                else:self.ivs[axis]=old_axis
                self.conv_origin=old_origin
                if axis!='m':
                    if old_m is None:self.ivs.pop('m',None)
                    else:self.ivs['m']=old_m
            elif isinstance(op,TileOriginOp):self.tile_origin(op)
            elif isinstance(op,ResidentInputOp):self.resident_input(op)
            elif isinstance(op,ResidentSlabOp):self.resident_slab(op)
            elif isinstance(op,GatherOp):self.gather(op)
            elif isinstance(op,IdentityTileOp):pass
            elif isinstance(op,FenceOp):
                asm=llvm.InlineAsmOp('fence','~{memory}',[],has_side_effects=True)
                asm.attributes.update(op.attributes)
                self.add(asm)
            else:
                rs1=self.operand(op.rs1.data)
                if op.instruction.data=='COMPUTE_AND_FLIP_CMD' and self.resident_a_addr is not None:
                    if int(op.rs1.data)&((1<<BUNDLES['ComputeRs']['fields']['local_addr']['slot_width'])-1):
                        raise ValueError('Resident compute has an unexpected local address')
                    rs1=self.binary(llvm.OrOp,rs1,self.resident_a_addr)
                    self.resident_a_addr=None
                asm=llvm.InlineAsmOp(TEMPLATES[op.instruction.data],'r,r,~{memory}',[rs1,self.operand(op.rs2.data)],has_side_effects=True)
                asm.attributes.update({k:v for k,v in op.attributes.items() if k in ('merlin.source_op_index','prov.region_id')})
                self.add(asm)

def emit(program,args):
    e=LLVMEmitter(args,program)
    e.add(llvm.InlineAsmOp('fence','~{memory}',[],has_side_effects=True));e.emit_ops(program.body.block.ops)
    e.add(llvm.InlineAsmOp('fence','~{memory}',[],has_side_effects=True));e.add(llvm.ReturnOp())
    # The contract entry owns the complete device CFG. The provided host
    # compiler renames this entry before composing independently lowered
    # regions, so no external implementation helper or weak entry is needed.
    # Keep one static command inventory across IR, object and final ELF. Small
    # bodies would otherwise be copied into the standalone ABI wrapper by -O2,
    # even though the source-bound model entry still owns the original body.
    function=llvm.FuncOp('gemmini_kernel',llvm.LLVMFunctionType([llvm.LLVMPointerType() for _ in args]),linkage=llvm.LinkageAttr('external'),body=e.region,other_props={'no_inline':UnitAttr()})
    # The standalone target harness has a separate public entry_symbol ABI.
    # It needs an exported function for linking and stack-usage accounting.
    # Model composition calls the renamed gemmini_kernel directly instead.
    harness_block=Block(arg_types=[llvm.LLVMPointerType() for _ in args])
    harness_block.add_op(llvm.CallOp('gemmini_kernel',*harness_block.args))
    harness_block.add_op(llvm.ReturnOp())
    harness=llvm.FuncOp('kernel_entry',llvm.LLVMFunctionType([llvm.LLVMPointerType() for _ in args]),linkage=llvm.LinkageAttr('weak'),body=Region(harness_block))
    complete_block=Block();complete_block.add_op(llvm.InlineAsmOp('fence','~{memory}',[],has_side_effects=True));complete_block.add_op(llvm.ReturnOp())
    # Identical invocation completion code may be merged across device objects.
    complete=llvm.FuncOp('kernel_complete',llvm.LLVMFunctionType([]),linkage=llvm.LinkageAttr('weak_odr'),body=Region(complete_block))
    globals=[]
    if e.zero is not None:
        value=DenseIntOrFPElementsAttr.from_list(TensorType(i8,[DIM]),[0]*DIM)
        globals.append(llvm.GlobalOp(llvm.LLVMArrayType(DIM,i8),'gemmini_zero_i8_'+str(DIM),'private',constant=True,value=value,alignment=64))
    if e.identity is not None:
        elements=[int(row==column) for row in range(DIM) for column in range(DIM)]
        value=DenseIntOrFPElementsAttr.from_list(TensorType(i8,[DIM*DIM]),elements)
        globals.append(llvm.GlobalOp(llvm.LLVMArrayType(DIM*DIM,i8),'gemmini_identity_i8_'+str(DIM),'private',constant=True,value=value,alignment=64))
    module=ModuleOp([*globals,function,harness,complete]);module.verify();return module

class ArtifactPrinter(Printer):
    """Use generic inline-asm syntax across MLIR/xDSL custom-parser versions.

    All operands, properties and source attributes are printed from verified IR.
    Generic syntax avoids the differing position of inline-asm attribute dicts
    in upstream LLVM and the provided host's xDSL reader.
    """
    def print_op(self,op):
        previous=self.print_generic_format
        if isinstance(op,llvm.InlineAsmOp):self.print_generic_format=True
        try:super().print_op(op)
        finally:self.print_generic_format=previous

def format_artifact(module):
    stream=StringIO();ArtifactPrinter(stream=stream).print_op(module)
    return stream.getvalue()
