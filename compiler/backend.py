"""Schedule logical command buffers as Gemmini tile programs."""
import math
from builder import Builder
from contraction import matmul, pack_resident_rhs, resident_matmul
from conv import conv2d
from elementwise import movement, residual_add, bias_add


def emit(buffer):
    tensors = buffer['tensors']
    args = buffer['compiler_pointer_abi']['arguments']
    b = Builder(args)
    ptrs = {name: '%arg' + str(index) for index,name in enumerate(args)}
    for name,spec in tensors.items():
        if name not in ptrs:
            ptrs[name] = b.alloc(math.prod(spec['shape']),spec['dtype'])
    residents = {}
    accumulators = {}
    resident_uses = {}
    for command in buffer['commands']:
        if command['opcode'] == 'MATMUL_RESIDENT':
            handle = command['operands']['rhs']
            resident_uses[handle] = resident_uses.get(handle, 0) + 1
    cached_handle = None
    cached_rhs = None

    def contract(a,w,out,m,k,n,attrs,transpose_rhs=False,resident_state=None):
        def compute(destination, configuration):
            if resident_state is None:
                matmul(b,a,w,destination,m,k,n,configuration,tensors,ptrs,transpose_rhs=transpose_rhs)
            else:
                resident_matmul(b,a,resident_state,destination,m,configuration,tensors,ptrs)
        if 'maxpool' in attrs.get('epilogue',[]):
            from pooling import pool
            temp = b.alloc(m*n,'i32')
            raw = dict(attrs)
            raw['epilogue'] = [s for s in attrs.get('epilogue',[]) if s in ('bias_add','bias')]
            raw['output_dtype'] = 'i32'
            raw.pop('acc_scale',None)
            raw.pop('acc_scale_by_column',None)
            compute(temp,raw)
            pool(b,temp,out,m,n,attrs)
        else:
            compute(out,attrs)

    for command in buffer['commands']:
        code = command['opcode']
        operands = command.get('operands',{})
        attrs = command.get('attributes',{})
        if code == 'RES_PACK':
            residents[operands['dst']] = operands['src']
        elif code == 'MATMUL_RESIDENT':
            accumulators[operands['dst']] = (operands['lhs'],residents[operands['rhs']],operands['rhs'])
        elif code == 'COMMIT':
            a,w,handle = accumulators[operands['src']]
            ashape,wshape = tensors[a]['shape'],tensors[w]['shape']
            state = None
            if resident_uses[handle] > 1:
                if cached_handle != handle:
                    cached_rhs = pack_resident_rhs(b,ptrs[w],wshape[-2],wshape[-1])
                    cached_handle = handle if cached_rhs is not None else None
                state = cached_rhs
            contract(ptrs[a],ptrs[w],ptrs[operands['dst']],ashape[-2],ashape[-1],wshape[-1],attrs,resident_state=state)
        elif code == 'EVICT':
            residents.pop(operands['handle'],None)
            if operands['handle'] == cached_handle:
                cached_handle = None
                cached_rhs = None
        elif code == 'CONV2D':
            w = residents.get(operands['weight'],operands['weight'])
            a,out = operands['ifm'],operands['dst']
            if 'maxpool' in attrs.get('epilogue',[]):
                from pooling import pool
                kh,kw,ci,co = attrs['kernel']
                batch,h,width,_ = tensors[a]['shape']
                sh,sw = attrs.get('stride',[1,1])
                dh,dw = attrs.get('dilation',[1,1])
                pt,pl,pb,pr = attrs.get('padding',[0,0,0,0])
                ho = (h+pt+pb-dh*(kh-1)-1)//sh+1
                wo = (width+pl+pr-dw*(kw-1)-1)//sw+1
                m = batch*ho*wo
                temp = b.alloc(m*co,'i32')
                raw = dict(attrs)
                raw['epilogue'] = [s for s in attrs.get('epilogue',[]) if s in ('bias_add','bias')]
                raw['output_dtype'] = 'i32'
                raw.pop('acc_scale',None)
                raw.pop('acc_scale_by_column',None)
                conv2d(b,ptrs[a],ptrs[w],temp,tensors[a]['shape'],raw,tensors,ptrs)
                pool(b,temp,ptrs[out],m,co,attrs)
            else:
                conv2d(b,ptrs[a],ptrs[w],ptrs[out],tensors[a]['shape'],attrs,tensors,ptrs)
        elif code == 'MOVEMENT':
            src,out = operands['src'],operands['dst']
            movement(b,ptrs[src],ptrs[out],tensors[src]['shape'],tensors[src]['dtype'],tensors[out]['dtype'],attrs,tensors,ptrs)
        elif code == 'RESIDUAL_ADD':
            lhs,rhs,out = operands['lhs'],operands['rhs'],operands['dst']
            residual_add(b,ptrs[lhs],ptrs[rhs],ptrs[out],tensors[out]['shape'],attrs,tensors,ptrs)
        elif code == 'BIAS_ADD':
            src,bias,out = operands['src'],operands['bias'],operands['dst']
            bias_add(b,ptrs[src],ptrs[bias],ptrs[out],tensors[out]['shape'],attrs,tensors,ptrs)
        elif code in ('ATTENTION_QK','ATTENTION_PV'):
            left,right = ('q','k') if code == 'ATTENTION_QK' else ('p','v')
            a,w,out = operands[left],operands[right],operands['dst']
            ashape,wshape = tensors[a]['shape'],tensors[w]['shape']
            trans = code == 'ATTENTION_QK'
            contract(ptrs[a],ptrs[w],ptrs[out],ashape[-2],ashape[-1],wshape[-2] if trans else wshape[-1],attrs,trans)
        elif code == 'BATCHED_MATMUL':
            a,w,out = operands['a'],operands['w'],operands['dst']
            shape = tensors[a]['shape']
            m,k,n = shape[-2],shape[-1],tensors[w]['shape'][-1]
            dtype = tensors[out]['dtype']
            def batch(index):
                contract(b.gep(ptrs[a],b.op('mul',index,m*k),'i8'),b.gep(ptrs[w],b.op('mul',index,k*n),'i8'),b.gep(ptrs[out],b.op('mul',index,m*n),dtype),m,k,n,attrs)
            b.loop(0,math.prod(shape[:-2]),1,batch)
        else:
            raise ValueError('Unsupported target command '+code)
    b.fence()
    return b.finish()
