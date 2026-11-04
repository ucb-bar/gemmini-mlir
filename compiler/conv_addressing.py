"""Pure address/control simplification for counted convolution coordinates.

Source loop bounds establish unsigned ranges. These permit exact division,
remainder and bounds-check simplifications without inspecting tensor payloads.
CSE is confined to one emitted basic block, so every reused SSA value dominates
its use. Singleton source axes become constants; nontrivial axes stay counted.
"""
from math import gcd


class ConvolutionAddressBuilder:
    def __init__(self, builder):
        self.builder=builder
        self.ranges={}
        self.expressions={}
        self.cache={}

    def __getattr__(self,name): return getattr(self.builder,name)

    def interval(self,value):
        if isinstance(value,int): return (value,value) if 0<=value<2**63 else None
        return self.ranges.get(value)

    def _cached(self,key,create):
        key=(self.builder.current_block,*key)
        if key not in self.cache: self.cache[key]=create()
        return self.cache[key]

    def multiple(self,value):
        if isinstance(value,int): return abs(value)
        expression=self.expressions.get(value)
        if expression is None: return 1
        name,a,b=expression
        if name=='mul': return self.multiple(a)*self.multiple(b)
        if name in ('add','sub'): return gcd(self.multiple(a),self.multiple(b))
        return 1

    def op(self,name,a,b,typ='i64'):
        if name in ('add','or','mul','and') and isinstance(a,int) and not isinstance(b,int):a,b=b,a
        if name in ('add','sub','or','shl','lshr') and b==0:return a
        if name in ('mul','udiv') and b==1:return a
        if name in ('mul','and') and (a==0 or b==0):return 0
        if typ=='i1' and name=='and':
            if a is True or a==1:return b
            if b is True or b==1:return a
        ar,br=self.interval(a),self.interval(b)
        # Only use divisibility where the source-bound arithmetic cannot wrap.
        if name=='urem' and isinstance(b,int) and b>0:
            if b==1 or (ar is not None and self.multiple(a)%b==0):return 0
            if ar is not None and ar[1]<b:return a
        if name=='udiv' and isinstance(b,int) and b>0 and ar is not None:
            if ar[1]<b:return 0
            expression=self.expressions.get(a)
            if expression and expression[0]=='mul':
                x,y=expression[1:]
                if y==b:return x
                if x==b:return y
        value=self._cached(('op',name,a,b,typ),lambda:self.builder.op(name,a,b,typ))
        if isinstance(value,str):
            self.expressions[value]=(name,a,b)
            result=None
            if ar is not None and br is not None:
                al,ah=ar;bl,bh=br
                if name=='add':result=(al+bl,ah+bh)
                elif name=='sub' and al>=bh:result=(al-bh,ah-bl)
                elif name=='mul':result=(al*bl,ah*bh)
                elif name=='udiv' and bl>0:result=(al//bh,ah//bl)
                elif name=='urem' and bl==bh and bl>0:result=(0,min(ah,bl-1))
                elif name=='and' and typ=='i1':result=(0,1)
            if result is not None and 0<=result[0]<=result[1]<2**63:self.ranges[value]=result
        return value

    def cmp(self,pred,a,b,ty='i64'):
        ar,br=self.interval(a),self.interval(b)
        if isinstance(a,int) and isinstance(b,int):
            if pred.startswith('u'):a,b=a%(1<<64),b%(1<<64)
            return {'eq':a==b,'ne':a!=b,'ult':a<b,'ule':a<=b,'ugt':a>b,'uge':a>=b,
                    'slt':a<b,'sle':a<=b,'sgt':a>b,'sge':a>=b}[pred]
        if a==b:return pred in ('eq','ule','uge','sle','sge')
        if ar is not None and br is not None:
            al,ah=ar;bl,bh=br
            if pred in ('ult','slt'):
                if ah<bl:return True
                if al>=bh:return False
            elif pred in ('ule','sle'):
                if ah<=bl:return True
                if al>bh:return False
            elif pred in ('eq','ne') and (ah<bl or bh<al):return pred=='ne'
        return self._cached(('cmp',pred,a,b,ty),lambda:self.builder.cmp(pred,a,b,ty))

    def select(self,condition,a,b,ty='i64'):
        if isinstance(condition,int):return a if condition else b
        if a==b:return a
        return self._cached(('select',condition,a,b,ty),lambda:self.builder.select(condition,a,b,ty))

    def loop(self,start,stop,step,callback):
        if all(isinstance(x,int) for x in (start,stop,step)):
            if step<=0:raise ValueError('convolution coordinate loop requires positive step')
            if stop<=start:return
            if start+step>=stop:
                callback(start);return
        def body(index):
            if all(isinstance(x,int) for x in (start,stop,step)) and 0<=start<stop<2**63:
                self.ranges[index]=(start,stop-1)
            callback(index)
        self.builder.loop(start,stop,step,body)

    def branch(self,condition,then_callback,else_callback=None):
        if isinstance(condition,int):
            if condition:return then_callback()
            if else_callback is not None:return else_callback()
            return
        return self.builder.branch(condition,then_callback,else_callback)

    def ptr_offset(self,base,offset):
        if offset==0:return base
        return self._cached(('ptr',base,offset),lambda:self.builder.ptr_offset(base,offset))

    def add(self,a,b,ty='i64'):return self.op('add',a,b,ty)
    def sub(self,a,b,ty='i64'):return self.op('sub',a,b,ty)
    def mul(self,a,b,ty='i64'):return self.op('mul',a,b,ty)
    def udiv(self,a,b,ty='i64'):return self.op('udiv',a,b,ty)
    def urem(self,a,b,ty='i64'):return self.op('urem',a,b,ty)
    def shl(self,a,b,ty='i64'):return self.op('shl',a,b,ty)
    def and_(self,a,b,ty='i64'):return self.op('and',a,b,ty)
    def or_(self,a,b,ty='i64'):return self.op('or',a,b,ty)
