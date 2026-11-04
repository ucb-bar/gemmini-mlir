"""Exact affine radix decoding for original counted convolution coordinates.

This changes only pure unsigned constant div/rem with a source-bound,
nonnegative, nonwrapping affine numerator. It neither moves instructions nor
changes counted iteration/effect order. Existing address/LICM selection remains
in force, including its unchanged fallback for every unproved expression.
"""
from pathlib import Path
from conv_addressing import ConvolutionAddressBuilder
from channel_index_hoisting import HoistedConvolutionAddressBuilder, wrap as _wrap
from merlin.compile.tuning_knobs import knob

VALUES=('off','affine_radix')
DECLARATION=dict(values=list(VALUES),mechanism='exact nonwrapping affine quotient/remainder decomposition for counted convolution coordinates',note='Default off. Only constant unsigned divisors and source-proven bounded affine numerators; original loops, effects and completion unchanged.')


def selected():
    return knob('convolution_index_strength',default='off',values=VALUES,root=Path(__file__).parent)


class _AffineRadix:
    def __init__(self,builder):
        super().__init__(builder)
        self.strength_receipt=dict(rewritten_udiv=0,rewritten_urem=0,proofs=[])

    def _affine(self,value):
        """Return nonnegative coefficient terms, each with an exact SSA range.

        Reject missing ranges and trees with a potentially wrapped intermediate.
        Existing builder arithmetic has no nsw/nuw flags; signed-positive source
        ranges below 2**63 are a sufficient stronger premise for unsigned math.
        """
        terms={};constant=0;budget=[128]
        def visit(v,multiplier):
            nonlocal constant
            budget[0]-=1
            if budget[0]<0 or multiplier>=2**63:return False
            span=self.interval(v)
            if span is None or not 0<=span[0]<=span[1]<2**63:return False
            if isinstance(v,int):constant+=multiplier*v;return constant<2**63
            expression=self.expressions.get(v)
            if expression:
                op,a,b=expression
                if op=='add':return visit(a,multiplier) and visit(b,multiplier)
                if op=='mul':
                    if isinstance(b,int) and b>=0:return visit(a,multiplier*b)
                    if isinstance(a,int) and a>=0:return visit(b,multiplier*a)
            terms[v]=terms.get(v,0)+multiplier
            return len(terms)<=32 and terms[v]<2**63
        if not visit(value,1):return None
        if constant+sum(c*self.interval(v)[1] for v,c in terms.items())>=2**63:return None
        return constant,terms

    def _decode(self,numerator,divisor,remainder):
        span=self.interval(numerator)
        if span is None or not 0<=span[0]<=span[1]<2**63:return None
        affine=self._affine(numerator)
        if affine is None:return None
        constant,terms=affine
        qconstant,rconstant=divmod(constant,divisor)
        quotient={};residual={}
        for value,coefficient in terms.items():
            q,r=divmod(coefficient,divisor)
            if q:quotient[value]=q
            if r:residual[value]=r
        lo=rconstant+sum(c*self.interval(v)[0] for v,c in residual.items())
        hi=rconstant+sum(c*self.interval(v)[1] for v,c in residual.items())
        qlo,qhi=lo//divisor,hi//divisor
        if qlo!=qhi:return None
        # Avoid expanding a numerator that contains no extractable radix digit.
        # A constant residual quotient alone is already handled by range folding.
        if not quotient:return None
        def materialize(values,initial):
            out=initial
            for value,coefficient in values.items():
                term=value if coefficient==1 else self.op('mul',value,coefficient)
                out=term if out==0 else self.op('add',out,term)
            return out
        if remainder:
            value=materialize(residual,rconstant)
            if qlo:value=self.op('sub',value,qlo*divisor)
            result_range=(lo-qlo*divisor,hi-qlo*divisor)
        else:
            value=materialize(quotient,qconstant+qlo)
            result_range=(span[0]//divisor,span[1]//divisor)
        if isinstance(value,str):self.ranges[value]=result_range
        kind='urem' if remainder else 'udiv'
        self.strength_receipt['rewritten_'+kind]+=1
        # Bounded compile-time witness; never depends on tensor values.
        if len(self.strength_receipt['proofs'])<32:
            self.strength_receipt['proofs'].append(dict(op=kind,divisor=divisor,numerator_range=span,residual_range=(lo,hi),residual_quotient=qlo,terms=len(terms),nonwrapping_unsigned=True))
        return value

    def op(self,name,a,b,typ='i64'):
        if typ=='i64' and name in ('udiv','urem') and isinstance(b,int) and 1<b<2**63 and not isinstance(a,int):
            key=(self.builder.current_block,'affine_radix',name,a,b,typ)
            if key in self.cache:return self.cache[key]
            value=self._decode(a,b,name=='urem')
            if value is not None:
                self.cache[key]=value
                return value
        return super().op(name,a,b,typ)


class AffineConvolutionAddressBuilder(_AffineRadix,ConvolutionAddressBuilder):
    pass


class AffineHoistedConvolutionAddressBuilder(_AffineRadix,HoistedConvolutionAddressBuilder):
    pass


def wrap(builder,*,mode=None):
    """Drop-in replacement for channel_index_hoisting.wrap.

    Preserve its existing choice and decline unfamiliar builder extensions.
    No extra wrapper or motion is introduced in default-off mode.
    """
    mode=selected() if mode is None else mode
    if mode not in VALUES:raise ValueError('unknown convolution index strength mode')
    original=_wrap(builder)
    if mode=='off':return original
    if type(original) is HoistedConvolutionAddressBuilder:return AffineHoistedConvolutionAddressBuilder(builder)
    if type(original) is ConvolutionAddressBuilder:return AffineConvolutionAddressBuilder(builder)
    return original
