"""Provider-owned raw convolution wrapper; numeric readout is owned by Merlin."""
from merlin.llvmlower.integer_readout import derive, evaluate, fixedpoint_candidate, emit_readout


def emit_conv_wrapper(proof,symbol,kernel_symbol,count,*,fixedpoint=False,saturation_first=False,packet=1):
    """Raw ABI: A,B,i8 output, caller-owned distinct i32 scratch, scratch elements.

    Kernel must write count row-major i32 accumulators and fence before returning.
    Caller grants exclusive scratch ownership for this call. No static scratch or
    heap allocation is introduced. A/B/output/scratch overlap is forbidden.
    """
    if type(count) is not int or count<=0:raise ValueError('positive static output count required')
    readout=symbol+'_readout'
    return emit_readout(proof,readout,fixedpoint=fixedpoint,saturation_first=saturation_first,packet=packet)+f'''
extern void {kernel_symbol}(const int8_t*,const int8_t*,int32_t*);
void {symbol}(const int8_t*a,const int8_t*b,int8_t*out,int32_t*scratch,size_t scratch_elements){{
 if(!a || !b || !out || !scratch || scratch_elements<{count} || ((uintptr_t)scratch&3))__builtin_trap();
 {kernel_symbol}(a,b,scratch);
 {readout}(scratch,out,{count});
}}
'''
