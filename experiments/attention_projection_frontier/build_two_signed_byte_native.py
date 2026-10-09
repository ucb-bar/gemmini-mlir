"""One fixed two-signed-byte residual-aware attention native screen.

No center-only observer substitution. Original source/FMA/nonlinear graph and
replay stay selected. Nonzero representation errors use existing checked bounds;
the zero-error RMS4 fast path remains guarded by actual numeric equality.
"""
from pathlib import Path
base=Path(__file__).resolve().parents[2]
s=(base/'experiments/attention_projection_frontier/capture_initial_bounds.py').read_text()
s=s.replace("h=base/'out/observation_frontier/initial_bounds'","h=base/'out/observation_frontier/two_signed_byte_native'")
needle="(dest/'provider.c').write_text(s)"
replacement='''from merlin.llvmlower.two_signed_byte_block_float import c_header
from merlin.llvmlower.radix_product_groups import plan_radix_product_groups
from merlin.llvmlower.radix_integer_reconstruct import c_fused_header
(dest/'two_signed_byte_block_float.h').write_text(c_header())
start=s.index('#ifndef MERLIN_RADIX_FUSED_INTEGER_RECONSTRUCT_H')
end=s.index('\\n#endif',start)+len('\\n#endif')
# The first endif closes the binary-format preprocessor check; the second
# closes the complete independently generated reconstruction helper.
end=s.index('\\n#endif',end)+len('\\n#endif')
s=s[:start]+c_fused_header(plan_radix_product_groups(radix_bits=8,digits=2,reduction_length=192))+s[end:]
s=s.replace('#include "bf16_radix_pack.h"','#include "bf16_radix_pack.h"\\n#include "two_signed_byte_block_float.h"')
old="""merlin_bf16_radix_row_widen(&environment,source+row*k,k,1,rd+row*k,1,
    planes+(transpose?row:row*k),m*k,transpose?m:1,3,&step,
    lower?lower+row*k:0,upper?upper+row*k:0,flags+row)"""
new="""merlin_two_signed_byte_row(&environment,source+row*k,k,rd+row*k,
    planes+(transpose?row:row*k),m*k,transpose?m:1,&step,
    lower?lower+row*k:0,upper?upper+row*k:0,flags+row)"""
assert s.count(old)==1;s=s.replace(old,new)
(dest/'provider.c').write_text(s)
'''
s=s.replace(needle,replacement)
s=s.replace('shape=(3*m*k,)).reshape(3,m,k)','shape=(2*m*k,)).reshape(2,m,k)').replace('shape=(3*k*n,)).reshape(3,k,n)','shape=(2*k*n,)).reshape(2,k,n)').replace('for ad in range(3):','for ad in range(2):').replace('if 0<=bd<3:','if 0<=bd<2:')
s=s.replace('product_calls==23040','product_calls==13824').replace("scope='Diagnostic snapshots before eager final quantizer refinement; original refinement remains active. RMS4 intervals are approximate, not rigorous source certificates.'","scope='Fixed2 signed-byte radix256 selected representation; actual residual norm bounds, original source replay/observer retained.4 MAC terms/3 callback readouts; old private3-plane operand capacity retained and counted. No source equality inferred from selected representation.'")
s=s.replace("assert snapshots==48 and not errors and product_calls==13824 and not record['bit_mismatches'] and counts[1]==0","assert not errors\nprint('SELECTED_COVERAGE',snapshots,product_calls,list(counts),flush=True)")
# Generated driver provenance is retained before any native observation.
out=base/'out/two_signed_byte_driver.py';out.write_text(s)
exec(compile(s,str(__file__),'exec'))
