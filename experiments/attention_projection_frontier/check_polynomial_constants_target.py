"""Existing rational word oracle applied to the admitted constant-context helper."""
from pathlib import Path
B=Path(__file__).resolve().parents[2]
s=(B/'experiments/attention_projection_frontier/qualify_monotone_target.py').read_text()
s=s.replace("w=base/'out/rounded_monotone_target_v3'", "w=base/'out/artifacts/probes/prepared-polynomial-constants/target_independent'")
s=s.replace("d=base/'out/rounded_monotone_group/candidate/target_numeric'", "d=base/'out/artifacts/probes/prepared-polynomial-constants/candidate/target_numeric'")
s=s.replace('#include "monotone_bit_polynomial.h"','#define MERLIN_POLYNOMIAL_BOUNDED_FLOOR 1\n#include "prepared_polynomial_constants.h"')
s=s.replace('if(p.fast_valid!=(mode==0))return 2;', 'if(p.fast_valid!=(mode==0)||merlin_polynomial_constants_admit(&p)!=(mode==0))return 2;')
a='merlin_f32_interval interval=merlin_monotone_bit_polynomial_apply_words(merlin_interval(x,x),&p);'
b='merlin_f32_interval xs[4],ys[4];for(int j=0;j<4;j++)xs[j]=merlin_interval(x,x);merlin_polynomial_constants_four(xs,&p,merlin_polynomial_constants_admit(&p),ys);merlin_f32_interval interval=ys[0];'
assert s.count(a)==1;s=s.replace(a,b)
s=s.replace('checks++;','''float other;unsigned next=(i+1)%(sizeof(inputs)/sizeof(inputs[0]));memcpy(&other,&inputs[next],4);
   float lo=x<other?x:other,hi=x<other?other:x;
   for(int j=0;j<4;j++)xs[j]=merlin_interval(lo,hi);
   merlin_polynomial_constants_four(xs,&p,merlin_polynomial_constants_admit(&p),ys);
   uint32_t lower=expected[i]<expected[next]?expected[i]:expected[next],upper=expected[i]>expected[next]?expected[i]:expected[next];
   for(int j=0;j<4;j++)if(!ys[j].valid||merlin_interval_bits(ys[j].lo)!=lower||merlin_interval_bits(ys[j].hi)!=upper)return 8;
   checks++;''')
s=s.replace("json.loads((d/'compile.json').read_text())['commands'][0]", "json.loads((d.parent/'build.json').read_text())['roles']['target_numeric']['commands'][0]")
output=B/'out/artifacts/probes/prepared-polynomial-constants/target_driver.py';output.write_text(s);exec(compile(s,str(__file__),'exec'))
