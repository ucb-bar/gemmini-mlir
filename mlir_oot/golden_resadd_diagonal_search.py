"""Exact feasibility search for i8 diagonal products plus one f32 readout.

This is a proof experiment only: it does not bind or rewrite any graph.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

CPP = r'''
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <random>
#include <vector>
struct Pair { int a,b,y; };
int main(int argc,char **argv) {
  if(argc!=2)return 2;
  std::ifstream input(argv[1],std::ios::binary);
  std::vector<int16_t> expected(65536);
  input.read((char*)expected.data(),expected.size()*sizeof(int16_t));
  if(!input)return 3;
  std::vector<Pair> pairs;
  for(int a=-128;a<128;a++)for(int b=-128;b<128;b++)
    pairs.push_back({a,b,expected[(a+128)*256+b+128]});
  std::mt19937 rng(20261005);std::shuffle(pairs.begin(),pairs.end(),rng);
  // Widened real-product intervals contain every value whose f32 rounding,
  // nearest-even integer rounding, and ReLU/i8 clipping produce y.
  double lower[128],upper[128];
  for(int y=0;y<128;y++) {
    float l=float(y)-0.5f,u=float(y)+0.5f;
    lower[y]= y==0 ? 0.0 : (y%2 ? (double(l)+std::nextafterf(l,INFINITY))/2 :
                                                  (double(l)+std::nextafterf(l,-INFINITY))/2);
    upper[y]= y==127 ? INFINITY : (y%2 ? (double(u)+std::nextafterf(u,-INFINITY))/2 :
                                                    (double(u)+std::nextafterf(u,INFINITY))/2);
  }
  unsigned rejected=0,survivors=0,exact=0;uint64_t tested_scales=0;
  std::printf("{\"accepted\":[");
  for(int p=0;p<=127;p++)for(int q=0;q<=127;q++) {
    double lo=0,hi=INFINITY;bool impossible=false;
    for(auto v:pairs) {
      int acc=v.a*p+v.b*q;
      if(acc<=0) {if(v.y!=0){impossible=true;break;}continue;}
      // One outward binary64 ulp makes division bounds conservative.
      lo=std::max(lo,std::nextafter(lower[v.y]/acc,-INFINITY));
      hi=std::min(hi,std::nextafter(upper[v.y]/acc,INFINITY));
      if(lo>hi){impossible=true;break;}
    }
    if(impossible){rejected++;continue;}
    survivors++;
    float first=float(lo),last=float(hi);
    if(double(first)<lo)first=std::nextafterf(first,INFINITY);
    if(double(last)>hi)last=std::nextafterf(last,-INFINITY);
    if(first<=0)first=std::nextafterf(0,INFINITY);
    for(float scale=first;scale<=last && std::isfinite(scale);scale=std::nextafterf(scale,INFINITY)) {
      tested_scales++;bool match=true;
      for(auto v:pairs) {
        volatile float product=float(v.a*p+v.b*q)*scale;
        int result=int(std::nearbyintf(product));
        result=std::min(127,std::max(0,result));
        if(result!=v.y){match=false;break;}
      }
      if(match) {
        if(exact++)std::printf(",");
        std::printf("{\"p\":%d,\"q\":%d,\"scale\":%.17g}",p,q,double(scale));
        break; // One exact representative establishes feasibility for p,q.
      }
      if(tested_scales>100000000){std::fprintf(stderr,"unbounded candidate interval\n");return 4;}
    }
  }
  std::printf("],\"integer_pairs\":16384,\"interval_rejected\":%u,\"interval_survivors\":%u,\"f32_scales_tested\":%llu}\n",rejected,survivors,(unsigned long long)tested_scales);
}
'''


def source_values(source: dict) -> np.ndarray:
    if not source['relu']:
        raise ValueError('this feasibility experiment supports ReLU residuals only')
    lhs, rhs, output = [np.float32(source[k]) for k in ('lhs_scale','rhs_scale','output_scale')]
    if not all(np.isfinite(x) and x>0 for x in (lhs,rhs,output)):
        raise ValueError('positive finite scalar f32 qparams required')
    a=np.arange(-128,128,dtype=np.float32)[:,None]
    b=np.arange(-128,128,dtype=np.float32)[None,:]
    value=np.add(a*lhs,b*rhs,dtype=np.float32)
    value=np.maximum(value,np.float32(0))
    return np.clip(np.rint(value*np.float32(1.0/float(output))),0,127).astype('<i2')


def search(source: dict, engine: Path, work: Path) -> dict:
    expected=source_values(source)
    path=work/'expected_pairs.bin';path.write_bytes(expected.tobytes())
    result=json.loads(subprocess.check_output([str(engine),str(path)],text=True))
    a=np.arange(-128,128,dtype=np.int32)[:,None]
    b=np.arange(-128,128,dtype=np.int32)[None,:]
    for candidate in result['accepted']:
        value=(a*candidate['p']+b*candidate['q']).astype(np.float32)*np.float32(candidate['scale'])
        got=np.clip(np.rint(value),0,127).astype(np.int16)
        if not np.array_equal(got,expected):
            raise AssertionError('independent exhaustive candidate verification failed')
    # Retain a concrete contradictory interval for the closest bounded ratio.
    pp,qq=np.meshgrid(np.arange(1,128),np.arange(1,128),indexing='ij')
    index=np.argmin(np.abs(pp/qq-source['lhs_scale']/source['rhs_scale']))
    p0,q0=int(pp.flat[index]),int(qq.flat[index])
    acc=a*p0+b*q0
    positive=acc>0
    y=expected.astype(np.int32)
    low=(y-.5).astype(np.float32);high=(y+.5).astype(np.float32)
    low_direction=np.where(y%2, np.float32(np.inf),np.float32(-np.inf))
    high_direction=-low_direction
    lower=(low.astype(np.float64)+np.nextafter(low,low_direction))/2
    upper=(high.astype(np.float64)+np.nextafter(high,high_direction))/2
    lower[y==0]=0;upper[y==127]=np.inf
    lows=np.full(y.shape,-np.inf);highs=np.full(y.shape,np.inf)
    lows[positive]=np.nextafter(lower[positive]/acc[positive],-np.inf)
    highs[positive]=np.nextafter(upper[positive]/acc[positive],np.inf)
    li=np.unravel_index(np.argmax(lows),y.shape);ui=np.unravel_index(np.argmin(highs),y.shape)
    def witness(index):
        i,j=index
        return dict(lhs=int(i)-128,rhs=int(j)-128,accumulator=int(acc[index]),source=int(y[index]))
    result['closest_ratio_interval']=dict(p=p0,q=q0,lower=float(lows[li]),upper=float(highs[ui]),
        lower_witness=witness(li),upper_witness=witness(ui),contradictory=bool(lows[li]>highs[ui]))
    result.update(source=source,pairs=65536,expected_pair_sha256=hashlib.sha256(expected.tobytes()).hexdigest())
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('proof_report',type=Path);p.add_argument('--workdir',type=Path,required=True)
    args=p.parse_args();args.workdir.mkdir(parents=True,exist_ok=False)
    cpp=args.workdir/'search.cc';cpp.write_text(CPP);engine=args.workdir/'search'
    subprocess.run(['c++','-O3','-std=c++17','-ffp-contract=off',str(cpp),'-o',str(engine)],check=True)
    # Exact positive controls include a nontrivial rational coefficient ratio.
    controls=[]
    for source in ({'lhs_scale':.5,'rhs_scale':.25,'output_scale':1.,'relu':True},
                   {'lhs_scale':1.,'rhs_scale':1.,'output_scale':1.,'relu':True},
                   {'lhs_scale':127/256,'rhs_scale':113/256,'output_scale':1.,'relu':True}):
        control=search(source,engine,args.workdir)
        if not control['accepted']:raise AssertionError('positive control failed')
        controls.append(control)
    report=json.loads(args.proof_report.read_text());results=[]
    for row in report['accepted']+report['refused']:
        result=search(row['source'],engine,args.workdir);result.update(region=row['region'],shape=row['shape'])
        results.append(result)
        print(row['region'],len(result['accepted']),'exact candidates',flush=True)
    result={'schema':'residual_diagonal_feasibility_v1','scope':'p,q integers0..127; exact integer accumulation; one positive finite f32 scale; ReLU/i8 readout; all65536 signed-i8 pairs; no graph rewrite',
            'proof_report':str(args.proof_report),'proof_report_sha256':hashlib.sha256(args.proof_report.read_bytes()).hexdigest(),
            'search_source_sha256':hashlib.sha256(CPP.encode()).hexdigest(),'controls':controls,'results':results}
    (args.workdir/'result.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
