"""Bind generic immutable polynomial context to the qualified private provider.

This experiment adapter retains the complete admitted span/maximum source body;
only the helper selection is replaced. Generic arithmetic lives in Merlin.
"""
from pathlib import Path
import hashlib
from merlin.llvmlower.prepared_polynomial_constants import c_header
from merlin.llvmlower.rounded_polynomial_monotonicity import RoundedPolynomialMonotonicity,consume_rounded_polynomial_monotonicity
from merlin.llvmlower.source_numeric_capability import SourceNumericContract
B=Path(__file__).resolve().parents[2]

def bind_source(source):
 changes=[('static int soft_details(const float*q,','#include "prepared_polynomial_constants.h"\nstatic int soft_details(const float*q,'),('  return soft_details_checked(q,k,mask,lo,hi,p,pl,ph,dl,dh,alpha,rows,counts,yl,yh,maxima);','  return soft_details_checked(q,k,mask,lo,hi,p,pl,ph,dl,dh,alpha,rows,counts,yl,yh,maxima);\n const int polynomial_constants_admitted=merlin_polynomial_constants_admit(&root_prepared);'),('    merlin_polynomial_words_four(xs,&root_prepared,ys);','    merlin_polynomial_constants_four(xs,&root_prepared,polynomial_constants_admitted,ys);')]
 for old,new in changes:
  if source.count(old)!=1:raise ValueError('qualified source context changed')
  source=source.replace(old,new)
 return source

def bind_headers(destination):
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();M=B/'out/observation_frontier/rounded_monotonicity';h=destination/'monotone_bit_polynomial.h'
 proof=RoundedPolynomialMonotonicity((0xc2aeac50,0x3fb8aa3b,0xbda235d5,0xbe65b8f5,0x3e9b69f0,0x38e077a1,0x4b000000,0x4e7e0000),1118743633,sha(M/'independent_qualification.json'),sha(M/'independent.c'),sha(h),True,True,True)
 contract=SourceNumericContract(*([True]*7),standard_floor_values=True,floor_interposition_unobserved=True)
 original=h.read_text();(destination/'prepared_polynomial_constants.h').write_text(c_header(original,proof,contract));h.write_text(consume_rounded_polynomial_monotonicity(original,proof,contract))
 return dict(proof=vars(proof),contract=vars(contract))
