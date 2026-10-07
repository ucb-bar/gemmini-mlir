"""Attribute existing complete-group execution; never rerun timing for debug."""
from pathlib import Path
B=Path(__file__).resolve().parents[2]
s=(B/'experiments/attention_projection_frontier/polynomial_constants_attribution.py').read_text()
s=s.replace('prepared-polynomial-constants','source-polynomial-prefix-table')
exec(compile(s,str(__file__),'exec'))
