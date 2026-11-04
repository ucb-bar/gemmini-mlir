"""Opt-in ordinary-product choices from array tile geometry.

These performance guards do not establish residency or numerical legality.
Each selected lowering still checks its own complete source/storage contract.
"""
from hashlib import sha256
from pathlib import Path
import json

from merlin.transforms.pipeline_policy import PolicyFacts, ResourcePredicate
from merlin.transforms.pipeline_state import AnalysisStamp, PipelineContext
from . import isa


def ordinary_rhs_choice(ctx, members):
    """Select only the raw, single-array-row class covered by paired trials."""
    shapes = []
    for row in members:
        a,b,c = (ctx.spec(row[key]) for key in ('lhs','rhs','dst'))
        attrs = row.get('attributes', {})
        if (a['dtype'] != 'i8' or b['dtype'] != 'i8' or c['dtype'] != 'i32'
                or len(a['shape']) != 2 or len(b['shape']) != 2
                or a['shape'][1] != b['shape'][0]
                or c['shape'] != [a['shape'][0],b['shape'][1]]
                or attrs.get('epilogue') or attrs.get('bias')
                or attrs.get('accumulator_init') or attrs.get('relu')
                or attrs.get('acc_scale_by_column') is not None
                or attrs.get('requant_shift') is not None
                or attrs.get('output_dtype','i32') != 'i32'
                or float(attrs.get('acc_scale',1)) != 1):
            return 'independent', {'reason':'outside paired raw-product class'}
        shapes.append((int(a['shape'][0]),int(a['shape'][1]),int(b['shape'][1])))
    if not shapes or any(min(s) <= 0 or s[1:] != shapes[0][1:] for s in shapes):
        return 'independent', {'reason':'incompatible product geometry'}
    d = isa.DIM
    resources = {'single_array_row':int(all((m+d-1)//d == 1 for m,_,_ in shapes)),
                 'multiple_column_tiles':int((shapes[0][2]+d-1)//d > 1)}
    facts = PolicyFacts(AnalysisStamp(sha256(json.dumps(shapes).encode()).hexdigest(),
        PipelineContext(resources=sha256(Path(isa.__file__).read_bytes()).hexdigest()),
        'source-products'),shapes=tuple(shapes),resources=tuple(sorted(resources.items())))
    if not ResourcePredicate('single_array_row',1).match(facts):
        selected = 'independent'
    elif ResourcePredicate('multiple_column_tiles',1).match(facts):
        selected = 'panels'
    else:
        selected = 'resident'
    return selected, {'selected':selected,'resources':resources,'array_edge':d,
        'basis':'paired single-array-row raw products; column tiling boundary',
        'legality':'selected lowering independently verifies source and storage'}
