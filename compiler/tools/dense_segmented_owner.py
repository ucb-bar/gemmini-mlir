"""Streamed segmented decode schedules explored through the ordinary manifest owner."""
import importlib.util
from pathlib import Path

_spec=importlib.util.spec_from_file_location('dense_ordinary_owner',Path(__file__).with_name('schedule_owner.py'))
owner=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(owner)

def binding(shape,caps):
    from dense import plan_dense
    from merlin.transforms.pipeline_state import PipelineContext
    base=owner.binding(shape,caps)
    mapping={'original':('regional_ready_transfers','balanced'),
             'identical':('regional_ready_transfers','balanced'),
             'batched':('segmented_ready_dma','balanced'),
             'ready':('segmented_ready_dma','balanced')}
    bias=bool(base['capsule']['operation'].get('attributes',{}).get('bias'))
    _,original=plan_dense(*shape,bias=bias,alternative=mapping['original'][0])
    modes={id(original):mapping['original']}
    def transform(result,mode):
        if owner.pin(base["original_source"])["sha256"] != base["source_sha256"]:
            raise ValueError("original source changed during segmented exploration")
        schedule,panel=mapping[mode]
        if mode in ('original','identical'):return original
        _,result=plan_dense(*shape,bias=bias,alternative=schedule,panel_shape=panel)
        result.revalidate();modes[id(result)]=(schedule,panel)
        return result
    def emit(result):
        result.revalidate();schedule,panel=modes[id(result)]
        payload,evidence=owner.manifest_emit(owner.PACKAGE,base['original_source'],schedule,panel)
        evidence.update(case=base['case'],panel_shape=panel,actual_checked_kernel_sha256=result.kernel.digest(),recipe_aliases=mapping)
        owner._PROGRAMS[owner.sha(payload)]={'schedule':schedule,'panel_shape':panel,'case':base['case'],
          'source':base['original_source'],'evidence':evidence,'shape':tuple(shape),
          'suite':'scalability' if Path(base['original_source']).parent.name=='public_sources' else 'core'}
        return payload.decode(),evidence
    base.update(source=original,transform=transform,emit=emit,
        context=PipelineContext(hardware=owner.sha((owner.PACKAGE/'data/facts.json').read_bytes()),
          resources=owner.sha({'caps':caps,'bundle':mapping}),numerical_policy='i8-i8-modular-i32:source-epilogue',
          execution='ordinary-capsule-Spike-complete-call',library=owner.library_pin()))
    return base

def identity(shape,bound):return owner.identity(shape,bound)
def evaluate(program,receipt,identity):return owner.evaluate(program,receipt,identity)
