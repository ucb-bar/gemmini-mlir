import importlib.util
from pathlib import Path
import numpy as np


def load_quality():
    spec=importlib.util.spec_from_file_location('whole_probe',Path(__file__).with_name('fused_whole_model_probe.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.quality


def test_default_quality_requires_source_bits():
    quality=load_quality();source=np.array([0.,1.],np.float32)
    changed=np.array([-0.,1.],np.float32)
    result=quality(changed,source)
    assert result['allclose'] and not result['exact_equal'] and not result['quality_pass']


def test_explicit_bounded_quality_keeps_original_metrics():
    quality=load_quality();source=np.array([1.,2.],np.float32);changed=np.array([1.01,2.01],np.float32)
    result=quality(changed,source,allow_bounded=True,atol=.02,rtol=0.)
    assert result['quality_pass'] and result['allclose'] and not result['exact_equal']
    assert result['relative_l2']>0 and result['max_abs_error']>0
    assert not quality(changed,source,allow_bounded=True,atol=.001,rtol=0.)['quality_pass']
