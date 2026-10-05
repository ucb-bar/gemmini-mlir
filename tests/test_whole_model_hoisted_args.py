import importlib.util
from pathlib import Path

import numpy as np
import pytest

from merlin.llvmlower.quant_hoist import HoistedArg, write_plan, write_values


def probe():
    spec=importlib.util.spec_from_file_location('whole_probe',Path(__file__).with_name('fused_whole_model_probe.py'))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hoisted_native_arguments_keep_plan_order_and_bytes(tmp_path,monkeypatch):
    module=probe()
    original=np.zeros((1,4),dtype=np.int8)
    monkeypatch.setattr(module,'resolve_forward_args',lambda capture:[original])
    a=np.arange(6,dtype=np.int8).reshape(2,3)
    b=np.arange(3,dtype=np.float32)
    write_plan(tmp_path,[HoistedArg('b',b.shape,'f32'),HoistedArg('a',a.shape,'i8')])
    write_values(tmp_path,{'a':a,'b':b})
    args=module.forward_args(tmp_path,tmp_path)
    assert args[0] is original
    assert np.array_equal(args[1],b) and args[1].dtype==b.dtype
    assert np.array_equal(args[2],a) and args[2].dtype==a.dtype
    assert all(x.flags.c_contiguous for x in args)


@pytest.mark.parametrize('mismatch',['missing','shape','dtype'])
def test_corrupt_hoisted_native_argument_refused(tmp_path,monkeypatch,mismatch):
    module=probe()
    monkeypatch.setattr(module,'resolve_forward_args',lambda capture:[])
    write_plan(tmp_path,[HoistedArg('weight',(2,3),'i8')])
    values={} if mismatch=='missing' else {'weight':np.zeros((3,2) if mismatch=='shape' else (2,3),dtype=np.float32 if mismatch=='dtype' else np.int8)}
    write_values(tmp_path,values)
    with pytest.raises(ValueError,match='hoisted native argument'):
        module.forward_args(tmp_path,tmp_path)
