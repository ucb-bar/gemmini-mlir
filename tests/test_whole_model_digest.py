import hashlib
import importlib.util
from pathlib import Path
import numpy as np
import pytest


def verifier():
    spec=importlib.util.spec_from_file_location('whole_digest',Path(__file__).with_name('fused_whole_model_probe.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.verify_output_digest


def test_digest_covers_complete_tail():
    values=np.array([0.,-0.,1.,4.],dtype=np.float32)
    digest=hashlib.sha256(values.astype('<f4').tobytes()).hexdigest()
    record=f'OUT_SHA256 f32le 4 16 {digest}'
    assert verifier()(record,values)['output_bytes']==16
    values[-1]=5.
    with pytest.raises(ValueError,match='differs'):verifier()(record,values)


@pytest.mark.parametrize('record',['OUT_SHA256 bad','OUT_SHA256 f32le 1 4 '+'0'*64,'OUT_SHA256 f32le 2 4 '+'0'*64])
def test_malformed_or_incomplete_digest_refused(record):
    with pytest.raises(ValueError):verifier()(record,np.array([1.],np.float32))


def test_absent_optional_digest_and_duplicate_records():
    assert verifier()('OUT 1 0',np.zeros(1,np.float32)) is None
    with pytest.raises(ValueError,match='exactly one'):verifier()('OUT_SHA256 x\nOUT_SHA256 y',np.zeros(1,np.float32))
