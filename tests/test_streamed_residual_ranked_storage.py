"""Explicit panel transaction boundaries and actual ranked output checks."""

import os
import shutil
import subprocess
from itertools import pairwise

import pytest
from merlin.llvmlower.quantized_affine_pair import derive

from mlir_oot.golden_streamed_resadd import panel_storage, ranked_adapter


def route(m=48):
    return {
        "symbol": "streamed_residual",
        "kernel": "streamed_kernel",
        "m": m,
        "n": 64,
        "single_output_guard": True,
        "proof": derive(
            0.011258588172495365,
            0.00940733402967453,
            0.011643771082162857,
            p=298,
            q=249,
            scale=0.0032446938566863537,
            relu=True,
        ),
    }


@pytest.mark.parametrize("m", [16, 32, 48, 80, 128])
@pytest.mark.parametrize("granule", [16, 32, 64, 128, 256, 512, 1024])
def test_declared_transaction_blocks_are_partitioned(m, granule):
    plan = panel_storage(m, coherence_granule_bytes=granule, dma_max_request_bytes=64)
    bound = plan["required_output_alignment"]
    for phase in (0, bound, 7 * bound):
        blocks = []
        for j in range(plan["panels"]):
            begin, end = phase + 1024 * j, phase + 1024 * (j + 1)
            blocks.append(set(range(begin // bound, (end - 1) // bound + 1)))
        for a, b in pairwise(blocks):
            assert not a & b
    assert plan["output_bytes"] == m * 64


@pytest.mark.parametrize(
    "m,coherence,dma",
    [
        (0, 64, 64),
        (17, 64, 64),
        (True, 64, 64),
        (1 << 60, 64, 64),
        (16, None, 64),
        (16, 0, 64),
        (16, 63, 64),
        (16, 64, True),
        (16, 64, 96),
        (16, 2048, 64),
        (16, 64, 2048),
    ],
)
def test_unknown_or_unpartitioned_bounds_refuse(m, coherence, dma):
    with pytest.raises(ValueError):
        panel_storage(m, coherence_granule_bytes=coherence, dma_max_request_bytes=dma)


def test_ranked_adapter_preserves_shared_checks_and_removes_serial_scan():
    selected = route()
    code, proof = ranked_adapter(
        selected, coherence_granule_bytes=64, dma_max_request_bytes=64
    )
    assert "strides[0]!=64" in code and "UINTPTR_MAX-3072" in code
    assert "_overlap(pc,pa)" in code and "_overlap(pc,pb)" in code
    assert "c->offset!=0 || c->allocated!=c->aligned" in code
    assert "((uintptr_t)pc&63)" in code
    assert code.count("streamed_kernel(pa,pb,pc,") == 1
    assert "_correct(" not in code and "*r=*c;" in code
    assert proof["required_output_alignment"] == 64
    changed = route()
    changed["proof"]["pairs"] = 65535
    with pytest.raises(ValueError, match="certificate"):
        ranked_adapter(changed, coherence_granule_bytes=64, dma_max_request_bytes=64)


def test_native_descriptor_alignment_offset_owner_and_overlap_checks(tmp_path):
    compiler = os.environ.get("MERLIN_CLANG") or shutil.which("clang")
    if not compiler:
        pytest.skip("native C compiler absent")
    code, _ = ranked_adapter(
        route(32), coherence_granule_bytes=64, dma_max_request_bytes=64
    )
    source = tmp_path / "adapter.c"
    source.write_text(
        code
        + r"""
#include <stdlib.h>
#include <string.h>
static int calls;
void streamed_kernel(const int8_t*a,const int8_t*b,int8_t*c,const int8_t*w){
 (void)w;calls++;for(size_t i=0;i<2048;i++)c[i]=a[i]+b[i];
}
static int8_t a[2048] __attribute__((aligned(64))),b[2048] __attribute__((aligned(64)));
static int8_t output[2112] __attribute__((aligned(64)));
int main(int argc,char**argv){
 int mode=argc>1?atoi(argv[1]):0;
 for(size_t i=0;i<2048;i++){a[i]=(int8_t)(i%23);b[i]=(int8_t)(i%31);}
 joint_memref2 da={a,a,0,{32,64},{64,1}},db={b,b,0,{32,64},{64,1}};
 joint_memref2 dc={output,output,0,{32,64},{64,1}},result={0};
 if(mode==1)dc.offset=1;
 if(mode==2)dc.allocated=dc.aligned=output+1;
 if(mode==3)dc.allocated=output+64;
 if(mode==4)dc.allocated=dc.aligned=a;
 if(mode==5)dc.sizes[0]=31;
 _mlir_ciface_streamed_residual(&result,&da,&db,&dc);
 if(calls!=1||memcmp(&result,&dc,sizeof(dc)))return 2;
 for(size_t i=0;i<2048;i++)if(output[i]!=a[i]+b[i])return 3;
 return 0;
}
"""
    )
    executable = tmp_path / "adapter"
    subprocess.run(
        [compiler, "-std=c11", "-O2", str(source), "-o", str(executable)], check=True
    )
    assert subprocess.run([str(executable)], check=False).returncode == 0
    for mode in range(1, 6):
        assert subprocess.run([str(executable), str(mode)], check=False).returncode != 0
