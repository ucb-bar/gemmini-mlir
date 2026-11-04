"""Selected target binding of typed pure movement to unchanged raw Encoder.

Nonshrinking ACC MVIN (overwrite, no accumulation), followed by FULL ACC MVOUT
copies the original 32 bits. No compute/preload/scaling/narrowing instruction is
used. Source f32 remains f32; i32 is only the command's raw storage selector.
Each transfer retains original disjoint full-buffer and completion obligations.
"""
from pathlib import Path
from hashlib import sha256
import json
from merlin.compile.scheduling.raw_storage_view import raw_storage_view


def emit_raw_run(enc, source, output, run, dtype):
    view=raw_storage_view(dtype)
    enc.load(enc.ptr_offset(source,enc.mul(run.source_element,view.itemsize)),
        1<<31,run.rows,run.columns,run.source_stride*view.itemsize,
        shrunk=view.storage_dtype=='i8')
    enc.fence()
    enc.store(enc.ptr_offset(output,enc.mul(run.output_element,view.itemsize)),
        0,run.rows,run.columns,run.output_stride*view.itemsize,dtype=view.storage_dtype)
    enc.fence()


def storage_receipt(dtype):
    view=raw_storage_view(dtype)
    return dict(view.record(),
        transfer='ACC overwrite MVIN; FULL raw32 MVOUT for 32-bit payload',
        load_shrink=view.storage_dtype=='i8',load_accumulate=False,
        output_full=view.storage_dtype=='i32',payload_compute=False,
        encoder_sha256=sha256(Path(__file__).with_name('encoding.py').read_bytes()).hexdigest(),
        adapter_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        qualification='selected target raw storage semantics; native IEEE and arbitrary raw32 fixtures; no floating arithmetic authority')


def raw_storage_admission(dtype, facts):
    if dtype != 'f32':
        return dict(source_dtype=dtype,interpretation='existing integer storage')
    try:
        p=Path(__file__).with_name('data')/'raw_storage_contract.json'
        proof=json.loads(p.read_text())
        from encoding import ACC, FULL
        if (proof['schema']!='authored.selected_raw32_storage_law.v1'
            or proof['encoding_facts_sha256']!=sha256(json.dumps(facts,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            or proof['bias_preload']['status']!='derived'
            or proof['bias_preload']['bits']!=32 or proof['bias_preload']['rounding']!='none'
            or proof['full_width_readout']['status']!='derived'
            or proof['full_width_readout']['dtype']!='i32'
            or proof['accumulator_load_address']!=ACC
            or proof['raw_store_address']!=ACC|FULL
            or proof['load_accumulate'] is not False or proof['payload_arithmetic'] is not False):
            return None
        return dict(source_dtype='f32',storage_dtype='i32',bit_preservation='structurally derived unmodified 32-bit lanes',
            target_contract_sha256=proof['target_contract_sha256'],fir_sha256=proof['firrtl']['fir_sha256'],
            storage_contract_sha256=sha256(p.read_bytes()).hexdigest())
    except (OSError,KeyError,TypeError,ValueError):
        return None
