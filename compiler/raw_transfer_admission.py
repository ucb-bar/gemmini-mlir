"""Selected stream0 ACC raw-copy RAW; retain retirement before reconfiguration.

No full asynchronous epoch is granted: queue routing by itself does not prove
per-command configuration capture across changing tail/stride configurations.
The narrower pair route retires every original store before the next run.
"""
from pathlib import Path
from hashlib import sha256
import json
from functools import lru_cache


def qualify_raw_pair(selected,facts_bytes,encoding_bytes):
    try:
        order=selected['ordering']
        if (selected['base_facts']['sha256']!=sha256(facts_bytes).hexdigest()
            or selected['ordering_sha256']!=sha256(json.dumps(order,sort_keys=True).encode()).hexdigest()
            or order['status']!='derived'):
            return dict(admitted=False,reason='selected ordering/facts identity mismatch')
        cross={(x['older'],x['younger'],x['space']):x for x in order['cross_queue']}
        raw=('MVIN.spad_addr','MVOUT.acc_addr','accumulator')
        if cross.get(raw,{}).get('overlap')!='completion' or cross[raw].get('tracked')is not True:
            return dict(admitted=False,reason='actual stream0 ACC MVIN-to-MVOUT RAW completion unavailable')
        access=order['accesses']
        if (access['MVIN.spad_addr']['funct']!=2 or access['MVIN.spad_addr']['queue']!='load'
            or access['MVOUT.acc_addr']['funct']!=3 or access['MVOUT.acc_addr']['queue']!='store'
            or order['local_address']['accumulator_select']!=1<<31):
            return dict(admitted=False,reason='actual stream0 ACC access binding unavailable')
        # Exact unchanged existing Encoder, including default stream0/raw flags;
        # there is no permission for another instruction or alternate selector.
        expected='e71675c2d2f6900e80993cf07b2ac070777146db7be49af05276d149f5b780f9'
        if sha256(encoding_bytes).hexdigest()!=expected:
            return dict(admitted=False,reason='raw transfer Encoder changed')
        return dict(admitted=True,route='raw_pair_with_store_retirement',
            required_completion_edge=list(raw),ordering_sha256=selected['ordering_sha256'],
            load_stream=0,accumulator_row=0,full_epochs_admitted=False,
            retained='store-completion drain before every next load/reconfiguration/source continuation',
            full_epoch_refusal='no independently selected per-command configuration capture law for changing tail/stride configurations')
    except (KeyError,TypeError,ValueError):
        return dict(admitted=False,reason='incomplete actual raw-copy ordering evidence')

@lru_cache(maxsize=1)
def raw_pair_admission():
    root=Path(__file__).parent
    return qualify_raw_pair(json.loads((root/'data/command_ordering.json').read_bytes()),
        (root/'data/facts.json').read_bytes(),(root/'encoding.py').read_bytes())
