"""Deterministic evidence from receipts returned by actually consumed emitters.

This sidecar never alters the semantic command buffer or emitted LLVM. Hashes
bind source, complete receipts, and emitted bytes. Identities are extracted from
those receipts; an available catalogue is never evidence of an applied pass.
"""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import hashlib
import json
import math
from pathlib import Path


def _plain(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('nonfinite emitter evidence is not JSON serializable')
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name:_plain(getattr(value,field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        if any(not isinstance(key,str) for key in value):
            raise TypeError('emitter receipt keys must be strings')
        return {key:_plain(item) for key,item in value.items()}
    if isinstance(value,(tuple,list)):
        return [_plain(item) for item in value]
    if isinstance(value,(set,frozenset)):
        return sorted((_plain(item) for item in value),key=_canonical)
    raise TypeError('unsupported emitter evidence value '+type(value).__qualname__)


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)


def _sha(data):
    return hashlib.sha256(data if isinstance(data,bytes) else data.encode('utf-8')).hexdigest()


def _identities(value,path=''):
    found = {}
    if isinstance(value,dict):
        for key,item in value.items():
            location = f'{path}.{key}' if path else key
            if isinstance(item,(str,int,float,bool)) and (
                    'sha256' in key or 'identity' in key or key in (
                    'library','mechanism','mode','dense_schedule','requested_retention',
                    'qualification','pass_name','pass','schema','route','catalog_row') or
                    (key == 'name' and '.passes[' in path)):
                found[location]=item
            found.update(_identities(item,location))
    elif isinstance(value,list):
        for index,item in enumerate(value):
            found.update(_identities(item,f'{path}[{index}]'))
    return found


def make_evidence(source,command_buffer,lowered,receipts):
    """Build compact JSON-compatible analysis of the same emitted compilation.

    Receipts must be those returned by emitters whose effects were appended to
    `lowered`. Empty/unavailable receipt lists remain explicitly unavailable.
    This is compiler construction evidence, never executed numerical/timing proof.
    """
    actual = _plain(receipts)
    if not isinstance(actual,list):
        raise TypeError('actual emitter receipts must be a sequence')
    products = []
    for index,receipt in enumerate(actual):
        products.append({'index':index,'receipt_sha256':_sha(_canonical(receipt)),
            'identities':_identities(receipt)})
    from merlin.compile.tuning_knobs import consulted
    unavailable = not actual or any(isinstance(r,dict) and r.get('status')=='unavailable' for r in actual)
    result = {'schema':'gemmini.emitted_analysis.v1',
        'source_sha256':_sha(source),
        'command_buffer_sha256':_sha(_canonical(_plain(command_buffer))),
        'emitted_llvm_sha256':_sha(lowered),
        'consumed_receipts_sha256':_sha(_canonical(actual)),
        'construction_evidence':'unavailable' if unavailable else 'actual emitter receipts',
        'execution_evidence':'not measured by this artifact',
        # CLI handles one source per process. Retain the API's precise scope for
        # embedding callers that compile several sources in one process.
        'process_consulted_knobs':_plain(consulted()),
        'products':products}
    # Keep actual typed-effect observations, including explicit unavailable
    # reasons. Receipts alone remain custody evidence rather than timing proof.
    activity = []
    def collect(value, path='receipts'):
        if isinstance(value, dict):
            if value.get('schema') == 'gemmini.schedule_activity.v1':
                activity.append(dict(receipt_path=path, observation=value))
            else:
                for key,item in value.items(): collect(item,path+'.'+key)
        elif isinstance(value,list):
            for index,item in enumerate(value): collect(item,f'{path}[{index}]')
    collect(actual)
    result['schedule_activity'] = activity
    result['schedule_activity_scope'] = 'observed emitter receipts only; host work and unobserved adapters remain unknown'
    return result


def write_evidence(path,source,command_buffer,lowered,receipts):
    """Write one sidecar without modifying the candidate's semantic artifacts."""
    evidence=make_evidence(source,command_buffer,lowered,receipts)
    Path(path).write_text(json.dumps(evidence,sort_keys=True,indent=2,allow_nan=False)+'\n')
    return evidence
