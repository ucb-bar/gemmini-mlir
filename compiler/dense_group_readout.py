"""Shared-LHS raw producers followed by original, checked scalar i8 exits.

This is a real materialized composition, not direct compact-group readout
qualification. The checked sibling schedule writes complete private i32 matrices;
only after its completion does the separately checked column_readout schedule
reload each matrix and apply its original scalar scale/ReLU once. Source primary
operations and regions remain on both pieces inside the actual coalesced task.
The caller proves original independent products and disjoint ABI buffers. No
source-dependent member, bias, transpose, per-column scale or pooling is admitted.
The full-K sibling capacity checker remains authoritative. Timing is unmeasured.
"""
from dataclasses import dataclass, replace
import hashlib
import json
import math

from dense_siblings import SiblingProduct, build_siblings, emit_siblings
from column_readout import plan_column_readout
from source_epilogue import declared_path
from column_producer import raw_attributes

KNOB_DECLARATION = {'dense_sibling_materialized_readout': {
    'values': ['off', 'raw_then_grouped_readout'],
    'mechanism': 'checked shared-LHS raw i32 sibling producers plus checked grouped scalar hardware readouts',
    'note': 'Experimental real private BSS spill/reload; preserves original i8 boundaries and source ownership; off incumbent; no cycle qualification.'}}


def _record(products, identity):
    return json.dumps({'source_identity': identity, 'products': [vars(p) for p in products]},
                      sort_keys=True, separators=(',', ':'))


def _path_proof():
    """Bind the existing positive-scale theorem to the declared physical path."""
    path = declared_path()
    stages = [s for s in path['readout'] if s['kind'] != 'pool']
    observed = [(s['kind'], s.get('function') if s['kind'] == 'activation'
                 else s.get('to', s.get('dtype'))) for s in stages]
    expected = [('activation', 'relu'), ('convert', 'f32'),
                ('scale', 'f32'), ('requant', 'i8')]
    if (observed != expected or path['accumulator']['dtype'] != 'i32'
            or not path['exits']['accumulator'] or not path['exits']['requant']
            or any(s.get('rounding') != 'half_even' for s in stages[1:])
            or stages[-1].get('clamp') != [-128, 127]
            or stages[2].get('granularity') != 'per_command'):
        raise ValueError('declared target output path is outside checked scalar column readout theorem')
    return {'declared_output_path_sha256': hashlib.sha256(
        json.dumps(path, sort_keys=True).encode()).hexdigest(),
        'binding': 'raw accumulator exit; optional accumulator ReLU; f32 convert/multiply; half-even signed i8 saturation',
        'source_equivalence': 'column_readout full-i32 positive-scale transition proof',
        'direct_group_readout_qualification': False}


@dataclass
class GroupReadoutSchedule:
    products: tuple
    raw_plan: object
    readouts: tuple
    source_identity: str
    source_record: str
    path_proof: dict
    realization: dict

    @property
    def choice(self): return self.raw_plan.choice

    @property
    def kernel(self): return self.raw_plan.kernel

    def revalidate(self):
        if _record(self.products, self.source_identity) != self.source_record:
            raise ValueError('original sibling readout source changed after admission')
        if _path_proof() != self.path_proof:
            raise ValueError('declared readout path changed after admission')
        self.raw_plan.revalidate()
        if len(self.readouts) != len(self.products):
            raise ValueError('original sibling readout roster changed')
        for p, readout in zip(self.products, self.readouts):
            expected = plan_column_readout(p.m, p.n, (p.attrs['acc_scale'],) * p.n,
                activation='relu' if 'relu' in p.attrs['epilogue'] else (), group_runs=True)
            if readout.identity != expected.identity:
                raise ValueError('readout no longer applies the original member scale/activation')
            readout.revalidate()
        return self


def build_group_readout(products, resources, storage, adapter, *, source_identity,
                        external_buffers_disjoint=False, static_budget=256):
    """Admit every source boundary before allocating scratch or issuing commands."""
    if type(products) is not tuple or not 2 <= len(products) <= 8 or any(
            type(p) is not SiblingProduct for p in products):
        raise ValueError('materialized readout requires two to eight typed sibling products')
    proof = _path_proof()
    original_names = {name for p in products for name in (p.lhs, p.rhs, p.output)}
    outputs = {p.output for p in products}
    if len(outputs) != len(products):
        raise ValueError('original sibling outputs must be distinct')
    raw_products, readouts = [], []
    for index, p in enumerate(products):
        a = p.attrs
        if p.lhs in outputs or p.rhs in outputs or outputs.intersection(p.dependencies):
            raise ValueError('dependent original products cannot form materialized siblings')
        if (p.bias is not None or not p.zero_seed or p.rhs_transposed
                or a.get('acc_scale_by_column') is not None or a.get('source_epilogue')
                or a.get('output_dtype', 'i32') != 'i8'
                or a.get('epilogue') not in (['acc_scale'], ['acc_scale', 'relu'])):
            raise ValueError('materialized siblings require unbiased ordinary scalar i8 scale/optional ReLU exits')
        scale = a.get('acc_scale')
        if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
            raise ValueError('materialized sibling scale must be original finite positive scalar')
        readouts.append(plan_column_readout(p.m, p.n, (scale,) * p.n,
            activation='relu' if 'relu' in a['epilogue'] else (), group_runs=True))
        raw_name = '__private_shared_lhs_raw_' + str(index)
        while raw_name in original_names:
            raw_name += '_'
        original_names.add(raw_name)
        raw_attrs = raw_attributes(a)
        raw_products.append(replace(p, output=raw_name, attrs=raw_attrs))
    # The library proves full-K source products, their actual scratch placement,
    # complete raw outputs and common-LHS effect reuse. It receives no synthetic
    # reviewed readout declaration: narrowing is a separate real consumer.
    raw = build_siblings(tuple(raw_products), resources, storage, adapter,
        source_identity=source_identity, external_buffers_disjoint=external_buffers_disjoint,
        static_budget=static_budget)
    spill_bytes = sum(p.m * p.n * 4 for p in products)
    import yaml
    from pathlib import Path
    contract = yaml.safe_load((Path(__file__).parent / 'contracts' / 'target_contract.yaml').read_text())
    budget = contract['logical_harness']['storage_budget_bytes']
    if spill_bytes > budget:
        raise ValueError('materialized sibling matrices exceed selected logical storage budget')
    report = {'mechanism': 'shared_lhs_raw_then_grouped_scalar_readout',
        'private_raw_bytes': spill_bytes, 'source_identity': source_identity,
        'source_numerical_boundary': 'complete i32 products; one original scalar i8 readout per member before dependent source consumers',
        'declared_path': proof, 'raw_producer': raw.realization,
        'readouts': [r.report() for r in readouts], 'performance_status': 'unmeasured opt-in'}
    return GroupReadoutSchedule(products, raw, tuple(readouts), source_identity,
        _record(products, source_identity), proof, report).revalidate()


def emit_group_readout(builder, plan, pointers, *, index_unroll=None):
    """Consume the checked raw producer and every checked readout, using real BSS."""
    if type(plan) is not GroupReadoutSchedule:
        raise ValueError('typed materialized sibling schedule required')
    plan.revalidate()
    count = len(plan.products)
    expected = {'lhs', *(f'rhs{i}' for i in range(count)), *(f'output{i}' for i in range(count))}
    if set(pointers) != expected or len(set(pointers.values())) != len(pointers):
        raise ValueError('materialized sibling ABI must bind every original disjoint input/output')
    def scope(p):
        return dict(task_index=p.attrs.get('source_task_index'),
            source_op_index=p.attrs.get('source_primary_op_index'), region_id=p.attrs.get('prov.region_id'))
    raw_pointers = dict(pointers)
    with builder.task_scope(**scope(plan.products[0])):
        scratch = [builder.global_buffer(p.m * p.n, 'i32') for p in plan.products]
        for i, ptr in enumerate(scratch): raw_pointers[f'output{i}'] = ptr
        producer = emit_siblings(builder, plan.raw_plan, raw_pointers, index_unroll=index_unroll)
        readout_receipts = []
        for i, p in enumerate(plan.products):
            with builder.task_scope(**scope(p)):
                readout_receipts.append(plan.readouts[i].emit(builder, scratch[i], pointers[f'output{i}']))
    plan.revalidate()
    result = {**plan.realization, 'raw_producer': producer,
              'readouts': readout_receipts, 'raw_storage': 'actual private global i32 buffers, complete accelerator spill/reload'}
    if not hasattr(builder, 'dense_group_readout_receipts'): builder.dense_group_readout_receipts = []
    builder.dense_group_readout_receipts.append(result)
    return result
