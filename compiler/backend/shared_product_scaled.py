"""Source-bound scaled i8 exits for the original checked shared-LHS producer.

The raw schedule retains complete producer and retention ownership. This adapter
rechecks the actual narrowed output pointers, parameter-bearing store calls and
all member lifetimes with the shared checker. Merlin's exact integer scalar
readout describes binary32 conversion/multiply, RNE and signed-i8 saturation;
existing dense_readout.store_tile implements the same selected hardware path.
No host output arithmetic or new execution/completion premise is introduced.
"""
from dataclasses import dataclass, replace
from hashlib import sha256
from math import isfinite
from pathlib import Path
import json, struct

from merlin.sched.ir import Const, Loop, Ptr, as_expr
from merlin.compile.scheduling.block_schedule import Store
from merlin.compile.scheduling.compact_dense import _digest
from merlin.compile.scheduling.compact_dense_group import CompactDenseGroupResult
from merlin.compile.scheduling.compact_dense_group_check import verify_compact_dense_group
from merlin.compile.scheduling.dense_contracts import DenseIntegerReadout
from merlin.llvmlower.integer_readout import emit_exact_scalar_ir

# Selected target_contract.yaml output_epilogue.output_path, audited against
# the already selected store_tile/config_st path. This identity is source
# custody, not an independent execution qualification or target certificate.
SELECTED_OUTPUT_CONTRACT_SHA256 = '7c1732429fc748ec54eb689c57738259eac71115749228a35900d9f03ef26c5d'


def output_attributes(attrs, *, allow_scaled):
    """Normalize only the exact source scalar scale/optional rectifier exits."""
    ep = attrs.get('epilogue', [])
    dtype = attrs.get('output_dtype', 'i32')
    if (attrs.get('bias') or attrs.get('accumulator_init')
            or attrs.get('acc_scale_by_column') is not None
            or attrs.get('requant_shift') is not None
            or type(ep) not in (list, tuple)):
        return None
    scale = attrs.get('acc_scale',1.0)
    if type(scale) not in (float,int):
        return None
    try:
        scale = struct.unpack('<f',struct.pack('<f',float(scale)))[0]
    except (OverflowError,ValueError):
        return None
    if not isfinite(scale) or scale <= 0:
        return None
    relu = bool(attrs.get('relu',False) or 'relu' in ep)
    if dtype == 'i32':
        return dict(output_dtype='i32',epilogue=[],acc_scale=1.0) if not ep and scale==1 and not relu else None
    if (not allow_scaled or dtype != 'i8'
            or any(x not in ('acc_scale','relu') for x in ep)
            or len(set(ep)) != len(ep)
            or ('relu' in ep and 'acc_scale' in ep and ep.index('relu') < ep.index('acc_scale'))):
        return None
    return dict(output_dtype='i8',epilogue=['acc_scale']+(['relu'] if relu else []),
                acc_scale=scale,relu=relu)


@dataclass(frozen=True)
class SharedScaledResult:
    original: object
    kernel: object
    realization: dict
    attributes_json: str
    _guard: object

    @property
    def choice(self):
        return self.original.choice

    def revalidate(self):
        self.original.revalidate()
        self._guard(self)

    def emission_attributes(self, member):
        self.revalidate()
        return json.loads(self.attributes_json)[member]


def bind_scaled_outputs(original, problems, resources, storages, adapter, *,
                        encode_member, project_member, retention_law, attributes):
    """Preserve every producer effect and recheck each actual typed output exit."""
    if type(original) is not CompactDenseGroupResult:
        raise ValueError('scaled shared exits require the original checked raw producer')
    original.revalidate()
    attributes = tuple(output_attributes(a,allow_scaled=True) for a in attributes)
    if len(attributes) != len(problems) or any(a is None for a in attributes):
        raise ValueError('shared output needs exact source scalar scale/activation attributes')
    if all(a['output_dtype']=='i32' for a in attributes):
        return original
    readouts=[];proofs=[]
    for attrs in attributes:
        if attrs['output_dtype']=='i32':
            readouts.append(None);proofs.append(None);continue
        scalar_ir, proof = emit_exact_scalar_ir((attrs['acc_scale'],),relu=attrs['relu'])
        readout = DenseIntegerReadout(scalar_ir,'i8','gemmini_mvout_scaled_i8',
            (('dram_row',64),('dram_col',64),('rows',64),('cols',64),('acc',64),
             ('readout_scale_bits',32),('readout_relu',1)))
        graph=readout.graph()
        if graph.input_bits != (32,) or graph.output_bits != 8:
            raise ValueError('shared readout lacks complete unary i32-to-i8 scalar semantics')
        readouts.append(readout);proofs.append(proof)
    readouts=tuple(readouts)
    def parameters(member):
        attrs=attributes[member]
        bits=struct.unpack('<I',struct.pack('<f',attrs['acc_scale']))[0]
        return (('readout_scale_bits',Const(bits)),('readout_relu',Const(int(attrs['relu']))))
    def transform_store(call,member,op):
        if readouts[member] is None:return call
        pointer=Ptr(f'output{member}',as_expr(op.dram_row)*problems[member].n+op.dram_col)
        return replace(call,args=tuple((key,pointer if key=='external_pointer' else value)
                                      for key,value in call.args)+parameters(member))
    def transform(body):
        result=[]
        for item in body:
            if type(item) is Loop:
                result.append(replace(item,body=transform(item.body)));continue
            member,op=project_member(item)
            result.append(transform_store(item,member,op) if type(op) is Store else item)
        return tuple(result)
    outputs={f'output{i}':a['output_dtype'] for i,a in enumerate(attributes)}
    proposal=replace(original.kernel,body=transform(original.kernel.body),
        args=tuple(replace(a,dtype=outputs[a.name]) if a.name in outputs else a
                   for a in original.kernel.args))
    def project_actual(call):
        scaled = any(key.startswith('readout_') for key,_ in call.args)
        stripped=replace(call,args=tuple((k,v) for k,v in call.args if not k.startswith('readout_')))
        member,op=project_member(stripped)
        if type(op) is not Store:
            if scaled:raise ValueError('shared readout parameters attached to non-store effects')
            return member,op
        canonical=encode_member(member,op)
        expected=transform_store(canonical,member,op)
        if call != expected:
            raise ValueError('actual shared store differs from original pointer, width, scale or activation')
        return member,replace(op,readout=None if readouts[member] is None else readouts[member].identity)
    # Compose exact physical load coalescing with the ACTUAL scaled output
    # product, then independently verify every final band and scaled store.
    # SharedWideCodec is only an address decoder here: it grants no result-type
    # admission or execution law. The original raw producer remains the owner.
    from merlin.compile.tuning_knobs import knob
    selection=knob('SHARED_PRODUCT_WIDE',default='off',values=('off','on'),
        root=Path(__file__).resolve().parent.parent)
    checked_kernel=proposal
    wide_receipt=dict(applied=False,selection=selection)
    codec=None
    if selection=='on':
        from .shared_product_wide import SharedWideCodec
        from .dense_wide_loads import selected_addressing_evidence
        from merlin.compile.scheduling.tile_load_coalescing import propose_tile_load_coalescing
        from merlin.compile.scheduling.panel_tile_loads import expand_panel_tile_loads
        facts=json.loads(Path(__file__).with_name('dense_load_facts.json').read_text())
        evidence=selected_addressing_evidence(facts)
        codec=SharedWideCodec(problems,storages,adapter,encode_member,project_actual,evidence['identity'])
        candidate,coalescing=propose_tile_load_coalescing(proposal,
            project=codec.project,encode=codec.encode,max_tiles=4,
            max_static_nodes=4096,coalesce_tail=True)
        if coalescing['changes']:
            checked_kernel,projection=expand_panel_tile_loads(candidate,
                project=codec.project,encode_band=codec.encode_band,max_static_nodes=4096)
            wide_receipt=dict(applied=True,selection=selection,addressing=evidence,
                proposal=coalescing,tile_load_projection=projection,
                original_scaled_kernel_sha256=proposal.digest(),kernel_sha256=candidate.digest(),
                checking_expansion_sha256=checked_kernel.digest())
            proposal=candidate
        else:
            wide_receipt=dict(applied=False,selection=selection,proposal=coalescing,
                refusal='no contiguous load-only tile run')
    qualification='original-store_tile-selected-output-path:'+SELECTED_OUTPUT_CONTRACT_SHA256
    report=verify_compact_dense_group(checked_kernel,problems,resources,storages,adapter,
        project_member=project_actual,retention_law=retention_law,dtypes=('i8','i8','i32'),
        static_budget=256,external_buffers_disjoint=True,timeout_ms=30000,max_members=8,
        output_readouts=readouts,
        readout_qualification_identities=tuple(None if r is None else qualification for r in readouts))
    original.revalidate()
    receipt=dict(schema='submission.checked_shared_scaled_outputs.v1',applied=True,
        original_kernel_sha256=original.kernel.digest(),kernel_sha256=proposal.digest(),
        source_attributes=attributes,readout_identities=tuple(None if r is None else r.identity for r in readouts),
        source_numeric_proofs=proofs,selected_output_contract_sha256=SELECTED_OUTPUT_CONTRACT_SHA256,
        implementation='unchanged dense_readout.store_tile and isa.config_st',
        completion='original pure staging/execute epochs and fenced readout',
        qualification='caller-owned source and selected readout contract; independent native validation required',
        complete_call_cycles='UNKNOWN',target_execution='not_run')
    realization=dict(report,scaled_outputs=receipt,kernel_sha256=proposal.digest(),wide_loads=wide_receipt)
    attr_json=json.dumps(attributes,sort_keys=True)
    paths=tuple(dict.fromkeys((*original.implementation_paths(),Path(__file__).resolve(),
        Path(__file__).with_name('dense_readout.py'),Path(__file__).with_name('isa.py'),
        Path(__file__).with_name('shared_product_wide.py'),Path(__file__).with_name('dense_wide_loads.py'),
        Path(__file__).with_name('dense_load_coalescing.py'),Path(__file__).with_name('dense_load_facts.json'))))
    pins=tuple((p,sha256(p.read_bytes()).hexdigest()) for p in paths)
    def functions():
        return (output_attributes,bind_scaled_outputs,SharedScaledResult.revalidate,
                SharedScaledResult.emission_attributes,emit_exact_scalar_ir,verify_compact_dense_group,
                *((SharedWideCodec.project,SharedWideCodec.encode,SharedWideCodec.encode_band,
                   propose_tile_load_coalescing,expand_panel_tile_loads,selected_addressing_evidence)
                  if codec is not None else ()))
    callbacks=functions();codes=tuple(f.__code__ for f in callbacks)
    frozen=(proposal.digest(),_digest(realization),attr_json,
            tuple(None if r is None else r.identity for r in readouts))
    def guard(current):
        if (current.original is not original or callbacks != functions()
                or codes != tuple(f.__code__ for f in functions())
                or pins != tuple((p,sha256(p.read_bytes()).hexdigest()) for p in paths)
                or (current.kernel.digest(),_digest(current.realization),current.attributes_json,
                    tuple(None if r is None else r.identity for r in readouts)) != frozen):
            raise ValueError('shared scaled producer, actual output, numeric binding or implementation changed')
        if wide_receipt['applied']:
            actual_expanded,actual_projection=expand_panel_tile_loads(current.kernel,
                project=codec.project,encode_band=codec.encode_band,max_static_nodes=4096)
            if (actual_expanded.digest()!=wide_receipt['checking_expansion_sha256'] or
                    _digest(actual_projection)!=_digest(wide_receipt['tile_load_projection'])):
                raise ValueError('actual scaled wide bands or readout-aware checking projection changed')
    result=SharedScaledResult(original,proposal,realization,attr_json,guard)
    result.revalidate()
    return result
