"""Merlin composition plans realized as Gemmini contraction schedules.

Merlin owns the semantic proofs: the constant-weight output-stationary value
bound (``merlin.targetgen.native_mac_bound``), the integer residual coefficient
decomposed into digit tiles (``merlin.xdsl_dialects.lowering.k_extension``) and
the readout boundary (``output_fusion``), all reached through
``merlin.compile.composition_fusion.plan_composition``.  This module supplies
the target facts those proofs consume, selects the provider schedule, and
lets the shared replay (``contraction_proof``) admit the emitted stream with
the dataflows and terms the re-derived plan licenses.

A refused premise returns None, which keeps the weight-stationary producer and
the host residual and readout boundary exactly as before.
"""

from __future__ import annotations

from dataclasses import dataclass


#: Dataflow names in Merlin's composition vocabulary.
_DATAFLOWS = {'WS': 'weight_stationary', 'OS': 'output_stationary'}


def _dtype_bits(dtype):
    digits = ''.join(ch for ch in str(dtype) if ch.isdigit())
    return int(digits) if digits else None


def _dataflow_codes():
    """CONFIG_EX dataflow codes, with RTL-derived modes cross-checked when present."""
    from ..tables import isa
    from .readout_encoding import contract, field
    from merlin.targetgen.rtl.facts import load_facts

    roles = contract()['rocc_operand_roles']['classes']['CONFIG_EX']['fields']
    if not any(row.get('name') == 'dataflow' for row in roles):
        return {}
    _, width = field('ConfigExRs1', 'dataflow')
    codes = {'WS': isa.WEIGHT_STATIONARY, 'OS': isa.OUTPUT_STATIONARY}
    if width is None or any(not 0 <= v < 1 << width for v in codes.values()) or len(set(codes.values())) != 2:
        return {}
    timing = (load_facts(contract()['name']).get('facts') or {}).get('timing_structure') or {}
    modes = (timing.get('dataflow') or {}).get('modes')
    if modes:
        derived = {label: int(value) for value, label in modes.items()}
        if any(derived.get(_DATAFLOWS[name]) != code for name, code in codes.items()):
            return {}
    return codes


def facts(semantics):
    """Composition facts from the selected numeric semantics and the synced contract."""
    from merlin.targetgen.rtl.facts import load_facts
    from .readout_encoding import contract

    semantics = semantics or {}
    internal = semantics.get('internal_arithmetic') or {}
    datapaths = {row.get('name'): row.get('dtype') for row in
                 (load_facts(contract()['name']).get('facts') or {}).get('datapaths', ())}
    operand_bits, accumulator_bits = _dtype_bits(datapaths.get('input')), _dtype_bits(datapaths.get('accumulator'))
    out = dict(dataflows=['WS'])
    mesh = internal.get('mac_result_bits')
    if (type(mesh) is int and mesh > 1 and internal.get('mac_result_overflow') == f'wrap_to_{mesh}_bits'):
        out['mesh_signed_bits'] = mesh
        declared = (semantics.get('native_unit_contract') or {}).get('mesh_mac') or {}
        if ('OS' in declared.get('dataflows', ()) and declared.get('signed_bits') == mesh
                and 'OS' in _dataflow_codes()):
            out['dataflows'].append('OS')
    if (operand_bits and accumulator_bits and internal.get('signed_operand_bits') == operand_bits
            and _dtype_bits(semantics.get('accumulator_dtype')) == accumulator_bits):
        out['digit_bits'] = operand_bits
        out['accumulator_bits'] = accumulator_bits
        # Every extended-K prefix of signed operand products stays an exact accumulator value.
        out['max_extended_k'] = ((1 << (accumulator_bits - 1)) - 1) // (1 << (operand_bits - 1)) ** 2
    return out


def _plan(module, op, options, route):
    """Merlin's plan for ``route`` rooted at ``op``, or None on any refused premise."""
    from merlin.compile.composition_fusion import composition_proposals, plan_composition

    block = (options or {}).get('compositions')
    if not block:
        return None
    operations = tuple(module.walk())
    root = operations.index(op)
    try:
        for found, members, detail, _ in composition_proposals(module, block, max_region_operations=len(operations)):
            if found != route or detail.get('root_ordinal') != root:
                continue
            candidate = dict(route=route, operation_ordinals=[operations.index(m) for m in members], detail=detail)
            selected, path = facts(options.get('numerical_semantics')), options['path']
            if route == 'k_concat' and 'max' in detail['composition'].get('reduction_kinds', ()):
                from .pool_plan import composition_facts
                selected.update(composition_facts())
                # StoreController's out-of-image taps select hardware zero.
                path = {**path, 'readout': [
                    {**stage, 'padding': 'zero_tap'} if stage['kind'] == 'pool' else dict(stage)
                    for stage in path['readout']]}
            return plan_composition(module, candidate, path=path, scheme=options['scheme'],
                                    facts=selected)
    except ValueError:
        return None  # Any unproved premise keeps the ordinary schedule.
    return None


def output_stationary(module, op, options, *, batch, runtime_seed):
    """Admit OS for a constant-weight rank-two product whose mesh partials are bounded.

    Merlin's interval includes the IR initializer, as a D preload would.  This
    schedule seeds the accumulator instead and starts the mesh at zero, so the
    proof is re-read without the seed for the partials the mesh actually holds.
    """
    if options is None or batch or runtime_seed:
        return None
    plan = _plan(module, op, options, 'output_stationary')
    if plan is None:
        return None
    from merlin.targetgen.native_mac_bound import constant_tensor
    from .recipes.output_stationary import resident_rows

    decision = plan.decision
    m, k, n = decision['shape_m_k_n']
    try:
        resident_rows(m, n, k)
        seed = constant_tensor(op.operands[2])
    except ValueError:
        return None
    lower = [int(v) - int(seed[:, c].min()) for c, v in enumerate(decision['column_lower'])]
    upper = [int(v) - int(seed[:, c].max()) for c, v in enumerate(decision['column_upper'])]
    limit = 1 << (decision['mesh_signed_bits'] - 1)
    if min(lower) < -limit or max(upper) >= limit:
        return None
    return dict(plan=plan.record(), mesh_partial_lower=lower, mesh_partial_upper=upper)


def dataflows(module, op, options, *, batch, runtime_seed):
    """Dataflow codes an admitted replay may accept, re-derived from the source alone.

    Output-stationary is accepted only when the same plan and seed-free mesh
    bound that selected it at emission hold again; otherwise only WS.
    """
    from ..tables import isa
    planned = output_stationary(module, op, options, batch=batch, runtime_seed=runtime_seed)
    codes = {isa.WEIGHT_STATIONARY}
    if planned is not None:
        codes.add(isa.OUTPUT_STATIONARY)
    return frozenset(codes), planned


@dataclass
class Extension:
    plan: object
    readout: object       # the plan's readout boundary, as a narrow composition plan
    chain: object
    residual: object      # the i8 source tensor of the residual dequantize
    digits: tuple


def _integer_residual(op):
    """Match the closed i32 ``acc + coefficient * extsi(residual)`` mesh suffix."""
    from .match import etype, shape, generic_maps, identity_map, iterators, _scalar_const

    users = list(op.results[0].uses)
    if len(users) != 1:
        return None
    suffix = users[0].operation
    if (suffix.name != 'linalg.generic' or len(suffix.operands) != 3 or len(suffix.results) != 1
            or suffix.operands[0] is not op.results[0] or etype(suffix.operands[1]) != 'i8'
            or any(shape(v) != shape(op.results[0]) for v in (*suffix.operands, *suffix.results))
            or any(etype(v) != 'i32' for v in (suffix.operands[0], suffix.operands[2], suffix.results[0]))
            or not all(identity_map(m) for m in generic_maps(suffix))
            or any(it != 'parallel' for it in iterators(suffix))):
        return None
    block = suffix.regions[0].blocks[0]
    body = list(block.ops)
    if [s.name for s in body] != ['arith.extsi', 'arith.constant', 'arith.muli', 'arith.addi', 'linalg.yield']:
        return None
    ext, constant, product, add, yielded = body
    if (list(ext.operands) != [block.args[1]] or str(ext.results[0].type) != 'i32'
            or str(constant.results[0].type) != 'i32'
            or set(product.operands) != {ext.results[0], constant.results[0]}
            or set(add.operands) != {block.args[0], product.results[0]}
            or list(yielded.operands) != [add.results[0]]):
        return None
    coefficient = _scalar_const(constant)
    return (suffix, coefficient) if type(coefficient) is int and coefficient != 0 else None


def _integer_extension(module, op, options):
    """Realize a raw accumulator residual using Merlin's all-prefix digit proof."""
    from merlin.compile.composition_fusion import CompositionPlan, composition_routes, _closed_members
    from merlin.targetgen.software_spec import validate_compositions
    from merlin.targetgen.output_path import validate
    from merlin.targetgen import model_coverage
    from merlin.targetgen.native_mac_bound import constant_tensor, signed_matmul_shape
    from merlin.transforms.certificates import ir_digest
    from merlin.verify.lean_contract import document_digest
    from merlin.xdsl_dialects.lowering import output_fusion as OF
    from merlin.xdsl_dialects.lowering.k_extension import lower_k_extension
    from .match import shape

    found = _integer_residual(op)
    if found is None or not options.get('compositions'):
        return None
    suffix, coefficient = found
    try:
        path = validate(options['path'])
        declarations = validate_compositions(options['compositions'])['chains']
        if (path['accumulator']['dtype'] != 'i32' or not path['exits'].get('accumulator')
                or not any('k_concat' in composition_routes(d) and 'accumulator' in d.get('exits', ())
                           and 'contraction' in ({d['producer']} if isinstance(d['producer'], str) else set(d['producer']))
                           and d.get('readout_stages') == 'from_output_path'
                           and all(requirement != 'required' or any(s['kind'] == kind for s in path['accumulator_stages'])
                                   for kind, requirement in d.get('accumulator_stages', {}).items())
                           for d in declarations)):
            return None
        # The same producer premises Merlin's _k_concat checks on the requant route.
        _, k, n = signed_matmul_shape(op)
        constant_tensor(op.operands[2])
        _closed_members((op, suffix))
        selected = facts(options.get('numerical_semantics'))
        if selected.get('digit_bits') != 8 or selected.get('accumulator_bits') != 32:
            return None
        values = OF.Values()
        chain = OF.Chain(op, shape(op.results[0])[-1], [], [suffix])
        bound = OF._bound(chain, values, model_coverage._contraction_extents(module))
        decision = lower_k_extension(coefficient, source_k=k,
            residual_width=n, digit_bits=selected['digit_bits'],
            max_extended_k=selected['max_extended_k'], accumulator_bits=selected['accumulator_bits'],
            source_bound=max(abs(v) for v in bound), residual_range=(-128, 127))
    except (ValueError, OF._Refused):
        return None
    operations = tuple(module.walk())
    ordinals = tuple(operations.index(member) for member in (op, suffix))
    readout = dict(cut='accumulator', stages=[], proof={})
    identity = dict(scheme_sha256=document_digest(options['scheme']), facts_sha256=document_digest(selected),
                    output_path_sha256=document_digest(path))
    plan = CompositionPlan('k_concat', ir_digest(module), ordinals,
                           {**decision, **identity, 'readout': readout})
    view = CompositionPlan('int32_readout', plan.source_ir_sha256, ordinals, {**readout, **identity})
    return Extension(plan, view, chain, suffix.operands[1], tuple(decision['tile_digits']))


def k_extension(module, op, options, *, batch):
    """Admit a K-concat residual whose integer coefficient Merlin decomposed into digits."""
    if options is None or batch:
        return None
    integer = _integer_extension(module, op, options)
    if integer is not None:
        return integer
    plan = _plan(module, op, options, 'k_concat')
    if plan is None:
        return None
    from merlin.compile.composition_fusion import CompositionPlan
    from merlin.xdsl_dialects.lowering import output_fusion as OF
    from .match import shape
    from .readout_encoding import supports_narrow_path
    from .stream import DIM

    decision = plan.decision
    readout = decision['readout']
    chain = OF.source_chain(op, OF.Values())
    width = readout['proof'].get('columns_per_scale')
    pooled = readout['cut'] == 'pool'
    if (readout['cut'] not in ('requant', 'pool') or not supports_narrow_path(options['path'])
            or (not pooled and (len(chain.ops) != len(chain.stages)
                                or shape(chain.ops[-1].results[0]) != shape(op.results[0])))
            or readout['proof'].get('output_column_partition')
            or (width is not None and (width < DIM or width % DIM))
            or sum(decision['tile_digits']) != decision['coefficient']):
        return None
    allowed = ('convert', 'scale', 'bias', 'residual', 'relu', 'quantize') + (('pad', 'pool') if pooled else ())
    if any(s.kind not in allowed for s in chain.stages):
        return None
    residual = next(s for s in chain.stages if s.kind == 'residual').detail['operand'].owner.operands[0]
    # Merlin's definitional f32 boundary is not a reassociation proof, for either cut.
    if not _source_exact_extension(module, chain, options, readout):
        return None
    view = CompositionPlan('conv_pool' if pooled else 'k_concat', plan.source_ir_sha256, plan.operation_ordinals,
                           {**readout, **{key: decision[key] for key in
                                          ('scheme_sha256', 'facts_sha256', 'output_path_sha256')}})
    return Extension(plan, view, chain, residual, tuple(int(d) for d in decision['tile_digits']))


def _source_exact_extension(module, chain, options, readout):
    """Delegate sequential source f32 equality to Merlin's threshold theorem.

    The scheme's definitional single-product boundary alone does not authorize
    reassociating a residual into the accumulator, so both the requant and the
    pool cut must reproduce the source's per-pixel quantized codes exactly for
    every residual value. Pooling then preserves equal per-pixel codes.
    """
    import numpy as np
    from merlin.targetgen import model_coverage
    from merlin.xdsl_dialects.lowering import output_fusion as OF
    from merlin.runtime.exact_readout import f32_readout

    values = OF.Values()
    try:
        table = OF._residual(chain, options['path'], values, absorbing=True, integer_extension=True)
        lo, hi = OF._bound(chain, values, model_coverage._contraction_extents(module))
    except OF._Refused:
        return False
    bits = np.asarray(readout['proof']['multiplier_bits'], np.uint32).view(np.float32)
    offsets = np.asarray(readout['proof']['offsets'], np.int64)
    quantize = chain.stages[-1]
    levels = np.arange(quantize.detail['min'], quantize.detail['max'] + 2)[:, None]
    shape = (len(levels), chain.channels)
    for shift, residual in table.items():
        # Batch the output levels of Merlin's monotone threshold theorem. The
        # offset is fixed by the emitted preload, so no offset search is needed.
        source = lambda x: OF._evaluate(chain.stages, np.float32, x, residual=residual)
        target = lambda x: f32_readout(x + shift + offsets, bits, 'i8', relu='relu' in readout['stages'])
        if not np.array_equal(OF._threshold(source, levels, lo, hi, shape),
                              OF._threshold(target, levels, lo, hi, shape)):
            return False
    return True


def digit_term(term, consts, *, residual, n, position, c0, cols):
    """The digit a replayed mesh term adds to a residual element, or a refusal.

    A term reading residual row r, columns a0.., through stationary rows that
    hold ``d * I`` adds ``d * residual[r, a0 + c]`` to accumulator column c, so
    it lands on output column ``n0`` exactly when ``a0 + c0 == n0``.
    """
    import numpy as np
    from .stream_proof import is_zero, require
    from .stream import DIM

    a_tag, w_tags, depth, width = term
    _, row, n0 = position
    require(a_tag.name == residual and a_tag.width >= depth, 'residual term reads unloaded columns')
    a_row, a0 = divmod(a_tag.flat, n)
    require(a_row == row and a0 + c0 == n0 and c0 + cols <= depth <= width, 'residual term leaves its output tile')
    first = w_tags[0]
    require(not is_zero(first) and first.name in consts and all(t.name == first.name for t in w_tags),
            'residual term weights are not one constant')
    data = np.asarray(consts[first.name]).reshape(-1).astype(np.int64)
    rows = np.stack([data[t.flat:t.flat + depth] for t in w_tags])
    digit = int(rows[0, 0])
    require(rows.shape == (depth, depth) and np.array_equal(rows, digit * np.eye(depth, dtype=np.int64)),
            'residual term weights are not a digit times identity')
    require(all(t.width >= depth for t in w_tags) and depth <= DIM, 'residual digit tile exceeds the mesh')
    return digit


def require_digits(found, extension):
    """Each proved digit tile exactly once, summing to the proved coefficient."""
    from .stream_proof import require
    require(sorted(found) == sorted(extension.digits) and sum(found) == extension.plan.decision['coefficient'],
            'residual digit tiles do not sum to the proved coefficient exactly once')


def is_scheduled(meta):
    return meta.get('dataflow_plan') is not None


def recipe(dataflow_plan, extension=None, n=None):
    """The schedule and extra kernel inputs for an admitted composition."""
    from functools import partial
    from .recipes.gemm import KExtension, blocked_matmul
    from .recipes.output_stationary import os_matmul

    if dataflow_plan is not None:
        return os_matmul, ()
    if extension is not None:
        from .match import shape
        from .stream import Arg
        residual = Arg('residual', 'i8', tuple(shape(extension.residual)), 'input')
        return partial(blocked_matmul, extension=KExtension('residual', n, 'kext_digits', extension.digits)), (residual,)
    return None, ()


def annotate(kernel, dataflow_plan, extension=None):
    """Record the admitted plan; widen the accumulator interval by the residual terms."""
    if dataflow_plan is not None:
        kernel.meta.update(dataflow_plan=dataflow_plan, dataflow='OS')
    if extension is not None:
        from .match import etype
        bits = _dtype_bits(etype(extension.residual))
        reach = abs(extension.plan.decision['coefficient']) * (1 << (bits - 1))
        lo, hi = kernel.meta['accumulator_min'] - reach, kernel.meta['accumulator_max'] + reach
        if not -(1 << 31) <= lo <= hi < (1 << 31):
            raise ValueError('K-extension accumulator interval exceeds i32')
        kernel.meta.update(k_extension_plan=extension.plan.record(), accumulator_min=lo, accumulator_max=hi)
