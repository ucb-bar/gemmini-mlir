"""Original-source aliases selected by the shared exact preparation pass.

The library transforms a provenance-tagged clone. Its actual removals alone
form this adapter's alias roster; arithmetic equivalence and floating point
legality are never reconstructed here. Original operations remain available
for source placement and output attribution.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
import os

from merlin.llvmlower.tensor_provenance_cse import deduplicate_tensor_preparation
from merlin.xdsl_dialects._common import text
from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, IntegerAttr, i64
from xdsl.ir import Operation, SSAValue

_MARKER = 'prov.source_preparation_adapter_index'


def enabled():
    """Explicit experiment selection; ordinary compilation keeps the incumbent."""
    choice = os.environ.get('SOURCE_PREPARATION')
    if choice is not None:
        if choice not in ('off', 'exact'):
            raise ValueError('SOURCE_PREPARATION must be off or exact')
        return choice == 'exact'
    path = Path(__file__).with_name('DENSE_SCHEDULE.json')
    settings = json.loads(path.read_text()) if path.exists() else {}
    choice = settings.get('share_exact_tensor_preparations', False)
    if type(choice) is not bool:
        raise ValueError('share_exact_tensor_preparations must be Boolean')
    return choice


def _digest(module):
    return sha256(text(module, generic=True).encode()).hexdigest()


@dataclass(frozen=True)
class PreparationAlias:
    source_op_index: int
    retained_source_op_index: int
    operation: Operation
    retained_operation: Operation
    result_pairs: tuple[tuple[SSAValue, SSAValue], ...]


@dataclass(frozen=True)
class PreparationPlan:
    source_sha256: str
    original_module_sha256: str
    transformed_module_sha256: str
    library_sha256: str
    aliases: tuple[PreparationAlias, ...]
    pass_counts: tuple[tuple[str, int], ...]

    @property
    def duplicate_ops(self):
        return frozenset(alias.operation for alias in self.aliases)

    @property
    def value_aliases(self):
        return {duplicate: retained for alias in self.aliases
                for duplicate, retained in alias.result_pairs}

    def resolve(self, value):
        for alias in self.aliases:
            for duplicate, retained in alias.result_pairs:
                if value is duplicate:
                    return retained
        return value

    def revalidate(self, module):
        if _digest(module) != self.original_module_sha256:
            raise ValueError('source preparation aliases are stale; plan before source mutation')
        import merlin.llvmlower.tensor_provenance_cse as library
        if sha256(Path(library.__file__).read_bytes()).hexdigest() != self.library_sha256:
            raise ValueError('source preparation library changed; replan')

    def receipt(self):
        return dict(schema='authored.exact_source_preparation_aliases.v1',
            source_sha256=self.source_sha256,
            original_module_sha256=self.original_module_sha256,
            transformed_module_sha256=self.transformed_module_sha256,
            library_sha256=self.library_sha256,
            pass_counts=dict(self.pass_counts),
            aliases=[dict(source_op_index=a.source_op_index,
                retained_source_op_index=a.retained_source_op_index,
                operation=a.operation.name, results=len(a.result_pairs)) for a in self.aliases],
            source_mutation='none; emission aliases consume actual library removals',
            numerical_execution='not_established', target_execution='not_established')


def plan_tensor_preparations(module, *, source_sha256, operations=None):
    """Plan before ABI/allocation, preserving every original placement index.

    Only one sequential function is supported by this compiler adapter. The
    shared pass retains returned preparations, strictfp, differing initializers,
    operand identity, and unsupported operations by its own admission rules.
    """
    functions = [op for op in module.body.block.ops if op.name == 'func.func']
    if len(functions) != 1 or len(functions[0].regions[0].blocks) != 1:
        raise ValueError('preparation adapter requires one sequential source function')
    original = tuple(op for op in functions[0].regions[0].block.ops if op.name != 'func.return')
    if operations is not None and tuple(operations) != original:
        raise ValueError('preparation roster differs from original source positions')
    if (type(source_sha256) is not str or len(source_sha256) != 64
            or any(c not in '0123456789abcdef' for c in source_sha256)):
        raise ValueError('preparation plan requires the actual source SHA256')
    if any(_MARKER in str(value) or key == _MARKER
           for op in module.walk() for key, value in (*op.attributes.items(), *op.properties.items())):
        raise ValueError('source already uses the adapter provenance namespace')
    before = _digest(module)
    clone = module.clone()
    function = next(op for op in clone.body.block.ops if op.name == 'func.func')
    copied = tuple(op for op in function.regions[0].block.ops if op.name != 'func.return')
    for index, op in enumerate(copied):
        metadata = {key: value for key, value in op.attributes.items()
                    if key.startswith('prov.') and key != 'prov.cse_sources'}
        metadata[_MARKER] = IntegerAttr(index, i64)
        previous = op.attributes.get('prov.cse_sources')
        if previous is not None and not isinstance(previous, ArrayAttr):
            raise ValueError('existing preparation provenance records must be an array')
        op.attributes['prov.cse_sources'] = ArrayAttr(
            (() if previous is None else previous.data) + (DictionaryAttr(metadata),))
    counts = deduplicate_tensor_preparation(clone)
    positions = {op: index for index, op in enumerate(copied)}
    aliases = []
    removed = set()
    for survivor in function.regions[0].block.ops:
        if survivor not in positions:
            continue
        retained = positions[survivor]
        for record in survivor.attributes['prov.cse_sources']:
            if not isinstance(record, DictionaryAttr) or _MARKER not in record.data:
                continue
            index = int(record.data[_MARKER].value.data)
            if index == retained:
                continue
            if index <= retained or copied[index].parent is not None or index in removed:
                raise ValueError('library preparation removal lost its dominating source correspondence')
            old, keep = original[index], original[retained]
            if len(old.results) != len(keep.results):
                raise ValueError('library preparation alias changed result arity')
            aliases.append(PreparationAlias(index, retained, old, keep,
                tuple(zip(old.results, keep.results, strict=True))))
            removed.add(index)
    actual = {index for index, op in enumerate(copied) if op.parent is None}
    if actual != removed or len(removed) != sum(counts.values()):
        raise ValueError('library preparation report and actual alias roster disagree')
    if _digest(module) != before:
        raise ValueError('preparation planning mutated original source')
    import merlin.llvmlower.tensor_provenance_cse as library
    return PreparationPlan(source_sha256, before, _digest(clone),
        sha256(Path(library.__file__).read_bytes()).hexdigest(),
        tuple(sorted(aliases, key=lambda alias: alias.source_op_index)), tuple(counts.items()))


def bind_pointer_aliases(plan, pointers):
    """After allocating retained values, bind all duplicate SSA pointers."""
    for alias in plan.aliases:
        for duplicate, retained in alias.result_pairs:
            if retained in pointers:
                pointers[duplicate] = pointers[retained]


def bind_runtime_alias(plan, operation, values, *, materialized_results=None):
    """At an elided source site, bind the exact dominating host tensor value."""
    for alias in plan.aliases:
        if alias.operation is operation:
            for duplicate, retained in alias.result_pairs:
                values[duplicate] = values[retained]
                if materialized_results is not None and retained in materialized_results:
                    materialized_results.add(duplicate)
            return alias
    return None
