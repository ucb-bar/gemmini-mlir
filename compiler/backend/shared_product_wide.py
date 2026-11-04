"""Checked shared-input wide loads under the original group's complete owner.

Only adjacent tile loads coalesce. Every actual wide band is independently
projected to its original member pointer and verified for source coverage,
private-window ownership, retention and accumulator lifetime by the shared
scheduler's checker. The selected RTL's audited 64-byte limit is unchanged.
"""
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path

from merlin.sched.ir import Call, Const, Ptr, as_expr
from merlin.compile.scheduling.block_schedule import Load
from merlin.compile.scheduling.compact_dense import _digest
from merlin.compile.scheduling.compact_dense_group import CompactDenseGroupResult
from merlin.compile.scheduling.compact_dense_group_check import verify_compact_dense_group
from merlin.compile.scheduling.panel_tile_loads import expand_panel_tile_loads
from merlin.compile.scheduling.tile_load_coalescing import (
    TileLoadAddressLaw, TileLoadMapping, _linear, propose_tile_load_coalescing,
)
from .dense_wide_loads import selected_addressing_evidence
from .dense_load_coalescing import _facts_digest


@dataclass(frozen=True)
class SharedWideCodec:
    problems: tuple
    storages: tuple
    adapter: object
    encode_member: object
    project_member: object
    identity: str

    def project(self, call):
        member, operation = self.project_member(call)
        if type(operation) is not Load:
            return None
        if (operation.role not in ('lhs', 'weight') or operation.step != Const(1)
                or self.encode_member(member, operation) != call):
            raise ValueError('shared wide load needs exact original member codec and pointer')
        left = operation.role == 'lhs'
        if (left and member is not None) or (not left and (type(member) is not int or not 0 <= member < len(self.problems))):
            raise ValueError('shared wide load changed original member ownership')
        index = 0 if left else member
        p, storage = self.problems[index], self.storages[index]
        pitch = p.k if left else p.n
        tensor = 'lhs' if left else f'rhs{member}'
        pointer = dict(call.args).get('external_pointer')
        expected = Ptr(tensor, as_expr(operation.dram_row) * pitch + operation.dram_col)
        if pointer != expected:
            raise ValueError('shared wide load changed original external pointer identity')
        view = storage.lhs if left else storage.rhs
        if ((p.tile_m,p.tile_k,p.tile_n,p.lhs_bytes,p.rhs_bytes) != (16,16,16,1,1)
                or view.backing != 'spad' or view.unit_bytes != 16):
            raise ValueError('shared wide loads require original contiguous i8 DIM16 windows')
        law = TileLoadAddressLaw(16,16,64,1,pitch,16,self.identity)
        return TileLoadMapping(tensor, view.backing, pointer.offset, operation.row,
                               operation.rows, operation.cols, law)

    def encode(self, first, columns):
        self.project(first)
        if type(columns) is not int or not 0 < columns <= 64:
            raise ValueError('shared wide MVIN exceeds selected 64-byte limit')
        member, operation = self.project_member(first)
        return self.encode_member(member, replace(operation, cols=Const(columns)))

    def encode_band(self, actual, band):
        mapping = self.project(actual)
        if mapping is None or band.law != mapping.law or band.source != mapping.source:
            raise ValueError('shared wide band changed original source/channel identity')
        ac,av = _linear(mapping.source_byte)
        bc,bv = _linear(band.source_byte)
        offset = bc-ac
        if av != bv or offset < 0 or offset % 16:
            raise ValueError('shared wide band is not an exact adjacent original tile')
        member, op = self.project_member(actual)
        return self.encode_member(member, replace(op,
            block=(op.block[0],op.block[1]+offset//16),
            dram_col=op.dram_col+offset, row=band.local_row,
            rows=band.rows, cols=band.columns))


@dataclass(frozen=True)
class SharedWideResult:
    original: object
    kernel: object
    realization: dict
    _guard: object

    @property
    def choice(self):
        return self.original.choice

    def revalidate(self):
        self.original.revalidate()
        self._guard(self)


def select_shared_wide_loads(original, problems, resources, storages, adapter, *,
                             encode_member, project_member, retention_law, facts):
    """Admission before completion rewriting; refusal leaves the incumbent intact."""
    if type(original) is not CompactDenseGroupResult:
        raise ValueError('shared wide loads require original checked shared-group owner')
    original.revalidate()
    evidence = selected_addressing_evidence(facts)
    codec = SharedWideCodec(problems,storages,adapter,encode_member,project_member,evidence['identity'])
    proposal, report = propose_tile_load_coalescing(original.kernel,
        project=codec.project, encode=codec.encode, max_tiles=4,
        max_static_nodes=4096, coalesce_tail=True)
    if not report['changes']:
        return original, dict(applied=False, refusal='no contiguous load-only tile run', proposal=report)
    expanded, projection = expand_panel_tile_loads(proposal, project=codec.project,
        encode_band=codec.encode_band, max_static_nodes=4096)
    checked = verify_compact_dense_group(expanded,problems,resources,storages,adapter,
        project_member=project_member,retention_law=retention_law,dtypes=('i8','i8','i32'),
        static_budget=256,external_buffers_disjoint=True,timeout_ms=5000,max_members=8)
    original.revalidate()
    receipt = dict(applied=True, addressing=evidence, proposal=report,
        tile_load_projection=projection, original_kernel_sha256=original.kernel.digest(),
        kernel_sha256=proposal.digest(), checking_expansion_sha256=expanded.digest(),
        complete_call_cycles='UNKNOWN',target_execution='not_run')
    realization = dict(original.realization, wide_loads=receipt,
        kernel_sha256=proposal.digest(),wide_source_lifetime_check=checked)
    # Freeze the actual owner, callbacks, facts and implementation. Revalidation
    # must not silently accept a different original, mapping or checking band.
    paths = tuple(dict.fromkeys((*original.implementation_paths(), Path(__file__).resolve(),
        Path(__file__).with_name('dense_wide_loads.py'),
        Path(__file__).with_name('dense_load_coalescing.py'))))
    pins = tuple((p,sha256(p.read_bytes()).hexdigest()) for p in paths)
    def functions():
        return (SharedWideCodec.project,SharedWideCodec.encode,SharedWideCodec.encode_band,
                SharedWideResult.revalidate,select_shared_wide_loads,
                propose_tile_load_coalescing,expand_panel_tile_loads,verify_compact_dense_group)
    callbacks = functions()
    codes = tuple(f.__code__ for f in callbacks)
    frozen = (proposal.digest(), expanded.digest(), _digest(projection), _facts_digest(facts),
              _digest(realization))
    def guard(current):
        if (current.original is not original
                or pins != tuple((p,sha256(p.read_bytes()).hexdigest()) for p in paths)
                or callbacks != functions() or codes != tuple(f.__code__ for f in functions())):
            raise ValueError('shared wide-load implementation or original owner changed')
        selected_addressing_evidence(facts)
        now_expanded, now_projection = expand_panel_tile_loads(current.kernel,
            project=codec.project,encode_band=codec.encode_band,max_static_nodes=4096)
        now = (current.kernel.digest(),now_expanded.digest(),_digest(now_projection),
               _facts_digest(facts),_digest(current.realization))
        if now != frozen:
            raise ValueError('shared wide-load actual commands, pointers, facts or checked realization changed')
    result = SharedWideResult(original,proposal,realization,guard)
    result.revalidate()
    return result, receipt
