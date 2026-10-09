"""Mixed-lane lowering: one program that drives the mesh AND the scalar lane.

A model does not belong to one lane.  Its int8 contractions are exactly what this target's
capability manifest admits on the 16x16 mesh; the normalizations, casts and elementwise maps
between them are families the mesh has no datapath for and belong on the scalar lane.  Leaving
the admitted work on the host is a compiler defect, not a placement choice, so this pass splits
the module's own dataflow into an ordered list of SEGMENTS -- a mesh contraction, or a run of
host ops -- and gives every value that crosses a segment boundary a DRAM buffer.

Nothing here is keyed on a capsule: which ops go to the mesh is decided by
`frontend.linalg_reader.place` (family + operand dtype against the RTL-derived datapath) and by
whether the op IS a contraction the tile schedule can express, and every extent is read from the
op's own operand types.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from xdsl.dialects.builtin import IntegerType, TensorType
from xdsl.ir import Operation, SSAValue

from ..contraction_patterns import match_integer_gemm
from ..codegen.host_linalg import HOST_LINALG_ELEMENT_BUDGET, estimate_cost
from ..frontend.linalg_reader import HOST_LANE, MESH_LANE, LinalgWorkload
from .plan import Buffer, Epilogue, LoweringDeclined, Plan, kernel_args


@dataclass
class HostSegment:
    """A run of host-lane ops, with the DRAM bindings that carry values in and out."""

    ops: list[Operation] = field(default_factory=list)
    inputs: list[tuple[SSAValue, str]] = field(default_factory=list)
    outputs: list[tuple[SSAValue, str]] = field(default_factory=list)
    regions: list[str] = field(default_factory=list)


def _shape(ty) -> tuple[int, ...]:
    if not isinstance(ty, TensorType):
        raise LoweringDeclined(f"expected a tensor type, got {ty}", op="model_lane")
    return tuple(int(d) for d in ty.get_shape())


def _elem(ty) -> str:
    ety = ty.get_element_type()
    if isinstance(ety, IntegerType):
        return f"i{int(ety.width.data)}"
    return str(ety)


def _func_of(module) -> Operation:
    for op in module.walk():
        if op.name == "func.func":
            return op
    raise LoweringDeclined("the module declares no func.func to lower", op="model_lane")


def _region_id(op: Operation) -> str:
    attr = op.attributes.get("prov.region_id")
    return getattr(attr, "data", "") or ""


def mesh_eligible(op: Operation, lane_of: dict[str, str]) -> bool:
    """Is this op a contraction the mesh both ADMITS and this backend can schedule?

    Both halves matter.  The placement rule says which (family, dtype) the datapath admits; this
    adds the second question a placement cannot answer -- whether the op's own shape is one the
    tile schedule expresses (a rank-2 contraction at the mesh operand dtype).
    """
    if lane_of.get(_region_id(op)) != MESH_LANE:
        return False
    matched = match_integer_gemm(op)
    return matched is not None and matched.batch == 1 and len(_shape(op.operands[0].type)) == 2


class MixedBuilder:
    """Splits one `linalg-on-tensors` function into mesh contractions and host segments."""

    def __init__(self, module, wl: LinalgWorkload):
        self.module = module
        self.wl = wl
        self.func = _func_of(module)
        self.block = self.func.regions[0].blocks[0]
        self.lane_of = {r.region_id: r.lane for r in wl.regions}
        self.buffers: dict[str, Buffer] = {}
        self.order: list[str] = []
        self.of_value: dict[SSAValue, str] = {}
        self.ordered: list[Any] = []
        self.commands: list[dict[str, Any]] = []
        self.residents: dict[str, str] = {}
        self._scratch = 0

    # -- buffers -----------------------------------------------------------------------------
    def declare(self, name: str, shape, dtype: str, role: str) -> str:
        self.buffers[name] = Buffer(name, [int(d) for d in shape] or [1], dtype, role)
        if name not in self.order:
            self.order.append(name)
        return name

    #: The role a compiler-owned intermediate is declared with.  NOT `input`: a leaf input is a
    #: tensor some command consumes and none produces, and the dataflow binder reads the buffer's
    #: leaves as the model's own inputs.  An intermediate IS produced -- by a COMMIT, or by the
    #: host segment that spills it -- so declaring it `input` would add a tensor nobody passes in
    #: to the list of things the runner thinks it has to supply.
    INTERMEDIATE_ROLE = "output"

    def buffer_for(self, value: SSAValue, role: str = "") -> str:
        """The DRAM buffer holding `value`, declaring an intermediate one on first use."""
        role = role or self.INTERMEDIATE_ROLE
        hit = self.of_value.get(value)
        if hit is not None:
            return hit
        name = f"t{self._scratch}"
        self._scratch += 1
        self.of_value[value] = self.declare(name, _shape(value.type), _elem(value.type), role)
        return name

    # -- entry point -------------------------------------------------------------------------
    def build(self) -> Plan:
        ops = [op for op in self.block.ops if op.name != "func.return"]
        returns = [op for op in self.block.ops if op.name == "func.return"]
        if not returns:
            raise LoweringDeclined("the entry function returns nothing to write", op="model_lane")
        result_values = list(returns[0].operands)

        mesh_ops = [op for op in ops if mesh_eligible(op, self.lane_of)]
        if not mesh_ops:
            raise LoweringDeclined(
                "no region of this module is a contraction the mesh admits", op="model_lane")
        cost = estimate_cost([op for op in ops if op not in mesh_ops])
        if cost > HOST_LINALG_ELEMENT_BUDGET:
            raise LoweringDeclined(
                f"the host-lane half of this module needs about {cost} straight-line element "
                f"evaluations, past this backend's {HOST_LINALG_ELEMENT_BUDGET} budget; the "
                f"emitted kernel is single-block so there is no loop to roll them into",
                op="model_lane",
                shape=list(_shape(result_values[0].type)) if result_values else [])

        # interface tensors: the entry's arguments, then its results, in declaration order
        rhs_values = {op.operands[1] for op in mesh_ops}
        for i, arg in enumerate(self.block.args):
            role = "weight" if arg in rhs_values else "input"
            self.of_value[arg] = self.declare(f"arg{i}", _shape(arg.type), _elem(arg.type), role)
        out_names: list[str] = []
        for i, value in enumerate(result_values):
            name = self.declare(f"Y{i}", _shape(value.type), _elem(value.type), "output")
            # A result the mesh commits to IS this buffer; a result the host computes is stored
            # into it.  Either way the interface's own name is the one the runner reads back.
            self.of_value.setdefault(value, name)
            if self.of_value[value] != name:
                self.of_value[value] = name
            out_names.append(name)

        segments: list[HostSegment] = []
        pending: list[Operation] = []
        mesh_set = set(id(op) for op in mesh_ops)
        schedule: list[Any] = []
        for op in ops:
            if id(op) in mesh_set:
                schedule.append(("host", pending))
                pending = []
                schedule.append(("mesh", op))
            else:
                pending.append(op)
        schedule.append(("host", pending))

        # Which values each host run must LEAVE in DRAM: whatever a later mesh op reads.
        needed_by_mesh: dict[SSAValue, None] = {}
        for op in mesh_ops:
            needed_by_mesh[op.operands[0]] = None
            needed_by_mesh[op.operands[1]] = None

        # Two different places a value can already be live when a host run needs it: in DRAM
        # (an interface tensor, or what a mesh contraction committed) or in the CPU lane's own
        # SSA (an earlier host run computed it and never had to spill it).  Only the first kind
        # needs a binding; confusing the two is how a run reloads a value nobody stored.
        in_dram: set[SSAValue] = set(self.block.args)
        in_ssa: set[SSAValue] = set()
        carry: list[Operation] = []
        for kind, payload in schedule:
            if kind == "mesh":
                self._mesh(payload, in_dram)
                in_dram.add(payload.results[0])
                continue
            host_ops = carry + [op for op in payload if op.name != "func.return"]
            defined = [r for op in host_ops for r in op.results]
            defined_set = set(defined)
            outputs: list[tuple[SSAValue, str]] = []
            for value in defined:
                if value in needed_by_mesh:
                    outputs.append((value, self.buffer_for(value)))
                elif value in result_values:
                    outputs.append((value, self.of_value[value]))
            if not outputs:
                # This run hands nothing to the mesh and nothing to the interface -- it is the
                # glue that initialises the next contraction's accumulator.  Carry its ops into
                # the next run rather than dropping them: a later run may still read them.
                carry = host_ops
                continue
            carry = []
            inputs: list[tuple[SSAValue, str]] = []
            seen: set[SSAValue] = set()
            for op in host_ops:
                for value in op.operands:
                    if value in defined_set or value in seen or value in in_ssa:
                        continue
                    if value not in in_dram:
                        raise LoweringDeclined(
                            f"the host run reads a value that is neither an interface tensor, a "
                            f"mesh result, nor computed by an earlier run: {value.type}",
                            op="model_lane")
                    seen.add(value)
                    inputs.append((value, self.of_value[value]))
            in_ssa |= defined_set
            in_dram |= {v for v, _ in outputs}
            segments.append(HostSegment(
                host_ops, inputs, outputs,
                sorted({_region_id(op) for op in host_ops if _region_id(op)})))
            self.ordered.append(segments[-1])

        # a mesh result that is the entry's own result needs no host store; one that is not is
        # read back by the segment after it, which the binding loop above already wired.
        cb = self._command_buffer(segments)
        return Plan("gemmini", self.buffers, self.ordered, cb,
                    kernel_args(cb, list(self.order)))

    # -- one mesh contraction ----------------------------------------------------------------
    def _mesh(self, op: Operation, in_dram: set[SSAValue]) -> None:
        from .plan import Contraction

        lhs_v, rhs_v = op.operands[0], op.operands[1]
        out_v = op.results[0]
        for operand in (lhs_v, rhs_v):
            if operand not in in_dram:
                raise LoweringDeclined(
                    "a mesh contraction reads an operand no earlier segment left in DRAM",
                    op="model_lane")
        lhs = self.of_value.get(lhs_v) or self.buffer_for(lhs_v)
        rhs = self.of_value.get(rhs_v) or self.buffer_for(rhs_v)
        dst = self.of_value.get(out_v) or self.buffer_for(out_v)
        m, k = _shape(lhs_v.type)
        _, n = _shape(rhs_v.type)
        out_dtype = _elem(out_v.type)
        handle = self.residents.get(rhs)
        if handle is None:
            handle = f"{rhs}_res"
            self.residents[rhs] = handle
            self.commands.append({"opcode": "RES_PACK",
                                  "operands": {"src": rhs, "dst": handle},
                                  "attributes": {"layout": "packed_rhs"}})
        acc = f"acc_{dst}"
        self.commands.append({"opcode": "MATMUL_RESIDENT",
                              "operands": {"lhs": lhs, "rhs": handle, "dst": acc}})
        self.commands.append({"opcode": "COMMIT",
                              "operands": {"src": acc, "dst": dst},
                              "attributes": {"epilogue": [], "output_dtype": out_dtype}})
        self.ordered.append(
            Contraction(lhs, rhs, dst, m, k, n, lhs_row_elems=k, rhs_row_elems=n,
                        epilogue=Epilogue(stages=[], output_dtype=out_dtype)))

    # -- serialisation -----------------------------------------------------------------------
    def _command_buffer(self, segments: list[HostSegment]) -> dict[str, Any]:
        # Every interface tensor a HOST segment touches also needs an address in the kernel's
        # argument list, and the ABI's resident-matmul row builds that list from the RES_PACK
        # sources.  Making them resident is what the kernel does anyway: the segment reads them
        # straight out of DRAM.
        host_touched: list[str] = []
        for seg in segments:
            for _, name in seg.inputs + seg.outputs:
                if name not in host_touched:
                    host_touched.append(name)
        head: list[dict[str, Any]] = []
        for name in host_touched:
            if name in self.residents or self.buffers[name].role == "output":
                continue
            if any(c["operands"].get("lhs") == name for c in self.commands):
                continue
            self.residents[name] = f"{name}_res"
            head.append({"opcode": "RES_PACK",
                         "operands": {"src": name, "dst": f"{name}_res"},
                         "attributes": {"layout": "host_lane_operand"}})
        commands = head + self.commands
        tensors = {n: {"shape": self.buffers[n].shape, "dtype": self.buffers[n].dtype,
                       "role": self.buffers[n].role} for n in self.order}
        placement = [{"region": r.region_id, "family": r.family, "op": r.op, "dtype": r.dtype,
                      "lane": r.lane, "reason": r.reason} for r in self.wl.regions]
        return {
            "abi_version": "0.1",
            "target": "gemmini",
            "backend": "mlir_oot_xdsl_gemmini",
            "tensors": tensors,
            "commands": commands,
            "params": {
                "lane_placement": placement,
                "mesh_regions": [r.region_id for r in self.wl.mesh_regions],
                "host_lane_regions": [r.region_id for r in self.wl.host_regions],
                "lanes": {
                    "reported": sorted({r.lane for r in self.wl.regions}),
                    MESH_LANE: [r.region_id for r in self.wl.mesh_regions],
                    HOST_LANE: [r.region_id for r in self.wl.host_regions],
                },
                "host_lane_segments": [
                    {"regions": seg.regions,
                     "reads": [n for _, n in seg.inputs],
                     "writes": [n for _, n in seg.outputs]} for seg in segments],
            },
        }


def build(module, wl: LinalgWorkload) -> Plan:
    return MixedBuilder(module, wl).build()
