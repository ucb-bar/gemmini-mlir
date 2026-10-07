"""Target-agnostic workload model extracted from the VERIFIED interface IR.

Nothing downstream reads interface text: the lowering, the command-buffer writer and the code
generator all consume this model. Every extent and every attribute in it comes from the capsule
that was handed in -- there is no default that silently stands in for a declaration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TensorDecl:
    name: str
    shape: tuple[int, ...]
    dtype: str
    role: str

    @property
    def elems(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


@dataclass
class Epilogue:
    """The readout the interface asked for, with every parameter it declared."""

    stages: tuple[str, ...] = ()
    output_dtype: str = "i32"
    acc_scale: float | None = None
    requant_shift: int | None = None
    bias: str | None = None
    pool_in_dims: tuple[int, ...] | None = None
    pool_size: tuple[int, ...] | None = None
    pool_stride: tuple[int, ...] | None = None
    pool_padding: tuple[int, ...] = (0, 0, 0, 0)
    pool_pad_value: int = 0

    @property
    def has(self) -> frozenset[str]:
        return frozenset(self.stages)


@dataclass
class Op:
    """One interface operation, normalised.

    ``kind`` is the interface mnemonic. ``operands`` maps the op's own operand names to either a
    tensor name or a resident-handle id. ``out`` is the produced tensor name (or the accumulator
    handle id for a matmul).
    """

    kind: str
    out: str
    operands: dict[str, str] = field(default_factory=dict)
    attrs: dict[str, Any] = field(default_factory=dict)
    epilogue: Epilogue | None = None
    #: extents the lowering needs, resolved from the operand/result types
    shape: dict[str, Any] = field(default_factory=dict)


@dataclass
class Workload:
    target: str
    abi_version: str
    tensors: dict[str, TensorDecl] = field(default_factory=dict)
    ops: list[Op] = field(default_factory=list)
    #: interface DECLARATION order of external tensors (input | weight | bias | output)
    decl_order: list[str] = field(default_factory=list)
    #: handle id -> the weight tensor it packs
    residents: dict[str, str] = field(default_factory=dict)
    #: kernel-INTERNAL buffers a rewrite introduced. They are addressed exactly like a DRAM tensor
    #: but the harness does not allocate them, so the kernel allocates them in its own frame.
    scratch: list[str] = field(default_factory=list)
    grammar: str = "merlin_iface"
    #: regions the compiler placed on a host lane, each with the reason it could not be accelerated
    lane_placement: list[dict[str, Any]] = field(default_factory=list)
    #: set when the whole program could not be lowered
    declined: dict[str, Any] | None = None
    #: ABI `params.im2col_recipes`: derived activations every engine materialises identically
    im2col_recipes: list[dict[str, Any]] = field(default_factory=list)
    #: for a host-lane program, the parsed region and its entry function, so the codegen can
    #: COMPILE the placement rather than only declare it
    host_module: Any = None
    host_entry: Any = None
    #: set when route Q rewrote a float contraction onto staged integer operands: the staging the
    #: kernel must allocate and the readout it must rescale (`lowering.quantize.QuantPlan`)
    quant: Any = None

    def declare(self, name: str, shape, dtype: str, role: str) -> TensorDecl:
        d = TensorDecl(name, tuple(int(x) for x in shape), dtype, role)
        self.tensors[name] = d
        if role in ("input", "weight", "bias", "output", "scale") and name not in self.decl_order:
            self.decl_order.append(name)
        return d
