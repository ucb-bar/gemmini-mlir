"""Explicit Gemmini addressing for readonly NR-panel rhs operands.

The exact permutation ``[K,N] -> [N/NR,K,NR]`` belongs to Merlin's
``weight_panel.pack_bytes``. This provider accepts that explicit physical
layout; it never infers layout from a pointer or workload. Dense GoldenGemm
remains unchanged. Only B load pointer arithmetic and B DMA row stride differ.
"""
from __future__ import annotations
from dataclasses import dataclass
from xdsl.dialects.builtin import StringAttr
from .golden_gemm import GoldenGemm, Shape
from .tables import rtl_facts as F


@dataclass(frozen=True)
class PackedRhs:
    k: int
    n: int
    nr: int

    def validate(self) -> None:
        if any(type(v) is not int or v <= 0 for v in (self.k, self.n, self.nr)):
            raise ValueError('packed rhs dimensions must be positive integers')
        if self.nr % F.DIM or self.nr > 4 * F.DIM or self.n % self.nr:
            raise ValueError('packed rhs requires full one-to-four-tile N panels')

    @property
    def shape(self) -> tuple[int, int, int]:
        self.validate()
        return self.n // self.nr, self.k, self.nr

    def element_offset(self, k: int, n: int) -> int:
        self.validate()
        if not 0 <= k < self.k or not 0 <= n < self.n:
            raise ValueError('packed rhs logical coordinate outside operand')
        return (n // self.nr * self.k + k) * self.nr + n % self.nr


class GoldenPackedRhsGemm(GoldenGemm):
    """Cached-A primitive schedule consuming one declared readonly B layout."""

    def __init__(self, shape: Shape, layout: PackedRhs):
        layout.validate()
        shape.validate()
        if (shape.k, shape.n) != (layout.k, layout.n):
            raise ValueError('packed rhs logical dimensions disagree with contraction')
        if (not shape.cache_a or not shape.wide_b or shape.cache_b or shape.reuse_b
                or min(shape.bn, 4) * F.DIM != layout.nr):
            raise ValueError('packed rhs requires cached A and matching physical wide-B subpanels')
        self.layout = layout
        super().__init__(shape)

    def _rocc(self, kind, attrs, pointer=None):
        if kind == 'config_ld' and attrs.get('load_id') == 1:
            attrs = dict(attrs, stride=self.layout.nr)
        super()._rocc(kind, attrs, pointer)

    def _load_b_panel(self, n0, k0, nr, kr, b_base):
        # _groups(N,bn) has no N tail under the checked layout contract.
        # n0 denotes a DIM-tile coordinate and is a multiple of bn.
        # packed offset = (ncol/NR)*K*NR + krow*NR = ncol*K + krow*NR.
        panel_tiles = self.layout.nr // F.DIM
        if len(nr) % panel_tiles or any(n != F.DIM for n in nr):
            raise ValueError('packed rhs scheduler produced a non-full N panel')
        krow = self._tile(k0, 0)
        for d in range(0, len(nr), panel_tiles):
            ncol = self._tile(n0, d)
            pointer = self._ptr(self.b, ncol, self.shape.k,
                               self.fb.mul_i(krow, self.fb.const(self.layout.nr)))
            self._rocc('mvin', {'local': (b_base + d) * F.DIM,
                               'rows': kr, 'cols': self.layout.nr, 'load_id': 1}, pointer)

    def _finish(self, symbol, batch=1):
        module = super()._finish(symbol, batch)
        module.attributes['gemmini.rhs_layout'] = StringAttr(
            f'n_panel:{self.layout.n // self.layout.nr}x{self.layout.k}x{self.layout.nr}:i8')
        return module
