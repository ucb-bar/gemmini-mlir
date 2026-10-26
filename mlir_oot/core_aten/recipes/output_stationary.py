"""Output-stationary matmul for constant weights whose whole-K partials fit the mesh.

C[M,N] = A[M,K] . B[K,N] (+ int32 seed) with every output tile's complete K
reduction held in the mesh PEs: one PRELOAD names the output only before the
last K tile, so the earlier partials never leave the array.  B is resident in
the scratchpad for the whole kernel; A row blocks alternate between two
scratchpad regions and output tiles between two accumulator halves.  The int32
seed is a full-width MVIN3 into the accumulator, which the final mesh output
adds to; D is never preloaded, so the mesh starts every tile from zero.

Only a caller holding a whole-prefix value bound for every mesh partial may
select this schedule (the constant-weight proof is Merlin's); the recipe itself
proves nothing about wraparound.
"""

from __future__ import annotations

from ..stream import ACC, ACC_ROWS, DIM, GARBAGE, NO_ACT, OS, SP, SPAD_ROWS, Stream


def _cdiv(a, b):
    return (a + b - 1) // b


def resident_rows(M: int, N: int, K: int) -> dict:
    """Local memory the schedule occupies, or ValueError when it does not fit."""
    Kt, Nt = _cdiv(K, DIM), _cdiv(N, DIM)
    weights = Kt * Nt * DIM
    rows = dict(weights=weights, operand=2 * Kt * DIM, accumulator=2 * Nt * DIM)
    if weights + rows['operand'] > SPAD_ROWS or rows['accumulator'] > ACC_ROWS:
        raise ValueError('output-stationary operands exceed resident local memory')
    return rows


def os_matmul(s: Stream, A: tuple[str, int], lda: int, B: tuple[str, int], ldb: int,
              C: tuple[str, int], ldc: int, M: int, K: int, N: int, *,
              bias: tuple[str, int] | None = None, scale: float = 1.0, relu: bool = False,
              out_i32: bool = False, configure: bool = True, bias_rows: bool = False) -> None:
    """Argument-compatible with ``blocked_matmul``; full-width int32 readout only."""
    if not out_i32 or scale != 1.0 or relu or not configure:
        raise ValueError('output-stationary schedule stores raw int32 and configures itself')
    rows = resident_rows(M, N, K)
    Mt, Nt, Kt = _cdiv(M, DIM), _cdiv(N, DIM), _cdiv(K, DIM)
    an, ao = A
    bn, bo = B
    cn, co = C
    s.config_ex(dataflow=OS)
    s.config_ld(0, lda)
    s.config_ld(1, ldb)
    s.config_ld(2, ldc * 4 if bias_rows else 0)
    s.config_st(ldc * 4, NO_ACT, 1.0)
    for k in range(Kt):
        kr = min(DIM, K - k * DIM)
        for j in range(0, Nt, 4):
            cols = min(min(4, Nt - j) * DIM, N - j * DIM)
            s.mvin(s.at(bn, bo + k * DIM * ldb + j * DIM), SP((k * Nt + j) * DIM), cols, kr, 1)
    for i in range(Mt):
        mr = min(DIM, M - i * DIM)
        a0 = rows['weights'] + (i & 1) * Kt * DIM
        acc0 = (i & 1) * Nt * DIM
        for k in range(0, Kt, 4):
            cols = min(min(4, Kt - k) * DIM, K - k * DIM)
            s.mvin(s.at(an, ao + i * DIM * lda + k * DIM), SP(a0 + k * DIM), cols, mr, 0)
        if bias is not None:
            for j in range(Nt):
                nc = min(DIM, N - j * DIM)
                offset = bias[1] + ((i * DIM * N if bias_rows else 0) + j * DIM) * 4
                s.mvin(s.at(bias[0], offset), ACC(acc0 + j * DIM), nc, mr, 2)
        for j in range(Nt):
            nc = min(DIM, N - j * DIM)
            for k in range(Kt):
                kr = min(DIM, K - k * DIM)
                out = ACC(acc0 + j * DIM, accumulate=bias is not None) if k == Kt - 1 else GARBAGE
                s.preload(GARBAGE, out, DIM, DIM, nc, mr)
                s.compute(SP(a0 + k * DIM), kr, mr, accumulate=k > 0,
                          bd=SP((k * Nt + j) * DIM), bd_cols=nc, bd_rows=kr)
        for j in range(Nt):
            nc = min(DIM, N - j * DIM)
            s.mvout(s.at(cn, co + (i * DIM * ldc + j * DIM) * 4), ACC(acc0 + j * DIM, full=True), nc, mr)
