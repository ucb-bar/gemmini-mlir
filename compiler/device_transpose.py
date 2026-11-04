"""Dense signed-byte transpose using bounded accumulator DMA copies."""
from __future__ import annotations
import re


def emit_byte_transpose(src, dst, rows, cols, tag):
    """Copy nonoverlapping src[rows,cols] into dst[cols,rows] exactly.

    Each command pair transports at most sixteen bytes as accumulator rows,
    with one column. Distinct load/store strides implement the transpose;
    no tensor value passes through host arithmetic. The load shrink setting
    accepts signed i8 accumulator inputs, and identity-scaled i8 readout
    preserves every byte, including -128 and 127.
    """
    if any(type(size) is not int or size <= 0 for size in (rows, cols)):
        raise ValueError('byte transpose requires positive static dimensions')
    stem = 'byte_transpose_' + re.sub(r'[^A-Za-z0-9_]', '_', str(tag))
    # Select the coordinate used as DMA rows by the number of command pairs.
    # On ties prefer contiguous input reads.
    along_rows = cols * ((rows + 15) // 16) < rows * ((cols + 15) // 16)
    length, outer = (rows, cols) if along_rows else (cols, rows)
    ld_stride, st_stride = (cols, 1) if along_rows else (1, rows)
    full, tail = divmod(length, 16)
    full *= 16
    i, j = stem + '_outer', stem + '_axis'
    body = ['{', f'gm_config_ld({ld_stride},0,1);',
            f'gm_config_st({st_stride},1.0f,0);',
            f'for(int {i}=0;{i}<{outer};{i}+=1) {{']

    def pair(coordinate, count):
        if along_rows:
            source = f'(gm_u64)({coordinate})*{cols}+(gm_u64){i}'
            dest = f'(gm_u64){i}*{rows}+(gm_u64)({coordinate})'
        else:
            source = f'(gm_u64){i}*{cols}+(gm_u64)({coordinate})'
            dest = f'(gm_u64)({coordinate})*{rows}+(gm_u64){i}'
        return [f'gm_mvin((const void*)((gm_u64)({src})+{source}),0x80000000u,{count},1,0);',
                f'gm_mvout((void*)((gm_u64)({dst})+{dest}),0x80000000u,{count},1);']

    if full:
        body += [f'for(int {j}=0;{j}<{full};{j}+=16) {{', *pair(j, 16), '}']
    if tail:
        body += pair(str(full), tail)
    body += ['}', 'gm_fence();', '}']
    return [], body
