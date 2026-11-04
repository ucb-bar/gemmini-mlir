"""Column-varying i8 readout through a transposed device contraction.

The host gathers bytes and repeats bias values only.  Gemmini performs every
multiply, accumulation, bias addition, activation and scaled conversion.
"""
from collections import Counter
from math import ceil, isfinite


def runtime_header():
    """The generated code reuses the primitive gm_* instruction helpers."""
    return ''


def emit_column_product(a, b, y, m, k, n, attrs, cb, names, tag, m_capacity=None, *, force=False):
    """Return (globals, body), or None when grouped GEMMs are preferable.

    Runtime M expressions are admitted when m_capacity gives their upper bound.
    Fixed staging storage is 10 KiB, independent of tensor sizes.
    """
    if attrs.get('output_dtype', 'i32') != 'i8':
        return None
    scales = attrs.get('acc_scale_by_column')
    if not scales or len(scales) != n or 'acc_scale' not in attrs.get('epilogue', []):
        return None
    if any(not isfinite(float(s)) for s in scales):
        return None
    groups = Counter(float(s) for s in scales)
    if len(groups) < 2:
        return None
    capacity = m if isinstance(m, int) else m_capacity
    if not isinstance(capacity, int) or capacity < 16 or k <= 0 or n <= 0:
        return None
    # Keep column regrouping within one 64-KiB host transport working set.
    # Beyond this bound, grouped lowering materializes full contraction-sized
    # weight/output gathers. The transposed schedule uses fixed tile staging
    # and device accumulator row readout, even when it issues more commands.
    max_group = max(groups.values())
    grouped_staging = (k + capacity) * max_group
    if any(stage in attrs.get('epilogue', []) for stage in ('bias', 'bias_add')):
        grouped_staging += 4 * max_group
    bounded_transport = grouped_staging >= 64 * 1024
    if not force and not bounded_transport and len(groups) < 4:
        return None
    # Compare primitive issue work for grouped ordinary GEMMs and transposed
    # row readout.  Distinct gains alone are insufficient: long rows require
    # extra one-row stores, which can outweigh modest grouping inefficiency.
    mt, nt, kt = ceil(capacity / 16), ceil(n / 16), ceil(k / 16)
    mb, kb = ceil(capacity / 64), ceil(k / 64)
    group_tiles = sum(ceil(count / 16) for count in groups.values())
    old_commands = 5 * len(groups) + mt * group_tiles * (4 * kt + 1)
    new_commands = 4 + nt * mb * (kb + kt) + 2 * nt * mt * kt + 2 * n * mt
    if not force and not bounded_transport and old_commands < 1.15 * new_commands:
        return None
    # Reuse the common semantic checks and bias-name resolution.  Importing at
    # call time avoids a module cycle when conv._product selects this lowering.
    from conv import _epilogue, _f32
    width, _, relu, bias, checked_scales = _epilogue(attrs, cb, names, n)
    assert width == 1 and checked_scales is not None
    prefix = tag + '_column'
    lhs, rhs, out = prefix + '_lhs', prefix + '_rhs', prefix + '_out'
    bp, gains = prefix + '_bias', prefix + '_gains'
    glob = [f'static signed char {lhs}[16*64] __attribute__((aligned(64)));',
            f'static signed char {rhs}[64*64] __attribute__((aligned(64)));',
            f'static signed char {out}[16*64] __attribute__((aligned(64)));',
            f'static const float {gains}[{n}] = {{{",".join(_f32(x) for x in scales)}}};']
    if bias != '0':
        glob.append(f'static int {bp}[16*64] __attribute__((aligned(64)));')
    bias_init = ''
    if bias != '0':
        bias_init = f'''for(int c=0;c<nc;c++) for(int r=0;r<mr;r++)
      {bp}[c*64+r]=({bias})[nb+c];
    for(int br=0;br<mr;br+=16) {{
      int bn=gm_min16(mr-br);
      gm_mvin({bp}+br,0x80000000u+br,nc,bn,2);
    }}'''
    # Use a C block so separate generated products have independent loop names.
    body = [f'''{{
  gm_config_ex();
  gm_config_ld(64,0,0);
  gm_config_ld(64,1,0);
  gm_config_ld(64*4,2,0);
  for(int nb=0;nb<{n};nb+=16) {{
    int nc=gm_min16({n}-nb);
    for(int rb=0;rb<({m});rb+=64) {{
      int mr=({m})-rb; if(mr>64) mr=64;
      {bias_init}
      for(int kb=0;kb<{k};kb+=64) {{
        int kr={k}-kb; if(kr>64) kr=64;
        /* Left tile is B^T, right tile is A^T. */
        for(int c=0;c<nc;c++) for(int q=0;q<kr;q++)
          {lhs}[c*64+q]=((const signed char*)({b}))[(gm_u64)(kb+q)*{n}+nb+c];
        for(int q=0;q<kr;q++) for(int r=0;r<mr;r++)
          {rhs}[q*64+r]=((const signed char*)({a}))[(gm_u64)(rb+r)*{k}+kb+q];
        gm_mvin({lhs},0,nc,kr,0);
        for(int q=0;q<kr;q+=16) {{
          int qn=gm_min16(kr-q);
          gm_mvin({rhs}+q*64,4096+(q/16)*64,qn,mr,1);
        }}
        for(int q=0;q<kr;q+=16) {{
          int qn=gm_min16(kr-q);
          for(int r=0;r<mr;r+=16) {{
            int rn=gm_min16(mr-r);
            gm_u32 ca=((kb || q || {int(bias != '0')})?0xc0000000u:0x80000000u)+r;
            gm_preload(4096+(q/16)*64+r,ca,qn,rn,nc);
            gm_compute(q,nc,qn);
          }}
        }}
        /* The next panel overwrites host DMA staging; wait before reuse. */
        if(kb+64<{k}) gm_fence();
      }}
      /* A local address selects an accumulator row.  Original columns are
         rows of B^T*A^T, allowing a separate scale for each original column. */
      for(int r=0;r<mr;r+=16) {{
        int rn=gm_min16(mr-r);
        for(int c=0;c<nc;c++) {{
          gm_config_st(64,{gains}[nb+c],{relu});
          gm_mvout({out}+c*64+r,0x80000000u+r+c,1,rn);
        }}
      }}
      gm_fence();
      for(int r=0;r<mr;r++) for(int c=0;c<nc;c++)
        ((signed char*)({y}))[(gm_u64)(rb+r)*{n}+nb+c]={out}[c*64+r];
    }}
  }}
}}''']
    return glob, body
