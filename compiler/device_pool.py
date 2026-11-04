"""Optional native StoreController rectangular i8 pooling schedule.

Only movement executes on the host. Each input rectangle is loaded into one
512-row accumulator bank, then one pooled MVOUT reduces all of its windows.

Public RTL basis:
* ConfigMvoutRs1 fields: rtl_facts/facts.json, ConfigMvoutRs1.
* StoreController.scala 76--80, 131, 136: local input row is
  (porow*stride+wrow)*ocols+(pocol*stride+wcol); destination pixel is
  porow*pool_out_dim+pocol, multiplied by the configured byte stride.
* StoreController.scala 184--186: first window tap starts the max; subsequent
  taps enable pooling, and only the last tap stores the result.
* ReservationStation.scala 242--251: a pooled read reserves its starting
  accumulator BANK. Restricting rectangles to 512 rows keeps dependencies
  complete, even though the whole accumulator contains 1024 rows.

Padding is loaded explicitly as the signed i8 neutral value -128. Native
out-of-bounds garbage reads are therefore never used (and their value need
not be assumed). No tensor comparisons or maximum operations run on host.
"""
from __future__ import annotations
import re


def _tile_shape(oh, ow, kernel, stride):
    """Choose a productive rectangle confined to the first accumulator bank."""
    candidates = []
    for th in range(1, min(oh, 255) + 1):
        ih = (th-1)*stride+kernel
        for tw in range(1, min(ow, 255) + 1):
            iw = (tw-1)*stride+kernel
            if ih*iw <= 512:
                # Maximize outputs per store, then minimize transfer volume,
                # then prefer fewer source row segments and square tiles.
                candidates.append(((th*tw, -ih*iw, -ih, -abs(th-tw)), th, tw))
    if not candidates:
        return None
    _, th, tw = max(candidates)
    return th, tw


def emit_pool_kernel(src, dst, B, H, W, C, OH, OW, KH, KW, SH, SW, PH, PW, unique):
    """Return (globals, statements), or None when the native path cannot apply.

    Input/output are NHWC i8 C pointer expressions. PH/PW are top/left padding;
    OH/OW explicitly determine bottom/right truncation. Padded cells contribute
    -128. Callers with another pad value must materialize that padding first.
    Supports square 2x2/3x3 windows, equal stride 1/2, all channel tails, batches,
    and output rectangles whose full row width fits the 8-bit pool_out_dim.
    """
    dims = (B,H,W,C,OH,OW,KH,KW,SH,SW,PH,PW)
    if any(not isinstance(x, int) or isinstance(x, bool) for x in dims):
        return None
    if min(B,H,W,C,OH,OW) <= 0 or min(PH,PW) < 0:
        return None
    if KH != KW or KH not in (2,3) or SH != SW or SH not in (1,2):
        return None
    if OW > 255:
        return None
    selected = _tile_shape(OH,OW,KH,SH)
    if selected is None:
        return None
    th,tw = selected
    ident = unique() if callable(unique) else unique
    tag = 'native_pool_' + re.sub(r'[^A-Za-z0-9_]', '_', str(ident))
    definition = f'''
GM_INLINE void {tag}(const signed char *X, signed char *Y) {{
  static const signed char neutral[16] __attribute__((aligned(64))) = {{
    -128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128}};
  gm_config_ld({C},0,1);
  gm_config_ld(0,1,1);
  for(int bn=0;bn<{B};++bn) for(int oy=0;oy<{OH};oy+={th}) for(int ox=0;ox<{OW};ox+={tw}) {{
    int rh=({OH}-oy<{th})?{OH}-oy:{th};
    int rw=({OW}-ox<{tw})?{OW}-ox:{tw};
    int ih=(rh-1)*{SH}+{KH}, iw=(rw-1)*{SW}+{KW};
    int sy=oy*{SH}-{PH}, sx=ox*{SW}-{PW};
    gm_u64 pool_cfg=2 | ((gm_u64){SH}<<4) | ((gm_u64){KH}<<6)
      | ((gm_u64){OW}<<24) | ((gm_u64)rh<<32) | ((gm_u64)rw<<40)
      | ((gm_u64)ih<<48) | ((gm_u64)iw<<56);
    GM_INSN(0,pool_cfg,((gm_u64)0x3f800000<<32)|(gm_u64){C});
    for(int c=0;c<{C};c+=16) {{
      int nc=gm_min16({C}-c);
      for(int iy=0;iy<ih;++iy) {{
        int gy=sy+iy;
        int lo=(sx<0)?-sx:0;
        int hi=({W}-sx<iw)?{W}-sx:iw;
        if(lo>iw) lo=iw;
        if(hi<0) hi=0;
        if(gy<0 || gy>={H} || hi<lo) lo=hi=0;
        for(int x=0;x<lo;x+=16) {{
          int nr=gm_min16(lo-x);
          gm_mvin(neutral,0x80000000u+(gm_u32)(iy*iw+x),nr,nc,1);
        }}
        for(int x=lo;x<hi;x+=16) {{
          int nr=gm_min16(hi-x);
          const signed char *p=X+(((gm_u64)bn*{H}+(unsigned)gy)*{W}+(unsigned)(sx+x))*{C}+c;
          gm_mvin(p,0x80000000u+(gm_u32)(iy*iw+x),nr,nc,0);
        }}
        for(int x=hi;x<iw;x+=16) {{
          int nr=gm_min16(iw-x);
          gm_mvin(neutral,0x80000000u+(gm_u32)(iy*iw+x),nr,nc,1);
        }}
      }}
      signed char *p=Y+(((gm_u64)bn*{OH}+oy)*{OW}+ox)*{C}+c;
      gm_mvout(p,0x80000000u,1,nc);
    }}
  }}
  gm_config_st({C},1.0f,0);
  gm_fence();
}}
'''
    return [definition], [f'{tag}((const signed char*)({src}), (signed char*)({dst}));']
