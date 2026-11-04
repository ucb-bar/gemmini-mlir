"""Direct-DMA NHWC convolution with no scalar tensor loads or im2col buffer.

Public LoadController.scala72/109 declares rs1==0 to be an all-zero transfer.
Each output row's receptive field therefore arrives directly in scratchpad,
with invalid coordinates represented by a null DMA source. This computes
padding on the accelerator instead of conditionally gathering host values.

The 16x64 output panel uses A scratchpad rows0..15, B8192..8255, and
accumulator rows0..63. Load-controller block_stride16 distributes a wide
64-column load over four adjacent sixteen-row blocks. No local allocation
or scalar tensor value access is emitted.
"""
from __future__ import annotations
import re


def _leaf(name, cb):
    seen = set()
    while name not in cb['tensors']:
        if name in seen:
            return None
        seen.add(name)
        command = next((c for c in cb['commands'] if c['opcode']=='RES_PACK' and c['operands']['dst']==name),None)
        if command is None:
            return None
        name = command['operands']['src']
    return name


def emit_direct_conv(cmd, cb, names, unique):
    """Emit a raw i8xi8->i32 convolution, or None for another operation form."""
    if cmd.get('opcode') != 'CONV2D':
        return None
    attrs = cmd.get('attributes', {})
    op = cmd['operands']
    if attrs.get('epilogue', []) or attrs.get('output_dtype','i32') != 'i32':
        return None
    if attrs.get('layout','nhwc') != 'nhwc' or 'groups' in attrs or attrs.get('seed_tensor'):
        return None
    a = _leaf(op['ifm'], cb)
    b = _leaf(op['weight'], cb)
    y = op['dst']
    if a is None or b is None:
        return None
    tensors=cb['tensors']
    sa,sb,sy = tensors[a],tensors[b],tensors[y]
    if sa['dtype']!='i8' or sb['dtype']!='i8' or sy['dtype']!='i32' or len(sa['shape'])!=4:
        return None
    batch,h,w,ci = sa['shape']
    kh,kw,kci,co = attrs['kernel']
    sh,sw = attrs.get('stride',[1,1])
    dh,dw = attrs.get('dilation',[1,1])
    pt,pl,pb,pr = attrs.get('padding',[0,0,0,0])
    if min(batch,h,w,ci,kh,kw,kci,co,sh,sw,dh,dw)<=0 or min(pt,pl,pb,pr)<0 or ci!=kci:
        return None
    oh=(h+pt+pb-(dh*(kh-1)+1))//sh+1
    ow=(w+pl+pr-(dw*(kw-1)+1))//sw+1
    m=batch*oh*ow
    if min(oh,ow)<=0 or sb['shape']!=[kh*kw*ci,co] or sy['shape']!=[m,co]:
        return None
    identifier=unique() if callable(unique) else unique
    tag='direct_conv_'+re.sub(r'[^A-Za-z0-9_]','_',str(identifier))
    definition=f'''
GM_INLINE void {tag}(const signed char *X,const signed char *W,int *Y) {{
  gm_config_ex();
  gm_config_ld(0,0,0);
  gm_config_ld({co},1,0);
  gm_config_ld(0,2,0);
  gm_config_st({co*4},1.0f,0);
  for(int ib=0;ib<{m};ib+=16) {{
    int mr=gm_min16({m}-ib);
    for(int jb=0;jb<{co};jb+=64) {{
      int nb=({co}-jb<64)?{co}-jb:64;
      /* Full-width i32 DMA is restricted to one 16-column block. */
      for(int jj=0;jj<nb;jj+=16) {{
        int nc=gm_min16(nb-jj);
        gm_mvin((const void*)0,0x80000000u+(gm_u32)jj,mr,nc,2);
      }}
      for(int ky=0;ky<{kh};++ky) for(int kx=0;kx<{kw};++kx) for(int kk=0;kk<{ci};kk+=16) {{
        int kr=gm_min16({ci}-kk);
        for(int row=0;row<mr;++row) {{
          int flat=ib+row;
          int bn=flat/{oh*ow}, oy=(flat/{ow})%{oh}, ox=flat%{ow};
          int iy=oy*{sh}-{pt}+ky*{dh}, ix=ox*{sw}-{pl}+kx*{dw};
          gm_u64 valid=(gm_u64)(((unsigned)iy<(unsigned){h}) & ((unsigned)ix<(unsigned){w}));
          gm_u64 offset=(((gm_u64)bn*{h}+(gm_u64)iy)*{w}+(gm_u64)ix)*{ci}+(unsigned)kk;
          gm_u64 address=((gm_u64)X+offset)*valid;
          gm_mvin((const void*)address,(gm_u32)row,1,kr,0);
        }}
        gm_u64 offset=((gm_u64)(ky*{kw}+kx)*{ci}+(unsigned)kk)*{co}+(unsigned)jb;
        gm_mvin((const void*)((gm_u64)W+offset),8192,kr,nb,1);
        for(int jj=0;jj<nb;jj+=16) {{
          int nr=gm_min16(nb-jj);
          gm_preload(8192+(gm_u32)jj,0xc0000000u+(gm_u32)jj,kr,nr,mr);
          gm_compute(0,mr,kr);
        }}
      }}
      for(int jj=0;jj<nb;jj+=16) {{
        int nr=gm_min16(nb-jj);
        gm_mvout(Y+(gm_u64)ib*{co}+(unsigned)(jb+jj),0xa0000000u+(gm_u32)jj,mr,nr);
      }}
    }}
  }}
  gm_fence();
}}
'''
    return [definition],[f'{tag}((const signed char*)({names[a]}),(const signed char*)({names[b]}),(int*)({names[y]}));']
