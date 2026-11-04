"""Authored Gemmini accumulator load/accumulate/readout elementwise kernels.

Register encodings are from rtl_facts/facts.json register_bundle_layouts;
LoadController FIRRTL lines 229271-229275 select narrow input with shrink=1.
Scratchpad lines 188800ff carry each load's accumulate bit to AccumulatorMem.
"""


def validate_movement(in_bytes, out_bytes, scale=1.0, relu=False):
    """Reject semantics absent from full-width hardware readout before emission."""
    if in_bytes not in (1, 4) or out_bytes not in (1, 4):
        raise ValueError("Gemmini movement supports i8 and i32 containers")
    if out_bytes == 4 and (float(scale) != 1.0 or relu):
        raise ValueError("Gemmini full-width i32 readout bypasses scale and activation")


def validate_residual(lhs_scale, rhs_scale, out_bytes=1, relu=False):
    """The authored scaled-load sum supports finite, nonnegative coefficients."""
    import math
    if any(not math.isfinite(float(s)) or float(s) < 0 for s in (lhs_scale, rhs_scale)):
        raise ValueError("Gemmini residual lowering requires finite nonnegative scales")
    if out_bytes not in (1, 4):
        raise ValueError("Gemmini residual supports i8 and i32 outputs")
    if out_bytes == 4 and relu:
        raise ValueError("Gemmini full-width i32 residual readout bypasses activation")


def runtime_header(tile_cols=16):
    """Emit a selectable 16/64-column DMA schedule; default remains 16.

    A wide narrow-input command lays successive 16-column chunks at
    accumulator rows 0, 16, 32, 48. Full-width DMA loads and readouts use
    separate commands per block (DMA.scala:267-274; StoreController.scala:225).
    """
    if type(tile_cols) is not int or tile_cols not in (16, 64):
        raise ValueError("elementwise tile_cols must be 16 or 64")
    header = (r'''
typedef unsigned long ew_u64;
typedef unsigned int ew_u32;
#define EW_ROCC(F,A,B) __asm__ volatile (".insn r 0x7b, 3, " #F ", x0, %0, %1" :: "r"((ew_u64)(A)), "r"((ew_u64)(B)) : "memory")
static __attribute__((always_inline)) inline ew_u32 ew_fbits(float x) { union {float f; ew_u32 u;} z; z.f=x; return z.u; }
static __attribute__((always_inline)) inline ew_u64 ew_shape(ew_u32 addr, int rows, int cols) {
  return (ew_u64)addr | ((ew_u64)(ew_u32)cols<<32) | ((ew_u64)(ew_u32)rows<<48);
}
static __attribute__((always_inline)) inline void ew_fence(void) { __asm__ volatile ("fence" ::: "memory"); }
static __attribute__((always_inline)) inline void ew_ldcfg(int state, int byte_stride, int narrow, float scale) {
  ew_u64 a=1 | ((ew_u64)(narrow!=0)<<2) | ((ew_u64)state<<3) |
             ((ew_u64)16<<16) | ((ew_u64)ew_fbits(scale)<<32);
  EW_ROCC(0,a,byte_stride);
}
static __attribute__((always_inline)) inline void ew_stcfg(int byte_stride, float scale, int relu) {
  EW_ROCC(0,2 | ((ew_u64)(relu!=0)<<2),
            (ew_u64)(ew_u32)byte_stride | ((ew_u64)ew_fbits(scale)<<32));
}
static __attribute__((always_inline)) inline void ew_load0(const void *x, ew_u32 a, int r, int c) { EW_ROCC(2,x,ew_shape(a,r,c)); }
static __attribute__((always_inline)) inline void ew_load1(const void *x, ew_u32 a, int r, int c) { EW_ROCC(1,x,ew_shape(a,r,c)); }
static __attribute__((always_inline)) inline void ew_store(void *x, ew_u32 a, int r, int c) { EW_ROCC(3,x,ew_shape(a,r,c)); }
/* StoreController permits wide narrow readouts, but full_data readouts must
   select a single 16-column chunk. Local chunk spacing follows block_stride. */
static __attribute__((always_inline)) inline void ew_readout(void *x, int r, int c, int out_bytes) {
  if(out_bytes==4) {
    for(int block=0;block<c;block+=16) {
      int cols=c-block<16?c-block:16;
      ew_store((unsigned char*)x+(ew_u64)block*4,0xa0000000u+(ew_u32)block,r,cols);
    }
  } else {
    ew_store(x,0x80000000u,r,c);
  }
}


/* Movement uses the accumulator for both exact widening and scaled readout.
   Full-width readout deliberately uses the unmodified full_data path. */
static __attribute__((always_inline)) inline void gm_movement(const void *X, void *Y, int M, int N,
                        int in_bytes, int out_bytes, float scale, int relu) {
  ew_fence();
  ew_ldcfg(0,N*in_bytes,in_bytes==1,1.0f);
  ew_stcfg(N*out_bytes,scale,relu);
  for(int i=0;i<M;i+=16) {
    int mr=M-i<16?M-i:16;
    for(int j=0;j<N;j+=EW_TILE_COLS) {
      int nc=N-j<EW_TILE_COLS?N-j:EW_TILE_COLS;
      ew_load0((const unsigned char*)X+((ew_u64)i*N+j)*in_bytes,0x80000000u,mr,nc);
      ew_readout((unsigned char*)Y+((ew_u64)i*N+j)*out_bytes,mr,nc,out_bytes);
    }
  }
  ew_fence();
}

/* A scaled narrow load rounds and saturates to i8 before widening. Splitting
   a positive coefficient at one avoids premature saturation when it exceeds
   one. Every full unit contribution remains exact in the i32 accumulator. */
static __attribute__((always_inline)) inline void ew_add_scaled(const signed char *X, int stride, int rows, int cols,
                          float scale, int first) {
  int whole=(int)scale;
  float frac=scale-(float)whole;
  if(whole>0) {
    ew_ldcfg(0,stride,1,1.0f);
    for(int k=0;k<whole;k++) {
      ew_load0(X,first?0x80000000u:0xc0000000u,rows,cols);
      first=0;
    }
  }
  if(frac!=0.0f || first) {
    ew_ldcfg(0,stride,1,frac);
    ew_load0(X,first?0x80000000u:0xc0000000u,rows,cols);
  }
}
static __attribute__((always_inline)) inline void gm_residual(const signed char *X,const signed char *R,void *Y,
                        int M,int N,int out_bytes,float lhs_scale,
                        float rhs_scale,int relu) {
  ew_fence();
  ew_stcfg(N*out_bytes,1.0f,relu);
  int simple=lhs_scale>=0.0f && lhs_scale<=1.0f && rhs_scale>=0.0f && rhs_scale<=1.0f;
  if(simple) {
    ew_ldcfg(0,N,1,lhs_scale);
    ew_ldcfg(1,N,1,rhs_scale);
  }
  for(int i=0;i<M;i+=16) {
    int mr=M-i<16?M-i:16;
    for(int j=0;j<N;j+=EW_TILE_COLS) {
      int nc=N-j<EW_TILE_COLS?N-j:EW_TILE_COLS;
      if(simple) {
        ew_load0(X+(ew_u64)i*N+j,0x80000000u,mr,nc);
        ew_load1(R+(ew_u64)i*N+j,0xc0000000u,mr,nc);
      } else {
        ew_add_scaled(X+(ew_u64)i*N+j,N,mr,nc,lhs_scale,1);
        ew_add_scaled(R+(ew_u64)i*N+j,N,mr,nc,rhs_scale,0);
      }
      ew_readout((unsigned char*)Y+((ew_u64)i*N+j)*out_bytes,mr,nc,out_bytes);
    }
  }
  ew_fence();
}

/* Flattened i32 elementwise sum packs consecutive groups of 16 elements
   into DMA rows. Both operands use a 64-byte stride; no broadcasting occurs.
   Full-width stores stay at 16 columns, as required by StoreController. */
static __attribute__((always_inline)) inline void gm_add_i32_flat(const int *X,const int *B,int *Y,int count) {
  ew_fence();
  ew_ldcfg(0,64,0,1.0f);
  ew_ldcfg(1,64,0,1.0f);
  ew_stcfg(64,1.0f,0);
  int full_rows=count/16;
  for(int row=0;row<full_rows;row+=16) {
    int rows=full_rows-row<16?full_rows-row:16;
    ew_u64 off=(ew_u64)row*16;
    ew_load0(X+off,0x80000000u,rows,16);
    ew_load1(B+off,0xc0000000u,rows,16);
    ew_store(Y+off,0xa0000000u,rows,16);
  }
  int tail=count%16;
  if(tail) {
    ew_u64 off=(ew_u64)full_rows*16;
    ew_load0(X+off,0x80000000u,1,tail);
    ew_load1(B+off,0xc0000000u,1,tail);
    ew_store(Y+off,0xa0000000u,1,tail);
  }
  ew_fence();
}

/* A zero DMA row stride broadcasts the bias vector to every accumulator row;
   the following full-width load adds the source using accumulate-on-load. */
static __attribute__((always_inline)) inline void gm_bias_add(const int *X,const int *B,int *Y,int M,int N) {
  ew_fence();
  ew_ldcfg(0,0,0,1.0f);
  ew_ldcfg(1,N*4,0,1.0f);
  ew_stcfg(N*4,1.0f,0);
  for(int i=0;i<M;i+=16) {
    int mr=M-i<16?M-i:16;
    for(int j=0;j<N;j+=EW_TILE_COLS) {
      int nc=N-j<EW_TILE_COLS?N-j:EW_TILE_COLS;
      ew_load0(B+j,0x80000000u,mr,nc);
      ew_load1(X+(ew_u64)i*N+j,0xc0000000u,mr,nc);
      ew_readout(Y+(ew_u64)i*N+j,mr,nc,4);
    }
  }
  ew_fence();
}
'''.replace('EW_TILE_COLS', str(tile_cols)))
    if tile_cols == 64:
        # Keep the baseline 16-column header byte-for-byte unchanged. Full-i32
        # DMA fixes its local block displacement to zero, so the wide variant
        # must explicitly address each accumulator block on input as on output.
        header = header.replace(
            '      ew_load0((const unsigned char*)X+((ew_u64)i*N+j)*in_bytes,0x80000000u,mr,nc);',
            r'''      if(in_bytes==4) {
        for(int block=0;block<nc;block+=16) {
          int cols=nc-block<16?nc-block:16;
          ew_load0((const unsigned char*)X+((ew_u64)i*N+j+block)*4,
                   0x80000000u+(ew_u32)block,mr,cols);
        }
      } else {
        ew_load0((const unsigned char*)X+((ew_u64)i*N+j)*in_bytes,0x80000000u,mr,nc);
      }''')
        header = header.replace(
            r'''      ew_load0(B+j,0x80000000u,mr,nc);
      ew_load1(X+(ew_u64)i*N+j,0xc0000000u,mr,nc);''',
            r'''      for(int block=0;block<nc;block+=16) {
        int cols=nc-block<16?nc-block:16;
        ew_load0(B+j+block,0x80000000u+(ew_u32)block,mr,cols);
        ew_load1(X+(ew_u64)i*N+j+block,0xc0000000u+(ew_u32)block,mr,cols);
      }''')
    return header
