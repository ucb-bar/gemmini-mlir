"""Explicit RV64GC directed binary32 certificate arithmetic capability."""

def emit_binary32_radius_header(*, host_isa: str) -> str:
    if host_isa != 'rv64gc':
        raise ValueError('explicit RV64GC host capability required')
    return '''#ifndef MERLIN_RV_BINARY32_RADIUS_H
#define MERLIN_RV_BINARY32_RADIUS_H
static inline float radius_fma_up(float a,float b,float c){float r;__asm__("fmadd.s %0,%1,%2,%3,rup":"=f"(r):"f"(a),"f"(b),"f"(c));return r;}
static inline float radius_add_up(float a,float b){float r;__asm__("fadd.s %0,%1,%2,rup":"=f"(r):"f"(a),"f"(b));return r;}
static inline float radius_add_down(float a,float b){float r;__asm__("fadd.s %0,%1,%2,rdn":"=f"(r):"f"(a),"f"(b));return r;}
#define MERLIN_F32_RADIUS_FMA_UP(a,b,c) radius_fma_up((a),(b),(c))
#define MERLIN_F32_RADIUS_ADD_UP(a,b) radius_add_up((a),(b))
#define MERLIN_F32_RADIUS_ADD_DOWN(a,b) radius_add_down((a),(b))
#endif
'''
