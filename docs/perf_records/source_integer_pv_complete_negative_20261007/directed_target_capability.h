#include <stdint.h>
static inline int8_t observer_rne(float x){float lo=-128.0f,hi=127.0f;long n;
 __asm__ volatile("fmax.s %0,%0,%2\n\tfmin.s %0,%0,%3\n\tfcvt.w.s %1,%0,rne":"+&f"(x),"=r"(n):"f"(lo),"f"(hi));return(int8_t)n;}
#define MERLIN_CLOSED_RNE_I8 observer_rne
#define MERLIN_SOURCE_BITCAST_COPY __builtin_memcpy
#ifndef PV_BOUNDS_OUTWARD_F64_H
#define PV_BOUNDS_OUTWARD_F64_H
#ifdef MERLIN_ORDERED_FMA_BOUNDS_H
#error "outward capability must precede every ordered-FMA header include"
#endif
#if defined(MERLIN_F64_OUTWARD_ADD_UP) || defined(MERLIN_F64_OUTWARD_ADD_DOWN) || defined(MERLIN_F64_OUTWARD_MUL_UP)
#error "outward scalar capability already selected"
#endif
/* Explicit IEEE binary64 directed arithmetic; source rounding is unchanged.
 * Nontrapping arithmetic and unobserved exception flags are required.
 * Include before ordered_fma_bounds.h; no accelerator instruction. */
static inline double pv_bounds_add_up(double a, double b) {
  double result;
  __asm__("fadd.d %0, %1, %2, rup"
          : "=f"(result) : "f"(a), "f"(b));
  return result;
}
static inline double pv_bounds_add_down(double a, double b) {
  double result;
  __asm__("fadd.d %0, %1, %2, rdn"
          : "=f"(result) : "f"(a), "f"(b));
  return result;
}
static inline double pv_bounds_mul_up(double a, double b) {
  double result;
  __asm__("fmul.d %0, %1, %2, rup"
          : "=f"(result) : "f"(a), "f"(b));
  return result;
}
#define MERLIN_F64_OUTWARD_ADD_UP(a,b) pv_bounds_add_up((a),(b))
#define MERLIN_F64_OUTWARD_ADD_DOWN(a,b) pv_bounds_add_down((a),(b))
#define MERLIN_F64_OUTWARD_MUL_UP(a,b) pv_bounds_mul_up((a),(b))
#if defined(MERLIN_F32_OUTWARD_FROM_F64_DOWN) || defined(MERLIN_F32_OUTWARD_FROM_F64_UP)
#error "outward narrowing capability already selected"
#endif
static inline float pv_bounds_cast_down(double value) {
  float result;
  __asm__("fcvt.s.d %0, %1, rdn" : "=f"(result) : "f"(value));
  return result;
}
static inline float pv_bounds_cast_up(double value) {
  float result;
  __asm__("fcvt.s.d %0, %1, rup" : "=f"(result) : "f"(value));
  return result;
}
#define MERLIN_F32_OUTWARD_FROM_F64_DOWN(value) pv_bounds_cast_down((value))
#define MERLIN_F32_OUTWARD_FROM_F64_UP(value) pv_bounds_cast_up((value))
#endif
