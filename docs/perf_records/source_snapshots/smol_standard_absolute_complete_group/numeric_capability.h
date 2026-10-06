#ifndef MERLIN_ADMITTED_SOURCE_NUMERIC_CAPABILITY_H
#define MERLIN_ADMITTED_SOURCE_NUMERIC_CAPABILITY_H
#ifdef MERLIN_SOURCE_F32_MATH_H
#error "source numeric capability must precede numeric headers"
#endif
#if !defined(__clang__) && !defined(__GNUC__)
#error "source numeric builtin capability unavailable"
#endif
#if !defined(__has_builtin)
#error "elementwise FMA compiler capability unavailable"
#elif !__has_builtin(__builtin_elementwise_fma)
#error "elementwise FMA compiler capability unavailable"
#endif
#define MERLIN_SOURCE_F32_FMA(a,b,c) __builtin_elementwise_fma((a),(b),(c))
#define MERLIN_SOURCE_BITCAST_COPY(d,s,n) __builtin_memcpy((d),(s),(n))
#if !defined(__has_builtin)
#error "finite classification compiler capability unavailable"
#elif !__has_builtin(__builtin_isfinite)
#error "finite classification compiler capability unavailable"
#endif
#define MERLIN_SOURCE_ISFINITE(x) __builtin_isfinite((x))
#if !defined(__has_builtin)
#error "absolute-value compiler capability unavailable"
#elif !__has_builtin(__builtin_fabsf) || !__has_builtin(__builtin_fabs)
#error "absolute-value compiler capability unavailable"
#endif
#define MERLIN_SOURCE_F32_ABS(x) __builtin_fabsf((x))
#define MERLIN_SOURCE_F64_ABS(x) __builtin_fabs((x))
#endif

#include "f32_floor_bits.h"
#define MERLIN_MONOTONE_F32_FLOOR(x) merlin_f32_floor_bits(x)
