/* Experimental generic min/max capability. Standard numeric result and
 * interposition/errno/nontrapping/unobserved-flags obligations are explicit.
 * Keep the actual source library operation for every zero and NaN operand,
 * preserving its returned representation. No source-zero/NaN waiver.
 * Compiler capability and actual target qualification are separate gates. */
#include <math.h>
#if !defined(__has_builtin)
#error "guarded min/max builtin capability unavailable"
#elif !__has_builtin(__builtin_fminf) || !__has_builtin(__builtin_fmaxf) || !__has_builtin(__builtin_fmin) || !__has_builtin(__builtin_fmax) || !__has_builtin(__builtin_isnan)
#error "guarded min/max builtin capability unavailable"
#endif
static inline float merlin_guarded_min32(float a, float b) {
  if(a==0 || b==0 || __builtin_isnan(a) || __builtin_isnan(b)) return fminf(a,b);
  return __builtin_fminf(a,b);
}
static inline float merlin_guarded_max32(float a, float b) {
  if(a==0 || b==0 || __builtin_isnan(a) || __builtin_isnan(b)) return fmaxf(a,b);
  return __builtin_fmaxf(a,b);
}
static inline double merlin_guarded_min64(double a, double b) {
  if(a==0 || b==0 || __builtin_isnan(a) || __builtin_isnan(b)) return fmin(a,b);
  return __builtin_fmin(a,b);
}
static inline double merlin_guarded_max64(double a, double b) {
  if(a==0 || b==0 || __builtin_isnan(a) || __builtin_isnan(b)) return fmax(a,b);
  return __builtin_fmax(a,b);
}
#define MERLIN_SOURCE_F32_MIN(a,b) merlin_guarded_min32((a),(b))
#define MERLIN_SOURCE_F32_MAX(a,b) merlin_guarded_max32((a),(b))
#define MERLIN_SOURCE_F64_MIN(a,b) merlin_guarded_min64((a),(b))
#define MERLIN_SOURCE_F64_MAX(a,b) merlin_guarded_max64((a),(b))
