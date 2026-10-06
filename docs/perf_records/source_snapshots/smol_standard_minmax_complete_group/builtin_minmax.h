/* Experiment only: explicit standard min/max, unobserved interposition,
 * errno, exception flags and min/max signed-zero/NaN-payload distinctions,
 * nontrapping execution. Does not grant arithmetic reassociation or a change
 * in the source-consumer accuracy gate. Actual platform independently checked. */
#if !defined(__has_builtin)
#error "standard min/max builtin capability unavailable"
#elif !__has_builtin(__builtin_fminf) || !__has_builtin(__builtin_fmaxf) || !__has_builtin(__builtin_fmin) || !__has_builtin(__builtin_fmax)
#error "standard min/max builtin capability unavailable"
#endif
#define MERLIN_SOURCE_F32_MIN(a,b) __builtin_fminf((a),(b))
#define MERLIN_SOURCE_F32_MAX(a,b) __builtin_fmaxf((a),(b))
#define MERLIN_SOURCE_F64_MIN(a,b) __builtin_fmin((a),(b))
#define MERLIN_SOURCE_F64_MAX(a,b) __builtin_fmax((a),(b))
