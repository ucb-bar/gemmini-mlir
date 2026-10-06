#include <stdint.h>
#include "merlin_model.h"
#include "htif.h"
#define CAP 4096
static uint64_t gaps[CAP], times[CAP], ids[CAP];
static uint64_t count, previous, total, device, tail;
static int active, overflow;
static inline uint64_t clock_now(void) {
 uint64_t v; __asm__ volatile("csrr %0, mcycle" : "=r"(v) :: "memory"); return v;
}
static void finish(uint64_t id, uint64_t begin, uint64_t end) {
 if (!active) return;
 if (count < CAP) { ids[count]=id; gaps[count]=begin-previous; times[count]=end-begin; }
 else overflow=1;
 ++count; device+=end-begin; previous=end;
}
extern void __real_merlin_run_multi(const merlin_arg_t*,int,const void*,void*const*,void*const*,merlin_descriptor_t*);
void __wrap_merlin_run_multi(const merlin_arg_t *a,int n,const void *w,void*const*i,void*const*o,merlin_descriptor_t*d) {
 count=0;device=0;overflow=0;active=1;uint64_t begin=clock_now();previous=begin;
 __real_merlin_run_multi(a,n,w,i,o,d);
 uint64_t end=clock_now();active=0;total=end-begin;tail=end-previous;
}
extern void __real_htif_exit(int) __attribute__((noreturn));
void __wrap_htif_exit(int code) {
 htif_line_flush(0);
 htif_puts("PROFILE_SUM ");htif_putd(total);htif_putc(' ');htif_putd(device);htif_putc(' ');
 htif_putd(total-device);htif_putc(' ');htif_putd(count);htif_putc(' ');htif_putd(overflow);htif_putc('\n');
 for(uint64_t j=0;j<count && j<CAP;j++) {
  htif_puts("PROFILE_CALL ");htif_putd(j);htif_putc(' ');htif_putd(ids[j]);htif_putc(' ');
  htif_putd(gaps[j]);htif_putc(' ');htif_putd(times[j]);htif_putc('\n');
 }
 htif_puts("PROFILE_TAIL ");htif_putd(tail);htif_putc('\n');htif_line_flush(1);
 __real_htif_exit(code);
}
extern void __real_gemmini_golden_29c4e0a80e5abd97(void *a0, void *a1, void *a2);
void __wrap_gemmini_golden_29c4e0a80e5abd97(void *a0, void *a1, void *a2) {
 uint64_t begin=clock_now(); __real_gemmini_golden_29c4e0a80e5abd97(a0, a1, a2);
 uint64_t end=clock_now(); finish(0,begin,end);
}
extern void __real_gemmini_golden_7b4380121ce5c4f3(void *a0, void *a1, void *a2);
void __wrap_gemmini_golden_7b4380121ce5c4f3(void *a0, void *a1, void *a2) {
 uint64_t begin=clock_now(); __real_gemmini_golden_7b4380121ce5c4f3(a0, a1, a2);
 uint64_t end=clock_now(); finish(1,begin,end);
}
extern void __real_gemmini_golden_a5705ab56e324ba1(void *a0, void *a1, void *a2);
void __wrap_gemmini_golden_a5705ab56e324ba1(void *a0, void *a1, void *a2) {
 uint64_t begin=clock_now(); __real_gemmini_golden_a5705ab56e324ba1(a0, a1, a2);
 uint64_t end=clock_now(); finish(2,begin,end);
}
extern void __real_gemmini_golden_e738f13ad92e3128(void *a0, void *a1, void *a2);
void __wrap_gemmini_golden_e738f13ad92e3128(void *a0, void *a1, void *a2) {
 uint64_t begin=clock_now(); __real_gemmini_golden_e738f13ad92e3128(a0, a1, a2);
 uint64_t end=clock_now(); finish(3,begin,end);
}
extern void __real_gemmini_golden_ef297329ec9dee28(void *a0, void *a1, void *a2);
void __wrap_gemmini_golden_ef297329ec9dee28(void *a0, void *a1, void *a2) {
 uint64_t begin=clock_now(); __real_gemmini_golden_ef297329ec9dee28(a0, a1, a2);
 uint64_t end=clock_now(); finish(4,begin,end);
}
