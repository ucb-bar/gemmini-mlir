"""This target's answer to an OPEN model's device dispatches: the C behind each ``merlin_dispatch_g<i>``.

:mod:`merlin.perf.whole_model_open` cuts every device group out of a model's host code and replaces it
with a call to ``merlin_dispatch_g<i>`` -- an external function with MLIR's C interface (a pointer to
the result descriptor, then one descriptor per argument). This module writes those functions for THIS
target, because what they call is this target's software environment: the vendor library's call, the
package kernel's pointer-argument ABI, and the core fallback when the machine cannot read a result out.
The generic builder decides nothing here about which buffer is which; it hands over, per dispatch, the
argument shapes, which argument is the activation and which the weight, and the route.

Every route writes the same thing: an int32 accumulator result in a fresh buffer the host code then
reads. Which route a group took is printed on its ``GM_GROUP`` line (``kind`` is ``<op>.<route>``) so
the per-route split is read off the console rather than assumed.

THE LOCAL CHECK (``verify='local'``), after each dispatch and outside its bracket: the result is an
exact integer function of the two operands the dispatch actually received, and it is checked on the
core by PROJECTION -- ``C r == A (B r)`` over two pseudo-random integer vectors ``r`` (Freivalds' check),
exact in 64-bit integers, O(MK + KN + MN) instead of the O(MKN) of recomputing it. A wrong element
survives one round with probability at most 1/9 (the entries of ``r`` are drawn from nine values), so
two rounds leave it at most 1/81 per wrong row, and every row is checked. ``mismatches`` counts ROWS
whose projection differs, which is zero exactly when no round saw a difference.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

#: The routes a dispatch can take, in the builder's own vocabulary.
PACKAGE, VENDOR, HOST = "package", "vendor", "host"

#: The library's host (CPU) dataflow token, and why it cannot answer an accumulator-committing dispatch.
LIBRARY_CPU = "CPU"
LIBRARY_CPU_NO_FULL_WIDTH = "library_cpu_path_has_no_full_width_result"

#: How many output values the program prints as bit patterns (the whole tensor is checked by GM_OUTPUT).
OUT_DUMP_CAP = 4096

#: Projection rounds of the local check, and the half-width of each vector entry's range.
LOCAL_ROUNDS = 2
LOCAL_SPAN = 4

_C_TYPES = {"i8": "int8_t", "i32": "int32_t", "i64": "int64_t", "f32": "float", "i16": "int16_t"}

PRELUDE = r"""
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "include/gemmini.h"

/* MLIR's ranked memref descriptor for rank 2, which is what every argument of a matmul dispatch is. */
typedef struct { void *allocated; void *aligned; int64_t offset; int64_t sizes[2]; int64_t strides[2]; } md2_t;

void *malloc(size_t);
extern uint64_t md_dispatch_cycles[];
extern uint64_t md_dispatch_count[];

static inline uint64_t md_cycles(void) { uint64_t c; __asm__ volatile("rdcycle %0" : "=r"(c)); return c; }

/* The operand's elements as ONE row-major block: the descriptor's own memory when it already is one,
   else a packed copy (counted, so a route that pays for relayout says so). */
static uint64_t md_copies;
static const void *md_packed(const md2_t *d, size_t elem) {
    const char *base = (const char *)d->aligned + d->offset * (int64_t)elem;
    if (d->strides[1] == 1 && d->strides[0] == d->sizes[1]) return base;
    char *out = (char *)malloc((size_t)(d->sizes[0] * d->sizes[1]) * elem + 64);
    out = (char *)(((uintptr_t)out + 63) & ~(uintptr_t)63);
    for (int64_t i = 0; i < d->sizes[0]; i++)
        for (int64_t j = 0; j < d->sizes[1]; j++)
            memcpy(out + (size_t)(i * d->sizes[1] + j) * elem,
                   base + (size_t)(i * d->strides[0] + j * d->strides[1]) * elem, elem);
    md_copies++;
    return out;
}

/* THE KERNEL'S POINTEE LAYOUT (the contract's kernel_abi.pointee_layout): row-major with every row padded
   to a whole number of tile edges. A dense operand whose row is not is copied into that layout,
   zero-padded, and a result is read back out of it; a row that already is passes through untouched. */
static void *md_scratch(size_t bytes) {
    char *raw = (char *)malloc(bytes + 64);
    return (void *)(((uintptr_t)raw + 63) & ~(uintptr_t)63);
}
static const void *md_pitched(const void *dense, int64_t rows, int64_t cols, int64_t pitch, size_t elem) {
    char *out = (char *)md_scratch((size_t)(rows * pitch) * elem);
    memset(out, 0, (size_t)(rows * pitch) * elem);
    for (int64_t r = 0; r < rows; r++)
        memcpy(out + (size_t)(r * pitch) * elem, (const char *)dense + (size_t)(r * cols) * elem, (size_t)cols * elem);
    md_copies++;
    return out;
}
static void md_unpitch(void *dense, const void *pitched, int64_t rows, int64_t cols, int64_t pitch, size_t elem) {
    for (int64_t r = 0; r < rows; r++)
        memcpy((char *)dense + (size_t)(r * cols) * elem, (const char *)pitched + (size_t)(r * pitch) * elem,
               (size_t)cols * elem);
}

static void *md_result(md2_t *out, int64_t rows, int64_t cols, size_t elem) {
    char *raw = (char *)malloc((size_t)(rows * cols) * elem + 64);
    void *aligned = (void *)(((uintptr_t)raw + 63) & ~(uintptr_t)63);
    out->allocated = raw; out->aligned = aligned; out->offset = 0;
    out->sizes[0] = rows; out->sizes[1] = cols; out->strides[0] = cols; out->strides[1] = 1;
    return aligned;
}
"""

#: TWO HARTS: the host code runs on one hart (a vector hart), and only the unit's own hart issues the
#: unit's instructions, so every device dispatch is handed to that hart through this mailbox and waited
#: for. One request is in flight at a time -- the host code needs the result before it goes on -- so the
#: dispatch's own code (its operand packing, the bump allocator, the local check) never runs on two harts
#: at once. The unit's hart serves on a stack of its own: the startup code gives each hart but the first a
#: small one.
RPC_HELPERS = r"""
typedef void (*md_body_t)(void **);
static md_body_t volatile md_rpc_body;
static void **volatile md_rpc_argv;
static uint64_t md_rpc_posted, md_rpc_served;
static void md_rpc(md_body_t body, void **argv) {
    md_rpc_body = body;
    md_rpc_argv = argv;
    uint64_t ticket = md_rpc_posted + 1;
    __atomic_store_n(&md_rpc_posted, ticket, __ATOMIC_RELEASE);
    while (__atomic_load_n(&md_rpc_served, __ATOMIC_ACQUIRE) != ticket) { }
}
void merlin_unit_serve(void) {
    uint64_t seen = 0;
    for (;;) {
        uint64_t posted;
        while ((posted = __atomic_load_n(&md_rpc_posted, __ATOMIC_ACQUIRE)) == seen) { }
        md_rpc_body(md_rpc_argv);
        seen = posted;
        __atomic_store_n(&md_rpc_served, seen, __ATOMIC_RELEASE);
    }
}
static unsigned char md_unit_stack[RPC_STACK_BYTES] __attribute__((aligned(16)));
void merlin_unit_serve_on_own_stack(void) {
    __asm__ volatile("mv sp, %0\n\ttail merlin_unit_serve" : : "r"(md_unit_stack + sizeof md_unit_stack) : "memory");
    for (;;) { }
}
"""

#: The unit hart's own stack: a dispatch's body calls a package kernel and the local check.
RPC_STACK_BYTES = 1 << 18

LOCAL_HELPERS = r"""
/* Projection check of an int8 x int8 -> int32 product (see the module docstring). */
static uint64_t lc_state;
static int64_t lc_next(void) {
    lc_state = lc_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (int64_t)((lc_state >> 33) % (2 * LC_SPAN + 1)) - LC_SPAN;
}
static void lc_project(const int8_t *A, const int8_t *B, const int32_t *C, int64_t M, int64_t K, int64_t N,
                       uint64_t seed, long long *mismatches, long long *first) {
    int64_t *r = (int64_t *)malloc((size_t)(N + K + 2 * M) * sizeof(int64_t));
    int64_t *x = r + N, *y = x + K, *z = y + M;
    *mismatches = 0; *first = -1;
    char *bad = (char *)malloc((size_t)M + 1);
    memset(bad, 0, (size_t)M);
    for (int round = 0; round < LC_ROUNDS; round++) {
        lc_state = seed * 0x9E3779B97F4A7C15ULL + (uint64_t)round + 1;
        for (int64_t j = 0; j < N; j++) r[j] = lc_next();
        for (int64_t k = 0; k < K; k++) {
            int64_t s = 0; const int8_t *b = B + k * N;
            for (int64_t j = 0; j < N; j++) s += (int64_t)b[j] * r[j];
            x[k] = s;
        }
        for (int64_t i = 0; i < M; i++) {
            int64_t s = 0, t = 0; const int8_t *a = A + i * K; const int32_t *c = C + i * N;
            for (int64_t k = 0; k < K; k++) s += (int64_t)a[k] * x[k];
            for (int64_t j = 0; j < N; j++) t += (int64_t)c[j] * r[j];
            if (s != t && !bad[i]) { bad[i] = 1; (*mismatches)++; if (*first < 0) *first = i; }
        }
    }
}
"""


def c_type(dtype: str) -> str:
    if dtype not in _C_TYPES:
        raise ValueError(f"no C element type for {dtype!r}")
    return _C_TYPES[dtype]


def _library_call(m: int, n: int, k: int, path: str) -> str:
    """The vendor library's matmul of an int8 pair into a full-width int32 result, on ``path``."""
    return (
        f"tiled_matmul_auto({m}, {n}, {k}, (const elem_t *)A, (const elem_t *)B, NULL, (void *)C, "
        f"{k}, {n}, {n}, {n}, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, NO_ACTIVATION, "
        f"ACC_SCALE_IDENTITY, 0, false, false, false, true, false, 0, {path});"
    )


def render_dispatch(
    dispatches: Sequence[Mapping[str, Any]],
    routes: Mapping[int, Mapping[str, Any]],
    *,
    uart: Mapping[str, str],
    verify: str = "none",
    samples: Mapping[int, Mapping[str, Any]] | None = None,
    words_helper: str = "",
    unit_rpc: bool = False,
    row_padding: int | None = None,
) -> dict[str, Any]:
    """``{"source": C text, "census": [...], "objects": [...]}`` for every dispatch.

    ``dispatches`` are :class:`merlin.perf.whole_model_open.Dispatch` dicts; ``routes[group]`` is
    ``{"on": package|vendor|host, ...}``: for ``package`` the kernel's ``symbol``, ``object`` and
    ``args`` (each a role: ``lhs``/``rhs``/``dst``); for ``vendor`` the library ``path`` (a dataflow
    token the library defines). A dispatch this target has no call for is written as the host route
    and the census says why. ``samples[group]`` (``{"indices": [...], "values": [...], "bound": b}``)
    makes the local build compare the activation it was handed with the oracle's at those positions.
    ``row_padding`` is the tile edge the kernel ABI pads every pointee row to (the builder derives it,
    ``whole_model_build.pointee_row_padding``); a package kernel is handed its operands and result in
    that layout, and when it is unknown (``None``) a package route whose rows would need padding is
    refused to the host rather than handed dense rows. ``unit_rpc`` writes a two-hart program: each device route's body runs on the unit's hart (which
    serves ``merlin_unit_serve_on_own_stack``), called and timed from the host code's hart; a host route
    runs where the host code does.
    """
    samples = samples or {}
    local = verify == "local"
    out = [PRELUDE]
    if unit_rpc:
        out.append(f"#define RPC_STACK_BYTES {RPC_STACK_BYTES}\n{RPC_HELPERS}")
    if local:
        out.append(f"#define LC_ROUNDS {LOCAL_ROUNDS}\n#define LC_SPAN {LOCAL_SPAN}\n{LOCAL_HELPERS}")
    census: list[dict[str, Any]] = []
    objects: list[str] = []
    reports: list[str] = []
    for slot, dispatch in enumerate(dispatches):
        group = int(dispatch["group"])
        route = dict(routes.get(group) or {"on": HOST})
        arguments = list(dispatch["arguments"])
        result = dispatch["result"]
        lhs_at, rhs_at = (list(dispatch.get("root_operands") or ()) + [None, None])[:2]
        shapes = [list(a["shape"]) for a in arguments]
        matmul = (
            len(result["shape"]) == 2
            and result["dtype"] == "i32"
            and lhs_at is not None
            and rhs_at is not None
            and all(len(s) == 2 for s in (shapes[lhs_at], shapes[rhs_at]))
            and arguments[lhs_at]["dtype"] == "i8"
            and arguments[rhs_at]["dtype"] == "i8"
        )
        m, n = (int(e) for e in result["shape"]) if len(result["shape"]) == 2 else (0, 0)
        k = int(shapes[lhs_at][1]) if matmul else 0
        if route["on"] != HOST and not (matmul and route.get("zero_init")):
            why = (
                "the dispatch accumulates into a value that is not stated zero, which no device route computes"
                if matmul
                else "this target states no device call for this dispatch's form"
            )
            route = {"on": HOST, "cause": "no_device_call_for_dispatch", "why": why, "declined": route.get("on")}
        if route["on"] == VENDOR and str(route.get("path")) == LIBRARY_CPU and result["dtype"] == "i32":
            # The library's own CPU matmul does not produce a full-width result: `tiled_matmul_auto`
            # prints "Not implemented: CPU matmul, full_C=1" and carries on, writing narrow elements
            # into the int32 buffer (measured on FireSim, job 1106). A loop-free library path for an
            # accumulator-committing dispatch therefore does not exist; its own IR answers it.
            route = {
                "on": HOST,
                "cause": LIBRARY_CPU_NO_FULL_WIDTH,
                "why": "the library's CPU matmul implements no full-width (int32) result, and it is the only "
                "loop-free library path this machine has",
                "declined": VENDOR,
            }
        pitch = {"k": k, "n": n}
        if route["on"] == PACKAGE and matmul:
            if row_padding:
                pitch = {
                    axis: -(-extent // int(row_padding)) * int(row_padding) for axis, extent in (("k", k), ("n", n))
                }
            else:
                # The edge is UNKNOWN, so whether these rows need padding is too: never hand the kernel
                # dense rows it may read at another pitch.
                route = {
                    "on": HOST,
                    "cause": "kernel_row_padding_unknown",
                    "why": "the kernel ABI pads every pointee row to the tile edge, and no edge was derived",
                    "declined": PACKAGE,
                }
        params = ", ".join(["md2_t *out", *(f"md2_t *a{i}" for i in range(len(arguments)))])
        host_args = ", ".join([*(f"a{i}" for i in range(len(arguments))), "out"])
        host_params = ", ".join([*(f"md2_t *a{i}" for i in range(len(arguments))), "md2_t *out"])
        body: list[str] = [f"void _mlir_ciface_merlin_host_g{group}({host_params});"]
        remote = unit_rpc and route["on"] != HOST
        if remote:
            # The body, on the unit's hart; its arguments arrive as one pointer array.
            body.append(f"static void md_body_g{group}(void **md_v) {{")
            body.append("    md2_t *out = (md2_t *)md_v[0];")
            body.extend(f"    md2_t *a{i} = (md2_t *)md_v[{i + 1}];" for i in range(len(arguments)))
        else:
            body.append(f"void _mlir_ciface_merlin_dispatch_g{group}({params}) {{")
            body.append("    uint64_t t0 = md_cycles();")
        if route["on"] == HOST:
            # The group's own body is a PUBLIC function of the lowered module, so the lowering has turned
            # its result into a caller-allocated out-parameter, LAST (the builder checks the arity).
            if len(result["shape"]) != 2:
                raise ValueError(f"group {group}: a rank-{len(result['shape'])} result has no descriptor here")
            element = 4 if result["dtype"] in ("i32", "f32") else 8 if result["dtype"] in ("i64", "f64") else 1
            body.append(f"    md_result(out, {result['shape'][0]}, {result['shape'][1]}, {element});")
            body.append(f"    _mlir_ciface_merlin_host_g{group}({host_args});")
        else:
            body.append(f"    const int8_t *A = (const int8_t *)md_packed(a{lhs_at}, 1);")
            body.append(f"    const int8_t *B = (const int8_t *)md_packed(a{rhs_at}, 1);")
            body.append(f"    int32_t *C = (int32_t *)md_result(out, {m}, {n}, 4);")
            if route["on"] == VENDOR:
                body.append(f"    {_library_call(m, n, k, str(route['path']))}")
            else:
                roles = {"lhs": "(void *)PA", "rhs": "(void *)PB", "dst": "(void *)PC"}
                call_args = [roles[str(role)] for role in route["args"]]
                out.append(f"extern void {route['symbol']}({', '.join('void *' for _ in call_args)});")
                body.append(
                    "    const int8_t *PA = "
                    + (f"(const int8_t *)md_pitched(A, {m}, {k}, {pitch['k']}, 1);" if pitch["k"] != k else "A;")
                )
                body.append(
                    "    const int8_t *PB = "
                    + (f"(const int8_t *)md_pitched(B, {k}, {n}, {pitch['n']}, 1);" if pitch["n"] != n else "B;")
                )
                body.append(
                    "    int32_t *PC = "
                    + (f"(int32_t *)md_scratch((size_t){m * pitch['n'] * 4}ULL);" if pitch["n"] != n else "C;")
                )
                body.append(f"    {route['symbol']}({', '.join(call_args)});")
                if pitch["n"] != n:
                    body.append(f"    md_unpitch(C, PC, {m}, {n}, {pitch['n']}, 4);")
                objects.append(str(route["object"]))
        if not remote:
            body.append(f"    md_dispatch_cycles[{slot}] += md_cycles() - t0; md_dispatch_count[{slot}]++;")
        if words_helper:
            # The result's bytes stay where they are after the window (the arena is not reclaimed inside
            # an invocation), so their digest is taken at report time, outside every bracket.
            element = 4 if result["dtype"] in ("i32", "f32") else 8 if result["dtype"] in ("i64", "f64") else 1
            body.append(
                f"    md_words_at[{slot}] = (const char *)out->aligned + out->offset * {element};"
                f" md_words_bytes[{slot}] = (unsigned long long)(out->sizes[0] * out->sizes[1]) * {element};"
            )
        if local and matmul:
            body.append(
                f"    {{ const int8_t *LA = (const int8_t *)md_packed(a{lhs_at}, 1);"
                f" const int8_t *LB = (const int8_t *)md_packed(a{rhs_at}, 1);"
                f" const int32_t *LC = (const int32_t *)md_packed(out, 4);"
                f" lc_project(LA, LB, LC, {m}, {k}, {n}, {group}ULL, &md_local[{slot}][0], &md_local[{slot}][1]); }}"
            )
        sample = samples.get(group)
        if local and sample and lhs_at is not None:
            indices = ",".join(str(int(i)) for i in sample["indices"])
            values = ",".join(str(int(v)) for v in sample["values"])
            body.append(
                f"    {{ static const int64_t si[] = {{{indices}}}; static const int64_t sv[] = {{{values}}};"
                f" const int8_t *SA = (const int8_t *)md_packed(a{lhs_at}, 1); long long worst = 0, over = 0;"
                f" for (int s = 0; s < {len(sample['indices'])}; s++) {{ long long d = (long long)SA[si[s]] - sv[s];"
                f" if (d < 0) d = -d; if (d > worst) worst = d; if (d > {int(sample['bound'])}) over++; }}"
                f" md_hostin[{slot}][0] = worst; md_hostin[{slot}][1] = over; }}"
            )
        body.append("}")
        if remote:
            # The dispatch the host code calls: hand the body to the unit's hart, wait, and time the whole
            # hand-off from the host code's hart, so the split's host-code share excludes it.
            argv = ", ".join(["(void *)out", *(f"(void *)a{i}" for i in range(len(arguments)))])
            body.append(f"void _mlir_ciface_merlin_dispatch_g{group}({params}) {{")
            body.append("    uint64_t t0 = md_cycles();")
            body.append(f"    void *md_v[] = {{{argv}}};")
            body.append(f"    md_rpc(md_body_g{group}, md_v);")
            body.append(f"    md_dispatch_cycles[{slot}] += md_cycles() - t0; md_dispatch_count[{slot}]++;")
            body.append("}")
        out.append("\n".join(body))
        kind = f"{dispatch.get('op') or 'dispatch'}.{route['on']}"
        census.append({"group": group, "slot": slot, "kind": kind, **route})
        report = f'    printf("GM_GROUP {group} {kind} %llu sum=UNKNOWN fnv1a=UNKNOWN\\n", (unsigned long long)md_dispatch_cycles[{slot}]);'
        if words_helper:
            report += (
                f'\n    printf("{_format("words", uart, group=group, bytes="%llu", digest="%llu")}\\n",'
                f" md_words_bytes[{slot}], words_digest(md_words_at[{slot}], (size_t)md_words_bytes[{slot}]));"
            )
        if local and matmul:
            report += f'\n    printf("GM_LOCAL {group} mismatches=%lld of={m} first=%lld\\n", md_local[{slot}][0], md_local[{slot}][1]);'
        if local and sample:
            report += (
                f'\n    printf("GM_HOSTIN {group} max_abs=%lld over=%lld of={len(sample["indices"])} bound={int(sample["bound"])}\\n",'
                f" md_hostin[{slot}][0], md_hostin[{slot}][1]);"
            )
        reports.append(report)
    count = max(1, len(dispatches))
    header = [
        f"uint64_t md_dispatch_cycles[{count}];",
        f"uint64_t md_dispatch_count[{count}];",
        f"static long long md_local[{count}][2];",
        f"static long long md_hostin[{count}][2];",
        f"static const void *md_words_at[{count}];",
        f"static unsigned long long md_words_bytes[{count}];",
        words_helper,
    ]
    out.insert(1, "\n".join(header))
    # THE SPLIT, per route, read off the brackets. Host code is what the whole window holds outside
    # every dispatch bracket; it is printed on its own line rather than folded into any route.
    sums = []
    for route in (PACKAGE, VENDOR, HOST):
        slots = [row["slot"] for row in census if row["on"] == route]
        terms = " + ".join(f"md_dispatch_cycles[{s}]" for s in slots) or "0"
        sums.append(f"    unsigned long long by_{route} = {terms}; int n_{route} = {len(slots)};")
    split = _format(
        UART_SPLIT_KEY,
        uart,
        authored="%llu",
        authored_groups="%d",
        im2col="0",
        im2col_groups="0",
        vendor="%llu",
        vendor_groups="%d",
    )
    out.append(
        "void merlin_dispatch_report(unsigned long long whole) {\n"
        + "\n".join(reports)
        + "\n"
        + "\n".join(sums)
        + "\n    unsigned long long bracketed = by_package + by_vendor + by_host;"
        + f'\n    printf("{split}\\n", by_package, n_package, by_vendor + by_host, n_vendor + n_host);'
        + f'\n    printf("{_format("bracket_sum", uart, cycles="%llu")}\\n", bracketed);'
        + f'\n    printf("{_format("uncounted", uart, cycles="%llu")}\\n", whole - bracketed);'
        + '\n    printf("FM open split package=%llu groups=%d vendor=%llu groups=%d host_dispatch=%llu groups=%d'
        ' host_code=%llu\\n", by_package, n_package, by_vendor, n_vendor, by_host, n_host, whole - bracketed);'
        + '\n    printf("GM_COPIES %llu\\n", (unsigned long long)md_copies);\n}'
    )
    return {"source": "\n\n".join(out), "census": census, "objects": objects}


#: The protocol key the per-family split line is printed under.
UART_SPLIT_KEY = "split"

#: The kernel bench's own summary line (its per-group lines are the model program's).
KBENCH_LINE = "GM_KBENCH package=%llu groups=%d vendor=%llu groups=%d macs=%llu"

KBENCH_HELPERS = r"""
/* Operands of a stated seed: every byte from a 64-bit LCG, the full int8 range, so a spike run and a
   board run of the same image hand each kernel the same bytes. */
static void kb_fill(int8_t *p, uint64_t n, uint64_t seed) {
    uint64_t s = seed * 0x9E3779B97F4A7C15ULL + 1;
    uint64_t i = 0;
    for (; i + 8 <= n; i += 8) { s = s * 6364136223846793005ULL + 1442695040888963407ULL; memcpy(p + i, &s, 8); }
    for (; i < n; i++) { s = s * 6364136223846793005ULL + 1442695040888963407ULL; p[i] = (int8_t)(s >> 56); }
}
/* Write a buffer larger than the caches the core shares with the unit, so the operands a kernel reads
   are not the lines the fill just left behind: a model's weights arrive cold. */
static volatile uint64_t *kb_evict_at; static uint64_t kb_evict_words;
static void kb_evict(void) { for (uint64_t i = 0; i < kb_evict_words; i++) kb_evict_at[i] = i; }
"""


def render_kernel_bench(
    dispatches: Sequence[Mapping[str, Any]],
    routes: Mapping[int, Mapping[str, Any]],
    *,
    uart: Mapping[str, str],
    words_helper: str,
    evict_bytes: int = 0,
    check: bool = True,
    only: Sequence[int] | None = None,
    row_padding: int | None = None,
) -> dict[str, Any]:
    """A program that runs every device dispatch of an open model ONCE, at its model shape, on operands
    of a stated seed, and nothing else: ``{"source": C text with main, "census": [...], "objects": [...]}``.

    It is the model's device part without its host code -- minutes on a board where the whole model
    takes hours. Each kernel is bracketed exactly as in the model program (``GM_GROUP``), checked by
    the same exact projection (``GM_LOCAL``) against the operands it read, and its result digested
    (``GM_WORDS``) so a board run is compared with a functional-simulator run of the same image. An
    int8 x int8 -> int32 kernel's cycles do not depend on its operands' values, so the bracket is the
    kernel's cost at that regime; what the bench does NOT reproduce is the cache state the model's own
    host code leaves behind (the operands are evicted first: the conservative side). Dispatches with
    no device route are not in the bench (the census lists them with ``on: host``).

    ``check=False`` drops the projection: on an elaborated-RTL model it costs more cycles than the kernel,
    and a run is then shown correct by its ``GM_WORDS`` agreeing with a checked run of the same seeds.
    ``only`` benches those groups alone (one per regime, say); the others are listed, not run.
    ``row_padding`` is the kernel ABI's row pitch, as in :func:`render_dispatch`: a package kernel's
    operands and result are laid out at it outside the bracket, and with no edge derived a package
    route whose rows would need it is not benched."""
    out = [PRELUDE, f"#define LC_ROUNDS {LOCAL_ROUNDS}\n#define LC_SPAN {LOCAL_SPAN}\n{LOCAL_HELPERS}", KBENCH_HELPERS]
    out.append("int printf(const char *, ...);\nvoid console_init(void);\nvoid htif_exit(int);")
    if words_helper:
        out.append(words_helper)
    census: list[dict[str, Any]] = []
    objects: list[str] = []
    calls: list[str] = []
    largest = {"a": 1, "b": 1, "c": 1}
    for dispatch in dispatches:
        group = int(dispatch["group"])
        route = dict(routes.get(group) or {"on": HOST})
        arguments = list(dispatch["arguments"])
        result = dispatch["result"]
        lhs_at, rhs_at = (list(dispatch.get("root_operands") or ()) + [None, None])[:2]
        matmul = (
            len(result["shape"]) == 2
            and result["dtype"] == "i32"
            and lhs_at is not None
            and rhs_at is not None
            and len(arguments[lhs_at]["shape"]) == 2
            and arguments[lhs_at]["dtype"] == "i8"
            and arguments[rhs_at]["dtype"] == "i8"
        )
        if only is not None and group not in set(int(g) for g in only):
            census.append({"group": group, "on": route["on"], "in_bench": False, "cause": "not_selected"})
            continue
        if route["on"] == HOST or not matmul or not dispatch.get("zero_init"):
            census.append({"group": group, "on": HOST, "in_bench": False, "cause": route.get("cause")})
            continue
        if route["on"] == VENDOR and str(route.get("path")) == LIBRARY_CPU:
            census.append({"group": group, "on": HOST, "in_bench": False, "cause": LIBRARY_CPU_NO_FULL_WIDTH})
            continue
        m, n = (int(e) for e in result["shape"])
        k = int(arguments[lhs_at]["shape"][1])
        largest = {"a": max(largest["a"], m * k), "b": max(largest["b"], k * n), "c": max(largest["c"], m * n)}
        slot = len(calls)
        kind = f"{dispatch.get('op') or 'dispatch'}.{route['on']}"
        layout, unpitch = "", ""
        if route["on"] == VENDOR:
            call = _library_call(m, n, k, str(route["path"]))
        else:
            if not row_padding:
                census.append({"group": group, "on": HOST, "in_bench": False, "cause": "kernel_row_padding_unknown"})
                continue
            kp, np_ = (-(-extent // int(row_padding)) * int(row_padding) for extent in (k, n))
            roles = {"lhs": "(void *)PA", "rhs": "(void *)PB", "dst": "(void *)PC"}
            call_args = [roles[str(role)] for role in route["args"]]
            out.append(f"extern void {route['symbol']}({', '.join('void *' for _ in call_args)});")
            call = f"{route['symbol']}({', '.join(call_args)});"
            objects.append(str(route["object"]))
            layout = (
                f"const int8_t *PA = {f'(const int8_t *)md_pitched(A, {m}, {k}, {kp}, 1)' if kp != k else 'A'};"
                f" const int8_t *PB = {f'(const int8_t *)md_pitched(B, {k}, {n}, {np_}, 1)' if np_ != n else 'B'};"
                f" int32_t *PC = {f'(int32_t *)md_scratch((size_t){m * np_ * 4}ULL)' if np_ != n else 'C'};"
            )
            unpitch = f"md_unpitch(C, PC, {m}, {n}, {np_}, 4);" if np_ != n else ""
        words = (
            f'\n    printf("{_format("words", uart, group=group, bytes="%llu", digest="%llu")}\\n",'
            f" (unsigned long long){m * n * 4}ULL, words_digest(C, (size_t){m * n * 4}ULL));"
            if words_helper
            else ""
        )
        calls.append(
            f"""static void kb_g{group}(int8_t *A, int8_t *B, int32_t *C) {{
    long long bad = 0, first = -1;
    kb_fill(A, {m * k}ULL, {2 * group + 1}ULL); kb_fill(B, {k * n}ULL, {2 * group + 2}ULL);
    memset(C, 0xA5, (size_t){m * n * 4}ULL);
    {layout}
    kb_evict();
    uint64_t t0 = md_cycles();
    {call}
    uint64_t t = md_cycles() - t0;
    kb_cycles[{slot}] = t;
    {unpitch}
    {"lc_project(A, B, C, " + f"{m}, {k}, {n}, {group}ULL, &bad, &first);" if check else "(void)bad; (void)first;"}
    printf("GM_GROUP {group} {kind} %llu sum=UNKNOWN fnv1a=UNKNOWN\\n", (unsigned long long)t);{words}
    {f'printf("GM_LOCAL {group} mismatches=%lld of={m} first=%lld\\n", bad, first);' if check else ""}
}}"""
        )
        census.append({"group": group, "slot": slot, "kind": kind, "in_bench": True, "m": m, "k": k, "n": n, **route})
    benched = [row for row in census if row.get("in_bench")]
    out.append(f"static uint64_t kb_cycles[{max(1, len(benched))}];")
    out.extend(calls)
    sums = []
    for route in (PACKAGE, VENDOR):
        slots = [row["slot"] for row in benched if row["on"] == route]
        sums.append(
            f"    unsigned long long by_{route} = 0;"
            f" for (int i = 0; i < {len(slots)}; i++) by_{route} += kb_cycles[{route}_slots[i]];"
        )
        out.append(f"static const int {route}_slots[{max(1, len(slots))}] = {{{', '.join(map(str, slots)) or '0'}}};")
    macs = sum(row["m"] * row["k"] * row["n"] for row in benched)
    invocations = "\n".join(f"    kb_g{row['group']}(A, B, C);" for row in benched)
    out.append(
        f"""int main(int hart) {{
    if (hart != 0) for (;;);
    console_init();
    printf("MERLIN_KBENCH dispatches={len(benched)} evict_bytes={int(evict_bytes)}\\n");
    int8_t *A = (int8_t *)md_result(&(md2_t){{0}}, 1, {largest["a"]}, 1);
    int8_t *B = (int8_t *)md_result(&(md2_t){{0}}, 1, {largest["b"]}, 1);
    int32_t *C = (int32_t *)md_result(&(md2_t){{0}}, 1, {largest["c"]}, 4);
    kb_evict_words = {int(evict_bytes) // 8}ULL;
    kb_evict_at = (volatile uint64_t *)md_result(&(md2_t){{0}}, 1, {max(1, int(evict_bytes))}, 1);
{invocations}
{chr(10).join(sums)}
    printf("{KBENCH_LINE}\\n", by_package, {sum(1 for r in benched if r["on"] == PACKAGE)}, by_vendor,
           {sum(1 for r in benched if r["on"] == VENDOR)}, {macs}ULL);
    printf("DONE\\n");
    htif_exit(0);
    return 0;
}}"""
    )
    # The helpers index these arrays; declare them ahead of every body.
    out.insert(1, "uint64_t md_dispatch_cycles[1];\nuint64_t md_dispatch_count[1];")
    return {"source": "\n\n".join(out), "census": census, "objects": objects, "macs": macs}


def _format(key: str, uart: Mapping[str, str], **conversions: str) -> str:
    return uart[key].format(**conversions)


def render_main(
    uart: Mapping[str, str],
    census: Sequence[Mapping[str, Any]],
    *,
    label: str = "whole_model_open",
    profile: bool = False,
    reference: Sequence[float] | None = None,
    atol: float = 0.0,
    rtol: float = 0.0,
    words_helper: str = "",
    host_hart: int = 0,
    unit_hart: int | None = None,
) -> str:
    """The program's ``main``: one measured invocation of the model, bracketed, then every report.

    ONE INVOCATION, stated as such: an open model's host code allocates from a bump arena that is not
    reclaimed inside an invocation, and a program of this size spends hours per invocation on an
    emulator, so a warm-up pass would double the cost of the only number the run exists to take. The
    output leaves as raw 32-bit patterns (``OUT <n> <bits...>``), exact, for the grader to decode.

    ``reference`` (the oracle's output) with ``atol``/``rtol`` (the capsule's numeric policy) makes the
    program state its own end result after the window: ``GM_OUTPUT within=<n> of=<m>``, the elements
    with ``|out - ref| <= atol + rtol * |ref|``. A few kilobytes of constants, never inside the window.

    ``host_hart`` runs the program (every other hart parks); with ``unit_hart`` (a two-hart program, see
    ``render_dispatch(unit_rpc=True)``) that hart serves the dispatches instead of parking."""
    check = ""
    if reference is not None:
        values = ",".join(f"{float(v)!r}f" for v in reference)
        output_line = _format("output", uart, within="%d", of="%d")
        check = f"""
    {{
        static const float REF[{len(reference)}] = {{{values}}};
        int within = 0, n = {len(reference)} < MERLIN_OUT_ELEMS ? {len(reference)} : MERLIN_OUT_ELEMS;
        for (int i = 0; i < n; i++) {{
            float d = OUT[i] - REF[i], r = REF[i];
            if (d < 0) d = -d;
            if (r < 0) r = -r;
            if (d <= {float(atol)!r}f + {float(rtol)!r}f * r) within++;
        }}
        printf("{output_line}\\n", within, (int){len(reference)});
    }}"""
    if words_helper:
        # THE WHOLE OUTPUT'S BYTES, digested: a reference arm (the model's own host code with exact
        # devices) is compared on this, bit for bit, whatever the tensor's size.
        digest_line = _format("output_digest", uart, bytes="%llu", digest="%llu")
        check += f"""
    printf("{digest_line}\\n", (unsigned long long)MERLIN_OUT_ELEMS * 4ULL,
           words_digest(OUT, (size_t)MERLIN_OUT_ELEMS * 4));"""
    serve = ""
    if unit_hart is not None:
        if int(unit_hart) == int(host_hart):
            raise ValueError("a two-hart program's unit hart is not the hart its host code runs on")
        serve = f"    if (hart == {int(unit_hart)}) merlin_unit_serve_on_own_stack();\n"
    return f"""#include <stdint.h>
#include <string.h>
#include "merlin_model.h"
#include "model_gen.h"
#include "model_io.h"

void console_init(void);
int printf(const char *, ...);
void htif_exit(int);
void merlin_dispatch_report(unsigned long long whole);
void merlin_unit_serve_on_own_stack(void);
{"void merlin_prof_dump(void);" if profile else ""}
{words_helper}

#ifndef MERLIN_WEIGHTS_BASE_ADDR
#error "the build states where the weights are"
#endif
#define OUT ((float *)MERLIN_OUTPUT_PTR[0])
static merlin_descriptor_t DESCS[MERLIN_N_ARGS];

static inline uint64_t mm_cycles(void) {{ uint64_t c; __asm__ volatile("rdcycle %0" : "=r"(c)); return c; }}

int main(int hart) {{
{serve}    if (hart != {int(host_hart)}) for (;;);
    console_init();
    printf("MERLIN_INVOCATIONS warmup=0 measured=1\\n");
    printf("{_format("window_begin", uart, label=label)}\\n");
    printf("{_format("measured_begin", uart)}\\n");
    merlin_reset_session();
    merlin_prepare_step(0);
    uint64_t c0 = mm_cycles();
    merlin_run_multi(MERLIN_ARGS, MERLIN_N_ARGS, (const void *)MERLIN_WEIGHTS_BASE_ADDR,
                     MERLIN_INPUT_PTR, MERLIN_OUTPUT_PTR, DESCS);
    uint64_t c1 = mm_cycles();
    printf("{_format("measured_end", uart)}\\n");
    printf("{_format("full_model", uart, cycles="%llu")}\\n", (unsigned long long)(c1 - c0));
    printf("{_format("metric", uart, cycles="%llu")}\\n", (unsigned long long)(c1 - c0));
    merlin_dispatch_report((unsigned long long)(c1 - c0));
{check}
    /* The output's first {OUT_DUMP_CAP} values, as exact bit patterns. A board's console prints ~100 B/s, so
       a large output is represented by GM_OUTPUT (the whole tensor, checked above) and a bounded prefix. */
    int dumped = MERLIN_OUT_ELEMS < {OUT_DUMP_CAP} ? MERLIN_OUT_ELEMS : {OUT_DUMP_CAP};
    printf("OUT %d", dumped);
    for (int i = 0; i < dumped; i++) {{ uint32_t bits; memcpy(&bits, &OUT[i], 4); printf(" %u", bits); }}
    printf("\\n");
    {"merlin_prof_dump();" if profile else ""}
    printf("{_format("window_end", uart, label=label)}\\n");
    printf("DONE\\n");
    htif_exit(0);
    return 0;
}}
"""
