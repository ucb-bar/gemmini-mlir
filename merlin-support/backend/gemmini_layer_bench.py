"""Per-layer bench programs that call this target's own C library kernels.

These give the same-design, same-shape library reference the performance loop compares a schedule
against (``tiled_conv_auto`` / ``tiled_matmul_auto`` from the curated ``gemmini.h``), measured on the
same engine and harness as a candidate. The program:

1. embeds its operands: a blob packed by ``merlin.perf.layer_bench.reference.pack_operands`` is pulled
   into ``.data`` with ``.incbin`` (a relative path; the build compiles inside the work directory), so
   the program generates nothing -- load it through the engine's backdoor and it costs no cycles;
2. writes its output PAST the image (``_end`` plus a guard above the crt's per-hart stack);
3. runs one untimed warm call, then the timed call (``warm_then_measured``) or only the timed call
   (``cold_single``);
4. prints ``LB_RECORD <label> cycles=<n> digest=<fnv1a64 over output words, top bit cleared>``.

Weight layout is ``[KH][KW][CI][CO]`` row-major (what ``tiled_conv_auto`` indexes with every transpose
flag false; the stock ResNet-50 driver declares e.g. ``conv_3_w[576][64]`` for a 3x3x64->64 conv).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from merlin.perf.layer_bench import reference as _ref

#: v3: operands embedded by .incbin (v2 filled them on the device from an LCG, which cost tens of
#: millions of scalar cycles on the large layers), and the digest runs over 64-bit words.
HARNESS_VERSION = "gemmini_library_layer_v3"
#: File name the operand blob must have inside the build directory.
OPERAND_BLOB = "operands.bin"
#: Distance from ``_end`` to the output buffer. The curated crt puts each hart's stack right after
#: ``_end`` (1 << STKSHIFT bytes, STKSHIFT = 17); 1 MiB clears it for up to 8 harts.
_OUTPUT_GUARD_BYTES = 1 << 20

#: What an output buffer holds before a window runs. Every byte is 0xA5, which the contract's int8
#: output CAN take (-91), so this is not a value the oracle is unable to produce -- what makes it work
#: is that a whole unwritten REGION of 0xA5 is astronomically unlikely to be the region the oracle
#: expects, while leaving the previous window's correct output in place would pass by construction.
POISON_WORD = 0xA5A5A5A5A5A5A5A5

_PRELUDE = (
    r"""
#include <stdint.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"

extern char _end[];
__asm__(".section .data.lb_operands,\"aw\",@progbits\n"
        ".balign @ALIGN@\n"
        ".globl lb_operands\n"
        "lb_operands:\n"
        ".incbin \"@BLOB@\"\n"
        ".previous\n");
extern uint8_t lb_operands[];

static uint64_t lb_fnv1a_words(const uint8_t *p, size_t n) {
    uint64_t h = @FNV_OFFSET@ULL;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) { uint64_t w; __builtin_memcpy(&w, p + i, 8); h ^= w; h *= @FNV_PRIME@ULL; }
    if (i < n) { uint64_t w = 0; __builtin_memcpy(&w, p + i, n - i); h ^= w; h *= @FNV_PRIME@ULL; }
    return h;
}
static inline uintptr_t lb_align(uintptr_t x, uintptr_t a) { return (x + a - 1) & ~(a - 1); }
static void lb_poison(void *p, size_t n) {
    uint8_t *q = (uint8_t *)p;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) { uint64_t w = @POISON@ULL; __builtin_memcpy(q + i, &w, 8); }
    for (; i < n; i++) { q[i] = (uint8_t)@POISON_BYTE@; }
}
""".replace("@ALIGN@", str(_ref.OPERAND_ALIGN))
    .replace("@BLOB@", OPERAND_BLOB)
    .replace("@FNV_OFFSET@", str(_ref.FNV_OFFSET))
    .replace("@FNV_PRIME@", str(_ref.FNV_PRIME))
    .replace("@POISON@", str(POISON_WORD))
    .replace("@POISON_BYTE@", str(POISON_WORD & 0xFF))
)


def _c_float(value: float) -> str:
    """The float32 nearest ``value``, as an exact C99 hex literal (no decimal double rounding)."""
    return float(np.float32(value)).hex() + "f"


def _output_ptr(n_out: str) -> str:
    return f"    elem_t *output = (elem_t *)lb_align((uintptr_t)_end + {_OUTPUT_GUARD_BYTES}, 64);\n"


def _conv(s: Mapping[str, Any], off: Mapping[str, int]) -> tuple[str, str, str]:
    b, n, ci, co = int(s["batch"]), int(s["in_dim"]), int(s["in_channels"]), int(s["out_channels"])
    k, st, pad = int(s["kernel"]), int(s["stride"]), int(s["padding"])
    out = (n + 2 * pad - k) // st + 1
    if out <= 0:
        raise ValueError("convolution produces no output")
    body = (
        f"    const size_t n_out = (size_t){b} * {out} * {out} * {co};\n"
        f"    elem_t *input = (elem_t *)(lb_operands + {int(off['input'])});\n"
        f"    elem_t *weights = (elem_t *)(lb_operands + {int(off['weights'])});\n"
        f"    acc_t *bias = (acc_t *)(lb_operands + {int(off['bias'])});\n" + _output_ptr("n_out")
    )
    call = (
        f"tiled_conv_auto({b}, {n}, {n}, {ci}, {co}, {out}, {out}, {st}, 1, 1, {pad}, {k},\n"
        "        false, false, false, false, false,\n"
        "        input, weights, bias, output,\n"
        f"        {'RELU' if s.get('relu') else 'NO_ACTIVATION'}, {_c_float(s['scale'])}, 0, 0, 0, WS);"
    )
    return body, "n_out", call


def _matmul(s: Mapping[str, Any], off: Mapping[str, int]) -> tuple[str, str, str]:
    i, j, kk = int(s["m"]), int(s["n"]), int(s["k"])
    body = (
        f"    const size_t n_out = (size_t){i} * {j};\n"
        f"    elem_t *a = (elem_t *)(lb_operands + {int(off['a'])});\n"
        f"    elem_t *b = (elem_t *)(lb_operands + {int(off['b'])});\n"
        f"    acc_t *d = (acc_t *)(lb_operands + {int(off['d'])});\n" + _output_ptr("n_out")
    )
    call = (
        f"tiled_matmul_auto({i}, {j}, {kk}, a, b, d, output, {kk}, {j}, {j}, {j},\n"
        "        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,\n"
        f"        {'RELU' if s.get('relu') else 'NO_ACTIVATION'}, {_c_float(s['scale'])}, 0, true,\n"
        "        false, false, false, false, 0, WS);"
    )
    return body, "n_out", call


def _conv_as_matmul(s: Mapping[str, Any], off: Mapping[str, int]) -> tuple[str, str, str]:
    """A 1x1, stride-1, unpadded conv issued as the NHWC matmul the library's own ResNet-50 issues for
    it (``tiled_matmul_nn_auto`` -> ``tiled_matmul_auto``): A = input ``[B*H*W][CI]``, B = weights
    ``[CI][CO]``, D = the bias row repeated. Same operands and same output bytes as ``_conv``, so the
    same expected digest checks it."""
    b, n, ci, co = int(s["batch"]), int(s["in_dim"]), int(s["in_channels"]), int(s["out_channels"])
    if int(s["kernel"]) != 1 or int(s["stride"]) != 1 or int(s["padding"]) != 0:
        raise ValueError("the matmul route computes the conv only for a 1x1, stride-1, unpadded conv")
    rows = b * n * n
    body = (
        f"    const size_t n_out = (size_t){rows} * {co};\n"
        f"    elem_t *input = (elem_t *)(lb_operands + {int(off['input'])});\n"
        f"    elem_t *weights = (elem_t *)(lb_operands + {int(off['weights'])});\n"
        f"    acc_t *bias = (acc_t *)(lb_operands + {int(off['bias'])});\n" + _output_ptr("n_out")
    )
    call = (
        f"tiled_matmul_auto({rows}, {co}, {ci}, input, weights, bias, output, {ci}, {co}, {co}, {co},\n"
        "        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,\n"
        f"        {'RELU' if s.get('relu') else 'NO_ACTIVATION'}, {_c_float(s['scale'])}, 0, true,\n"
        "        false, false, false, false, 0, WS);"
    )
    return body, "n_out", call


#: How a ``conv2d`` spec may be issued. ``conv`` is ``tiled_conv_auto``; ``matmul`` is the library
#: ResNet-50's own route for 1x1 stride-1 convs.
CONV_ROUTES = ("conv", "matmul")

_ELEM_BYTES = {"i8": 1, "i16": 2, "i32": 4, "f32": 4}


def render_package_layer(
    cb: Mapping[str, Any],
    *,
    offsets: Mapping[str, int],
    label: str,
    output: str,
    entry_symbol: str,
    protocol: str = "warm_then_measured",
) -> str:
    """C source measuring one layer compiled by an mlir_oot PACKAGE (e.g. a phase-1 submission).

    The package's kernel object is linked beside this program. Every ``read`` argument of the command
    buffer's ``kernel_abi`` must have an offset in the embedded operand blob (``offsets``); every
    ``write`` argument (the output and any intermediates the package declares, e.g. a host im2col
    buffer) is placed past ``_end``. The entry is called in ABI order: an untimed warm call, then the
    timed call (``warm_then_measured``), and ``output``'s bytes are digested exactly as the library
    programs digest theirs, so both kinds of program are checked against the same reference.
    """
    if not label or any(ch.isspace() for ch in label):
        raise ValueError("label must be one non-empty token")
    if protocol not in ("warm_then_measured", "cold_single"):
        raise ValueError(f"unknown protocol {protocol!r}")
    if not entry_symbol.isidentifier():
        raise ValueError(f"entry symbol {entry_symbol!r} is not a C identifier")
    tensors = cb["tensors"]
    args = (cb.get("kernel_abi") or {}).get("args") or []
    if not args:
        raise ValueError("command buffer declares no kernel ABI")
    decl, ptrs, sizes = [], [], {}
    cursor = f"lb_align((uintptr_t)_end + {_OUTPUT_GUARD_BYTES}, 64)"
    body = ["    uintptr_t lb_heap = " + cursor + ";\n"]
    for i, arg in enumerate(args):
        name = arg["tensor"]
        t = tensors[name]
        n = 1
        for d in t["shape"]:
            n *= int(d)
        nbytes = n * _ELEM_BYTES[t["dtype"]]
        sizes[name] = nbytes
        var = f"lb_arg{i}"
        if arg.get("access") == "read":
            if name not in offsets:
                raise ValueError(f"read argument {name!r} has no operand in the blob")
            body.append(f"    void *{var} = (void *)(lb_operands + {int(offsets[name])});\n")
        else:
            body.append(f"    void *{var} = (void *)lb_heap; lb_heap = lb_align(lb_heap + {nbytes}, 64);\n")
        decl.append("void *")
        ptrs.append(var)
        if name == output:
            body.append(f"    const uint8_t *lb_out = (const uint8_t *){var};\n")
    if output not in sizes:
        raise ValueError(f"output {output!r} is not a kernel argument")
    call = f"    {entry_symbol}({', '.join(ptrs)});\n    gemmini_fence();\n"
    warm = call if protocol == "warm_then_measured" else ""
    return (
        _PRELUDE
        + f"\nextern void {entry_symbol}({', '.join(decl)});\n\nint main(void) {{\n"
        + "".join(body)
        + warm
        + "    uint64_t t0 = read_cycles();\n"
        + call
        + "    uint64_t t1 = read_cycles();\n"
        f"    uint64_t dg = lb_fnv1a_words(lb_out, {sizes[output]}) & {_ref.DIGEST_MASK}ULL;\n"
        f'    printf("LB_RECORD {label} cycles=%llu digest=%llu\\n", (unsigned long long)(t1 - t0), (unsigned long long)dg);\n'
        # A FIXED final line. The record above carries a digest and a cycle count, so no two runs share
        # it and it cannot serve as the exact success marker a sealed FPGA receipt validates against.
        # Reaching here means the kernel ran and its output was digested; whether the digest is RIGHT is
        # decided off-device against the contract, which is the only place that can know.
        f'    printf("LB_DONE {label}\\n");\n'
        "    return 0;\n}\n"
    )


def render_library_layer(spec: Mapping[str, Any], *, offsets: Mapping[str, int]) -> str:
    """C source for one library-kernel layer measurement.

    ``spec['op']`` is ``conv2d`` or ``matmul``; ``offsets`` are the operand offsets inside the blob that
    ``reference.pack_operands`` produced for this spec, which the build directory must hold as
    ``OPERAND_BLOB``.
    """
    label = str(spec["label"])
    if not label or any(ch.isspace() for ch in label):
        raise ValueError("label must be one non-empty token")
    protocol = spec.get("protocol", "warm_then_measured")
    if protocol not in ("warm_then_measured", "cold_single"):
        raise ValueError(f"unknown protocol {protocol!r}")
    op = spec["op"]
    route = spec.get("route", "conv")
    if op == "conv2d" and route not in CONV_ROUTES:
        raise ValueError(f"unknown conv route {route!r} (known: {CONV_ROUTES})")
    if op == "conv2d" and route == "matmul":
        body, n_out, call = _conv_as_matmul(spec, offsets)
    elif op == "conv2d":
        body, n_out, call = _conv(spec, offsets)
    elif op == "matmul":
        body, n_out, call = _matmul(spec, offsets)
    else:
        raise ValueError(f"no library kernel for op {op!r}")
    return _PRELUDE + _measured_main(body, n_out, call, label=label, protocol=protocol)


def _measured_main(body: str, n_out: str, call: str, *, label: str, protocol: str) -> str:
    # The call is emitted inline (warm pass, then timed pass), never through a multi-line macro.
    stmt = f"    {call}\n"
    warm = stmt + "    gemmini_fence();\n" if protocol == "warm_then_measured" else ""
    return (
        "\nint main(void) {\n"
        + body
        + "    gemmini_flush(0);\n"
        + warm
        + "    uint64_t t0 = read_cycles();\n"
        + stmt
        + "    gemmini_fence();\n"
        "    uint64_t t1 = read_cycles();\n"
        f"    uint64_t dg = lb_fnv1a_words((const uint8_t *)output, {n_out} * sizeof(elem_t)) & {_ref.DIGEST_MASK}ULL;\n"
        f'    printf("LB_RECORD {label} cycles=%llu digest=%llu\\n", (unsigned long long)(t1 - t0), (unsigned long long)dg);\n'
        # A FIXED final line. The record above carries a digest and a cycle count, so no two runs share
        # it and it cannot serve as the exact success marker a sealed FPGA receipt validates against.
        # Reaching here means the kernel ran and its output was digested; whether the digest is RIGHT is
        # decided off-device against the contract, which is the only place that can know.
        f'    printf("LB_DONE {label}\\n");\n'
        "    return 0;\n}\n"
    )


def render_schedule_layer(
    spec: Mapping[str, Any], *, offsets: Mapping[str, int], kernel_c: str, symbol: str, arg_names: list[str]
) -> str:
    """C source measuring one layer through a SCHEDULED kernel (``merlin.sched``).

    Everything but the timed call is the library program's for the same spec: the operand blob and its
    pointers, the output placement, the protocol and the digest. The library call is replaced by a call
    of the emitted kernel function ``symbol`` (its C text is ``kernel_c``) on ``arg_names``, which must
    be pointers that program declares (``input/weights/bias/output`` for a conv, ``a/b/d/output`` for a
    matmul). So a schedule and the library are measured and checked by one harness."""
    label = str(spec["label"])
    if not label or any(ch.isspace() for ch in label):
        raise ValueError("label must be one non-empty token")
    protocol = spec.get("protocol", "warm_then_measured")
    if protocol not in ("warm_then_measured", "cold_single"):
        raise ValueError(f"unknown protocol {protocol!r}")
    if not symbol.isidentifier():
        raise ValueError(f"symbol {symbol!r} is not a C identifier")
    if spec["op"] == "conv2d":
        body, n_out, _ = _conv(spec, offsets)
        declared = ("input", "weights", "bias", "output")
    elif spec["op"] == "matmul":
        body, n_out, _ = _matmul(spec, offsets)
        declared = ("a", "b", "d", "output")
    else:
        raise ValueError(f"no layer program for op {spec['op']!r}")
    unknown = [a for a in arg_names if a not in declared]
    if unknown:
        raise ValueError(f"kernel arguments {unknown} are not pointers this program declares {declared}")
    return (
        _PRELUDE
        + "\n"
        + kernel_c
        + _measured_main(body, n_out, f"{symbol}({', '.join(arg_names)});", label=label, protocol=protocol)
    )


def render_external_layer(
    spec: Mapping[str, Any],
    *,
    offsets: Mapping[str, int],
    symbol: str,
    header: str,
    declared_shape: Mapping[str, Any],
) -> str:
    """C source measuring a THIRD-PARTY kernel on the same operands as the library and the package.

    An external kernel published beside a paper is usually specialised to one shape: its tile sizes,
    its scratchpad base addresses and sometimes a statically sized padding buffer are searched for
    that shape and are not parameters. Running it on any other extents silently measures something
    that is not the kernel, so ``declared_shape`` is the shape its own descriptor states and every
    field of it must equal the spec. The refusal is the point -- a bench that quietly reshaped the
    comparison would flatter or damn the wrong party.

    The operand set is the library's: ``(batch, input, weights, bias, output, activation, scale)``.
    A kernel with a different surface needs its own renderer rather than a reordering here.
    """
    label = str(spec["label"])
    if not label or any(ch.isspace() for ch in label):
        raise ValueError("label must be one non-empty token")
    if spec.get("op") != "conv2d":
        raise ValueError(f"no external kernel renderer for op {spec.get('op')!r}")
    mismatched = {key: (spec.get(key), value) for key, value in declared_shape.items() if spec.get(key) != value}
    if mismatched:
        raise ValueError(
            f"external kernel {symbol!r} declares {dict(declared_shape)} and would be measured on "
            f"{ {k: v[0] for k, v in mismatched.items()} }; a specialised kernel on other extents is "
            "not that kernel"
        )
    if not symbol or any(ch.isspace() for ch in symbol):
        raise ValueError("kernel symbol must be one non-empty token")
    protocol = spec.get("protocol", "warm_then_measured")
    if protocol not in ("warm_then_measured", "cold_single"):
        raise ValueError(f"unknown protocol {protocol!r}")
    body, n_out, _library_call = _conv(spec, offsets)
    call = (
        f"{symbol}({int(spec['batch'])}, input, weights, bias, output,\n"
        f"        {'RELU' if spec.get('relu') else 'NO_ACTIVATION'}, {_c_float(spec['scale'])});"
    )
    return _PRELUDE + f'\n#include "{header}"\n' + _measured_main(body, n_out, call, label=label, protocol=protocol)


# ----------------------------------------------------- a third party's kernel, compiled as it is written
#: The short instruction names the published Exo/AutoComp GEMM kernels are written in, each mapped onto
#: a macro THIS target's own curated ``gemmini.h`` already defines. Quoted from the ``auto-comp-v2``
#: branch of gemmini-rocc-tests that the AutoComp artifact names as its build dependency
#: (``include/gemmini.h``, the "defined functions" block), so no opcode, funct or field layout is
#: introduced here: every alias expands to a macro whose encoding comes from the curated header the
#: rest of this bench already compiles against.
AUTOCOMP_ALIASES = r"""
/* Instruction-name aliases used by the published Exo/AutoComp kernels. Quoted from gemmini-rocc-tests
   branch auto-comp-v2, include/gemmini.h, the "defined functions" block. Each expands to a macro the
   curated gemmini.h included above defines; nothing about the encoding is restated here. */
#define config_ex(dataflow, act, A_stride, A_transpose, B_transpose) \
    gemmini_extended_config_ex(dataflow, act, 0, A_stride, A_transpose, B_transpose)
#define config_ld(cols, scale, spad_block_stride, id) \
    gemmini_extended4_config_ld(cols, scale, false, spad_block_stride, id)
#define mvin(dram_addr, spad_addr, cols, rows)  gemmini_extended_mvin(dram_addr, spad_addr, cols, rows)
#define mvin2(dram_addr, spad_addr, cols, rows) gemmini_extended_mvin2(dram_addr, spad_addr, cols, rows)
#define mvin3(dram_addr, spad_addr, cols, rows) gemmini_extended_mvin3(dram_addr, spad_addr, cols, rows)
#define preload(B_spad_addr, C_acc_addr, B_cols, B_rows, C_cols, C_rows) \
    gemmini_extended_preload(B_spad_addr, C_acc_addr, B_cols, B_rows, C_cols, C_rows)
#define preload_zeros(C_acc_addr) gemmini_preload_zeros(C_acc_addr)
#define compute_preloaded(A_spad_addr, bias_spad_addr, A_cols, A_rows, bias_cols, bias_rows) \
    gemmini_extended_compute_preloaded(A_spad_addr, bias_spad_addr, A_cols, A_rows, bias_cols, bias_rows)
#define compute_accumulated(A_spad_addr, bias_spad_addr, A_cols, A_rows, bias_cols, bias_rows) \
    gemmini_extended_compute_accumulated(A_spad_addr, bias_spad_addr, A_cols, A_rows, bias_cols, bias_rows)
#define config_st(cols) gemmini_config_st(cols)
#define mvout(dram_addr, acc_addr, cols, rows) gemmini_extended_mvout(dram_addr, acc_addr, cols, rows)
#define fence() asm volatile("fence")
"""

#: The published harness's own preprocessor context (``harnesses/exo/test<N>.c``), which its
#: ``sol*_gemmini_baseline.c`` reads its extents and matrix names out of. Emitted from the measured
#: spec, so the shape a kernel is compiled for is the shape the bench asked for and never a literal
#: carried along beside it.
AUTOCOMP_HARNESS_DEFINES = """
/* The published harness's own defines (harnesses/exo/test<N>.c), which its gemmini_baseline solution
   reads its extents and operand names from. Values come from the measured spec. */
#define MAT_DIM_I {i}
#define MAT_DIM_K {k}
#define MAT_DIM_J {j}
#define A_TRANSPOSE 0
#define B_TRANSPOSE 0
#define NO_BIAS 1
#define REPEATING_BIAS 0
#define SUB_BIAS 0
#define A_MATRIX_NAME A
#define B_MATRIX_NAME B
#define C_MATRIX_NAME C
#define OUTPUT_MATRIX_NAME C_MATRIX_NAME
"""


def render_autocomp_kernel_unit(
    solution_c: str, *, symbol: str, m: int, n: int, k: int, declared_shape: Mapping[str, int]
) -> str:
    """ONE TRANSLATION UNIT holding a published Exo/AutoComp GEMM kernel, compiled as it is written.

    The published kernels are one C function, ``void solution(elem_t A[I][K], elem_t B[K][J], elem_t
    C[I][J])``, written against the harness's preprocessor context and against the short instruction
    names the ``auto-comp-v2`` ``gemmini.h`` adds. ``solution_c`` is that function's text VERBATIM:
    this renderer supplies the context around it and never rewrites a line of it, because a bench that
    edited a third party's kernel to make its numbers land is measuring its own edit.

    It is a translation unit of its OWN, not a block spliced into the window program, for two reasons
    that are properties of the published files rather than preferences: every one of them names its
    function ``solution``, and the library arm's file is written entirely in the harness's
    ``MAT_DIM_*`` macros, which take a different value for each measured shape. Six shapes in one
    program is therefore six units. The only thing this adds inside the unit is ``#define solution
    <symbol>``, a preprocessor rename that leaves the file's own bytes untouched and lets the window
    program call each shape's kernel by a name of its own.

    These kernels are specialised to one shape -- their tile counts and scratchpad bases are literals
    -- so ``declared_shape`` is what the published file's own signature states, and every extent must
    equal the spec's: a kernel whose tiling is a literal is not that kernel on other extents.
    """
    if not symbol.isidentifier():
        raise ValueError(f"symbol {symbol!r} is not a C identifier")
    if "void solution(" not in solution_c:
        raise ValueError("the published kernel text does not define solution()")
    want = {"m": int(m), "n": int(n), "k": int(k)}
    mismatched = {key: (want[key], int(value)) for key, value in declared_shape.items() if want[key] != int(value)}
    if mismatched:
        raise ValueError(
            f"the published kernel declares {dict(declared_shape)} and would be measured on "
            f"{ {key: got for key, (got, _) in mismatched.items()} }; a kernel whose tiling is a "
            "literal is not that kernel on other extents"
        )
    # ``gemmini.h``, not ``gemmini_testutils.h``: the kernels need only the ISA macros and the
    # library entry, and testutils defines ``rand``/``rand_double`` with EXTERNAL linkage, so six
    # published units including it would be six definitions of each and would not link.
    return (
        '#include <stdint.h>\n#include "include/gemmini.h"\n'
        + AUTOCOMP_ALIASES
        + AUTOCOMP_HARNESS_DEFINES.format(i=int(m), j=int(n), k=int(k))
        + f"\n#define solution {symbol}\n"
        + solution_c.rstrip()
        + "\n"
    )


def package_window_call(
    cb: Mapping[str, Any], *, offsets: Mapping[str, int], symbol: str, output: str, element_dtype: str
) -> tuple[str, str, str, str]:
    """``(extern declaration, pointer declarations, output-count expression, the one call)`` for a
    kernel an OUT-OF-TREE COMPILER PACKAGE emitted, linked beside the window program as an object.

    This is :func:`render_package_layer`'s argument placement, returned as a window instead of as a
    whole program: every ``read`` argument of the command buffer's ``kernel_abi`` is a pointer into
    the shared operand blob at the offset the caller packed it to, and every ``write`` argument (the
    output and any intermediate the package declares, e.g. a host im2col buffer) is carved out of the
    same region past ``_end`` the other arms write their output to. The entry is called in ABI order.

    ``element_dtype`` is the contract's own output dtype, passed in rather than assumed here: the
    frame digests ``n_out`` elements of ``elem_t``, so a package asked for a wider readout computes a
    different function and is refused rather than quietly measured beside the narrow rows.
    """
    if not symbol.isidentifier():
        raise ValueError(f"symbol {symbol!r} is not a C identifier")
    tensors = cb["tensors"]
    args = (cb.get("kernel_abi") or {}).get("args") or []
    if not args:
        raise ValueError("command buffer declares no kernel ABI")
    declarations: list[str] = [f"    uintptr_t lb_heap = lb_align((uintptr_t)_end + {_OUTPUT_GUARD_BYTES}, 64);\n"]
    types: list[str] = []
    pointers: list[str] = []
    out_elements = 0
    for index, arg in enumerate(args):
        name = arg["tensor"]
        declared = tensors[name]
        elements = 1
        for extent in declared["shape"]:
            elements *= int(extent)
        variable = f"lb_arg{index}"
        if arg.get("access") == "read":
            if name not in offsets:
                raise ValueError(f"read argument {name!r} has no operand in the blob")
            declarations.append(f"    void *{variable} = (void *)(lb_operands + {int(offsets[name])});\n")
        else:
            nbytes = elements * _ELEM_BYTES[declared["dtype"]]
            declarations.append(
                f"    void *{variable} = (void *)lb_heap; lb_heap = lb_align(lb_heap + {nbytes}, 64);\n"
            )
        types.append("void *")
        pointers.append(variable)
        if name == output:
            if declared["dtype"] != element_dtype:
                raise ValueError(
                    f"output {output!r} is declared {declared['dtype']}, not the contract's {element_dtype}; a "
                    "window frame digests elem_t, so a wider readout is a different function whose cycle count "
                    "belongs in its own run rather than in a column beside these"
                )
            declarations.append(f"    elem_t *output = (elem_t *){variable};\n")
            out_elements = elements
    if not out_elements:
        raise ValueError(f"output {output!r} is not a kernel argument")
    declaration = f"extern void {symbol}({', '.join(types)});\n"
    return declaration, "".join(declarations), str(out_elements), f"{symbol}({', '.join(pointers)});"


def autocomp_window_call(
    *, symbol: str, offsets: Mapping[str, int], m: int, n: int, k: int
) -> tuple[str, str, str, str]:
    """``(extern declaration, pointer declarations, output-count expression, the one call)``.

    The published kernel takes its operands as 2-D arrays, which decay to row pointers, so the window
    declares them at exactly the types the file's own signature states, reading A and B out of the
    shared operand blob and writing C into the same buffer past ``_end`` that every other arm's output
    goes to -- one placement, one digest, one oracle.
    """
    if not symbol.isidentifier():
        raise ValueError(f"symbol {symbol!r} is not a C identifier")
    declaration = f"extern void {symbol}(elem_t (*)[{int(k)}], elem_t (*)[{int(n)}], elem_t (*)[{int(n)}]);\n"
    body = (
        f"    const size_t n_out = (size_t){int(m)} * {int(n)};\n"
        f"    elem_t (*A)[{int(k)}] = (elem_t (*)[{int(k)}])(lb_operands + {int(offsets['a'])});\n"
        f"    elem_t (*B)[{int(n)}] = (elem_t (*)[{int(n)}])(lb_operands + {int(offsets['b'])});\n"
        + _output_ptr("n_out")
        + f"    elem_t (*C)[{int(n)}] = (elem_t (*)[{int(n)}])output;\n"
    )
    return declaration, body, "n_out", f"{symbol}(A, B, C);"


# --------------------------------------------------------------- the sealable window frame
#: EVERY PROTOCOL LINE A SEALABLE PROGRAM PUBLISHES, ONCE.  These are not this module's spellings to
#: choose: ``MERLIN_INVOCATIONS warmup=1 measured=1``, the four ``MERLIN_PROFILE`` lines and
#: ``METRIC cycles N`` are what ``merlin.perf.firesim_receipt._verify_uart`` accepts and what
#: ``merlin.perf.warm_profile_harness`` renders for every other workload, and the
#: ``MERLIN_BATCH``/``MERLIN_WINDOW`` frame is what ``merlin.perf.firesim_batch`` reads.  A program
#: that prints its own spelling of the measurement cannot be sealed, however right its arithmetic:
#: that is exactly why no run of the GSIM layer-bench harness could ever have produced a receipt.
WINDOW_UART = {
    "invocations": "MERLIN_INVOCATIONS warmup=1 measured=1",
    "batch_begin": "MERLIN_BATCH begin id={batch} windows={windows}",
    "batch_end": "MERLIN_BATCH end id={batch}",
    "window_begin": "MERLIN_WINDOW begin label={label}",
    "warm_begin": "MERLIN_PROFILE warmup begin",
    "warm_end": "MERLIN_PROFILE warmup end rc=0",
    "record": "LB_RECORD {label} cycles={cycles} digest={digest}",
    "done": "LB_DONE {label}",
    "measured_begin": "MERLIN_PROFILE measured begin",
    "metric": "METRIC cycles {cycles}",
    "measured_end": "MERLIN_PROFILE measured end rc=0",
    "window_end": "MERLIN_WINDOW end label={label}",
}

#: How a v2 validation policy must declare this harness's checksum line.  ``LB_RECORD <label>
#: cycles=<n> digest=<d>`` already parses as one: the label is the group and ``digest`` the value.
CHECKSUM_LINE = {"prefix": "LB_RECORD", "value_key": "digest", "group_token_offset": 1}


#: The exact correctness marker a window publishes.  Fixed text, so a v2 policy can name it; the
#: NUMBER that proves the window was right is the digest on the ``LB_RECORD`` line above, which is
#: compared against the off-device integer oracle.  A marker alone is the FireSim job 730 shape.
def window_marker(label: str) -> str:
    return WINDOW_UART["done"].format(label=label)


def _window_label(label: object) -> str:
    if not isinstance(label, str) or not label or label != label.strip():
        raise ValueError("a window label must be one nonempty, unpadded token")
    if any(character.isspace() for character in label) or "=" in label:
        raise ValueError(f"window label {label!r} must be whitespace-free and contain no '='")
    return label


def _window_call(window: Mapping[str, Any]) -> tuple[str, str, str]:
    """``(declarations, output-element-count expression, the one call)`` for one measured window.

    A window carrying ``external`` supplies all three itself: its declarations must define ``output``
    (the ``elem_t *`` the frame digests) and whatever pointers its call needs, and its ``kernel_c`` is
    the declaration of the symbol it calls, defined in another translation unit linked beside the
    program. That is how a window measures code this repo did not render -- a third party's published
    kernel, or an object a compiler package emitted -- on the same operands, placement and oracle.
    """
    external = window.get("external")
    if external is not None:
        for field in ("body", "n_out", "call"):
            if not isinstance(external.get(field), str) or not external[field].strip():
                raise ValueError(f"an external window must state a non-empty {field!r}")
        if "elem_t *output" not in external["body"]:
            raise ValueError("an external window's declarations must define the elem_t *output the frame digests")
        return external["body"], external["n_out"], external["call"]
    spec, offsets = window["spec"], window["offsets"]
    symbol = window.get("symbol")
    if symbol is None:
        if spec["op"] == "matmul":
            return _matmul(spec, offsets)
        if spec["op"] == "conv2d":
            route = spec.get("route", "conv")
            if route not in CONV_ROUTES:
                raise ValueError(f"unknown conv route {route!r} (known: {CONV_ROUTES})")
            return _conv_as_matmul(spec, offsets) if route == "matmul" else _conv(spec, offsets)
        raise ValueError(f"no library kernel for op {spec['op']!r}")
    if not str(symbol).isidentifier():
        raise ValueError(f"symbol {symbol!r} is not a C identifier")
    if spec["op"] == "matmul":
        body, n_out, _library = _matmul(spec, offsets)
        declared = ("a", "b", "d", "output")
    elif spec["op"] == "conv2d":
        body, n_out, _library = _conv(spec, offsets)
        declared = ("input", "weights", "bias", "output")
    else:
        raise ValueError(f"no layer program for op {spec['op']!r}")
    names = list(window.get("arg_names") or declared)
    unknown = [name for name in names if name not in declared]
    if unknown:
        raise ValueError(f"kernel arguments {unknown} are not pointers this program declares {declared}")
    return body, n_out, f"{symbol}({', '.join(names)});"


def render_window_program(
    windows: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    *,
    batch_id: str | None = None,
    protocol: str = "warm_then_measured",
) -> str:
    """One program containing N measured windows, framed so a queue-owned run can be SEALED.

    WHY N WINDOWS IN ONE PROGRAM AND NOT N PROGRAMS.  ``merlin.perf.execution_policy`` derives the
    FireSim queue lifecycle from the daemon's own trace as exactly ``kill -> infrasetup ->
    runworkload -> kill``, one ``runworkload`` per job, so "hold one session and replay N ELFs" is
    inadmissible by construction and can never produce a receipt.  The admissible shape is one
    bootbinary with N windows, which is what this renders.

    Every window shares ONE embedded operand blob (``OPERAND_BLOB``); a caller packs the members'
    operands into it and passes each window's offsets INTO that blob.  Windows are otherwise
    independent: each declares its own pointers, runs its own warm call, opens and closes its own
    cycle window, and digests its own output before the next window starts.

    ``batch_id`` frames the whole program as a batch.  Omit it for a single window, whose UART is
    then a strict prefix of the batched shape and is accepted by the SOLO receipt path
    (:func:`merlin.perf.firesim_receipt.parse_queued_firesim_receipt`) -- which requires exactly one
    ``METRIC cycles`` line and therefore cannot read a batch.

    Ordering is not cosmetic.  ``_verify_uart`` requires the correctness markers to follow
    ``MERLIN_PROFILE measured begin`` and PRECEDE the metric publication, so the record and its
    digest are printed before ``METRIC cycles``; and the digest is computed AFTER the closing cycle
    read, so host-side checking costs the measurement nothing.
    """
    if protocol not in ("warm_then_measured", "cold_single"):
        raise ValueError(f"unknown protocol {protocol!r}")
    if not windows:
        raise ValueError("a window program needs at least one measured window")
    labels = [_window_label(window["label"]) for window in windows]
    if len(set(labels)) != len(labels):
        raise ValueError("window labels must be unique within one program")
    if batch_id is not None:
        batch_id = _window_label(batch_id)

    definitions: list[str] = []
    seen_definitions: set[str] = set()
    bodies: list[str] = []
    for index, (label, window) in enumerate(zip(labels, windows)):
        kernel_c = window.get("kernel_c")
        if kernel_c and kernel_c not in seen_definitions:
            seen_definitions.add(kernel_c)
            definitions.append(kernel_c)
        body, n_out, call = _window_call(window)
        statement = f"    {call}\n    gemmini_fence();\n"
        warm = statement if protocol == "warm_then_measured" else ""
        bodies.append(
            f"static void lb_window_{index}(void) {{\n"
            + body
            + f'    printf("{WINDOW_UART["window_begin"].format(label=label)}\\n");\n'
            # POISON FIRST, and outside the measured region. Every window writes its output to the same
            # buffer past _end, so a kernel that fills only part of its output would inherit the rest
            # from the window before it and print that window's correct digest -- the exact shape of
            # silent wrongness the digest admission exists to catch. Filling the buffer with a value the
            # oracle cannot produce makes any unwritten element change the digest.
            + f"    lb_poison(output, {n_out} * sizeof(elem_t));\n"
            + "    gemmini_flush(0);\n"
            + f'    printf("{WINDOW_UART["warm_begin"]}\\n");\n'
            + warm
            + f'    printf("{WINDOW_UART["warm_end"]}\\n");\n'
            + f'    printf("{WINDOW_UART["measured_begin"]}\\n");\n'
            + "    const uint64_t t0 = read_cycles();\n"
            + statement
            + "    const uint64_t t1 = read_cycles();\n"
            + f"    const uint64_t dg = lb_fnv1a_words((const uint8_t *)output, {n_out} * sizeof(elem_t))"
            + f" & {_ref.DIGEST_MASK}ULL;\n"
            + '    printf("'
            + WINDOW_UART["record"].format(label=label, cycles="%llu", digest="%llu")
            + '\\n", (unsigned long long)(t1 - t0), (unsigned long long)dg);\n'
            + f'    printf("{WINDOW_UART["done"].format(label=label)}\\n");\n'
            + '    printf("'
            + WINDOW_UART["metric"].format(cycles="%llu")
            + '\\n", (unsigned long long)(t1 - t0));\n'
            + f'    printf("{WINDOW_UART["measured_end"]}\\n");\n'
            + f'    printf("{WINDOW_UART["window_end"].format(label=label)}\\n");\n'
            + "}\n"
        )

    main = [f'int main(void) {{\n    printf("{WINDOW_UART["invocations"]}\\n");\n']
    if batch_id is not None:
        opened = WINDOW_UART["batch_begin"].format(batch=batch_id, windows=len(labels))
        main.append(f'    printf("{opened}\\n");\n')
    main.extend(f"    lb_window_{index}();\n" for index in range(len(labels)))
    if batch_id is not None:
        main.append(f'    printf("{WINDOW_UART["batch_end"].format(batch=batch_id)}\\n");\n')
    main.append("    return 0;\n}\n")
    return _PRELUDE + "\n" + "\n".join(definitions) + "\n" + "\n".join(bodies) + "\n" + "".join(main)
