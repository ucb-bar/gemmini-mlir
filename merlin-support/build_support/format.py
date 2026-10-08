"""Pure target buffer declarations and storage words; no backend discovery or execution."""
from __future__ import annotations

from dataclasses import dataclass


class CodegenError(RuntimeError):
    pass

OPERAND_DTYPE, OPERAND_CTYPE = "i8", "elem_t"


def ceil_dim(x: int, dim: int) -> int:
    """Existing legacy row padding, with its target dimension supplied explicitly."""
    return ((x + dim - 1) // dim) * dim


def pad_rowmajor(data, rows: int, cols: int, prows: int, pcols: int):
    """Zero-pad a row-major rows x cols matrix into prows x pcols."""
    out = [0] * (prows * pcols)
    for r in range(rows):
        base, pbase = r * cols, r * pcols
        out[pbase:pbase + cols] = data[base:base + cols]
    return out


def buffer_extent(spec: dict, *, name: str) -> tuple[int, int]:
    """Legacy caller matrix view: flatten leading axes; retain final columns."""
    shape = spec.get("shape")
    if not isinstance(shape, list) or not shape or any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in shape):
        raise CodegenError(
            f"host-lane tensor {name!r} needs a non-empty shape of positive extents, got {shape!r}")
    rows = 1
    for dim in shape[:-1]:
        rows *= dim
    return rows, shape[-1]

@dataclass(frozen=True)
class Container:
    """How one harness buffer of a declared dtype is spelled in C: storage type, row-alignment macro,
    and how one element is printed onto the ``OUT`` line."""
    ctype: str
    align: str
    cast: str
    conv: str
    word_bytes: int
    signed: bool

    def decl(self, symbol: str, elems: int, *, const: bool = False,
             initializer: str | None = None) -> str:
        body = f" = {{{initializer}}}" if initializer is not None else ""
        return (f"static {'const ' if const else ''}{self.ctype} {symbol}[{elems}] "
                f"{self.align}(1){body};")

    def printf_element(self, expr: str) -> str:
        return f'printf(" {self.conv}", ({self.cast}){expr});'


def container_for(dtype: str) -> Container:
    """The C container a harness buffer of ``dtype`` is allocated and printed in — DERIVED from that
    dtype's own storage width, never tabulated per spelling.

    What a buffer must get right is how many bytes one element occupies (the kernel's stores and the
    harness's reads have to agree on the stride) and how its rows are aligned. Both follow from the
    width, so a dtype nobody has thought about is sized correctly or REFUSED, not quietly given four
    bytes. Sizing every non-i8 destination as ``int32_t`` was the actual defect a bf16 result hit: the
    kernel stores 2 bytes per element and the readback walked it at 4.

    A FLOAT dtype lands in the UNSIGNED integer container of its own width and is printed as its
    stored bit PATTERN. That is what makes a float result deliverable over a console whose ``printf``
    has no float formatting: the pattern is lossless, and the value is recovered at the readback from
    the same declared dtype (:func:`merlin.runtime.backends.base.decode_float_readback`). Unsigned so
    a top-bit-set pattern is printed as the pattern rather than through an implementation-defined
    conversion.
    """
    from merlin.common.quant_formats import storage_bits
    from merlin.runtime.fp8_formats import float_format_of
    if dtype == OPERAND_DTYPE:
        return Container(OPERAND_CTYPE, "row_align", "int", "%d", 1, True)
    bits = storage_bits(dtype)                    # fails closed on an unregistered spelling
    if bits % 8 or bits not in (8, 16, 32, 64):
        # A SUB-BYTE LOGICAL WIDTH IS NOT A STORAGE WIDTH. `storage_bits` answers how many bits the
        # VALUE occupies; what a harness buffer needs is the stride the kernel actually stores at.
        # For `i1` those differ: the compiler stores one whole byte per element and declares so in
        # four places -- llvmlower.c_runtime.DT_BYTES["i1"] == 1 and C_OF["i1"] == "signed char",
        # llvmlower.abi (ctypes.c_bool), llvmlower.kernel_backend (np.int8). SmolVLA's
        # `prefix_pad_masks` is i1, so refusing here stopped a whole model whose stride was never
        # in doubt. Read that declaration rather than restating it: a dtype the compiler does NOT
        # size still falls through to the refusal, which is what keeps a guessed width out.
        from merlin.llvmlower.c_runtime import DT_BYTES
        declared = DT_BYTES.get(dtype)
        if declared and declared * 8 in (8, 16, 32, 64):
            bits = declared * 8
        else:
            raise CodegenError(
                f"a harness buffer of dtype {dtype!r} stores {bits} bits per element and the "
                f"compiler declares no whole-byte storage width for it; this harness lays out "
                f"whole 8/16/32/64-bit containers only, and guessing a width mis-strides the "
                f"buffer")
    if float_format_of(dtype) is not None:
        return Container(f"uint{bits}_t", "row_align_acc",
                         "unsigned long long" if bits > 32 else "unsigned", "%llu" if bits > 32 else "%u",
                         bits // 8, False)
    return Container(f"int{bits}_t", "row_align_acc",
                     "long long" if bits > 32 else "int", "%lld" if bits > 32 else "%d", bits // 8, True)


def container_words(values, dtype: str) -> list[int]:
    """``values`` as the integer words a ``container_for(dtype)`` buffer holds.

    An integer dtype contributes its values; a float dtype contributes its stored CODE PATTERNS, via
    the one registry that defines the code<->value mapping — so what the harness embeds and what a
    readback decodes are inverse by construction rather than by two hand-written conversions.
    """
    from merlin.runtime.fp8_formats import float_format_of
    fmt = float_format_of(dtype)
    if fmt is None:
        return [int(v) for v in values]
    from merlin.runtime import fp8_formats as _ff
    return [int(c) for c in _ff.float_to_codes(list(values), fmt)]
