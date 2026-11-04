"""Decode a closed full-value OUT_BIN packet against a caller-owned output roster.

This is a wire decoder, not evidence that a native program completed. The
caller must independently establish the selected process/ELF/readback binding
before presenting these actual values to its numerical checker.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from dataclasses import dataclass

from merlin.runtime.out_bin import (
    BinaryFrame,
    OutBinError,
    binary_console_diagnostics,
    parse_binary_console_details,
)


class OutPacketError(ValueError):
    """The packet or its declared output roster is not complete and exact."""


@dataclass(frozen=True)
class ExpectedOutput:
    name: str
    rows: int
    cols: int
    logical_dtype: str
    container_signed: bool
    container_max_bytes: int
    logical_shape: tuple[int, ...]


@dataclass(frozen=True)
class DecodedOutput:
    logical_dtype: str
    logical_shape: tuple[int, ...]
    values: object
    wire_bytes: int
    wire_signed: bool
    f32_bits: tuple[int, ...] | None


_INTEGER_BITS = {
    "i1": 1,
    "i8": 8,
    "i16": 16,
    "i32": 32,
    "i64": 64,
    "u8": 8,
    "u16": 16,
    "u32": 32,
    "u64": 64,
}
_MAX_PACKET_BYTES = 256 * 1024 * 1024
_END_LINE = b"OUT_BIN_END v1 0123456789abcdef\n"
_DONE_LINE = b"DONE\n"


def _validate_expected(expected: Sequence[ExpectedOutput]) -> tuple[ExpectedOutput, ...]:
    if not isinstance(expected, (tuple, list)) or not expected:
        raise OutPacketError("OUT_BIN packet needs a nonempty closed output roster")
    rows = tuple(expected)
    names: set[str] = set()
    for row in rows:
        if type(row) is not ExpectedOutput:
            raise OutPacketError("OUT_BIN output roster has an invalid descriptor")
        if (
            type(row.name) is not str
            or not row.name
            or not row.name.isascii()
            or not all(char.isalnum() or char == "_" for char in row.name)
            or row.name in names
        ):
            raise OutPacketError("OUT_BIN output roster has a duplicate or invalid name")
        names.add(row.name)
        if type(row.rows) is not int or type(row.cols) is not int or row.rows <= 0 or row.cols <= 0:
            raise OutPacketError("OUT_BIN output roster has invalid frame geometry")
        if row.rows * row.cols > (1 << 64) - 1:
            raise OutPacketError("OUT_BIN output roster exceeds the packet word count")
        if type(row.logical_dtype) is not str or row.logical_dtype not in (*_INTEGER_BITS, "f32"):
            raise OutPacketError("OUT_BIN output roster has unsupported logical dtype")
        if type(row.container_signed) is not bool or row.container_max_bytes not in (1, 2, 4, 8):
            raise OutPacketError("OUT_BIN output roster has invalid container type")
        if type(row.container_max_bytes) is not int:
            raise OutPacketError("OUT_BIN output roster has invalid container width")
        signed_dtype = row.logical_dtype.startswith("i") and row.logical_dtype != "i1"
        if row.container_signed != signed_dtype:
            raise OutPacketError("OUT_BIN output roster signedness differs from logical dtype")
        logical_bytes = 4 if row.logical_dtype == "f32" else (_INTEGER_BITS[row.logical_dtype] + 7) // 8
        if row.container_max_bytes < logical_bytes:
            raise OutPacketError("OUT_BIN output container cannot retain its logical dtype")
        if row.rows * row.cols * row.container_max_bytes > (1 << 64) - 1:
            raise OutPacketError("OUT_BIN output roster exceeds the packet byte bound")
        if type(row.logical_shape) is not tuple or any(type(dim) is not int or dim <= 0 for dim in row.logical_shape):
            raise OutPacketError("OUT_BIN output roster has invalid logical shape")
        count = 1
        for dim in row.logical_shape:
            count *= dim
        if count != row.rows * row.cols:
            raise OutPacketError("OUT_BIN logical shape differs from frame geometry")
    return rows


def out_bin_packet_capacity(expected: Sequence[ExpectedOutput]) -> int:
    """Bound a complete packet using every declared output's maximum wire width.

    The source-selected narrower width may use fewer payload bytes and decimal
    digits. This is a capacity bound, not proof of an actual emitted frame.
    """

    roster = _validate_expected(expected)
    total = len(_DONE_LINE)
    for row in roster:
        payload = row.rows * row.cols * row.container_max_bytes
        header = (
            f"OUT_BIN_BEGIN v1 {row.name} {row.rows} {row.cols} "
            f"{row.container_max_bytes} {'s' if row.container_signed else 'u'} {payload}\n"
        ).encode("ascii")
        total += len(header) + payload + len(_END_LINE)
        if total > _MAX_PACKET_BYTES:
            raise OutPacketError("OUT_BIN packet capacity exceeds the bounded host arena")
    return total


def _require_closed_wire(data: bytes, expected: tuple[ExpectedOutput, ...], frames: dict[str, BinaryFrame]) -> None:
    diagnostics = binary_console_diagnostics(data)
    lines = diagnostics.splitlines(keepends=True)
    if len(lines) != 2 * len(expected) + 1 or lines[-1] != _DONE_LINE:
        raise OutPacketError("OUT_BIN packet lacks exact frames and literal DONE wire closure")
    for index, row in enumerate(expected):
        frame = frames[row.name]
        count = frame.rows * frame.cols * frame.word_bytes
        header = (
            f"OUT_BIN_BEGIN v1 {frame.name} {frame.rows} {frame.cols} "
            f"{frame.word_bytes} {'s' if frame.signed else 'u'} {count}\n"
        ).encode("ascii")
        if lines[2 * index] != header:
            raise OutPacketError("OUT_BIN packet frame order or header is not exact")
        footer = lines[2 * index + 1]
        prefix = b"OUT_BIN_END v1 "
        if (
            not footer.startswith(prefix)
            or len(footer) != len(prefix) + 16 + 1
            or footer[-1:] != b"\n"
            or any(byte not in b"0123456789abcdef" for byte in footer[len(prefix) : -1])
        ):
            raise OutPacketError("OUT_BIN packet has no canonical END frame")


def _reshape(flat: tuple[int | float, ...], shape: tuple[int, ...]) -> object:
    if not shape:
        return flat[0]
    if len(shape) == 1:
        return flat
    stride = len(flat) // shape[0]
    return tuple(_reshape(flat[index * stride : (index + 1) * stride], shape[1:]) for index in range(shape[0]))


def decode_out_bin_packet(packet: bytes, expected: Sequence[ExpectedOutput]) -> dict[str, DecodedOutput]:
    """Return every actual logical value after exact frame and roster checks.

    ``f32_bits`` retains original 32-bit patterns even for negative zero and
    NaNs; floating ``values`` are a convenience reinterpretation, not a
    replacement for the full-bit output. Native exit/DONE is a separate proof.
    """

    roster = _validate_expected(expected)
    if type(packet) is not bytes:
        raise OutPacketError("OUT_BIN packet requires raw bytes")
    if len(packet) > out_bin_packet_capacity(roster):
        raise OutPacketError("OUT_BIN packet exceeds its declared maximum capacity")
    try:
        decoded, metrics, frames = parse_binary_console_details(packet)
    except OutBinError as exc:
        raise OutPacketError(str(exc)) from exc
    if metrics or tuple(decoded) != tuple(row.name for row in roster) or set(frames) != set(decoded):
        raise OutPacketError("OUT_BIN packet differs from the exact output roster")
    try:
        _require_closed_wire(packet, roster, frames)
    except OutBinError as exc:
        raise OutPacketError(str(exc)) from exc

    result: dict[str, DecodedOutput] = {}
    for row in roster:
        frame = frames[row.name]
        if frame.rows != row.rows or frame.cols != row.cols or frame.word_bytes > row.container_max_bytes:
            raise OutPacketError("OUT_BIN frame geometry or wire width differs from output declaration")
        if frame.signed and not row.container_signed:
            raise OutPacketError("OUT_BIN signed wire differs from output declaration")
        flat = tuple(word for line in decoded[row.name] for word in line)
        if row.logical_dtype == "f32":
            if frame.signed or any(word < 0 or word > 0xFFFFFFFF for word in flat):
                raise OutPacketError("OUT_BIN f32 payload is not unsigned 32-bit source bits")
            bits = flat
            values: tuple[int | float, ...] = tuple(struct.unpack("<f", word.to_bytes(4, "little"))[0] for word in bits)
        else:
            bits = None
            width = _INTEGER_BITS[row.logical_dtype]
            minimum = -(1 << (width - 1)) if row.container_signed else 0
            maximum = (1 << (width - 1)) - 1 if row.container_signed else (1 << width) - 1
            if any(word < minimum or word > maximum for word in flat):
                raise OutPacketError("OUT_BIN actual value exceeds declared logical dtype")
            values = flat
        result[row.name] = DecodedOutput(
            logical_dtype=row.logical_dtype,
            logical_shape=row.logical_shape,
            values=_reshape(values, row.logical_shape),
            wire_bytes=frame.word_bytes,
            wire_signed=frame.signed,
            f32_bits=bits,
        )
    return result
