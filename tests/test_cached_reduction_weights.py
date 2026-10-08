"""Complete source ownership and K order for full-HWIO weight residency."""

from collections import Counter

import pytest
from test_resident_conv_command_loops import executed_commands

from mlir_oot.golden_conv import ConvShape
from mlir_oot.golden_flat_conv import GoldenFlatConv, choose_band_rows
from mlir_oot.tables import isa
from mlir_oot.tables import rtl_facts as F


def packet(value):
    return value & 0xFFFFFFFF, value >> 48, (value >> 32) & 65535


def contraction_stream(s):
    g = GoldenFlatConv(
        s,
        wide_a=True,
        separate_b_bank=False,
        band_rows=choose_band_rows(s, virtual_padding=True),
        virtual_padding=True,
        cached_reduction_weights=True,
    )
    scratch = {}
    acc = {}
    configs = {}
    active = None
    destination = None
    outputs = set()
    counts = Counter()
    traffic = Counter()
    stride = 1
    for name, encoded, pointers in executed_commands(g.build()):
        counts[name] += 1
        if encoded is None:
            continue
        funct, left, right = encoded
        if name == "gemmini.config_ex":
            stride = (left >> 16) & 65535
        elif name == "gemmini.config_ld":
            load = (left >> 3) & 3
            configs[load] = (right, (left >> 16) & 65535)
        elif name == "gemmini.mvin":
            local, rows, cols = packet(right)
            (ptr,) = pointers
            load = 0 if funct == isa.K_MVIN else (1 if funct == isa.K_MVIN2 else 2)
            row_stride, block_stride = configs[load]
            assert rows <= F.DIM and cols <= 4 * F.DIM
            if ptr.base is not None:
                assert ptr.base == load
                traffic[ptr.base] += rows * cols
            else:
                traffic["zero_fill"] += rows * cols
            for block in range((cols + 15) // 16):
                width = min(16, cols - block * 16)
                for row in range(rows):
                    address = local + block * block_stride + row
                    assert 0 <= address < F.SPAD_ROWS
                    if ptr.base is None:
                        tokens = (None,) * width
                    else:
                        offset = ptr.offset.value + row * row_stride + block * 16
                        assert (
                            0
                            <= offset
                            <= (
                                (s.h * s.w * s.cin)
                                if ptr.base == 0
                                else 9 * s.cin * s.cout
                            )
                            - width
                        )
                        tokens = tuple((ptr.base, offset + i) for i in range(width))
                    scratch[address] = tokens
        elif name == "gemmini.preload":
            b, brows, bcols = packet(left)
            c, crows, ccols = packet(right)
            assert c & isa.ACC_ADDR_BIT and crows <= 16 and ccols <= 16
            destination = (c & 0x3FFF, crows, ccols, bool(c & isa.ACC_ACCUMULATE_BIT))
            if b != isa.GARBAGE_ADDR:
                active = tuple(scratch[b + k][:bcols] for k in range(brows))
        elif name == "gemmini.compute":
            a, arows, acols = packet(left)
            base, rows, cols, add = destination
            assert active is not None and arows == rows and acols == 16
            for lane in range(rows):
                address = base + lane
                if not add:
                    acc[address] = []
                assert address in acc
                acc[address].append((scratch[a + lane * stride][:acols], active))
        elif name == "gemmini.mvout":
            local, rows, cols = packet(right)
            (ptr,) = pointers
            assert ptr.base == 2 and local & isa.ACC_ADDR_BIT
            full = bool(local & isa.ACC_FULL_ROW_BIT)
            assert full == (s.output_dtype == "i32")
            for block in range((cols + 15) // 16):
                width = min(16, cols - block * 16)
                for lane in range(rows):
                    first = (
                        ptr.offset.value // (4 if full else 1)
                        + lane * s.cout
                        + block * 16
                    )
                    pixel, channel = divmod(first, s.cout)
                    y, x = divmod(pixel, s.ow)
                    assert 0 <= y < s.oh and channel + width <= s.cout
                    address = (local & 0x3FFF) + block * 16 + lane
                    work = acc[address]
                    assert len(work) == 9 * (s.cin // 16)
                    for index, (atokens, btokens) in enumerate(work):
                        start = index * 16
                        tap, ci = divmod(start, s.cin)
                        kh, kw = divmod(tap, 3)
                        iy = y * s.stride + kh - 1
                        ix = x * s.stride + kw - 1
                        expected_a = tuple(
                            (0, (iy * s.w + ix) * s.cin + ci + k)
                            if 0 <= iy < s.h and 0 <= ix < s.w
                            else None
                            for k in range(16)
                        )
                        assert atokens == expected_a
                        assert btokens == tuple(
                            tuple(
                                (1, (start + k) * s.cout + channel + n)
                                for n in range(width)
                            )
                            for k in range(16)
                        )
                    cells = set(range(first, first + width))
                    assert not outputs.intersection(cells)
                    outputs.update(cells)
    assert stride == 1 and outputs == set(range(s.oh * s.ow * s.cout))
    return counts, traffic


@pytest.mark.parametrize(
    "s",
    [
        ConvShape(1, 1, 16, 17, bn=2),
        ConvShape(5, 21, 32, 19, stride=2, bn=2),
        ConvShape(7, 23, 48, 33, bn=3),
        ConvShape(19, 35, 32, 67, stride=2, bn=4),
    ],
)
def test_every_actual_primitive_closes_original_source_K_output_and_bounds(s):
    contraction_stream(s)


@pytest.mark.parametrize(
    "s", [ConvShape(19, 35, 32, 67, stride=2, bn=4), ConvShape(7, 23, 48, 33, bn=3)]
)
def test_retained_spatial_commands_preserve_cached_weight_pointers(s):
    kw = {
        "wide_a": True,
        "band_rows": choose_band_rows(s, virtual_padding=True),
        "virtual_padding": True,
        "cached_reduction_weights": True,
    }
    assert executed_commands(GoldenFlatConv(s, **kw).build()) == executed_commands(
        GoldenFlatConv(s, **kw, loop_spatial=True).build()
    )


@pytest.mark.parametrize("value", [0, 1, None, "yes"])
def test_weight_residency_requires_typed_boolean(value):
    with pytest.raises(ValueError, match="boolean"):
        GoldenFlatConv(
            ConvShape(5, 21, 32, 19),
            virtual_padding=True,
            cached_reduction_weights=value,
        )


@pytest.mark.parametrize(
    "shape,options",
    [
        (ConvShape(5, 21, 17, 19), {}),
        (ConvShape(5, 21, 1024, 256, bn=4), {}),
        (ConvShape(5, 21, 32, 19), {"separate_b_bank": True}),
        (ConvShape(5, 21, 32, 19), {"pingpong_b": True}),
    ],
)
def test_unproved_layout_competing_placement_or_capacity_refuses(shape, options):
    with pytest.raises(ValueError):
        GoldenFlatConv(
            shape, virtual_padding=True, cached_reduction_weights=True, **options
        )


def test_complete_option_clone_preserves_selected_weight_owner():
    s = ConvShape(19, 35, 32, 67, stride=2, bn=4)
    g = GoldenFlatConv(
        s,
        wide_a=True,
        band_rows=choose_band_rows(s, virtual_padding=True),
        virtual_padding=True,
        cached_reduction_weights=True,
    )
    assert g.emission_options.cached_reduction_weights
    assert str(g.build()) == str(g.with_emission_options().build())


def test_current_geometry_weights_and_activation_are_disjoint():
    s = ConvShape(
        56,
        56,
        128,
        128,
        stride=2,
        bn=4,
        output_dtype="i8",
        scale=0.0015212674625217915,
        relu=True,
        wide_b=True,
    )
    g = GoldenFlatConv(
        s,
        wide_a=True,
        band_rows=choose_band_rows(s, virtual_padding=True),
        virtual_padding=True,
        cached_reduction_weights=True,
    )
    assert g.weight_rows == 9216 and g.weight_base == 7168 and g.shape.bm * 64 == 896
