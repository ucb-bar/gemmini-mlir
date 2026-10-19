"""Hardware maxpool readout of an i32 accumulator image in DRAM."""

from builder import float_bits


ACC = 0x80000000
ONE = 0x3F800000
DIM = 16


def _descriptor(b, local, columns, rows=0):
    geometry = (int(columns) << 32) | (int(rows) << 48)
    return b.op("or", local, geometry)


def _segments(b, start, end, callback):
    """Keep transfer extents literal while emitting runtime full-block loops."""
    full_end = start + ((end - start) // DIM) * DIM
    if full_end > start:
        b.loop(start, full_end, DIM, lambda offset: callback(offset, DIM))
    if full_end < end:
        callback(full_end, end - full_end)


def _pointer(b, pointer):
    return b.cast("ptrtoint", pointer, "ptr", "i64")


def _configure_store(b, cols, scale, relu, size, stride,
                     height, width, oh, ow, top=0, left=0):
    first = (2 | (4 if relu else 0) | (stride << 4) | (size << 6)
             | (top << 8) | (left << 10) | (ow << 24)
             | (oh << 32) | (ow << 40) | (height << 48) | (width << 56))
    second = (int(scale) << 32) | int(cols)
    b.rocc(0, first, second)


def pool(b, src_i32, out, rows, cols, attrs):
    """Pool on Gemmini, with CPU code restricted to address calculation.

    The source already includes any bias but retains full i32 values. Scaling,
    activation, saturation and max reduction happen in the store datapath.
    Large images use one gathered window at a time, bounding SRAM requirements.
    """
    if attrs.get("output_dtype", "i8") != "i8":
        raise ValueError("hardware pooling readout requires i8 output")
    hi, wi = [int(x) for x in attrs["pool_in_dims"]]
    ph, pw = [int(x) for x in attrs["pool_size"]]
    sh, sw = [int(x) for x in attrs["pool_stride"]]
    pt, pl, pb, pr = [int(x) for x in attrs.get("pool_padding", [0, 0, 0, 0])]
    if min(hi, wi, ph, pw, sh, sw) <= 0 or min(pt, pl, pb, pr) < 0:
        raise ValueError("invalid pooling geometry")
    if rows % (hi * wi):
        raise ValueError("pooling input rows must contain complete images")
    if max(ph, pw) > 3:
        raise ValueError("pooling window exceeds the target's 2-bit native size")
    if any([pt, pl, pb, pr]) and attrs.get("pool_pad_value") != 0:
        raise ValueError("target pooling padding requires explicit zero pad value")
    ho = (hi + pt + pb - ph) // sh + 1
    wo = (wi + pl + pr - pw) // sw + 1
    if min(ho, wo) <= 0:
        raise ValueError("pooling output is empty")
    batches = rows // (hi * wi)
    relu = "relu" in attrs.get("epilogue", [])
    scales = attrs.get("acc_scale_by_column")
    if scales:
        if len(scales) != cols:
            raise ValueError("pool scale vector length does not match channels")
        if len(set(float(x) for x in scales)) == 1:
            attrs = dict(attrs, acc_scale=float(scales[0]))
            scales = None
    fast = (ph == pw and sh == sw and sh <= 3 and hi * wi <= 1024
            and max(hi, wi, ho, wo) <= 255 and max(pt, pl) <= 3)
    size = max(ph, pw)
    zero = None if fast else b.alloc(DIM, "i32")
    # Full-width mvin to accumulator; a zero shrink bit selects i32 inputs.
    b.rocc(0, (ONE << 32) | (DIM << 16) | (1 << 8) | 1,
           cols * 4)

    def image(batch):
        input_base = b.op("mul", batch, hi * wi * cols)
        output_base = b.op("mul", batch, ho * wo * cols)

        def channels(column, nr, scale):
            if fast:
                def load_rows(row, count):
                    idx = b.op("add", input_base,
                               b.op("add", b.op("mul", row, cols), column))
                    pointer = b.gep(src_i32, idx, "i32")
                    local = b.op("add", ACC, row)
                    b.rocc(2, _pointer(b, pointer), _descriptor(b, local, nr, count))
                _segments(b, 0, hi * wi, load_rows)
                b.fence()
                _configure_store(b, cols, scale, relu, ph, sh, hi, wi, ho, wo, pt, pl)
                target = b.gep(out, b.op("add", output_base, column))
                b.rocc(3, _pointer(b, target), _descriptor(b, ACC, nr))
                b.fence()
            else:
                def output_pixel(pixel):
                    oy = b.op("udiv", pixel, wo)
                    ox = b.op("urem", pixel, wo)
                    iy0 = b.op("sub", b.op("mul", oy, sh), pt)
                    ix0 = b.op("sub", b.op("mul", ox, sw), pl)

                    def patch_pixel(tap):
                        # Repeated taps extend rectangular windows to squares
                        # without introducing any new value into their maximum.
                        ky = b.op("urem", b.op("udiv", tap, size), ph)
                        kx = b.op("urem", b.op("urem", tap, size), pw)
                        iy = b.op("add", iy0, ky)
                        ix = b.op("add", ix0, kx)
                        valid = b.op("and", b.cmp("ult", iy, hi),
                                     b.cmp("ult", ix, wi), "i1")

                        local = b.op("add", ACC, tap)
                        descriptor = _descriptor(b, local, nr, 1)

                        def load_valid():
                            source = b.op("add", input_base,
                                          b.op("mul", b.op("add", b.op("mul", iy, wi), ix), cols))
                            idx = b.op("add", source, column)
                            pointer = b.gep(src_i32, idx, "i32")
                            b.rocc(2, _pointer(b, pointer), descriptor)

                        def load_zero():
                            b.rocc(2, _pointer(b, zero), descriptor)

                        b.branch_if(valid, load_valid, load_zero)
                    b.loop(0, size * size, 1, patch_pixel)
                    b.fence()
                    _configure_store(b, cols, scale, relu, size, 1, size, size, 1, 1)
                    dst = b.op("add", output_base, b.op("add", b.op("mul", pixel, cols), column))
                    b.rocc(3, _pointer(b, b.gep(out, dst)), _descriptor(b, ACC, nr))
                    b.fence()
                b.loop(0, ho * wo, 1, output_pixel)

        if scales:
            # Equal adjacent scale runs share a static configuration and keep
            # their complete channel tiles as loops instead of unrolling N.
            first = 0
            while first < cols:
                last = first + 1
                scale = float_bits(scales[first])
                while last < cols and float_bits(scales[last]) == scale:
                    last += 1
                _segments(b, first, last,
                          lambda column, count: channels(column, count, scale))
                first = last
        else:
            scale = float_bits(attrs.get("acc_scale", 1.0))
            _segments(b, 0, cols,
                      lambda column, count: channels(column, count, scale))

    b.loop(0, batches, 1, image)
    b.fence()
