"""NHWC convolution with DMA-gathered patches and Gemmini contraction."""

from contraction import (matmul, _tiles, _matrix, _address, _mvin_config,
                         _mvout_config, DIM, ACC, ACC_ADD, ACC_FULL,
                         GARBAGE, ONE)


def conv2d(b, ifm, w, out, input_shape, attrs, tensors, ptrs):
    """Gather each patch into SRAM using DMA; no CPU tensor loads or stores."""
    if attrs.get("layout", "nhwc") != "nhwc":
        raise ValueError("conv2d requires NHWC layout")
    if len(input_shape) != 4:
        raise ValueError("conv2d requires a rank-four NHWC input")
    batch, height, width, channels = [int(x) for x in input_shape]
    kh, kw, ci, co = [int(x) for x in attrs["kernel"]]
    sh, sw = [int(x) for x in attrs.get("stride", [1, 1])]
    dh, dw = [int(x) for x in attrs.get("dilation", [1, 1])]
    pt, pl, pb, pr = [int(x) for x in attrs.get("padding", [0, 0, 0, 0])]
    if channels != ci:
        raise ValueError("conv2d kernel/input channel mismatch")
    if min(batch, height, width, ci, co, kh, kw, sh, sw, dh, dw) <= 0:
        raise ValueError("conv2d dimensions and strides must be positive")
    if min(pt, pl, pb, pr) < 0:
        raise ValueError("conv2d padding must be nonnegative")
    ho = (height + pt + pb - dh * (kh - 1) - 1) // sh + 1
    wo = (width + pl + pr - dw * (kw - 1) - 1) // sw + 1
    if min(ho, wo) <= 0:
        raise ValueError("conv2d has an empty output extent")
    epilogue = attrs.get("epilogue", [])
    if "maxpool" in epilogue:
        raise ValueError("maxpool must use the pooling-aware convolution wrapper")
    rows = batch * ho * wo
    depth = kh * kw * ci
    dtype = attrs.get("output_dtype", "i32")
    if dtype not in ("i8", "i32"):
        raise ValueError("conv2d readout must be i8 or i32")
    scales = attrs.get("acc_scale_by_column")
    if scales and len(scales) != co:
        raise ValueError("conv2d scale vector must match output channels")

    # No gather is necessary when the input is already the required matrix.
    # A nonuniform readout uses the direct conv path to avoid CPU scattering.
    uniform = not scales or len(set(float(x) for x in scales)) == 1
    if kh == kw == sh == sw == 1 and not any([pt, pl, pb, pr]) and uniform:
        matmul(b, ifm, w, out, rows, depth, co, attrs, tensors, ptrs)
        return

    bias_name = attrs.get("bias")
    bias = ptrs[bias_name] if bias_name else None
    zero_acc = b.alloc(DIM, "i32") if bias is None else None
    zero_input = b.alloc(DIM, "i8")
    relu = "relu" in epilogue
    b.rocc(0, 4 | (1 << 16) | (ONE << 32), 1 << 48)
    _mvin_config(b, 0, ci)
    _mvin_config(b, 1, co)
    _mvin_config(b, 2, 0)

    def output_rows(mi, mr):
        def output_columns(nj, nr, scale):
            seed = b.gep(bias, nj, "i32") if bias else zero_acc
            b.fence()
            b.rocc(14, _address(b, seed), _matrix(b, ACC, mr, nr))

            def kernel_y(ky):
                def kernel_x(kx):
                    kernel_base = b.op("mul", b.op("add", b.op("mul", ky, kw), kx), ci)

                    def input_channels(kk, kr):
                        def gather_row(ri):
                            row = b.op("add", mi, ri)
                            image = b.op("udiv", row, ho * wo)
                            y = b.op("urem", b.op("udiv", row, wo), ho)
                            x = b.op("urem", row, wo)
                            iy = b.op("add", b.op("sub", b.op("mul", y, sh), pt),
                                      b.op("mul", ky, dh))
                            ix = b.op("add", b.op("sub", b.op("mul", x, sw), pl),
                                      b.op("mul", kx, dw))
                            valid = b.op("and", b.cmp("ult", iy, height),
                                         b.cmp("ult", ix, width), "i1")
                            descriptor = b.op("or", ri, (kr << 32) | (1 << 48))

                            def load_pixel():
                                spatial = b.op("add", b.op("mul", iy, width), ix)
                                image_offset = b.op("mul", image, height * width * ci)
                                channel_offset = b.op("add", b.op("mul", spatial, ci), kk)
                                source = b.gep(ifm, b.op("add", image_offset, channel_offset))
                                b.rocc(2, _address(b, source), descriptor)

                            def load_zero():
                                b.rocc(2, _address(b, zero_input), descriptor)

                            b.branch_if(valid, load_pixel, load_zero)

                        b.loop(0, mr, 1, gather_row)
                        weight_row = b.op("add", kernel_base, kk)
                        weight_idx = b.op("add", b.op("mul", weight_row, co), nj)
                        wp = b.gep(w, weight_idx)
                        b.rocc(1, _address(b, wp), _matrix(b, DIM, kr, nr))
                        b.rocc(6, _matrix(b, DIM, kr, nr), _matrix(b, ACC_ADD, mr, nr))
                        b.rocc(4, _matrix(b, 0, mr, kr), _matrix(b, GARBAGE, mr, nr))
                        b.fence()

                    _tiles(b, ci, input_channels)

                b.loop(0, kw, 1, kernel_x)

            b.loop(0, kh, 1, kernel_y)
            output_idx = b.op("add", b.op("mul", mi, co), nj)
            if dtype == "i32":
                _mvout_config(b, co * 4, 1.0, False)
                dst = b.gep(out, output_idx, "i32")
                b.rocc(3, _address(b, dst), _matrix(b, ACC_FULL, mr, nr))
            else:
                _mvout_config(b, co, scale, relu)
                dst = b.gep(out, output_idx)
                b.rocc(3, _address(b, dst), _matrix(b, ACC, mr, nr))
            b.fence()

        if scales and dtype == "i8":
            # Literal-scale runs retain static readout commands. They also
            # avoid a CPU scatter or a payload scale-table load.
            first = 0
            while first < co:
                scale = float(scales[first])
                last = first + 1
                while last < co and float(scales[last]) == scale:
                    last += 1
                run = last - first
                start = first
                _tiles(b, run, lambda offset, count: output_columns(
                    b.op("add", start, offset), count, scale))
                first = last
        else:
            _tiles(b, co, lambda column, count: output_columns(
                column, count, attrs.get("acc_scale", 1.0)))

    _tiles(b, rows, output_rows)
    b.fence()
