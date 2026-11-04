"""Exact original-NHWC mapping for private horizontal-K pool producers."""
from merlin.compile.scheduling.horizontal_convolution import plan_horizontal_convolution_panels


def prepare(source):
    p = source.producer
    ci = p.window.input_shape[-1]
    if (p.row_tile != p.primitive or ci >= p.primitive or p.window.dilation[1] != 1
            or p.window.kernel[1]*ci > 4*p.primitive):
        return None
    result = plan_horizontal_convolution_panels(p, max_row_blocks=1,
        maximum_transfer_columns=4*p.primitive, private_rows_owned=True)
    if (result is None or result.reduction_tiles >= p.window.kernel[0]*p.window.kernel[1]
            or result.reduction_tiles > 16 or result.operand_rows > p.rhs_row):
        return None
    return result


def input_fragments(plan, builder, fields, transfer):
    """Map original padded coordinates to one real row span or zero source.

    The caller admits horizontally interior original output spans. Only Y may
    be padded; each local row is initialized once before its complete reduction.
    """
    w = plan.source.window
    batch, height, width, ci = w.input_shape
    pt, pl, pb, pr = w.padding
    virtual_width = width+pl+pr
    plane = (height+pt+pb)*virtual_width*ci
    first = fields['first_element']
    image = builder.op('udiv', first, plane)
    rem = builder.op('urem', first, plane)
    y = builder.op('udiv', rem, virtual_width*ci)
    x = builder.op('urem', rem, virtual_width*ci)
    valid = builder.op('and', builder.cmp('ule',pt,y), builder.cmp('ult',y,pt+height),'i1')
    def stage(zero):
        base = 0 if zero else builder.op('add', builder.op('mul',
            builder.op('add',builder.op('mul',image,height),builder.op('sub',y,pt)),width*ci),
            builder.op('sub',x,pl*ci))
        for chunk in range(plan.horizontal_chunks):
            offset = chunk*plan.source.primitive
            transfer(dict(zero=zero,
                first_element=0 if zero else builder.op('add',base,offset),
                row_stride_elements=fields['row_stride_elements'],
                local_row=builder.op('add',fields['local_row'],chunk*fields['block_stride_rows']),
                rows=fields['rows'], columns=min(plan.source.primitive,plan.horizontal_columns-offset)))
    builder.branch(valid, lambda:stage(False), lambda:stage(True))
