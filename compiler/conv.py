"""Layout-only lowerings for contractions; arithmetic is delegated to Gemmini."""
from __future__ import annotations
import re


def _ident(unique):
    value = unique() if callable(unique) else unique
    return 'cv_' + re.sub(r'[^A-Za-z0-9_]', '_', str(value))


def _leaf(name, cb):
    seen = set()
    while name not in cb['tensors']:
        if name in seen:
            raise ValueError('cyclic resident handle')
        seen.add(name)
        pack = next((c for c in cb['commands'] if c['opcode'] == 'RES_PACK' and c['operands']['dst'] == name), None)
        if pack is None:
            raise ValueError(f'unknown tensor/resident handle {name}')
        name = pack['operands']['src']
    return name


def _ptr(name, cb, names):
    return names.get(name, names[_leaf(name, cb)])


def _f32(value):
    text = repr(float(value))
    return text + 'f'


def _epilogue(attrs, cb, names, n):
    ep = attrs.get('epilogue', [])
    if 'requant' in ep:
        raise ValueError('integer requant requires round-half-up; Gemmini mesh shift rounds ties to even')
    allowed = {'bias', 'bias_add', 'acc_scale', 'relu', 'maxpool'}
    if any(x not in allowed for x in ep):
        raise ValueError('unsupported contraction epilogue')
    if 'maxpool' in ep and ep[-1] != 'maxpool':
        raise ValueError('maxpool must be the final contraction epilogue stage')
    bias = '0'
    if 'bias' in ep or 'bias_add' in ep:
        if 'bias' not in attrs:
            raise ValueError('bias stage without bias tensor')
        tensor = cb['tensors'][attrs['bias']]
        if tensor['shape'] != [n] or tensor['dtype'] != 'i32':
            raise ValueError('bias must be an i32 output-column vector')
        bias = '(const int*)' + _ptr(attrs['bias'], cb, names)
    scales = None
    scale = '1.0f'
    if 'acc_scale' in ep:
        if 'acc_scale_by_column' in attrs:
            scales = attrs['acc_scale_by_column']
            if len(scales) != n:
                raise ValueError('column scale vector width mismatch')
        elif 'acc_scale' in attrs:
            scale = _f32(attrs['acc_scale'])
        else:
            raise ValueError('acc_scale stage without multiplier')
    dtype = attrs.get('output_dtype', 'i32')
    if dtype not in ('i8', 'i32'):
        raise ValueError('Gemmini contraction supports i8/i32 output')
    return (1 if dtype == 'i8' else 4), scale, int('relu' in ep), bias, scales


def _product(a, b, y, m, k, n, attrs, cb, names, tag, m_capacity=None):
    width, scale, relu, bias, scales = _epilogue(attrs, cb, names, n)
    import json
    from pathlib import Path
    settings_path = Path(__file__).resolve().parent / 'settings.json'
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    rows_allocated = m if m_capacity is None else m_capacity
    if scales is None or len(set(float(x) for x in scales)) == 1:
        if scales is not None:
            scale = _f32(scales[0])
        if settings.get('matmul_schedule') == 'static' or (
            settings.get('source_matmul_schedule') == 'static' and cb.get('params',{}).get('global_program_plan')):
            if settings.get('static_reuse') == 'ab_resident':
                from device_static_abresident import emit_static_product
            elif settings.get('static_reuse') == 'a_resident':
                from device_static_aresident import emit_static_product
            elif settings.get('static_reuse') == 'combined_weight_reuse':
                from device_static_reuse import emit_static_product
            elif settings.get('static_reuse') == 'panel_weight_reuse':
                from device_static_panelreuse import emit_static_product
            elif settings.get('static_reuse') == 'weight_reuse':
                from device_static_weightreuse import emit_static_product
            elif settings.get('static_reuse') == 'b_resident':
                from device_static_bresident import emit_static_product
            else:
                from device_static_resident import emit_static_product
            return emit_static_product(a,b,y,m,k,n,width,scale,relu,bias,tag)
        return [], [f'gm_matmul((const signed char*)({a}), (const signed char*)({b}), (void*)({y}), {m}, {k}, {n}, {width}, {scale}, {relu}, {bias});']
    if settings.get('column_schedule') == 'dma':
        from device_columns_dma import emit_column_product
    else:
        from device_columns import emit_column_product
    column_product = emit_column_product(a,b,y,m,k,n,attrs,cb,names,tag,m_capacity,
        force=settings.get('column_schedule') == 'transpose')
    if column_product is not None:
        return column_product
    # Hardware scaling is per readout. Gather equal-scale columns into a matrix,
    # run each scale group independently, then scatter bytes without arithmetic.
    groups = {}
    for j, s in enumerate(scales):
        groups.setdefault(float(s), []).append(j)
    maxcols = max(len(v) for v in groups.values())
    glob = [f'static signed char {tag}_w[{k*maxcols}] __attribute__((aligned(64)));',
            f'static unsigned char {tag}_out[{rows_allocated*maxcols*width}] __attribute__((aligned(64)));']
    if bias != '0':
        glob.append(f'static int {tag}_bias[{maxcols}] __attribute__((aligned(64)));')
    body = []
    for group, (s, columns) in enumerate(groups.items()):
        count = len(columns)
        glob.append(f'static const int {tag}_cols_{group}[{count}] = {{{",".join(map(str, columns))}}};')
        body += [f'for (int i=0; i<{k}; ++i) for (int j=0; j<{count}; ++j) {tag}_w[i*{count}+j] = ((const signed char*)({b}))[i*{n}+{tag}_cols_{group}[j]];']
        bp = '0'
        if bias != '0':
            body.append(f'for (int j=0; j<{count}; ++j) {tag}_bias[j] = ({bias})[{tag}_cols_{group}[j]];')
            bp = tag + '_bias'
        body.append(f'gm_matmul((const signed char*)({a}), {tag}_w, {tag}_out, {m}, {k}, {count}, {width}, {_f32(s)}, {relu}, {bp});')
        body.append(f'for (int i=0; i<{m}; ++i) for (int j=0; j<{count}; ++j) for (int z=0; z<{width}; ++z) ((unsigned char*)({y}))[(i*{n}+{tag}_cols_{group}[j])*{width}+z] = {tag}_out[(i*{count}+j)*{width}+z];')
    return glob, body


def emit_conv(cmd, cb, names, unique):
    """Emit NHWC im2col with static storage, followed by a device matmul."""
    import json
    from pathlib import Path
    settings_path = Path(__file__).resolve().parent / 'settings.json'
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    if settings.get('conv_schedule') in ('direct', 'direct_small'):
        from device_conv import emit_direct_conv
        output = cb['tensors'][cmd['operands']['dst']]
        direct = emit_direct_conv(cmd, cb, names, unique) if (
            settings.get('conv_schedule') == 'direct' or output['shape'][0] <= 16) else None
        if direct is not None:
            return direct
    op, attrs = cmd['operands'], cmd.get('attributes', {})
    src = _leaf(op['ifm'], cb)
    weight = _leaf(op['weight'], cb)
    shape = cb['tensors'][src]['shape']
    if len(shape) != 4 or cb['tensors'][src]['dtype'] != 'i8' or cb['tensors'][weight]['dtype'] != 'i8':
        raise ValueError('CONV2D requires rank-four NHWC i8 input and i8 weight')
    if attrs.get('layout', 'nhwc') != 'nhwc' or 'groups' in attrs:
        raise ValueError('CONV2D requires ungrouped NHWC layout')
    batch, h, w, ci = shape
    kh, kw, kci, co = attrs['kernel']
    sh, sw = attrs.get('stride', [1, 1])
    dh, dw = attrs.get('dilation', [1, 1])
    pt, pl, pb, pr = attrs.get('padding', [0, 0, 0, 0])
    if min(kh, kw, sh, sw, dh, dw, ci, co) <= 0 or min(pt, pl, pb, pr) < 0 or ci != kci:
        raise ValueError('invalid CONV2D geometry')
    oh = (h + pt + pb - (dh*(kh-1)+1)) // sh + 1
    ow = (w + pl + pr - (dw*(kw-1)+1)) // sw + 1
    m, k = batch*oh*ow, kh*kw*ci
    if min(oh, ow) <= 0 or cb['tensors'][weight]['shape'] != [k, co]:
        raise ValueError('CONV2D weight/output geometry mismatch')
    if 'maxpool' not in attrs.get('epilogue', []) and cb['tensors'][op['dst']]['shape'] != [m, co]:
        raise ValueError('CONV2D output shape mismatch')
    tag = _ident(unique)
    inp, rhs, out = _ptr(src, cb, names), _ptr(weight, cb, names), _ptr(op['dst'], cb, names)
    glob, body = [], []
    direct = kh == kw == 1 and sh == sw == dh == dw == 1 and pt == pl == pb == pr == 0
    # Keep im2col within a 64 KiB row block (or one 16-row tile for
    # exceptionally long reductions). The final block carries its actual M.
    block_rows = min(m, max(16, (65536 // k // 16) * 16))
    if direct:
        matrix = inp
    else:
        matrix = tag + '_im2col'
        glob.append(f'static signed char {matrix}[{block_rows*k}] __attribute__((aligned(64)));')
    pool = 'maxpool' in attrs.get('epilogue', [])
    product_out = out
    if pool:
        if attrs.get('output_dtype', 'i32') != 'i8':
            raise ValueError('device maxpool currently requires i8 output')
        if attrs.get('pool_in_dims') != [oh, ow]:
            raise ValueError('CONV2D pool_in_dims disagrees with convolution geometry')
        product_out = tag + '_prepool'
        glob.append(f'static signed char {product_out}[{m*co}] __attribute__((aligned(64)));')
    if direct:
        pg, code = _product(matrix, rhs, product_out, m, k, co, attrs, cb, names, tag)
        glob += pg
        body += code
    else:
        row_base = tag + '_base'
        zero = tag + '_zero'
        glob.append(f'static const signed char {zero} = 0;')
        output_bytes = 1 if attrs.get('output_dtype', 'i32') == 'i8' else 4
        full_end = (m // block_rows) * block_rows
        tail_rows = m - full_end
        # Partition packing blocks into a bounded periodic set when batching
        # spatial DMA rows. Each phase has a known output-x coordinate, and
        # every contraction retains literal full/tail row counts.
        from math import lcm
        period = lcm(block_rows,ow)
        phased = settings.get('im2col_schedule') == 'dma_batched' and period//block_rows <= 8
        blocks = []
        if phased:
            cycle_end = m//period*period
            if cycle_end:
                for offset in range(0,period,block_rows):
                    opening = (f'for(int {row_base}=0; {row_base}<{cycle_end}; {row_base}+={period}) {{\n'
                               if offset == 0 else '') + '{'
                    closing = '}' + ('\n}' if offset+block_rows == period else '')
                    blocks.append((block_rows,f'({row_base}+{offset})',tag+f'_phase_{offset}',opening,closing,offset%ow))
            for offset in range(cycle_end,m,block_rows):
                blocks.append((min(block_rows,m-offset),str(offset),tag+f'_remainder_{offset}','{','}',offset%ow))
        else:
            if full_end:
                blocks.append((block_rows,row_base,tag+'_full',
                    f'for(int {row_base}=0; {row_base}<{full_end}; {row_base}+={block_rows}) {{','}',None))
            if tail_rows:
                blocks.append((tail_rows,str(full_end),tag+'_tail','{','}',None))
        for count, base, block_tag, opening, closing, row_mod_ow in blocks:
            block_out = f'((unsigned char*)({product_out})+(gm_u64)({base})*{co*output_bytes})'
            pg, code = _product(matrix, rhs, block_out, count, k, co, attrs, cb, names,
                                block_tag, m_capacity=count)
            glob += pg
            body.append(opening)
            if settings.get('im2col_schedule') in ('dma','dma_batched'):
                if phased:
                    from device_im2col_batched import emit_im2col
                else:
                    from device_im2col import emit_im2col
                kwargs = {'row_mod_ow':row_mod_ow} if phased else {}
                pg, packing = emit_im2col(matrix,inp,shape,attrs['kernel'],[sh,sw],[dh,dw],
                    [pt,pl,pb,pr],oh,ow,base,count,block_tag,**kwargs)
                glob += pg
                body += packing
            else:
                body.append(f'''for(int row=0; row<{count}; ++row) {{
      int flat={base}+row;
      int bn=flat/{oh*ow}, oy=(flat/{ow})%{oh}, ox=flat%{ow};
      for(int ky=0;ky<{kh};++ky) for(int kx=0;kx<{kw};++kx) {{
        int iy=oy*{sh}-{pt}+ky*{dh}, ix=ox*{sw}-{pl}+kx*{dw};
        int pos=row*{k}+(ky*{kw}+kx)*{ci};
        gm_u64 valid=((unsigned)iy<{h}) & ((unsigned)ix<{w});
        for(int c=0;c<{ci};++c) {{
          gm_u64 offset=(((gm_u64)bn*{h}+(gm_u64)iy)*{w}+(gm_u64)ix)*{ci}+(gm_u64)c;
          gm_u64 address=((gm_u64)({inp})+offset)*valid+(gm_u64)&{zero}*(1-valid);
          {matrix}[pos+c]=*((const signed char*)address);
        }}
      }}
    }}''')
            body += code
            body.append(closing)
    if pool:
        pg, code = emit_pool(product_out, out, m, co, attrs, tag,
                             batch=batch, dst_shape=cb['tensors'][op['dst']]['shape'])
        glob += pg
        body += code
    return glob, body


def emit_pool(src_expr, dst_expr, m, n, attrs, unique, *, batch=1, dst_shape=None):
    """Reduce an already scaled i8 [M,N] tensor with the device maxpool path.

    The caller owns the unpooled source allocation. Returns (globals, body).
    ``pool_in_dims`` describes each batch's input spatial dimensions.
    """
    if attrs.get('output_dtype', 'i8') != 'i8':
        raise ValueError('device maxpool currently requires i8 output')
    ep = attrs.get('epilogue', ['maxpool'])
    if 'maxpool' in ep and ep[-1] != 'maxpool':
        raise ValueError('maxpool must be the final epilogue stage')
    oh, ow = attrs['pool_in_dims']
    if min(batch, oh, ow, n) <= 0 or m != batch*oh*ow:
        raise ValueError('pool_in_dims and batch do not match the unpooled matrix rows')
    tag = _ident(unique)
    glob, body = [], []
    product_out, out, co = src_expr, dst_expr, n
    ph, pw = attrs['pool_size']
    psh, psw = attrs['pool_stride']
    ppt, ppl, ppb, ppr = attrs.get('pool_padding', [0, 0, 0, 0])
    poh = (oh + ppt + ppb - ph) // psh + 1
    pow_ = (ow + ppl + ppr - pw) // psw + 1
    if min(ph, pw, psh, psw, poh, pow_) <= 0 or min(ppt, ppl, ppb, ppr) < 0:
        raise ValueError('invalid maxpool geometry')
    if dst_shape is not None and list(dst_shape) != [batch*poh*pow_, co]:
        raise ValueError('pooled output shape mismatch')
    if any((ppt, ppl, ppb, ppr)) and 'pool_pad_value' not in attrs:
        raise ValueError('padded pooling requires pool_pad_value')
    padvalue = max(-128, min(127, int(attrs.get('pool_pad_value', -128))))
    poolh, poolw = oh, ow
    if padvalue != -128 and any((ppt, ppl, ppb, ppr)):
        # A literal copy/pad realizes the declared identity value. Reduction
        # still runs on Gemmini, with no host comparisons of tensor values.
        padded = tag + '_poolpad'
        poolh, poolw = oh+ppt+ppb, ow+ppl+ppr
        glob.append(f'static signed char {padded}[{batch*poolh*poolw*co}] __attribute__((aligned(64)));')
        body.append(f'''for(int bn=0;bn<{batch};++bn) for(int py=0;py<{poolh};++py) for(int px=0;px<{poolw};++px) for(int c=0;c<{co};++c)
  {padded}[((bn*{poolh}+py)*{poolw}+px)*{co}+c]=(py>={ppt} && py<{ppt+oh} && px>={ppl} && px<{ppl+ow}) ? ((const signed char*)({product_out}))[((bn*{oh}+py-{ppt})*{ow}+px-{ppl})*{co}+c] : {padvalue};''')
        product_out = padded
        ppt, ppl = 0, 0
    import json
    from pathlib import Path
    settings_path = Path(__file__).resolve().parent / 'settings.json'
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    if settings.get('pool_schedule') == 'rectangles':
        from device_pool import emit_pool_kernel
        native = emit_pool_kernel(product_out,out,batch,poolh,poolw,co,poh,pow_,ph,pw,psh,psw,ppt,ppl,tag)
        if native is not None:
            return glob+native[0], body+native[1]
    body.append(f'gm_maxpool2d((const signed char*)({product_out}), (signed char*)({out}), {batch}, {poolh}, {poolw}, {co}, {poh}, {pow_}, {ph}, {pw}, {psh}, {psw}, {ppt}, {ppl});')
    return glob, body


def emit_attention(cmd, cb, names, unique):
    """QK explicitly transposes row-per-key storage; PV is a direct product."""
    op, attrs = cmd['operands'], cmd.get('attributes', {})
    qk = cmd['opcode'] == 'ATTENTION_QK'
    left, right = (op['q'], op['k']) if qk else (op['p'], op['v'])
    lspec, rspec = cb['tensors'][left], cb['tensors'][right]
    if lspec['dtype'] != 'i8' or rspec['dtype'] != 'i8' or len(lspec['shape']) != 2 or len(rspec['shape']) != 2:
        raise ValueError('attention contraction requires rank-two i8 operands')
    m, k = lspec['shape']
    n, rk = rspec['shape'] if qk else rspec['shape'][::-1]
    if rk != k or cb['tensors'][op['dst']]['shape'] != [m, n]:
        raise ValueError('attention contraction geometry mismatch')
    tag = _ident(unique)
    a, b, y = _ptr(left, cb, names), _ptr(right, cb, names), _ptr(op['dst'], cb, names)
    glob, body = [], []
    if qk:
        trans = tag + '_transpose'
        glob.append(f'static signed char {trans}[{k*n}] __attribute__((aligned(64)));')
        import json
        from pathlib import Path
        settings_file = Path(__file__).resolve().parent / 'settings.json'
        settings = json.loads(settings_file.read_text()) if settings_file.exists() else {}
        if settings.get('attention_transpose') == 'dma' and min(n, k) > 0:
            from device_transpose import emit_byte_transpose
            tg, tb = emit_byte_transpose(b, trans, n, k, tag + '_pack')
            glob.extend(tg)
            body.extend(tb)
        else:
            body.append(f'for (int j=0; j<{n}; ++j) for (int i=0; i<{k}; ++i) {trans}[i*{n}+j]=((const signed char*)({b}))[j*{k}+i];')
        b = trans
    pg, code = _product(a, b, y, m, k, n, attrs, cb, names, tag)
    return glob+pg, body+code
