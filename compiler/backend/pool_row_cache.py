"""Exact standalone i8 pooling from bounded linear accumulator row epochs.

Adjacent pooled Y positions share their source rows. A private linear epoch
loads their union once from original source rows or exact immutable i8 padding,
then consumes each window through the existing pool traversal. Real
load and store completion fences own every epoch; no circular storage, rolling
source assumption, host payload operation or asynchronous admission is used.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from merlin.compile.scheduling.llvm_program import FnBuilderControlAdapter

from . import isa
from .pool_stripes import PoolStripes, _plan as plan_stripes


@dataclass(frozen=True)
class PoolRowCache:
    stripes: object
    output_rows: int

    def projection(self):
        """Prove the actual linear source and ACC mapping for every window.

        The horizontal map is the incumbent's exact checked partition. Along
        Y, union row r denotes source (block_y*SH-top+r); output j reads r=
        j*SH+ky. These are the original (block_y+j)*SH-top+ky coordinates,
        including out-of-bounds rows, which use the original declared pad.
        The finite local domain is bounded by physical accumulator capacity.
        """
        p = self.stripes
        if type(p) is not PoolStripes:
            raise ValueError('pool row cache requires the exact original source stripe mapping')
        base = p.projection()
        s = p.source
        ph, _ = s.window
        sh, _ = s.stride
        oh, _ = s.output_hw
        if (type(self.output_rows) is not int or not 2 <= self.output_rows <= oh
                or not 1 <= sh < ph):
            raise ValueError('row retention requires overlapping original Y windows')
        local_rows = (self.output_rows - 1) * sh + ph
        max_acc = 0
        for x, outputs in p.stripes:
            width, _ = p.horizontal(x, outputs)
            if local_rows * width > isa.ACC_ROWS:
                raise ValueError('linear pool row union exceeds private ACC capacity')
            addresses = set()
            for j in range(self.output_rows):
                for ky in range(ph):
                    local_y = j * sh + ky
                    if not 0 <= local_y < local_rows:
                        raise ValueError('pool row cache read escapes its loaded union')
                    # Coefficients of block_y and constant source offsets
                    # must match; arbitrary padding then receives the same
                    # in-bounds test and the exact same i8 literal.
                    if (sh, local_y - s.padding[0]) != (sh, j * sh + ky - s.padding[0]):
                        raise ValueError('pool row cache changed an original source coordinate')
                    addresses.add(local_y)
            if addresses != set(range(local_rows)):
                raise ValueError('pool row union has a missing or unused source row')
            max_acc = max(max_acc, local_rows * width)
        rectangular = sh == s.stride[1] and local_rows <= 255 and self.output_rows <= 255
        full, tail = divmod(oh, self.output_rows)
        loaded_rows = full * local_rows + ((tail - 1) * sh + ph if tail else 0)
        # Full and peeled epochs partition [0,OH); batch epochs are separate.
        if full * self.output_rows + tail != oh:
            raise ValueError('pool row cache does not partition original output Y')
        receipt = dict(schema='backend.pool_row_cache.v1',
            source_sha256=base['source_sha256'], source=base,
            output_rows_per_epoch=self.output_rows, union_rows_per_full_epoch=local_rows,
            private_accumulator_rows=max_acc,
            input_rows_per_stripe_and_channel=loaded_rows,
            incumbent_input_rows_per_stripe_and_channel=oh * ph,
            output_partition='separate image/channel/stripe epochs partition original Y with peeled tail',
            source_projection='(block_y+j)*SH-top+ky = block_y*SH-top+(j*SH+ky)',
            local_projection='pool row j/ky reads union row j*SH+ky via shifted ACC base or equal-stride rectangle',
            rectangular_readout=rectangular,
            physical_mapping='linear ACC rows; incumbent explicit disjoint halo/data/halo DMA intervals',
            completion='real load drain before store/config; real store drain before epoch reuse',
            hardware_implicit_padding=False, host_payload_operations=0, timing='UNKNOWN')
        receipt['schedule_sha256'] = hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()
        return receipt


def geometry_candidates(p):
    """Construct real source-exact stripe/Y plans, without a timing selector.

    The finite alternatives change only the checked horizontal partition and
    capacity-bounded retained rows. Every choice passes the same source proof
    and the same terminal encoder; observations decide promotion externally.
    """
    s = p.source
    ph, pw = s.window
    sh, sw = s.stride
    if not 1 <= sh < ph:
        return {}
    widths = {
        'incumbent': p.stripe,
        'dma_row': min(p.stripe, max(1, (isa.DIM - pw) // sw + 1)),
        'half_width': max(1, (p.stripe + 1) // 2),
    }
    candidates = {}
    for name, stripe in widths.items():
        alternative = PoolStripes(s, p.batches, p.channels, stripe)
        max_width = max(alternative.horizontal(x, outputs)[0]
                        for x, outputs in alternative.stripes)
        max_rows = isa.ACC_ROWS // max_width
        if sh == sw:
            max_rows = min(max_rows, 255)
        outputs = min(s.output_hw[0], (max_rows - ph) // sh + 1)
        if outputs < 2:
            continue
        try:
            plan = PoolRowCache(alternative, outputs)
            receipt = plan.projection()
            # Count actual interval-sized transfers, with the original row
            # bounds distinguishing vertical halo rows from valid rows.
            load_calls = loaded_cells = epochs = stores = 0
            oh = s.output_hw[0]
            full, tail = divmod(oh, outputs)
            period = outputs * sh
            overlap = ph - sh
            total_rows = full * (period + overlap)
            valid_rows = max(0, min(s.input_hw[0], full * period - s.padding[0]))
            # Full epochs are a disjoint period plus at most PH-SH rows.
            # Count those shared rows by arithmetic progression rather than
            # expanding the original OH-sized runtime loop at compilation.
            for k in range(overlap):
                first = max(1, -(-(s.padding[0] - k) // period))
                last = min(full, (s.input_hw[0] - 1 + s.padding[0] - k) // period)
                valid_rows += max(0, last - first + 1)
            if tail:
                union = (tail - 1) * sh + ph
                start = full * period - s.padding[0]
                total_rows += union
                valid_rows += max(0, min(s.input_hw[0], start + union) - max(0, start))
            for x, count in alternative.stripes:
                width, intervals = alternative.horizontal(x, count)
                stripe_epochs = full + bool(tail)
                epochs += stripe_epochs
                stores += stripe_epochs if receipt['rectangular_readout'] else oh
                valid_calls = sum((stop - start + isa.DIM - 1) // isa.DIM
                                  for start, stop, _ in intervals)
                pad_calls = (width + isa.DIM - 1) // isa.DIM
                load_calls += valid_rows * valid_calls + (total_rows - valid_rows) * pad_calls
                loaded_cells += total_rows * width
            receipt.update(geometry=name, source_stripe_outputs=stripe,
                screening=dict(basis='exact emitted loop-trip/interval census per image/channel tile',
                    load_calls=load_calls, loaded_spatial_cells=loaded_cells,
                    store_calls=stores, load_store_drains=epochs + stores,
                    timing='UNKNOWN; census is not a latency observation'))
            candidates[name] = plan, receipt
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            continue
    return candidates


def geometry_pass_family():
    """Library exploration hook for the actual checked source-plan choices.

    This produces schedule plans, not emitted ISA or timing evidence. The
    ordinary lowering consumes exactly the same geometry_candidates result.
    """
    from merlin.transforms.pass_exploration import PassFamily, ExplorationRefused
    from merlin.transforms.pipeline_policy import RegisteredPass
    from merlin.transforms.pipeline_state import PassSpec
    def bind(options):
        def action(session):
            choice = geometry_candidates(session.module).get(options['geometry'])
            if choice is None:
                raise ExplorationRefused('source/capacity does not admit retained row geometry',
                                         layer='construction')
            return {'pool_geometry': choice}
        return RegisteredPass(PassSpec('pool_row_geometry', 0,
            produces=('pool_geometry',), mutates_ir=False, ir_level='source_pool'), action)
    return PassFamily('pool_row_geometry', bind,
        (('geometry', ('incumbent', 'dma_row', 'half_width')),))


def _plan(ctx, src, dst, attrs, *, require_full_rows=False):
    base = plan_stripes(ctx, src, dst, attrs)
    if base is None:
        return None
    from pathlib import Path
    from merlin.compile.tuning_knobs import knob
    choice = knob('POOL_ROW_GEOMETRY', default='incumbent',
        values=('incumbent', 'dma_row', 'half_width'),
        root=Path(__file__).resolve().parent.parent)
    # Auto is a source/capacity admission, not a timing selector. It uses the
    # measured incumbent geometry and refuses before any emission unless all
    # original output Y positions share one bounded linear union.
    candidate = geometry_candidates(base[0]).get('incumbent' if require_full_rows else choice)
    if (candidate is not None and require_full_rows
            and candidate[0].output_rows != candidate[0].stripes.source.output_hw[0]):
        return None
    return candidate


def can_lower(ctx, src, dst, attrs):
    return _plan(ctx, src, dst, attrs) is not None


def lower(ctx, src, dst, attrs, *, require_full_rows=False):
    """Return False before mutation when the exact row-union route is refused."""
    prepared = _plan(ctx, src, dst, attrs, require_full_rows=require_full_rows)
    if prepared is None:
        return False
    plan, receipt = prepared
    p = plan.stripes
    s = p.source
    h, w = s.input_hw
    oh, ow = s.output_hw
    ph = s.window[0]
    sh, sw = s.stride
    top = s.padding[0]
    n = p.channels
    control = FnBuilderControlAdapter(ctx.fb)
    op = control.op
    ctx.fence()
    pad_name = None
    if any(s.padding):
        pad_name = '_pool_row_cache_padding_' + str(len(ctx.tensors))
        while pad_name in ctx.tensors:
            pad_name += '_'
        ctx.constant(pad_name, [isa.DIM], 'i8', [s.pad_value] * isa.DIM)
    ctx.emit(*isa.config_ld(n, channel=0, shrink=True))
    if pad_name:
        ctx.emit(*isa.config_ld(0, channel=1, shrink=True))

    def chunks(start, stop, body):
        full, tail = divmod(stop - start, isa.DIM)
        if full:
            control.loop(0, full, 1,
                lambda i: body(op('add', start, op('mul', i, isa.DIM)), isa.DIM))
        if tail:
            body(start + full * isa.DIM, tail)

    for x, outputs in p.stripes:
        width, intervals = p.horizontal(x, outputs)
        ctx.fence()
        if not receipt['rectangular_readout']:
            ctx.emit(*isa.config_st(n, scale=1.0, activation=0, pool_stride=sw,
                pool_size=ph, pool_out_dim=ow, porows=1, pocols=outputs,
                orows=ph, ocols=width))

        def image_body(image):
            def panel(channel, columns):
                def epoch(y, count):
                    union_rows = (count - 1) * sh + ph
                    base_y = op('sub', op('mul', y, sh), top)
                    def load_row(local_y):
                        iy = op('add', base_y, local_y)
                        acc_y = op('mul', local_y, width)
                        def pad_row(start, stop):
                            if start == stop:
                                return
                            if pad_name is None:
                                raise ValueError('row cache reached an unproved padding path')
                            chunks(start, stop, lambda local, rows: ctx.emit(*isa.mvin(
                                ctx.ptr(pad_name), isa.acc_addr(op('add', acc_y, local), fb=ctx.fb),
                                rows, columns, channel=1, fb=ctx.fb)))
                        def valid_row():
                            for start, stop, source_x in intervals:
                                if start == stop:
                                    continue
                                if source_x is None:
                                    pad_row(start, stop)
                                else:
                                    def load(local, rows, start=start, source_x=source_x):
                                        ix = op('add', source_x, op('sub', local, start))
                                        pixel = op('add', op('mul', op('add', op('mul', image, h), iy), w), ix)
                                        offset = op('add', op('mul', pixel, n), channel)
                                        ctx.emit(*isa.mvin(ctx.ptr(src, offset),
                                            isa.acc_addr(op('add', acc_y, local), fb=ctx.fb),
                                            rows, columns, fb=ctx.fb))
                                    chunks(start, stop, load)
                        if any(s.padding):
                            control.branch(control.cmp('ult', iy, h), valid_row,
                                lambda: pad_row(0, width))
                        else:
                            valid_row()
                    control.loop(0, union_rows, 1, load_row)
                    ctx.fence()
                    def store(j):
                        output_y = op('add', y, j)
                        pixel = op('add', op('mul', op('add', op('mul', image, oh), output_y), ow), x)
                        offset = op('add', op('mul', pixel, n), channel)
                        acc = op('mul', op('mul', j, sh), width)
                        ctx.emit(*isa.mvout(ctx.ptr(dst, offset), isa.acc_addr(acc, fb=ctx.fb),
                            0, columns, fb=ctx.fb))
                        ctx.fence()
                    if receipt['rectangular_readout']:
                        # The selected pool traversal has one shared spatial
                        # stride. When SH=SW its j/ky row is exactly j*SH+ky,
                        # so one rectangular store consumes the loaded union.
                        ctx.emit(*isa.config_st(n, scale=1.0, activation=0,
                            pool_stride=sw, pool_size=ph, pool_out_dim=ow,
                            porows=count, pocols=outputs, orows=union_rows, ocols=width))
                        pixel = op('add', op('mul', op('add', op('mul', image, oh), y), ow), x)
                        offset = op('add', op('mul', pixel, n), channel)
                        ctx.emit(*isa.mvout(ctx.ptr(dst, offset), isa.acc_addr(0), 0, columns, fb=ctx.fb))
                        ctx.fence()
                    else:
                        control.loop(0, count, 1, store)
                full, tail = divmod(oh, plan.output_rows)
                if full:
                    control.loop(0, full, 1, lambda i: epoch(op('mul', i, plan.output_rows), plan.output_rows))
                if tail:
                    epoch(full * plan.output_rows, tail)
            full, tail = divmod(n, isa.DIM)
            if full:
                control.loop(0, full, 1, lambda i: panel(op('mul', i, isa.DIM), isa.DIM))
            if tail:
                panel(full * isa.DIM, tail)
        control.loop(0, p.batches, 1, image_body)
    ctx.fence()
    if any(receipt.get(key) != value for key, value in plan.projection().items()):
        raise ValueError('row cache source mapping changed during emission')
    receipt = dict(receipt, input=src, output=dst,
        target_encoder='backend.isa', payload_arithmetic='accelerator_store_maxpool')
    ctx.metadata.setdefault('pool_row_cache', []).append(receipt)
    return receipt
