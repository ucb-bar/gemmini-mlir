"""Wide state0 scratchpad loads for one contiguous NHWC convolution tap.

Caller configures LD0 stride0/block_stride16, identity i8 load, and raw
scratchpad ST. A wide load places channel block c in local row16*c; each
16-column readout restores its original byte position in im2col DRAM.
"""
from xdsl.dialects import llvm
from xdsl.dialects.builtin import i8
from xdsl.ir import Block

from . import isa


def pack_tap(f, source, destination, zero, source_base, target_base,
             channels: int, valid=None):
    """Move exact i8 tap bytes; padding reads at most16 bytes from zero.

    Source/target bases are i64 indices. If valid is supplied, branch before
    issuing any source DMA. Valid channels use groups of at most64 bytes,
    padded channels retain individual16-byte zero loads. No host values move.
    """
    if type(channels) is not int or channels <= 0:
        raise ValueError('convolution tap channels must be a positive integer')

    def valid_channels():
        def group(channel, width):
            pointer = f.gep(source, f.add_i(source_base, channel), i8)
            isa.mvin(f, pointer, 0, width, 1, state=0)
            for offset in range(0, width, isa.DIM):
                index = f.add_i(target_base, f.add_i(channel, f.const(offset)))
                isa.mvout(f, f.gep(destination, index, i8), offset,
                          min(isa.DIM, width - offset), 1)
        full = channels // 64 * 64
        if full:
            f.for_loop(0, full, 64, lambda channel: group(channel, 64), retain_loop=True)
        if full < channels:
            group(f.const(full), channels - full)

    if valid is None:
        valid_channels()
        return
    inside, outside, done = Block(), Block(), Block()
    for block in (inside, outside, done):
        f.region.add_block(block)
    f.add(llvm.CondBrOp(valid, inside, [], outside, []))
    f.blk = inside
    valid_channels()
    f.add(llvm.BrOp(done))
    f.blk = outside

    def zero_channels(channel):
        width = f.smin(f.const(16), f.sub_i(f.const(channels), channel))
        isa.mvin(f, zero, 0, width, 1, state=0)
        target = f.gep(destination, f.add_i(target_base, channel), i8)
        isa.mvout(f, target, 0, width, 1)

    f.for_loop(0, channels, 16, zero_channels, retain_loop=True)
    f.add(llvm.BrOp(done))
    f.blk = done
