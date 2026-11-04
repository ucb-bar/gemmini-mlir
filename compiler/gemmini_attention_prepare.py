"""Bounded loop tiling around the shared exact transpose emitter's body.

Only iteration order changes. The common emitter still owns every byte load,
destination index and store; this adapter contains no payload arithmetic.
"""
from merlin.llvmlower.host_epilogue import HostEpilogueEmitter
from types import SimpleNamespace


class BlockedTransposeBuilder:
    def __init__(self, builder, rows, columns, *, tile_rows=16, tile_columns=16):
        if any(type(v) is not int or v <= 0 for v in
               (rows, columns, tile_rows, tile_columns)):
            raise ValueError('transpose tiling requires positive static extents')
        self.builder=builder
        self.rows,self.columns=rows,columns
        self.tile_rows,self.tile_columns=tile_rows,tile_columns
        self.depth=0
        self.column_interval=None

    def __getattr__(self,name):
        return getattr(self.builder,name)

    def _tiles(self,extent,tile,body):
        full,tail=divmod(extent,tile)
        if full:
            self.builder.for_loop(0,full,1,
                lambda block:body(self.builder.mul_i(block,self.builder.const(tile)),tile),
                retain_loop=True)
        if tail:body(self.builder.const(full*tile),tail)

    def for_range(self,count,body):
        if self.depth==1:
            if count!=self.columns or self.column_interval is None:
                raise ValueError('shared transpose inner loop contract changed')
            base,width=self.column_interval
            self.builder.for_loop(0,width,1,
                lambda j:body(self.builder.add_i(base,j)),retain_loop=True)
            return
        if self.depth!=0 or count!=self.rows:
            raise ValueError('shared transpose outer loop contract changed')
        def row_tile(row_base,height):
            def column_tile(column_base,width):
                self.column_interval=(column_base,width)
                self.depth=1
                self.builder.for_loop(0,height,1,
                    lambda i:body(self.builder.add_i(row_base,i)),retain_loop=True)
                self.depth=0
                self.column_interval=None
            self._tiles(self.columns,self.tile_columns,column_tile)
        self._tiles(self.rows,self.tile_rows,row_tile)


def emit_blocked_transpose(builder,source,destination,rows,columns):
    if source==destination:
        raise ValueError('transpose requires disjoint private destination')
    adapter=BlockedTransposeBuilder(builder,rows,columns)
    pointers={'source':source,'destination':destination}
    emitter=HostEpilogueEmitter(adapter,pointers.__getitem__,None,
        row_alignment=1,operand_dtype='i8',pool_floor=-128)
    emitter.emit_transpose(SimpleNamespace(bufs=('source','destination'),
        attrs={'rows':rows,'cols':columns,'dtype':'i8'}))
    return dict(schema='merlin.attention_blocked_preparation.v1',
        source_shape=[rows,columns],destination_shape=[columns,rows],
        tile_shape=[adapter.tile_rows,adapter.tile_columns],
        loop_order=['row_tile','column_tile','row','column'],
        payload_owner='merlin.llvmlower.host_epilogue.HostEpilogueEmitter.emit_transpose',
        exact_byte_permutation=True,copy_bytes=rows*columns,
        timing=None,target_execution_verified=False)
