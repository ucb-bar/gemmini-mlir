"""Checked composition of two source-qualified fresh spatial consumers.

This is not a larger ConvolutionReadoutEpochLaw. Each original member retains
its issued bounds. The outer composition rebases only private local cells and
checks its actual effect vocabulary; it permits no new external observer,
configuration effect, RHS write, array stay, or source reduction permutation.
Finite selected queues backpressure both members; their pending bounds do not
change. The caller must execute the returned real group ownership drain.
"""
from dataclasses import dataclass
from .convolution_output_groups import plan_convolution_output_group
from .affine_gather_runs import ConvolutionGatherRuns
from .window_transfer import ConvolutionReadoutEpochLaw, ConvolutionInputRetentionLaw, ConvolutionOutputTile


@dataclass(frozen=True)
class DisjointConvolutionMGroup:
    source: object
    gather: object
    group: object
    readout_law: object
    compute_law: object
    independence: object

    def __post_init__(self):
        p=self.source.plan;w=p.window;g=self.group
        g.__post_init__();self.gather.__post_init__()
        self.readout_law.__post_init__();self.compute_law.__post_init__()
        self.independence.__post_init__()
        b,h,width,ci=w.input_shape;pt,pl,pb,pr=w.padding
        expected=ConvolutionGatherRuns((b,h+pt+pb,width+pl+pr,ci),
            (b,*w.output_spatial),w.stride,w.dilation,w.kernel,p.primitive)
        if (g!=plan_convolution_output_group(g.panel,b*w.output_spatial[0]*w.output_spatial[1],max_row_blocks=2)
                or g.row_blocks!=2 or self.gather!=expected or p.row_tile!=p.primitive
                or g.panel.tile!=p.primitive or p.lhs_row!=0
                or g.panel.accumulator_capacity_rows!=p.accumulator_rows-p.accumulator_row
                or self.independence.max_live_tiles<g.row_blocks*g.panel.column_blocks
                or type(self.readout_law)is not ConvolutionReadoutEpochLaw
                or type(self.independence)is not ConvolutionInputRetentionLaw
                or not all((self.independence.input_preserved,self.independence.independent_outputs,
                    self.independence.independent_output_rows,self.independence.no_intermediate_observers,
                    self.independence.immutable_mapped_disjoint_buffers))):
            raise ValueError('two-M composition requires exact source gather, private group and independent original outputs')
        for law in (self.readout_law,self.compute_law,self.independence):
            if (law.source_identity!=self.source.identity
                    or law.hardware_identity!=self.compute_law.hardware_identity
                    or law.numerical_policy_identity!=self.compute_law.numerical_policy_identity):
                raise ValueError('two-M composition changed source or selected target contracts')
        # No local writer for one member can reach the other's primitive LHS
        # or ACC window. Full-K RHS is immutable throughout the group.
        if g.lhs_row(0)+p.primitive>g.lhs_row(1):
            raise ValueError('two-M composition has overlapping private LHS cells')

    def require(self,counts,widths):
        self.__post_init__();p=self.source.plan;g=self.group
        if (type(counts)is not tuple or not 0<len(counts)<=2
                or any(type(r)is not int or not 0<r<=p.primitive for r in counts)
                or any(r!=p.primitive for r in counts[:-1])):
            raise ValueError('two-M composition requires complete original tiles and at most one final tail')
        spans=[]
        for rb,rows in enumerate(counts):
            tiles=tuple(ConvolutionOutputTile(n*p.primitive,c,
                g.accumulator_row(rb,n,active_column_blocks=len(widths)))for n,c in enumerate(widths))
            # The original per-member work limits are unchanged. This outer
            # composition proves separation, not a larger issued/pending epoch.
            self.readout_law.require(p,rows,tiles)
            self.compute_law.require(max(2,g.panel.reduction_tiles*len(widths)))
            for t in tiles:
                lo=t.accumulator_row;hi=lo+rows
                if any(lo<b and a<hi for a,b in spans):raise ValueError('two-M ACC members overlap')
                spans.append((lo,hi))

    def source_work_bounds(self,counts,widths):
        """Closed source work, not additional physical queue capacity.

        One original readout command publishes each active N tile of each M
        member. Each has the original source reduction_tiles product pairs.
        """
        self.require(counts,widths)
        issued=sum(len(widths) for _ in counts)
        return dict(max_issued_readouts=issued,
            max_compute_pairs=self.group.panel.reduction_tiles*issued,
            qualification='derived closed source work; not pending capacity or a new target assertion')

    def member_issue(self,builder,issue,*,row_block,rows,widths):
        """Rebase actual unchanged canonical member effects into its checked map.

        The caller is the already source-qualified original fresh-M consumer.
        Only its exact terminal effect vocabulary is accepted; configuration and
        ownership drains remain outside this callback. All source addresses,
        masks, K coordinates and scalar store states pass through unchanged.
        """
        if not 0<=row_block<2:raise ValueError('unknown two-M member')
        self.require((rows,),widths)
        g=self.group;p=self.source.plan
        schemas={
            'load_input':{'zero','first_element','local_row','rows','columns','row_stride_elements'},
            'initialize_bias':{'output_row','output_column','rows','columns','accumulator_row'},
            'store':{'output_row','output_column','rows','columns','accumulator_row'},
            'compute':{'output_row','output_column','rows','columns','accumulator_row','lhs_row','rhs_row','depth','first_reduction','bias','fresh_weights'},
        }
        def lower(kind,fields):
            if kind not in schemas or set(fields)!=schemas[kind]:
                raise ValueError('unqualified effect inside two-M composition: '+kind)
            f=dict(fields)
            if kind=='load_input':
                # Relative row expression is emitted by the original exact
                # gather. Translation alone preserves all its masks/bounds.
                f['local_row']=builder.op('add',f['local_row'],g.lhs_row(row_block))
            else:
                if type(f['accumulator_row'])is not int:
                    raise ValueError('two-M consumer requires original static ACC cells')
                n=f['accumulator_row']//p.primitive
                if (f['accumulator_row']!=g.accumulator_row(0,n,active_column_blocks=len(widths))
                        or f['rows']!=rows or f['columns']!=widths[n]):
                    raise ValueError('two-M consumer changed its original canonical output cell')
                f['accumulator_row']=g.accumulator_row(row_block,n,active_column_blocks=len(widths))
                if kind=='compute':
                    if (f['lhs_row']!=g.lhs_row(0)or f['fresh_weights']is not True
                            or f['bias']!=p.bias):
                        raise ValueError('two-M composition changed canonical LHS or introduced array retention')
                    f['lhs_row']=g.lhs_row(row_block)
            issue(kind,f)
        return lower

    def emit_group(self,builder,issue,complete,*,first,counts,widths,initialize,member):
        """Consume original member callbacks and emit the real ownership drain.

        Both callbacks see canonical original tile cells. Initializers are
        completely issued before either producer. The exact checked rebasing
        wrapper is the only effect outlet; configuration/RHS mutation cannot
        enter this group. Each member must publish each original N cell once.
        """
        bounds=self.source_work_bounds(counts,widths)
        closed_readouts=0
        def effect_out(rb,rows,allowed):
            checked=self.member_issue(builder,issue,row_block=rb,rows=rows,widths=widths)
            stores=[]
            def emit(kind,f):
                if kind not in allowed:raise ValueError('effect crosses a two-M source stage: '+kind)
                if kind in ('store','initialize_bias'):
                    if f['accumulator_row'] in stores:raise ValueError('two-M member repeats an output cell')
                    stores.append(f['accumulator_row'])
                checked(kind,f)
            return emit,stores
        if self.source.plan.bias:
            for rb,rows in enumerate(counts):
                outlet,cells=effect_out(rb,rows,{'initialize_bias'})
                initialize(builder.op('add',first,rb*self.group.panel.tile),rows,outlet)
                if len(cells)!=len(widths):raise ValueError('two-M initializer is not closed')
            complete('spatial_initializer_ready',{})
        for rb,rows in enumerate(counts):
            outlet,cells=effect_out(rb,rows,{'load_input','compute','store'})
            member(builder.op('add',first,rb*self.group.panel.tile),rows,outlet)
            if len(cells)!=len(widths):raise ValueError('two-M member is not a closed original output group')
            closed_readouts+=len(cells)
        if closed_readouts!=bounds['max_issued_readouts']:
            raise ValueError('two-M composition did not close its derived total readout bound')
        # Actual terminal adapter supplies its real full-device completion.
        complete('spatial_output_last_use',{})
