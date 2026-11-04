"""Finite small-geometry policy over the unchanged DMA epoch emitter.

Capacity/admission establish legality. Frozen completed gSIM observations below
motivate this narrow profitability class; neighboring shapes remain unmeasured.
Ordinary compilation never reads feedback or runs a simulator.
"""
from hashlib import sha256
from pathlib import Path

from merlin.transforms.pipeline_policy import (PipelinePolicy,PipelineVariant,
    PolicyFacts,PropertyPredicate,RegisteredPass,execute_policy)
from merlin.transforms.pipeline_state import PipelineContext,PipelineSession,PassSpec

# Offline equality checks against the full immutable epoch040 compiler products.
# These identify checked products, not source-selection predicates.
PRODUCT_EQUIVALENCE = (
    dict(prepool_shape=(1,7,5,13), llvm_bytes=119943,
         llvm_sha256='dc715ea7cc4c064a4cb95e63fb61ee212f1f58ec3d2d92fb4fc439481a657ecf'),
    dict(prepool_shape=(1,10,10,12), llvm_bytes=110088,
         llvm_sha256='a5f92d4f96f5d913eedc0c5b4161d0474cdfb8ec1b0072bf2417eb0cb6bb4f05'),
)

# Frozen observation provenance only, never source-name selection predicates.
OBSERVATIONS = ({'evidence_path': 'artifacts/experiments/pool_epochs_edge040/evidence.json',
  'evidence_sha256': '6021c679994e270fcc556ea1b6d284a55af0d70db5323047d43a4613094587df',
  'scope': 'measured gSIM whole original connected region',
  'before': {'request_id': '13_1793282321792600464_1b6a1f377a1fb180c9d2eebdb8b1114e',
             'source_sha256': '33709f0d3c24d6bb91dd6eaea573a20d323e1a81b7753bb12a4eaeafb90da3c8',
             'case': 'CH_conv_pool_edge_c3h13w9k7s2n13_p3s2',
             'barrier_cycles': 141350,
             'barrier_tier': 'L3',
             'numerical_pass': True,
             'barrier_timing_identity': {'engine': 'gsim',
                                         'simulator_path_sha256': '6667da986f860848236418725dfb2ee38c78476462c01afdf529a191f7f7d3fc',
                                         'simulator_sha256': '76ddbfdc5b3beadef6b6b37768a08522122327057b92d71d3539a9529e4ea651'},
             'structural_qualification': {'component': 'instruction_trace',
                                          'status': 'pass',
                                          'unavailable_requirements': [],
                                          'violations': []},
             'execution_digest': 'fbfa5db3778d7cc498e89154e04425e4703cd0e696728817f34751353a74dc45'},
  'after': {'request_id': '2_1793285939600965687_d2d98539f9b88e48f589aa10d2b66dbe',
            'source_sha256': '823c4a8415a849c10a42e96460aa849c3165b9a33dc7805dd5cbf49b1b1c2dd1',
            'case': 'CH_conv_pool_edge_c3h13w9k7s2n13_p3s2',
            'barrier_cycles': 87889,
            'barrier_tier': 'L3',
            'numerical_pass': True,
            'barrier_timing_identity': {'engine': 'gsim',
                                        'simulator_path_sha256': '6667da986f860848236418725dfb2ee38c78476462c01afdf529a191f7f7d3fc',
                                        'simulator_sha256': '76ddbfdc5b3beadef6b6b37768a08522122327057b92d71d3539a9529e4ea651'},
            'structural_qualification': {'component': 'instruction_trace',
                                         'status': 'pass',
                                         'unavailable_requirements': [],
                                         'violations': []},
            'execution_digest': '60b2d32c39c31feaa5b7bad3a50b53b431cda6ae138ad2bc43532f8473566f1a'}},
 {'evidence_path': 'artifacts/experiments/pool_epochs_residual040/evidence.json',
  'evidence_sha256': 'fba74cc94e7ee782ae7acb2e821e5b53b6bee4f38e02611684733c7a0dfbeea3',
  'scope': 'measured gSIM whole original connected region',
  'before': {'request_id': '14_1793282327165353655_d8bf78273d2251db2c25fbec58d9c595',
             'source_sha256': '33709f0d3c24d6bb91dd6eaea573a20d323e1a81b7753bb12a4eaeafb90da3c8',
             'case': 'CH_conv_pool_residual_cnn_c3h20w20k7s2n12_p3s2',
             'barrier_cycles': 327081,
             'barrier_tier': 'L3',
             'numerical_pass': True,
             'barrier_timing_identity': {'engine': 'gsim',
                                         'simulator_path_sha256': '6667da986f860848236418725dfb2ee38c78476462c01afdf529a191f7f7d3fc',
                                         'simulator_sha256': '76ddbfdc5b3beadef6b6b37768a08522122327057b92d71d3539a9529e4ea651'},
             'structural_qualification': {'component': 'instruction_trace',
                                          'status': 'pass',
                                          'unavailable_requirements': [],
                                          'violations': []},
             'execution_digest': '1b52fb5b74109d07ab3981462fafea9cbe06b37f69e18c0a98fb52e5feb40bab'},
  'after': {'request_id': '2_1793285939594039962_5a0560a4636d4bd4a29c6701aad660ea',
            'source_sha256': '823c4a8415a849c10a42e96460aa849c3165b9a33dc7805dd5cbf49b1b1c2dd1',
            'case': 'CH_conv_pool_residual_cnn_c3h20w20k7s2n12_p3s2',
            'barrier_cycles': 215673,
            'barrier_tier': 'L3',
            'numerical_pass': True,
            'barrier_timing_identity': {'engine': 'gsim',
                                        'simulator_path_sha256': '6667da986f860848236418725dfb2ee38c78476462c01afdf529a191f7f7d3fc',
                                        'simulator_sha256': '76ddbfdc5b3beadef6b6b37768a08522122327057b92d71d3539a9529e4ea651'},
            'structural_qualification': {'component': 'instruction_trace',
                                         'status': 'pass',
                                         'unavailable_requirements': [],
                                         'violations': []},
            'execution_digest': 'd8653d9fe91ab0946c24623b1f7a304692345068c395201e06bcc0e0b13b64ab'}})


def geometry(plan):
    from convolution import _capacities
    from conv_pool_panel import _dma_epoch_schedule
    edge,_,accumulator_rows=_capacities()
    w=plan.window;oh,ow=w.output_spatial;stages=plan.attrs.get('epilogue',())
    conditions=dict(
        single_output_channel_tile=0<plan.channels<=edge,
        bounded_input_channels=0<w.input_shape[-1]<=edge,
        measured_source_window_family=(w.kernel==(7,7) and w.stride==(2,2)
            and w.dilation==(1,1) and w.padding==(3,3,3,3)),
        measured_readout_family=bool(plan.bias) and list(stages)==['bias_add','acc_scale','relu','maxpool'],
        measured_pool_family=(tuple(plan.attrs.get('pool_size',()))==(3,3)
            and tuple(plan.attrs.get('pool_stride',()))==(2,2)
            and tuple(plan.attrs.get('pool_padding',()))==(1,1,1,1)
            and plan.attrs.get('pool_pad_value',-128)==-128),
        bounded_spatial_extents=oh<=edge and ow<=edge,
        complete_image_in_one_panel=plan.output_rows<=plan.rows,
    )
    admission=None;refusal=None
    if all(conditions.values()):
        try:
            _,admission=_dma_epoch_schedule(w,plan.output_rows,edge,accumulator_rows)
        except ValueError as error:
            refusal=str(error)
    footprint=admission['tile_rows']*admission['slots'] if admission else 0
    conditions['complete_image_within_acc_spatial_footprint']=0<plan.output_rows<=footprint
    conditions['single_capacity_phase']=admission is not None and admission['phases']==1
    return dict(conditions=conditions,eligible=all(conditions.values()),
        resources=dict(array_edge=edge,accumulator_rows=accumulator_rows,
            epoch_accumulator_rows=footprint,prepool_rows=plan.output_rows,
            panel_rows=plan.rows,output_channels=plan.channels,
            input_channels=w.input_shape[-1],prepool_height=oh,prepool_width=ow),
        legality_admission=admission,refusal=refusal,
        support='two completed same-provider gSIM whole-region pairs; same common emitter',
        unknown_coverage='Neighbor timing is unmeasured. Larger-class L2 numerics passed but completed timing is unavailable (unfinished raw ingress); larger images, multiple output channel tiles and other window/readout families retain incumbent.')


def select(plan,*,emit_route=None):
    """Execute a registered shared route, or return incumbent without effects.

    ``plan`` must come from conv_pool_panel.prepare; that routine remains the
    numerical/layout/pointer admission. This policy does not relax its guards.
    ``emit_route`` invokes the existing prepared DMA epoch/dense/pool emitter.
    """
    g=geometry(plan)
    identity=sha256(repr(plan).encode()).hexdigest()
    pin=sha256(Path(__file__).with_name('facts.json').read_bytes()).hexdigest()
    context=PipelineContext(hardware=pin,resources=pin,numerical_policy='original-Q1-before-signed-max')
    session=PipelineSession(plan,context=context,ir_level='source-window-panel',
        fingerprint=lambda p:sha256(repr(p).encode()).hexdigest(),verify_ir=lambda p:p.window.__post_init__())
    facts=PolicyFacts(session.stamp(),capabilities=(('small_dma_epoch_class',g['eligible']),),
        resources=tuple(sorted(g['resources'].items())))
    policy=PipelinePolicy((PipelineVariant('dma_epochs',('dma_epochs',),
        (PropertyPredicate('capability','small_dma_epoch_class'),)),
        PipelineVariant('incumbent',('incumbent',))),'incumbent')
    def action(name):
        return lambda _:dict(pool_panel_selection=dict(route=name,
            actual=emit_route() if name=='dma_epochs' and emit_route is not None else None))
    registry={name:RegisteredPass(PassSpec(name,0,produces=('pool_panel_selection',),
        mutates_ir=False,ir_level='source-window-panel'),action(name))
        for name in ('dma_epochs','incumbent')}
    result=execute_policy(session,policy,registry,facts,output='pool_panel_selection')
    return dict(mode='dma_epochs_small_geometry',selected_route=result.artifact['route'],
        actual=result.artifact['actual'],emitted=result.artifact['actual'] is not None,
        source_geometry_sha256=identity,geometry=g,policy=result.receipt,
        observations=OBSERVATIONS,product_equivalence=PRODUCT_EQUIVALENCE,
        qualification='finite authored geometry policy; two measured witnesses only; original observation identities retained; no strict extrapolated performance certification')
