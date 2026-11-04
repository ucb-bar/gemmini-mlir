"""Opt-in transport of Merlin's named preparation fusion to stock host MLIR.

Only the already selected host island enters here. The library inserts a second
elementwise fusion after named-op generalization, before buffer ownership. The
existing broadcast guard must be selected to avoid multiplying scalar producer
work. No arithmetic identities or persistent storage are introduced.
"""
from pathlib import Path
import hashlib

from merlin.llvmlower import impr_features
from host_storage_lifetimes import run_stock_pipeline as _incumbent


def selection():
    from merlin.compile.tuning_knobs import knob
    return knob('host_source_pipeline', default='off',
                values=('off', 'fuse_named_preparations'), root=Path(__file__).resolve().parent)


def _run_selected_pipeline(source, lowered, passes, *, tool, invoke, print_module, report=None):
    selected = selection()
    if selected == 'off':
        return _incumbent(source, lowered, passes, tool=tool, invoke=invoke,
                          print_module=print_module, report=report)
    from host_policy import selection as fusion_selection
    if fusion_selection() != 'guarded_fusion':
        raise ValueError('named preparation fusion requires the existing guarded_fusion policy')
    anchors = [i for i, p in enumerate(passes) if p == 'func.func(linalg-generalize-named-ops)']
    bufferizers = [i for i, p in enumerate(passes) if p.startswith('one-shot-bufferize{')]
    if len(anchors) != 1 or len(bufferizers) != 1 or anchors[0] >= bufferizers[0]:
        raise ValueError('named preparation fusion requires generalization before unique bufferization')
    source = Path(source)
    original = source.read_bytes()
    chosen = impr_features._fuse_after_generalize(list(passes))
    _incumbent(source, lowered, chosen, tool=tool, invoke=invoke,
               print_module=print_module, report=report)
    if source.read_bytes() != original:
        raise ValueError('named preparation fusion modified original source')
    if report is not None:
        report['source_pipeline'] = {
            'schema': 'merlin.stock_host_source_pipeline.v1', 'selection': selected,
            'mechanism': 'merlin.llvmlower.impr_features._fuse_after_generalize',
            'library_sha256': hashlib.sha256(Path(impr_features.__file__).read_bytes()).hexdigest(),
            'source_sha256': hashlib.sha256(original).hexdigest(),
            'lowered_sha256': hashlib.sha256(Path(lowered).read_bytes()).hexdigest(),
            'actual_pass_order': chosen,
            'scope': 'selected host island; tensor fusion before allocation/ownership',
            'timing': 'UNKNOWN',
        }


def run_stock_pipeline(source, lowered, passes, *, tool, invoke, print_module, report=None):
    """Consume original pure tensor islands before the unchanged stock CPU path.

    This entry has no borrowed-call, scalar-observation or mask proof carrier
    to rebind. Functions containing calls retain their original route; the
    admitted pure tensor recurrence keeps the same operation and axis order.
    """
    from merlin.compile.tuning_knobs import knob
    mode = knob('host_window_reduction', default='off', values=('off', 'ordered_ssa'),
                root=Path(__file__).resolve().parent)
    if mode == 'off':
        return _run_selected_pipeline(source, lowered, passes, tool=tool, invoke=invoke,
                                      print_module=print_module, report=report)
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.window_reduction_ssa import prepare_owner_source
    source = Path(source)
    original = source.read_bytes()
    module = parse_mlir_text(original.decode())
    # No effect carrier is available at this target-owned source boundary.
    # Pure tensor islands need no call rebinding. Keep any call-bearing source
    # unchanged so original native/borrowed ownership cannot be inferred here.
    has_calls = any(op.name in ('func.call', 'func.call_indirect') for op in module.walk())
    from tempfile import mkdtemp
    work = Path(mkdtemp(prefix='window-reduction-', dir=source.parent))
    selected, selection_report, products = prepare_owner_source(
        original.decode(), work, bound_effects=has_calls)
    actual = products['window_reduction_selected'] if selection_report['applied'] else source
    result = _run_selected_pipeline(actual, lowered, passes, tool=tool, invoke=invoke,
                                   print_module=print_module, report=report)
    if source.read_bytes() != original or actual.read_bytes().decode() != selected:
        raise ValueError('original or selected host reduction source changed during lowering')
    if report is not None:
        report['window_reduction'] = {
            **selection_report,
            'selected_source_consumed': True,
            'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'target_boundary_refusal': ('call-bearing source has no effect transport at this entry'
                                        if has_calls else None),
            'lowered_sha256': hashlib.sha256(Path(lowered).read_bytes()).hexdigest(),
            'products': {k: {'sha256': hashlib.sha256(Path(v).read_bytes()).hexdigest()}
                         for k, v in products.items()},
            'scope': 'already selected original pure tensor host island through stock CPU emitter',
            'owner_complete_execution': 'not_established',
        }
    return result
