"""Bind an explicit tensor-view preparation choice in ordinary host lowering.

The source tree declares the optional choice. No source identity, runtime data,
device-call effects or arithmetic permissions select or admit a transformation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def prepare_selected_views(source, work):
    """Prepare current routed host tensor SSA and retain the actual products."""
    from merlin.compile.tuning_knobs import knob

    # This module lives at TREE/compiler_library/merlin/llvmlower. The exact
    # candidate tree supplies its declaration; an installed stock library with
    # no declaration keeps the default. Never search unrelated source trees.
    tree = Path(__file__).resolve().parents[3]
    mode = knob('HOST_TENSOR_VIEWS', default='off',
                values=('off', 'fold_slices', 'fold_slices_im2col', 'fold_slices_im2col_affine'), root=tree)
    if mode == 'off':
        return source, None
    from .source_tensor_views import rewrite_text as rewrite_views

    selected = source
    im2col = None
    if mode in ('fold_slices_im2col', 'fold_slices_im2col_affine'):
        from .source_im2col import rewrite_text as rewrite_im2col
        selected, im2col = rewrite_im2col(selected)
    pointwise = None
    if mode == 'fold_slices_im2col_affine':
        from .source_pointwise_views import rewrite_text as rewrite_pointwise
        selected, pointwise = rewrite_pointwise(selected)
    selected, report = rewrite_views(selected)
    if pointwise is not None:
        report['pointwise'] = pointwise
    if im2col is not None:
        report['im2col'] = im2col
    digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
    modules = ['source_view_preparation.py', 'source_tensor_views.py']
    if im2col is not None:
        modules.append('source_im2col.py')
    if pointwise is not None:
        modules.append('source_pointwise_views.py')
    implementations = [dict(path=str(Path(__file__).with_name(name)),
        sha256=hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest())
        for name in modules]
    report.update(mode=mode, source_sha256=digest(source),
                  output_sha256=digest(selected),
                  implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  implementations=implementations,
                  binding_stage='ordinary lower_model routed tensor source before stock lowering',
                  hardware_cycles='not_measured')
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    (work / 'model.tensor-views.mlir').write_text(selected, encoding='utf-8')
    (work / 'source-tensor-views.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return selected, report
