#!/usr/bin/env python3
"""Audit ordinary terminal products without executing unqualified cleanup.

This evaluator deliberately submits no broker request: the missing selected
execution law prevents APPLIED_EMITTED_CHANGE. Counts are diagnostics only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parent
sys.path[:0] = [str(PACKAGE), str(PACKAGE/'compiler_library')]
from gemmini_conv_evaluate import manifest_emit, tree_identity
from gemmini_command_cleanup import audit_command_cleanup, cleanup_commands


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def parse_module(source):
    from xdsl.context import Context
    from xdsl.dialects import builtin, llvm
    from xdsl.parser import Parser
    ctx = Context(allow_unregistered=True)
    ctx.load_dialect(builtin.Builtin)
    ctx.load_dialect(llvm.LLVM)
    return Parser(ctx, source).parse_module()


def evaluate(*, submission, sources, output):
    from merlin.xdsl_dialects._common import text
    submission, output = Path(submission).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(submission):
        raise ValueError('use a fresh output outside the compiler')
    output.mkdir(parents=True)
    frozen_hash = tree_identity(submission)
    rows = []
    members = json.loads((PACKAGE.parent/'PUBLIC_CAPABILITIES.json').read_text())['members']
    for number, path in enumerate(map(Path, sources)):
        path = path.resolve()
        case = path.parent.name if path.name == 'capsule.interface.mlir' else path.stem
        source = path.read_bytes()
        expected = members[case]['source_analysis']['sha256']
        if _sha(source) != expected:
            raise ValueError('source differs from public declaration: '+case)
        payload, product = manifest_emit(submission, path, output/str(number))
        module = parse_module(payload.decode())
        before = text(module)
        audit = audit_command_cleanup(module)
        cleanup_commands(module, enabled=False)
        if text(module) != before:
            raise ValueError('disabled audit altered actual product')
        try:
            cleanup_commands(module, enabled=True)
        except ValueError as error:
            refusal = str(error)
        else:
            raise ValueError('unqualified cleanup unexpectedly executed')
        if text(module) != before:
            raise ValueError('refusal altered actual product')
        rows.append(dict(case=case, source_path=str(path), source_sha256=_sha(source),
                         ordinary_emission=product, audit=audit, enabled_refusal=refusal,
                         disabled_byte_identity=True, refusal_byte_identity=True,
                         actual_product_changed=False, target_requests_submitted=0))
    if tree_identity(submission) != frozen_hash:
        raise ValueError('compiler changed during audit; discard products')
    report = dict(schema='gemmini.command_cleanup_evaluation.v1',
                  compiler=str(submission), compiler_sha256=frozen_hash, cases=rows,
                  adapter_sha256=_sha((PACKAGE/'gemmini_command_cleanup.py').read_bytes()),
                  evaluator_sha256=_sha(Path(__file__).read_bytes()),
                  verdict='REFUSED_MISSING_TARGET_LAW', numerics='UNCHANGED_PRODUCT',
                  cycle_ranking='NOT_REQUESTED', default_promotion=False)
    (output/'report.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submission', required=True)
    parser.add_argument('--source', action='append', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = evaluate(submission=args.submission, sources=args.source, output=args.output)
    print(json.dumps(dict(verdict=result['verdict'],
        cases=[dict(case=r['case'], proposals=r['audit']['proposed_omissions'],
                    program_sha256=r['ordinary_emission']['program_sha256']) for r in result['cases']]), indent=2))
