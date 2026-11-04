"""Summarize public source/census work without interpreting functional ticks as time."""
import importlib.util
import hashlib
import json
from pathlib import Path
import re

WS = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('model_summary', WS / '.agents/skills/inspect-model/scripts/model_summary.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def main():
    result = {
        'schema': 'authored.public_model_work_map.v1',
        'scope': 'Public source declarations and latest available endpoint kernel replay report per model; no whole-chain elapsed timing.',
        'timing': 'UNKNOWN: functional replay ticks are not elapsed-cycle measurements or modeled reservations.',
        'captured_graphs': [], 'endpoint_programs': [],
    }
    for case in sorted((WS / 'model').glob('*/capsule.yaml')):
        data = summary.summarize(case.parent, WS)
        source = case.parent / 'capsule.interface.mlir'
        result['captured_graphs'].append({
            'name': data['name'], 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'numeric_policy': data.get('numeric_policy'),
            'quantization': data.get('quantization'),
            'declared_placement': data.get('placement'),
            'operator_groups_by_macs': data.get('operator_cases', []),
            'frontend_op_counts': data.get('frontend_ops'),
            'coverage_declaration': data.get('coverage'),
            'host_time_cycles': None,
        })
    latest = {}
    for file in sorted((WS / 'model_reports').glob('*.json'), key=lambda p: p.stat().st_mtime_ns):
        report = json.loads(file.read_text())
        if isinstance(report.get('report'), dict) and 'programs' in report['report']:
            latest[report['model']] = (file, report)
    for _, (file, report) in sorted(latest.items()):
        for program in report['report']['programs']:
            groups = []
            for kernel in program['kernels']:
                form = kernel['form']
                match = re.search(r'matmul M=(\d+) K=(\d+) N=(\d+)', form)
                occurrences = len(kernel.get('source_region_ids', []))
                macs = None
                if match:
                    m, k, n = map(int, match.groups())
                    macs = m * k * n
                else:
                    conv = re.search(r'conv2d (\d+)x(\d+)/s(\d+) Cin=(\d+) Cout=(\d+) on (\d+)x(\d+)(?: pad \[([\d, ]+)\])?', form)
                    if conv:
                        kh, kw, stride, ci, co, ih, iw = map(int, conv.groups()[:7])
                        padding = [int(x) for x in conv.group(8).split(',')] if conv.group(8) else [0]*4
                        # Public endpoint forms have batch1, dilation1, symmetric
                        # stride; four explicit padding values retain both axes.
                        oh = (ih + padding[0] + padding[1] - kh)//stride + 1
                        ow = (iw + padding[2] + padding[3] - kw)//stride + 1
                        macs = oh * ow * ci * co * kh * kw
                groups.append({
                    'symbol': kernel['symbol'], 'form': form,
                    'source_region_occurrences': occurrences,
                    'occurrence_basis': 'distinct source_region_ids in public endpoint census; not replay count',
                    'interface_sha256': kernel['interface_sha256'],
                    'shape_sha256': kernel['shape_sha256'],
                    'macs_per_occurrence': macs,
                    'static_macs': None if macs is None else macs * occurrences,
                    'replays': kernel.get('cases'), 'exact_replays': kernel.get('exact'),
                    'complete_region_elapsed_cycles': None,
                })
            groups.sort(key=lambda g: -(g['static_macs'] or 0))
            result['endpoint_programs'].append({
                'model': report['model'], 'program': program['program'],
                'compiler_source_sha256': report['source_sha256'],
                'report_path': str(file.relative_to(WS)),
                'report_sha256': hashlib.sha256(file.read_bytes()).hexdigest(),
                'report_status': report.get('status'),
                'gates': report['report'].get('gates'),
                'census': program.get('census'), 'kernel_groups': groups,
                'host_regions': 'UNKNOWN: endpoint kernel replay census does not enumerate complete host graph execution',
                'preparation_and_intermediate_lifetimes': 'UNKNOWN outside independently compiled connected public regions',
                'whole_program_elapsed_cycles': None,
            })
    (WS / 'submission/docs/model_work_map.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
