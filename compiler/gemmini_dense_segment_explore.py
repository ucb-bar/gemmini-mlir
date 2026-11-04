#!/usr/bin/env python3
"""Explore explicit N/K plans through the shared parallel schedule runner.

Each of three immutable packages must emit the same source using its requested
N/K choice. Checked schedules reach the ordinary terminal; manifest equality is
required. Numerical and cycle observations belong to separately pinned ordinary
self-check results, never to this source-declaration-only adapter.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from gemmini_dense_rhs_parallel import SourceBoundAdapter, digest, tree_identity


class SegmentedAdapter(SourceBoundAdapter):
    def __init__(self, *args, alternatives, **kwargs):
        super().__init__(*args, **kwargs)
        self.alternatives = alternatives

    def binding(self, shape, caps):
        from gemmini_dense import plan_dense, geometry, FACTS_PATH
        from gemmini_dense_longk_experiment import plan_longk_output
        from gemmini_dense_longk_wide import plan_wide
        from gemmini_dense_scalar import plan_fused_dense
        from gemmini_frontend import normalize_source
        from gemmini_readout import source_scales, _relu
        from merlin.transforms.pipeline_state import PipelineContext

        records = [r for r in self.sources if tuple(r['shape']) == tuple(shape)]
        if len(records) != 1:
            raise ValueError('requires one original source per geometry')
        record = records[0]
        source = Path(record['path']).read_text()
        model = normalize_source(source)
        products = [o for o in model['operations'] if o['category'] == 'product']
        readouts = [o for o in model['operations'] if o['category'] == 'readout']
        if len(products) != 1 or len(readouts) != 1:
            raise ValueError('requires one original dense product and readout')
        if products[0]['opcode'] not in ('MATMUL', 'MATMUL_RESIDENT'):
            raise ValueError('requires an original rank-two dense source')
        attrs = readouts[0]['attributes']
        dtype = attrs.get('output_dtype', 'i32')
        readout = json.dumps(dict(dtype=dtype, scale=float(source_scales(attrs)),
                                  relu=_relu(attrs)), sort_keys=True)
        base, adapter, problem = plan_dense(*shape, readout_identity=readout)
        candidates, owners = {}, {}

        def construct(name):
            if name not in candidates:
                option = self.alternatives[name]
                columns = option['columns']
                planner = (plan_wide if columns in (48, 64) else
                           plan_fused_dense if dtype in ('i8', 'int8') else
                           plan_longk_output)
                result, proof = planner(base, adapter, problem, geometry(), FACTS_PATH,
                    column_blocks=columns, k_tile_limit=option['k_tiles'],
                    attrs=attrs, output_dtype=dtype)
                result.revalidate()
                candidates[name] = result, proof
                owners[id(result)] = name
            return candidates[name][0]

        original = construct('original')

        def transform(result, branch):
            if result is not original:
                raise ValueError('branch lost its original source owner')
            return original if branch == 'identical' else construct(branch)

        def emit(result):
            from gemmini_dense import consume_checked_dense
            from gemmini_backend import compile_source
            name = owners[id(result)]
            with consume_checked_dense(result, adapter, problem, mode='panel_epochs',
                                       readout_identity=readout) as state:
                _, lowered = compile_source(source)
            payload = (lowered+'\n').encode()
            command = [sys.executable, str(self.packages[name]/'gemmini-opt'),
                '--convert-iface-to-gemmini', '--emit-target-artifact', record['path']]
            done = subprocess.run(command, capture_output=True, check=True)
            if done.stdout != payload or state['consumed'] != 1:
                raise ValueError('checked artifact differs from ordinary manifest: '+name)
            manifest = self.output/(record['capsule']+'-'+name+'.manifest.llvm.mlir')
            manifest.write_bytes(payload)
            row = dict(capsule=record['capsule'], shape=shape, branch=name,
                source_sha256=record['source_sha256'],
                original_effect_sha256=original.kernel.digest(),
                effect_sha256=result.kernel.digest(), program_sha256=digest(payload),
                manifest_path=str(manifest), manifest_command=command,
                manifest_byte_equal=True, consumed=state['consumed'],
                compiler_tree_sha256=self.package_pins[name],
                option=self.alternatives[name], admission=candidates[name][1])
            self.programs[(record['source_sha256'], name)] = row
            return payload.decode(), row

        binding = dict(source=original, transform=transform, emit=emit,
            shapes=((shape[0], shape[1]), (shape[1], shape[2])),
            context=PipelineContext(hardware=digest(FACTS_PATH.read_bytes()),
                resources=digest(asdict(geometry()[1])),
                numerical_policy=adapter.numerical_policy_identity,
                execution='ordinary-selfcheck:spike:scalability',
                library=tree_identity(self.packages['rhs_ready_wide']/'compiler_library')),
            record=record, model=model)
        self.bindings[tuple(shape)] = binding
        return binding


def alternative(text):
    """PACKAGE:COLUMNS:K_TILES; K is capacity, 4, 8 or 16."""
    try:
        package, columns, k = text.rsplit(':', 2)
        columns = int(columns)
        k = 'capacity' if k == 'capacity' else int(k)
        if columns not in (8, 16, 32, 48, 64) or k not in ('capacity', 4, 8, 16):
            raise ValueError()
    except ValueError as error:
        raise argparse.ArgumentTypeError('expected PACKAGE:{8,16,32,48,64}:{capacity,4,8,16}') from error
    return Path(package).resolve(), dict(columns=columns, k_tiles=k)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, action='append', required=True)
    for name in ('original', 'batched', 'ready'):
        parser.add_argument('--'+name, type=alternative, required=True,
                            metavar='PACKAGE:COLUMNS:K_TILES')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    names = ('original', 'batched', 'ready')
    packages = {n: getattr(args, n)[0] for n in names}
    alternatives = {n: getattr(args, n)[1] for n in names}
    # The inherited declaration-custody identity owner uses this library alias.
    packages['rhs_ready_wide'] = packages['original']
    sys.path[:0] = [str(packages['original']), str(packages['original']/'compiler_library')]
    from gemmini_frontend import normalize_source
    sources = []
    for path in args.source:
        path = path.resolve()
        model = normalize_source(path.read_text())
        products = [o for o in model['operations'] if o['category'] == 'product']
        if len(products) != 1:
            raise ValueError('requires one dense product per source')
        extents = products[0]['extents']
        sources.append(dict(path=str(path), capsule=path.stem,
            source_sha256=digest(path.read_bytes()), shape=[extents[d] for d in ('m', 'k', 'n')]))
    workspace = Path.cwd()
    owner = SegmentedAdapter(workspace, output, packages, sources, alternatives=alternatives)
    recipe_path = workspace/'.agents/skills/heavy-pass-recipes/scripts/parallel_schedule_recipe.py'
    spec = importlib.util.spec_from_file_location('segmented_shared_parallel_recipe', recipe_path)
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    with ThreadPoolExecutor(max_workers=2) as executor:
        result = recipe.run([tuple(r['shape']) for r in sources], {}, executor=executor,
                            max_in_flight=2, directory=output/'parallel', owner=owner)
    if any(tree_identity(packages[n]) != p for n, p in owner.package_pins.items()):
        raise ValueError('immutable compiler package changed')
    if any(digest(Path(r['path']).read_bytes()) != r['source_sha256'] for r in sources):
        raise ValueError('original source changed')
    report = dict(schema='gemmini.segmented_parallel_owner.v1', sources=sources,
        alternatives=alternatives, package_pins=owner.package_pins,
        recipe_sha256=digest(recipe_path.read_bytes()),
        adapter_sha256=digest(Path(__file__).read_bytes()),
        products=list(owner.programs.values()), completions=len(result['completions']),
        timing_requests=0, decision='UNKNOWN recipe observations; join separately pinned ordinary feedback')
    (output/'owner.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(report=str(output/'owner.json'), products=len(owner.programs),
                          completions=report['completions'], timing_requests=0)))


if __name__ == '__main__':
    main()
