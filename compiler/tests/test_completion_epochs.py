"""Safety witnesses on the selected public source programs; no new evaluator inputs."""
import json
from dataclasses import replace
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'compiler_library')]
import completion_epochs as ce
from dense import plan_matmul, dense_adapter, _geometry
from merlin.compile.scheduling.dense_panel_epochs import STREAM_FLAGS, schedule_dense_panel_epochs
from merlin.compile.scheduling.dense_panel_pipeline import DRAIN, _serial_projection
from merlin.sched.ir import Call, Loop


def shapes():
    # Read tensor semantics, never derive a geometry from a case's filename.
    for p in sorted((ROOT.parent / 'public-cases').glob('*.mlir')):
        text = p.read_text()
        n0, n = map(int, re.search(r'name = "W".*tensor<(\d+)x(\d+)xi8>', text).groups())
        m, k = map(int, re.search(r'name = "A0".*tensor<(\d+)x(\d+)xi8>', text).groups())
        assert n0 == k
        yield m, k, n


class CompletionEpochTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = dense_adapter()
        cls.programs = [(dims, plan_matmul(*dims, segmented=True),
                         plan_matmul(*dims, segmented=True, merge_k_completion_streams=True))
                        for dims in shapes()]

    def test_all_public_sources_change_only_completion(self):
        for dims, before, after in self.programs:
            with self.subTest(dims=dims):
                self.assertEqual(_serial_projection(before.kernel), _serial_projection(after.kernel))
                self.assertGreater(after.realization['panel_epochs']['removed_dynamic_drains'], 0)
                self.assertEqual(before.realization['source'], after.realization['source'])
                self.assertFalse(after.law.accumulator_result_visibility)
                self.assertTrue(after.realization['backend_completion_proof']['hardware_visibility_fact']['value'])
                ce._check_frontiers(before.kernel, self.adapter)
                ce._check_frontiers(after.kernel, self.adapter)
                after.revalidate()

    def test_all_unknown_stream_flags_keep_every_boundary(self):
        _, _, after = self.programs[0]
        original, storage = after.original, after._storage
        # The consumer only needs a K=1 block from the original decision.
        from merlin.compile.scheduling.schedule_ir import parse_dense_schedule_plan
        from merlin.transforms.dense_planning import DenseProblem, DenseResources, dense_blocks
        m,k,n = self.programs[0][0]
        edge,op_rows,acc_rows = _geometry()
        p = DenseProblem(m,k,n,edge,edge,edge,1,1,4)
        r = DenseResources(op_rows*edge,acc_rows*edge*4,max_m_blocks=8,max_k_blocks=1,max_n_blocks=8,
                           partial_output_tiles=True,partial_output_blocks=True)
        info = after.realization['backend_completion_proof']['block_tiles']
        block = next(b for b in dense_blocks(p,r) if (b.m,b.k,b.n)==tuple(info))
        for flag in STREAM_FLAGS:
            with self.subTest(flag=flag):
                result = schedule_dense_panel_epochs(original,p,block,r,storage,self.adapter,
                    original.admission,replace(after.law, **{flag:False}))
                self.assertEqual(result.kernel, original.kernel)
                self.assertEqual(result.realization['panel_epochs']['removed_dynamic_drains'], 0)

    def test_real_final_and_readout_completions_are_required(self):
        _,_,after = self.programs[0]
        def strip(body):
            return tuple(replace(s,body=strip(s.body)) if type(s) is Loop else s
                         for s in body if s != Call(DRAIN,()))
        with self.assertRaisesRegex(ValueError, 'closing drain'):
            ce._check_frontiers(replace(after.kernel,body=strip(after.kernel.body)),self.adapter)
        # A final completion alone cannot repair a missing readout frontier.
        bad = replace(after.kernel,body=(*strip(after.kernel.body),Call(DRAIN,())))
        with self.assertRaisesRegex(ValueError,'readout/output-block'):
            ce._check_frontiers(bad,self.adapter)

    def test_unknown_stale_and_missing_selected_inputs_fall_back(self):
        _,_,after = self.programs[0]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'rtl_facts').mkdir()
            paths = ('COMPLETION_PREMISES.json','COMMAND_ORDERING_SELECTION.json','rtl_facts/facts.json')
            for p in paths: (root/p).write_bytes((ROOT/p).read_bytes())
            original = (root/'COMPLETION_PREMISES.json').read_bytes()
            with patch.object(ce,'__file__',str(root/'completion_epochs.py')):
                self.assertIsNotNone(ce.selected_stream_law(self.adapter,after.admission))
                for flag in STREAM_FLAGS:
                    if flag in ('complete_device_drain','exclusive_private_state'): continue
                    for unknown in (False,None):
                        d=json.loads(original);d['premises'][flag]['value']=unknown
                        (root/'COMPLETION_PREMISES.json').write_text(json.dumps(d))
                        self.assertIsNone(ce.selected_stream_law(self.adapter,after.admission))
                (root/'COMPLETION_PREMISES.json').write_bytes(original)
                for field in ('coverage_caveats','accumulator_result_visibility'):
                    d=json.loads(original)
                    if field=='coverage_caveats':del d[field]
                    else:del d['premises'][field]
                    (root/'COMPLETION_PREMISES.json').write_text(json.dumps(d))
                    self.assertIsNone(ce.selected_stream_law(self.adapter,after.admission))
                (root/'COMPLETION_PREMISES.json').write_bytes(original)
                (root/'rtl_facts/facts.json').write_text('{}')
                self.assertIsNone(ce.selected_stream_law(self.adapter,after.admission))
                (root/'COMPLETION_PREMISES.json').unlink()
                self.assertIsNone(ce.selected_stream_law(self.adapter,after.admission))

    def test_checked_result_refuses_changed_kernel_or_law(self):
        _,_,after = self.programs[0]
        with self.assertRaises(ValueError):
            replace(after, law=replace(after.law,local_write_after_read=False)).revalidate()
        with self.assertRaises(ValueError):
            replace(after, kernel=after.original.kernel).revalidate()
        with self.assertRaisesRegex(ValueError,'implementation evidence changed'):
            replace(after, backend_source_pins=(('target.py','0'*64),)).revalidate()

    def test_noop_and_wrong_owner_completion_callbacks_are_refused(self):
        from target import Builder
        from merlin.compile.scheduling.llvm_program import FnBuilderControlAdapter
        _,_,after = self.programs[0]
        owner, other = Builder(['A','B','C']), Builder(['A','B','C'])
        for callback in (lambda:None, other.fence):
            with self.subTest(callback=callback), self.assertRaisesRegex(ValueError,'Builder.fence'):
                after.emit_index_program(FnBuilderControlAdapter(owner),lambda *args:None,
                                          complete_epoch=callback)

    def test_deleted_boundary_readback_does_not_permit_data_edits(self):
        _,before,after = self.programs[0]
        def edit(body):
            changed=False;result=[]
            for statement in body:
                if not changed and type(statement) is Loop:
                    nested,changed=edit(statement.body)
                    statement=replace(statement,body=nested)
                elif not changed and statement.instr=='gemmini_lhs_load':
                    args=dict(statement.args);args['dst']=args['dst']+1
                    statement=replace(statement,args=tuple(args.items()));changed=True
                result.append(statement)
            return tuple(result),changed
        body,changed=edit(after.kernel.body)
        self.assertTrue(changed)
        with self.assertRaisesRegex(ValueError,'actual data call'):
            ce._completion_difference(before.kernel,replace(after.kernel,body=body),self.adapter)


if __name__ == '__main__':
    unittest.main()
