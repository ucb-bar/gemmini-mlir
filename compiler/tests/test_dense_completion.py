"""Adversarial completion-law unit tests; no added training/model interfaces."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'compiler_library'), str(ROOT)]
import dense
from completion import selected_law, make_completion_plan, consume_completion_plan, select_completion
from merlin.compile.scheduling.block_schedule import Load, Preload, Compute, Store
from merlin.compile.scheduling.compact_dense import DenseLoopStorage, DenseMemoryView
from merlin.compile.scheduling.dense_completion import (
    CompletionNotApplicable, DenseCompletionOwnership, coalesce_dense_k_completions,
    verify_ordered_dense_completion, _erase_drains,
)
from merlin.compile.scheduling.dense_panel_pipeline import DRAIN
from merlin.compile.scheduling.schedule_ir import dense_schedule_text, parse_dense_schedule_plan
from merlin.sched.ir import Kernel, TensorArg, Loop, Call, Var, as_expr


class CompletionProofTests(unittest.TestCase):
    def setUp(self):
        self.adapter = dense.dense_adapter()
        self.law = selected_law(ROOT, self.adapter)
        self.storage = DenseLoopStorage(
            DenseMemoryView('left', 'spad', 16, 0, 32),
            DenseMemoryView('right', 'spad', 16, 64, 32),
            DenseMemoryView('sum', 'acc', 64, 0, 16))
        self.drain = Call(DRAIN, ())
        self.args = (TensorArg('lhs',(2,8),'i8','read'), TensorArg('rhs',(8,2),'i8','read'),
                     TensorArg('output',(2,2),'i32','write'))

    def step(self, k):
        k = as_expr(k)
        return tuple(self.adapter.encode(op) for op in (
            Load('lhs',(as_expr(0),k),as_expr(0),k*2,as_expr(2),as_expr(2),as_expr(0),as_expr(1)),
            Load('weight',(k,as_expr(0)),k*2,as_expr(0),as_expr(2),as_expr(2),as_expr(64),as_expr(1)),
            Preload(as_expr(64),(k,as_expr(0)),as_expr(0),as_expr(1),as_expr(2),as_expr(2),as_expr(2)),
            Compute(as_expr(0),(as_expr(0),k),as_expr(2),as_expr(2),True)))

    def fixture(self):
        store = self.adapter.encode(Store(*map(as_expr,(0,0,2,2,0))))
        kernel = Kernel('completion_fixture', self.args, (
            Loop('reduction',3,(Loop('step_epoch',1,(*self.step(Var('reduction')),self.drain)),)),
            Loop('last_epoch',1,(*self.step(3),self.drain)),
            Loop('readout',1,(store,self.drain))))
        return kernel

    def original(self, kernel=None):
        kernel = self.fixture() if kernel is None else kernel
        return SimpleNamespace(kernel=kernel,realization={},revalidate=lambda:None)

    def ownership(self, kernel):
        return DenseCompletionOwnership(kernel.digest(),True,'fixture full-drain callback',True)

    def verify(self, source, actual, **kwargs):
        return verify_ordered_dense_completion(source,actual,self.storage,self.adapter,self.law,
            self.ownership(source),**kwargs)

    def test_real_alternatives_preserve_all_data_and_required_drains(self):
        original = self.original()
        for mode in ('inner_stream','ordered_stream'):
            result = coalesce_dense_k_completions(original,self.storage,self.adapter,self.law,
                self.ownership(original.kernel),mode=mode)
            self.assertEqual(_erase_drains(original.kernel.body),_erase_drains(result.kernel.body))
            report = result.realization['completion_placement']
            self.assertGreater(report['load_reuse_row_edges'],0)
            self.assertGreater(report['protected_original_dynamic_drains'],0)
            self.assertEqual(report['max_issued_between_actual_drains']['compute_pairs'],4)
            result.revalidate()

    def test_joint_output_reduction_loop_is_not_a_k_stream(self):
        source=self.fixture()
        reduction=source.body[0]
        epoch=reduction.body[0]
        calls=[]
        for call in epoch.body:
            if call.instr=='gemmini_compute':
                call=replace(call,args=tuple((key,Var('reduction') if key=='tile0' else value)
                    for key,value in call.args))
            calls.append(call)
        source=replace(source,body=(replace(reduction,body=(replace(epoch,body=tuple(calls)),)),*source.body[1:]))
        original=self.original(source)
        with self.assertRaisesRegex(CompletionNotApplicable,'no repeated pure K'):
            coalesce_dense_k_completions(original,self.storage,self.adapter,self.law,self.ownership(source))

    def test_neighboring_nonreduction_group_is_untouched(self):
        source=self.fixture()
        short=(Loop('short_stage',1,(self.step(0)[0],self.drain)),
            Loop('short_compute',1,(*self.step(0)[1:],self.drain)),
            replace(source.body[-1],var='short_readout'))
        source=replace(source,body=(*short,*source.body))
        original=self.original(source)
        result=coalesce_dense_k_completions(original,self.storage,self.adapter,self.law,
            self.ownership(source),mode='ordered_stream')
        self.assertEqual(result.kernel.body[:3],short)
        self.assertNotEqual(result.kernel.body[3:],source.body[3:])

    def test_carried_completion_never_enters_raw_index_emitter(self):
        source=self.fixture()
        reduction=replace(source.body[0],iter_args=(('incoming','event0'),),
            yields=('incoming',),results=('outgoing',))
        source=replace(source,body=(reduction,*source.body[1:]))
        with self.assertRaises(CompletionNotApplicable):
            self.verify(source,source)

    def test_unknown_target_fact_refuses(self):
        source=self.fixture()
        with self.assertRaises(CompletionNotApplicable):
            verify_ordered_dense_completion(source,source,self.storage,self.adapter,
                replace(self.law,local_write_after_read=None),self.ownership(source))

    def test_missing_private_ownership_refuses(self):
        source=self.fixture()
        with self.assertRaises(CompletionNotApplicable):
            verify_ordered_dense_completion(source,source,self.storage,self.adapter,self.law,
                replace(self.ownership(source),exclusive_storage=False))

    def test_missing_real_full_drain_refuses(self):
        source=self.fixture()
        with self.assertRaises(CompletionNotApplicable):
            verify_ordered_dense_completion(source,source,self.storage,self.adapter,self.law,
                replace(self.ownership(source),full_drain_implementation=''))

    def test_load_load_overwrite_cannot_use_assumed_load_queue_order(self):
        load=self.step(0)[0]
        source=Kernel('unconsumed_waw',self.args,(
            Loop('first',1,(load,self.drain)),Loop('second',1,(load,self.drain))))
        actual=replace(source,body=(replace(source.body[0],body=(load,)),source.body[1]))
        with self.assertRaisesRegex(CompletionNotApplicable,'unconsumed load-load'):
            self.verify(source,actual)

    def test_pre_readout_completion_must_survive(self):
        source=self.fixture()
        last=source.body[1]
        actual=replace(source,body=(source.body[0],replace(last,body=last.body[:-1]),source.body[2]))
        with self.assertRaisesRegex(CompletionNotApplicable,'pre-readout'):
            self.verify(source,actual)

    def test_closing_readout_completion_must_survive(self):
        source=self.fixture()
        store=source.body[-1]
        actual=replace(source,body=(*source.body[:-1],replace(store,body=store.body[:-1])))
        with self.assertRaisesRegex(CompletionNotApplicable,'closing full drain'):
            self.verify(source,actual)

    def test_noncompletion_operand_change_refuses(self):
        source=self.fixture()
        last=source.body[1]
        call=last.body[0]
        changed=replace(call,args=tuple((k,as_expr(1) if k=='dst' else v) for k,v in call.args))
        actual=replace(source,body=(source.body[0],replace(last,body=(changed,*last.body[1:])),source.body[2]))
        with self.assertRaisesRegex(CompletionNotApplicable,'non-completion'):
            self.verify(source,actual)

    def test_proof_budget_is_fallback_and_not_a_resource_grant(self):
        source=self.fixture()
        with self.assertRaisesRegex(CompletionNotApplicable,'proof budget'):
            self.verify(source,source,max_dynamic_calls=1)

    def test_wide_transfer_without_span_law_refuses(self):
        source=self.fixture()
        def wide(body):
            return tuple(replace(s,body=wide(s.body)) if isinstance(s,Loop) else
                replace(s,args=tuple((k,as_expr(17) if k=='cols' else v) for k,v in s.args))
                if s.instr=='gemmini_lhs_load' else s for s in body)
        source=replace(source,body=wide(source.body))
        with self.assertRaisesRegex(CompletionNotApplicable,'transfer geometry'):
            self.verify(source,source)

    def test_schedule_dialect_roundtrip_and_stale_binding(self):
        source=self.fixture()
        plan=make_completion_plan(source,self.law.ordering_identity,'ordered_stream')
        parsed=parse_dense_schedule_plan(dense_schedule_text(plan))
        self.assertEqual(consume_completion_plan(parsed,source,self.law.ordering_identity),'ordered_stream')
        with self.assertRaisesRegex(ValueError,'stale'):
            consume_completion_plan(parsed,replace(source,name='different'),self.law.ordering_identity)

    def test_missing_target_files_selects_unchanged_registered_fallback(self):
        original=self.original()
        result,receipt=select_completion(original,self.storage,self.adapter,root=ROOT/'no_target',
            mode='ordered_stream',exclusive_storage=True,serialized_invocation=True,
            full_drain_implementation='fixture completion')
        self.assertIs(result,original)
        self.assertEqual(receipt['selected_variant'],'per_k')
        self.assertIsNotNone(receipt['refusal'])

if __name__=='__main__': unittest.main()
