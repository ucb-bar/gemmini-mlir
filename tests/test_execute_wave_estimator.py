from dataclasses import replace

import pytest
from xdsl.context import Context
from xdsl.dialects import llvm
from xdsl.dialects.builtin import Builtin
from xdsl.parser import Parser

from mlir_oot.execute_wave_estimator import (
    ExecuteWaveFacts, PrimitiveCommand, WaveProofError, WaveState,
    commands_from_function, derive_facts, estimate_commands, estimate_wave,
)
from mlir_oot.ir.gemmini_dialect import GEMMINI
from mlir_oot.tables import isa


RULE = '''
val total_rows = WireInit(block_size.U)
when (current_dataflow === Dataflow.WS.id.U && d_garbage &&
 !a_should_be_fed_into_transposer && !b_should_be_fed_into_transposer && !d_should_be_fed_into_transposer) {
 val rows_a = Mux(a_garbage, 1.U, a_rows)
 val rows_b = Mux(b_garbage, 1.U, b_rows)
 total_rows := maxOf(maxOf(rows_a, rows_b), 4.U)
}
val start_inputting_d = WireInit(false.B)
val d_garbage = d_address_rs1.is_garbage() || !start_inputting_d
val a_garbage = a_address_rs1.is_garbage() || !start_inputting_a
val b_garbage = b_address_rs2.is_garbage() || !start_inputting_b
val a_should_be_fed_into_transposer = Mux(current_dataflow === Dataflow.OS.id.U, !a_transpose, a_transpose)
val b_should_be_fed_into_transposer = current_dataflow === Dataflow.OS.id.U && bd_transpose
val d_should_be_fed_into_transposer = current_dataflow === Dataflow.WS.id.U && bd_transpose
when(perform_single_preload) { start_inputting_d := true.B }
.elsewhen(perform_mul_pre) { start_inputting_d := true.B }
.elsewhen(perform_single_mul) {
 start_inputting_a := !a_should_be_fed_into_transposer
 start_inputting_b := !b_should_be_fed_into_transposer
}
.elsewhen(DoComputes(0)) {
 start_inputting_a := !a_should_be_fed_into_transposer
 start_inputting_b := !b_should_be_fed_into_transposer
}
'''


def test_hardware_fact_limits_are_derived_and_comments_cannot_change_them(tmp_path):
    source = tmp_path / 'controller.scala'
    source.write_text(RULE.replace('4.U)', '7.U)') +
                     '\n// total_rows := unsupported\n/* nested /* done */ total_rows := 3.U */')
    facts = derive_facts(source, dimension=16)
    assert facts.minimum_resident_rows == 7
    assert facts.inactive_a_rows == facts.inactive_b_rows == 1
    assert estimate_wave(facts, state(a_rows=1))['rows'] == 7


@pytest.mark.parametrize('bad', [
    RULE.replace('&& d_garbage', ''),
    RULE.replace('!d_should_be_fed_into_transposer', 'd_should_be_fed_into_transposer'),
    RULE.replace('4.U)', 'other.U)'),
    RULE.replace('4.U)', '4.U)\n total_rows := 16.U'),
    RULE.replace('start_inputting_b := !b_should_be_fed_into_transposer', 'start_inputting_d := true.B'),
    RULE.replace('WireInit(false.B)', 'WireInit(true.B)'),
])
def test_changed_or_ambiguous_hardware_rule_refuses(tmp_path, bad):
    source = tmp_path / 'controller.scala'
    source.write_text(bad)
    with pytest.raises(WaveProofError):
        derive_facts(source, dimension=16)


FACTS = ExecuteWaveFacts(16, 1, 1, 4, 'fixture')


def state(**changes):
    return replace(WaveState('single_mul', isa.WEIGHT_STATIONARY,
        8, 16, False, True, True, False, False, False), **changes)


@pytest.mark.parametrize('rows,expected', [(1,4),(8,8),(16,16)])
def test_effective_ws_ports_use_variable_rows(rows, expected):
    assert estimate_wave(FACTS, state(a_rows=rows))['rows'] == expected


@pytest.mark.parametrize('changes,classification', [
    ({'mode':'mul_pre','d_garbage':False}, 'fixed_dimension'),
    ({'a_transposer':True}, 'fixed_dimension'),
    ({'dataflow':isa.OUTPUT_STATIONARY}, 'fixed_dimension'),
    ({'a_transposer':None}, 'uncertain'),
    ({'a_garbage':None}, 'uncertain'),
    ({'mode':None}, 'uncertain'),
])
def test_nongarbage_d_transpose_os_and_unresolved_state_retain_dimension(changes, classification):
    result = estimate_wave(FACTS, state(**changes))
    assert result['rows'] == 16
    assert result['classification'] == classification


def test_active_b_extent_and_invalid_port_claims():
    assert estimate_wave(FACTS, state(b_garbage=False,b_rows=12))['rows'] == 12
    with pytest.raises(WaveProofError,match='inactive D'):
        estimate_wave(FACTS,state(d_garbage=False))
    with pytest.raises(WaveProofError,match='extent'):
        estimate_wave(FACTS,state(a_rows=17))
    with pytest.raises(WaveProofError,match='booleans'):
        estimate_wave(FACTS,state(d_garbage=0))


def preload(address):
    return PrimitiveCommand('preload',dict(bd=address,bd_rows=16,bd_cols=16))


def compute(stay=False):
    return PrimitiveCommand('compute',dict(a=0,a_rows=1,bd=isa.GARBAGE_ADDR,bd_rows=16,accumulate=stay))


def test_flip_stay_weight_reuse_is_separate_from_effective_d_and_dispatch():
    cfg = PrimitiveCommand('config_ex',dict(dataflow=isa.WEIGHT_STATIONARY,a_transpose=False,b_transpose=False))
    commands=[cfg,preload(32),compute(),preload(64),compute(True),preload(isa.GARBAGE_ADDR),compute(True)]
    result=estimate_commands(commands,FACTS)
    records=result['compute_records']
    assert [r['active_weight']['address'] for r in records] == [32,32,32]
    assert records[0]['rows_min_possible'] == 4
    assert records[0]['rows_conservative'] == 16
    assert records[0]['possible_dispatch'][1]['operand_state']['d_garbage'] is False
    assert records[1]['rows_conservative'] == records[2]['rows_conservative'] == 4
    assert result['padded_compute_geometry'] == 48
    assert result['possible_compute_rows'] == 12
    assert result['conservative_compute_rows'] == 24
    assert result['raw_command_counts'] == {'config_ex':1,'preload':3,'compute':3}
    assert result['dispatch_mode_resolved'] is False
    assert estimate_commands(commands,FACTS,retain_records=False)['geometry_histogram'] == result['geometry_histogram']


def test_missing_configuration_is_not_a_weight_or_transposer_proof():
    result=estimate_commands([preload(32),compute()],FACTS)
    record=result['compute_records'][0]
    assert record['active_weight'] is None
    assert record['classification'] == 'uncertain_state'
    assert record['rows_conservative'] == 16


def test_actual_lowering_configuration_and_dynamic_a_use_generic_cfg_trace():
    ctx=Context()
    for dialect in (Builtin,llvm.LLVM,GEMMINI):ctx.load_dialect(dialect)
    module=Parser(ctx,'''builtin.module { llvm.func @trace() {
      gemmini.config_ex {dataflow = 1 : i64} : () -> ()
      %start = llvm.mlir.constant(0 : i64) : i64
      %one = llvm.mlir.constant(1 : i64) : i64
      %stop = llvm.mlir.constant(2 : i64) : i64
      llvm.br ^loop(%start : i64)
      ^loop(%iv: i64):
      gemmini.compute %iv {a_max=1:i64,a_reserved_rows=16:i64,a_rows=1:i64,a_cols=16:i64,accumulate=1:i64} : (i64) -> ()
      %next = llvm.add %iv, %one : i64
      %more = llvm.icmp "slt" %next, %stop : i64
      llvm.cond_br %more, ^loop(%next : i64), ^exit
      ^exit:
      gemmini.fence : () -> ()
      llvm.return
    } }''').parse_module()
    commands=list(commands_from_function(module.body.block.first_op,pointer_index_bits=64))
    assert [c.fields['a'] for c in commands if c.kind == 'compute'] == [0,1]
    assert commands[0].fields['a_transpose'] is False
    result=estimate_commands(commands,FACTS)
    assert result['raw_command_counts']['compute'] == 2
    assert result['possible_compute_rows'] == result['conservative_compute_rows'] == 8


def test_incomplete_command_generator_never_returns_completed_report():
    def commands():
        yield compute()
        raise WaveProofError('unresolved dynamic input')
    with pytest.raises(WaveProofError,match='unresolved'):
        estimate_commands(commands(),FACTS)
