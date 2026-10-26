"""Runtime scale SSA reaches a real integer load and the CONFIG rs2 register."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from xdsl.dialects import llvm

from merlin.targetgen.composition_capsules import _Source
from mlir_oot.core_aten.catalog import _stream_digest, binding_record
from mlir_oot.core_aten.emit import build_module, compile_to_object, encode, llvm_bin_dir
from mlir_oot.core_aten.integer_regions import _emit_readout_plan, contraction
from mlir_oot.core_aten.match import Match
from mlir_oot.core_aten import ops
from mlir_oot.core_aten.runtime_scale import RuntimeScaleRef, bind_scale_operand
from mlir_oot.core_aten.stream import ACC, Arg, Stream, simulate
from mlir_oot.core_aten.trace import stream_from_module
from mlir_oot.frontend.parse import parse_module
from test_runtime_scale_encoding import runtime_stream, scale_record

# Vendored byte-for-byte from the kextdiag selected-RTL facts (Merlin session f6514365, kextdiag/rtl-facts.json),
# sha256 a8d92927fad2f4aa56602b64672f2cf694e3d8ae5094e80413795fb8d270abe3.
FACTS_FILE = Path(__file__).resolve().parent / 'fixtures/runtime_scale/rtl-facts.json'
PATH = dict(schema='output_path_v1', accumulator=dict(dtype='i32'),
    accumulator_stages=[dict(kind='preload', dtype='i32')],
    readout=[dict(kind='activation', function='relu', domain='accumulator'),
             dict(kind='convert', to='f32', rounding='half_even'),
             dict(kind='scale', dtype='f32', rounding='half_even', granularity='per_command', columns_per_scale=16),
             dict(kind='requant', to='i8', rounding='half_even', clamp=[-128, 127])],
    exits=dict(accumulator=True, requant=True))


def source(*, relu=False):
    builder = _Source()
    a = builder.input('lhs', [3, 7], 'i8')
    b = builder.input('rhs', [7, 17], 'i8')
    builder.input('unused_scale', [1], 'f32')
    scale = builder.input('actual_scale', [1], 'f32')
    acc = builder.matmul('acc', a, b, 3, 7, 17)
    out = builder.readout('q', acc, 3, 17, [.25], 16, relu=relu, runtime_scale=scale)
    args = ','.join(f'%{row["name"]}:{builder.types[row["name"]]}' for row in builder.inputs)
    return '\n'.join([f'builtin.module {{ func.func @forward({args}) -> tensor<3x17xi8> {{',
                      *builder.lines, f'func.return %{out}:tensor<3x17xi8>', '}}'])


def planned_kernel(*, relu=False):
    """Use real source proof and RTL facts, without a sampled multiplier."""
    from merlin.compile.composition_fusion import CompositionPlan
    from merlin.targetgen import model_coverage
    from merlin.transforms.certificates import ir_digest
    from merlin.xdsl_dialects.lowering import output_fusion as OF
    from mlir_oot.core_aten.readout_encoding import runtime_scale_facts

    module = parse_module(source(relu=relu))
    root = next(op for op in module.walk() if contraction(op) is not None)
    values = OF.Values()
    chain = OF.source_chain(root, values)
    transport = runtime_scale_facts(json.loads(FACTS_FILE.read_text()), PATH)
    decision = OF.decide(chain, PATH, values, model_coverage._contraction_extents(module),
                         readout_strategy='f32_product_rne', runtime_scale_facts=transport)
    assert decision.cut == 'requant', decision.refusals
    members = [root, *chain.ops]
    ordinals = {op: i for i, op in enumerate(module.walk())}
    plan = CompositionPlan('epilogue', ir_digest(module), tuple(ordinals[op] for op in members), decision.record())
    kernel = ops.matmul('aten.mm.default', 3, 7, 17, s_out=None)
    kernel.meta.update(integer_producer=True, integer_readout=False)
    operands = [('a', root.operands[0]), ('b', root.operands[1])]
    consumed = [root]
    result = _emit_readout_plan(kernel, root, (plan, chain), consumed, module=module, operands=operands)
    match = Match(kernel.op, root, consumed, operands, ('out', result.results[0]), kernel)
    return module, match


@pytest.mark.parametrize('constant_gap', [False, True])
def test_runtime_scale_lowers_to_integer_load_zext_shift_or(tmp_path, constant_gap):
    stream = runtime_stream()
    if constant_gap:
        stream.args.insert(1, Arg('private', 'i8', (1,), 'const'))
        stream.consts['private'] = np.array([1], np.int8)
    stream.config_st(17, scale_operand=RuntimeScaleRef(3 if constant_gap else 2))
    module, _ = build_module(stream, 'runtime_slot')
    function = next(op for op in module.walk() if isinstance(op, llvm.FuncOp))
    load, = [op for op in module.walk() if isinstance(op, llvm.LoadOp)]
    assert str(load.results[0].type) == 'i32'
    assert load.alignment.value.data == 4
    ptr_to_int = load.ptr.owner.operands[0].owner
    assert isinstance(ptr_to_int, llvm.PtrToIntOp)
    assert ptr_to_int.operands[0] is function.body.blocks[0].args[2]
    names = [op.name for op in module.walk()]
    assert all(name in names for name in ['llvm.load', 'llvm.zext', 'llvm.shl', 'llvm.or', 'llvm.inline_asm'])
    asm, = [op for op in module.walk() if isinstance(op, llvm.InlineAsmOp)]
    assert isinstance(asm.operands[1].owner, llvm.OrOp)
    assert '.insn r ' in asm.asm_string.data
    artifact = compile_to_object(module, tmp_path, stem='runtime')
    assert artifact['nofsm'] == 'pass'
    ir = (tmp_path / 'runtime.ll').read_text()
    assert 'load i32' in ir and 'zext i32' in ir and 'shl i64' in ir and 'or i64' in ir
    assembly = tmp_path / 'runtime.s'
    subprocess.run([str(llvm_bin_dir() / 'llc'), '-mtriple=riscv64-unknown-elf', '-mattr=+m,+a,+f,+d,+c',
                    '-target-abi=lp64d', '-O2', str(tmp_path/'runtime.ll'), '-o', str(assembly)],
                   check=True, capture_output=True, text=True)
    text = assembly.read_text()
    # LLVM may remove zext when shift-by-32 discards all sign-extension bits.
    assert ('lwu' in text or '\tlw\t' in text) and 'slli' in text and '.insn r' in text
    assert ', 32' in text
    assert '0(a2)' in text  # Third runtime pointer, not the first same-typed scale.
    assert 'fcvt' not in text


@pytest.mark.parametrize('relu', [False, True])
def test_runtime_readout_rebinds_scale_without_specializing(tmp_path, relu):
    module, match = planned_kernel(relu=relu)
    stream = match.kernel.stream
    operand = next((name, value) for name, value in match.operands if name.startswith('readout_scale'))
    fn = next(op for op in module.walk() if op.name == 'func.func')
    assert operand[1] is fn.body.blocks[0].args[3]
    ref = RuntimeScaleRef(next(i for i,a in enumerate(stream.args) if a.name == operand[0]))
    assert bind_scale_operand(scale_record(3), module, match.operands, stream.args) == ref
    configs = [ins for ins in stream.ins if ins.kind == 'config_st']
    assert configs and all(ins.a('scale_operand') == ref for ins in configs)
    assert all(encode(ins).values[1] == 17 for ins in configs)
    assert all(ins.arg != ref.arg_index for ins in stream.ins if ins.kind in ('mvin', 'mvout'))
    proof = match.kernel.meta['composition_plan']['decision']['proof']
    assert proof['scale_operand']['source']['func_arg'] == 3
    assert not {'multiplier_bits', 'device_scale', 'columns_per_scale'} & proof.keys()
    assert stream.args[stream._index['out']].dtype == 'i8'
    ordinals = {op: i for i,op in enumerate(module.walk())}
    binding = binding_record(match, 'gemmini_ca_' + _stream_digest(match.kernel)[:16], ordinals, list(fn.body.blocks[0].args))
    assert binding['operands'][operand[0]] == {'func_arg': 3}
    # Different call inputs are deliberately never arguments to compilation.
    modules, artifacts, digests = [], [], []
    for index, value in enumerate((0., .25, .5, .3)):
        env = {operand[0]: np.array([value], np.float32)}
        assert env[operand[0]].dtype == np.float32
        device, _ = build_module(stream, 'same_runtime_kernel')
        modules.append(str(device))
        artifacts.append(compile_to_object(device, tmp_path / str(index))['object_sha256'])
        digests.append(_stream_digest(match.kernel))
    assert len(set(modules)) == len(set(artifacts)) == len(set(digests)) == 1


@pytest.mark.parametrize('columns', [1, 7, 17, 31])
def test_runtime_narrowing_keeps_tail_stride_and_dma_offsets(columns):
    stream = Stream([Arg('out', 'i32', (2,columns), 'output'), Arg('s', 'f32', (1,), 'input')])
    stream.config_st(columns * 4)
    stream.mvout(stream.at('out', columns * 4), ACC(0, full=True), min(columns,16), 1)
    stream.narrow_readout(relu=False, scale_operand=RuntimeScaleRef(1))
    config, store = stream.ins
    assert config.a('stride') == columns and store.off == columns
    assert store.arg == 0 and not store.a('local') & (1 << 29)
    assert config.a('scale_operand') == RuntimeScaleRef(1)


@pytest.mark.parametrize('aggregate', [False, True])
def test_dynamic_trace_refuses_instead_of_defaulting_to_identity(aggregate):
    stream = runtime_stream()
    stream.config_st(17, scale_operand=RuntimeScaleRef(1))
    module, _ = build_module(stream, 'runtime_trace')
    if aggregate:
        from xdsl.dialects.builtin import ModuleOp
        fn = next(op for op in module.walk() if isinstance(op, llvm.FuncOp))
        module = ModuleOp([fn.clone()])
    with pytest.raises(ValueError, match='unsupported dynamic trace'):
        stream_from_module(module, stream.args)


@pytest.mark.parametrize('data', [None, np.array([np.nan], np.float32), np.array([np.inf], np.float32),
    np.array([.25], np.float64), np.array([.25, .5], np.float32), np.array(.25, np.float32)])
def test_public_simulation_rejects_invalid_scale_input_before_byte_erasure(data):
    stream = runtime_stream()
    stream.config_st(17, scale_operand=RuntimeScaleRef(1))
    with pytest.raises(ValueError, match='finite f32'):
        simulate(stream, {} if data is None else {'scale0': data})


def test_selected_runtime_transport_rejects_missing_evidence():
    from mlir_oot.core_aten.readout_encoding import runtime_scale_facts
    facts = json.loads(FACTS_FILE.read_text())
    assert runtime_scale_facts(facts, PATH)['offset'] == 32
    with pytest.raises(ValueError, match='UNKNOWN'):
        runtime_scale_facts({}, PATH)


def test_runtime_register_bits_reload_on_each_native_invocation(tmp_path):
    """Capture emitted rs2 in a host buffer; real RoCC object is checked separately."""
    import ctypes
    import hashlib
    from xdsl.rewriter import Rewriter

    stream = Stream([Arg('capture', 'i64', (1,), 'output'),
                     Arg('other', 'f32', (1,), 'input'), Arg('scale', 'f32', (1,), 'input')])
    stream.config_st(17, scale_operand=RuntimeScaleRef(2))
    module, _ = build_module(stream, 'capture_register')
    fn = next(op for op in module.walk() if isinstance(op, llvm.FuncOp))
    asm = next(op for op in module.walk() if isinstance(op, llvm.InlineAsmOp))
    Rewriter.replace_op(asm, [llvm.StoreOp(asm.operands[1], fn.body.blocks[0].args[0], alignment=8)])
    module.verify()
    mlir, ir, obj, library = (tmp_path / ('capture' + suffix) for suffix in ('.mlir', '.ll', '.o', '.so'))
    mlir.write_text(str(module))
    bindir = llvm_bin_dir()
    commands = [
        [str(bindir/'mlir-translate'), '--mlir-to-llvmir', str(mlir), '-o', str(ir)],
        [str(bindir/'llc'), '-mtriple=x86_64-unknown-linux-gnu', '-relocation-model=pic', '-O2',
         '-filetype=obj', str(ir), '-o', str(obj)],
        ['cc', '-shared', str(obj), '-o', str(library)],
    ]
    for command in commands:
        subprocess.run(command, check=True, capture_output=True, text=True)
    original = hashlib.sha256(library.read_bytes()).hexdigest()
    function = ctypes.CDLL(str(library)).capture_register
    function.argtypes = [ctypes.c_void_p] * 3
    function.restype = None
    actual, other, scale = np.zeros(1, np.uint64), np.array([.75], np.float32), np.zeros(1, np.float32)
    for bits in (0, 0x80000000, 0x3e800000, 0x3f000000, 0x3e99999a, 1, 0x7f7fffff, 0xbf000000):
        scale.view(np.uint32)[0] = bits
        function(actual.ctypes.data, other.ctypes.data, scale.ctypes.data)
        assert int(actual[0]) == (bits << 32) | 17
        other[0] *= -1  # A same-typed unrelated input must not determine rs2.
    assert hashlib.sha256(library.read_bytes()).hexdigest() == original


@pytest.mark.parametrize('edge', ['source_argument', 'stream_reference', 'stream_argument_role'])
def test_runtime_binding_chain_mutations_are_authenticated(edge):
    import copy
    from mlir_oot.core_aten.catalog import authenticate_binding

    module, match = planned_kernel()
    fn = next(op for op in module.walk() if op.name == 'func.func')
    ordinal = {op: i for i, op in enumerate(module.walk())}
    symbol = 'gemmini_ca_' + _stream_digest(match.kernel)[:16]
    binding = binding_record(match, symbol, ordinal, list(fn.body.blocks[0].args))
    authenticate_binding(binding, match)
    scale_name = next(name for name, _ in match.operands if name.startswith('readout_scale'))
    if edge == 'source_argument':
        binding = copy.deepcopy(binding)
        binding['operands'][scale_name] = {'func_arg': 2}
    elif edge == 'stream_reference':
        config = next(i for i, ins in enumerate(match.kernel.stream.ins) if ins.kind == 'config_st')
        ins = match.kernel.stream.ins[config]
        attrs = dict(ins.attrs, scale_operand=RuntimeScaleRef(1))
        match.kernel.stream.ins[config] = ins._replace(attrs=tuple(sorted(attrs.items())))
    else:
        from dataclasses import replace
        stream = match.kernel.stream
        index = stream._index[scale_name]
        stream.args[index] = replace(stream.args[index], role='const')
        # Digest cannot authenticate a const without its constant bytes.
        with pytest.raises((ValueError, KeyError)):
            build_module(stream, symbol)
        return
    with pytest.raises(ValueError, match='binding'):
        authenticate_binding(binding, match)


@pytest.mark.parametrize('mutation', ['second_input_token', 'meta_source'])
def test_runtime_scale_admission_refuses_rebound_scale(mutation):
    """A second bound f32[1] input or a restated meta source must not pass contraction_proof.admit."""
    from mlir_oot.core_aten import contraction_proof
    from mlir_oot.core_aten.stream_proof import ProofError

    module, match = planned_kernel()
    selected = dict(path=PATH, scheme=dict(readout_strategy='f32_product_rne'), facts=json.loads(FACTS_FILE.read_text()))
    assert contraction_proof.admit(match, readout_options=selected) == 'f32_product_rne'
    if mutation == 'second_input_token':
        unused = next(op for op in module.walk() if op.name == 'func.func').body.blocks[0].args[2]
        stream = match.kernel.stream
        stream.args.append(Arg('readout_scale9', 'f32', (1,), 'input'))
        stream.__post_init__()
        match.operands.append(('readout_scale9', unused))
        for index, ins in enumerate(stream.ins):
            if ins.kind == 'config_st':
                attrs = dict(ins.attrs, scale_operand=RuntimeScaleRef(len(stream.args) - 1))
                stream.ins[index] = ins._replace(attrs=tuple(sorted(attrs.items())))
    else:
        meta = match.kernel.meta
        meta['scale_operand'] = dict(meta['scale_operand'], source=dict(func_arg=2, element_index=0))
    with pytest.raises((ProofError, ValueError)):
        contraction_proof.admit(match, readout_options=selected)


def test_runtime_v1_retains_nonzero_seed_readout_on_host():
    from mlir_oot.core_aten.integer_regions import _readout_plan
    text = source().replace('%acc_zero = arith.constant 0 : i32', '%acc_zero = arith.constant 1 : i32')
    module = parse_module(text)
    root = next(op for op in module.walk() if contraction(op) is not None)
    assert _readout_plan(module, root, dict(path=PATH, scheme=dict(readout_strategy='f32_product_rne'),
                                          facts=json.loads(FACTS_FILE.read_text()))) is None
