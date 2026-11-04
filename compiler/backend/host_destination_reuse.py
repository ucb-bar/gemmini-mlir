"""Qualify closed raw-DMA callbacks before exposing their destination identity.

This is an adapter to Merlin's descriptor rewriter and stock bufferization. It
does not lower tensor arithmetic. The bounded interpreter visits the *emitted*
LLVM CFG with symbolic pointer roots and static integers, and checks every DMA
span, initialized local cell, output write and drain. Unsupported instructions,
dynamic branches, extra functions and incomplete writes refuse the optimization.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from math import prod
from pathlib import Path
import re


class Refused(ValueError):
    pass


def _require(condition, reason):
    if not condition:
        raise Refused(reason)


def qualify(artifact, buffers, *, max_steps=1000000, max_elements=1000000, return_mapping=False):
    """Prove effects for one finite, full-width, two-pointer movement callback.

    Pointer values never become ordinary integers: ptrtoint retains its root and
    offset, and only a DMA address operand can consume it. No loads, stores,
    calls, returns of values, pointer arithmetic via integers or hidden functions
    are admitted. Thus no host alias can escape or be released. The selected
    backend's full fence drains the only admitted DMA instructions before return.
    """
    from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context
    from xdsl.parser import Parser
    from xdsl.dialects.builtin import IntegerType
    from . import isa
    cb = artifact.command_buffer or {}
    _require(artifact.format == 'llvm-dialect', 'requires emitted LLVM dialect')
    _require(len(buffers) == 2 and [b.kind for b in buffers] == ['input', 'output'],
             'requires one input and one distinct output')
    _require(all(b.dtype in ('f32', 'i32') for b in buffers), 'requires raw 32-bit movement')
    _require(len(cb.get('commands', ())) == 1 and cb['commands'][0]['opcode'] in
             ('MOVEMENT', 'STRIDED_MOVEMENT'), 'requires one checked movement command')
    counts = [prod(b.shape) for b in buffers]
    _require(all(0 < n <= max_elements for n in counts), 'movement span exceeds proof budget')
    payload = Path(artifact.path).read_bytes()
    module = Parser(make_llvm_context(), payload.decode()).parse_module()
    module.verify()
    functions = tuple(module.body.block.ops)
    _require(len(functions) == 1 and functions[0].name == 'llvm.func', 'requires closed single function')
    function = functions[0]
    _require(function.sym_name.data == artifact.symbol and len(function.body.blocks) > 0,
             'callback definition differs from selected symbol')
    entry = function.body.blocks.first
    _require(len(entry.args) == 2 and all(str(a.type) == '!llvm.ptr' for a in entry.args),
             'requires original two-pointer ABI')
    allowed = {'llvm.mlir.constant', 'llvm.add', 'llvm.mul', 'llvm.and', 'llvm.or',
               'llvm.shl', 'llvm.icmp', 'llvm.getelementptr', 'llvm.ptrtoint',
               'llvm.br', 'llvm.cond_br', 'llvm.inline_asm', 'llvm.return'}
    _require(all(o.name in allowed for b in function.body.blocks for o in b.ops),
             'callback contains an unqualified instruction')
    env = {entry.args[0]: (0, 0), entry.args[1]: (1, 0)}
    writes = bytearray(counts[1])
    initialized = {}
    mapping = [-1] * counts[1]
    pending = False
    ld = st = None
    steps = dma = fences = flushes = 0
    block = entry
    while True:
        jumped = False
        for op in block.ops:
            steps += 1
            _require(steps <= max_steps, 'callback static CFG exceeds proof budget')
            args = [env[a] for a in op.operands]
            name, props = op.name, op.properties
            result = None
            if name == 'llvm.mlir.constant':
                _require(isinstance(op.results[0].type, IntegerType), 'noninteger constant')
                _require(str(op.results[0].type) in ('i32', 'i64'), 'unsupported constant width')
                result = props['value'].value.data & ((1 << op.results[0].type.width.data) - 1)
            elif name in ('llvm.add', 'llvm.mul', 'llvm.and', 'llvm.or', 'llvm.shl'):
                _require(all(type(a) is int for a in args), 'pointer used as an integer')
                a, b = args
                width = op.results[0].type.width.data
                _require(width in (32, 64), 'unsupported integer width')
                if name == 'llvm.add': result = a + b
                elif name == 'llvm.mul': result = a * b
                elif name == 'llvm.and': result = a & b
                elif name == 'llvm.or': result = a | b
                else:
                    _require(0 <= b < width, 'undefined shift')
                    result = a << b
                _require(props.get('overflowFlags') is None or props['overflowFlags'].value.data == 0,
                         'unqualified integer overflow promise')
                result &= (1 << width) - 1
            elif name == 'llvm.icmp':
                _require(all(type(a) is int for a in args) and props['predicate'].value.data == 2,
                         'requires static signed-less-than loop')
                width = op.operands[0].type.width.data
                def signed(v): return v - (1 << width) if v & (1 << (width - 1)) else v
                result = int(signed(args[0]) < signed(args[1]))
            elif name == 'llvm.getelementptr':
                _require(len(args) == 2 and type(args[0]) is tuple and type(args[1]) is int
                         and str(props['elem_type']) == 'i32'
                         and props['noWrapFlags'].value.data == 0
                         and tuple(props['rawConstantIndices'].iter_values()) == (-2147483648,),
                         'requires one raw 32-bit GEP index')
                result = (args[0][0], args[0][1] + args[1] * 4)
            elif name == 'llvm.ptrtoint':
                _require(len(args) == 1 and type(args[0]) is tuple
                         and str(op.results[0].type) == 'i64', 'unqualified pointer cast')
                result = args[0]
            elif name in ('llvm.br', 'llvm.cond_br'):
                if name == 'llvm.br':
                    successor, passed = op.successors[0], args
                else:
                    _require(type(args[0]) is int and args[0] in (0, 1), 'dynamic branch')
                    n = len(op.successors[0].args)
                    successor = op.successors[0 if args[0] else 1]
                    passed = args[1:1+n] if args[0] else args[1+n:]
                _require(len(successor.args) == len(passed), 'branch argument mismatch')
                env.update(zip(successor.args, passed))
                block, jumped = successor, True
                break
            elif name == 'llvm.inline_asm':
                asm, constraints = props['asm_string'].data, props['constraints'].data
                _require(not op.results and 'has_side_effects' in props, 'unqualified asm effects')
                if asm == 'fence':
                    _require(not args and constraints == '~{memory}', 'unqualified completion fence')
                    pending = False
                    fences += 1
                else:
                    match = re.fullmatch(r'\.insn r 123, 3, (0|2|3|7), x0, \$0, \$1', asm)
                    _require(match is not None and constraints == 'r,r,~{memory}' and len(args) == 2,
                             'unqualified target instruction')
                    funct = int(match[1]); a, packed = args
                    if funct == 7:
                        _require(a == packed == 0 and dma == 0 and flushes == 0,
                                 'flush after DMA or nonzero flush argument')
                        flushes += 1
                    elif funct == 0:
                        _require(type(a) is int and type(packed) is int and not pending,
                                 'configuration while pending')
                        if a & 3 == 1:
                            _require(a == 0x3f80000000100101, 'nonidentity load configuration')
                            _require(packed % 4 == 0, 'unaligned load row stride')
                            ld = packed // 4
                        elif a == 2:
                            _require(packed >> 32 == 0x3f800000 and packed & 3 == 0,
                                     'nonidentity store configuration')
                            st = (packed & 0xffffffff) // 4
                        else: raise Refused('nonmovement target configuration')
                    else:
                        _require(type(a) is tuple and type(packed) is int and not pending
                                 and ld is not None and st is not None and flushes == 1,
                                 'unqualified DMA ordering or address')
                        root, byte_offset = a
                        addr, cols, rows = packed & 0xffffffff, packed >> 32 & 65535, packed >> 48
                        _require(0 < rows <= 16 and 0 < cols <= 16 and addr & 0x80000000
                                 and not addr & 0x40000000 and byte_offset % 4 == 0,
                                 'unqualified raw accumulator DMA')
                        local = addr & 0x1fffffff
                        _require(local + rows <= isa.ACC_ROWS, 'local accumulator span out of bounds')
                        _require(root == (0 if funct == 2 else 1)
                                 and bool(addr & 0x20000000) == (funct == 3),
                                 'DMA direction or raw width differs from contract')
                        for r in range(rows):
                            at = byte_offset // 4 + r * (ld if funct == 2 else st)
                            _require(0 <= at <= counts[root] - cols, 'DMA exceeds original argument span')
                            for c in range(cols):
                                cell = (local + r, c)
                                if funct == 2: initialized[cell] = at+c
                                else:
                                    _require(cell in initialized and writes[at+c] == 0,
                                             'uninitialized readout or repeated destination write')
                                    writes[at+c] = 1
                                    mapping[at+c] = initialized[cell]
                        dma += 1
                        pending = True
            elif name == 'llvm.return':
                _require(not args and not pending and dma > 0 and all(writes),
                         'incomplete destination or DMA at return')
                proof = dict(kernel_sha256=hashlib.sha256(payload).hexdigest(),
                            static_steps=steps, dma_instructions=dma, completion_fences=fences,
                            output_elements=counts[1], output_writes='exactly_once',
                            no_retained_aliases=True, no_releases=True, complete_before_return=True)
                proof['source_element_mapping_sha256'] = hashlib.sha256(json.dumps(mapping).encode()).hexdigest()
                if return_mapping:
                    proof['source_element_mapping'] = mapping
                return proof
            if op.results:
                _require(len(op.results) == 1 and result is not None, 'unsupported result')
                env[op.results[0]] = result
        _require(jumped, 'callback does not reach a qualified return')


def prepare(source, shims, artifacts, *, work):
    """Consume original callback proofs and the original descriptor forwarder."""
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.xdsl_dialects._common import text
    from merlin.llvmlower.device_shim import emit_logical_translation_unit
    from merlin.llvmlower.descriptor_result_identity import rewrite_descriptor_result_identities
    from merlin.llvmlower.borrowed_effects import BorrowedBufferEffects
    work = Path(work); work.mkdir(parents=True, exist_ok=True)
    shim = emit_logical_translation_unit('gemmini', shims,
        kernel_symbol_for=lambda s: s + '_kernel', descriptor_results=True)
    by_symbol = {a.symbol: a for a in artifacts}
    identities, rows = [], []
    candidates = {i.symbol for i in shim.result_identities}
    for symbol in shims:
        if symbol not in candidates:
            rows.append(dict(symbol=symbol, applied=False,
                             reason='requires one original descriptor result'))
    for identity in shim.result_identities:
        try:
            proof = qualify(by_symbol[identity.symbol + '_kernel'], shims[identity.symbol][0])
        except Refused as error:
            rows.append(dict(symbol=identity.symbol, applied=False, reason=str(error)))
            continue
        digest = hashlib.sha256(json.dumps(proof, sort_keys=True).encode()).hexdigest()
        identities.append(replace(identity, borrowed_effects=BorrowedBufferEffects(
            'authored-closed-dma-cfg:' + digest, True, True, True)))
        rows.append(dict(symbol=identity.symbol, applied=True, proof=proof))
    module = parse_mlir_text(source)
    report = rewrite_descriptor_result_identities(module, tuple(identities)) if identities else []
    for row in rows:
        if row['applied']:
            row['calls'] = sum(r['calls'] for r in report if r['symbol'] == row['symbol'])
            row['applied'] = row['calls'] > 0
    applied = any(row['applied'] for row in rows)
    selected = text(module) if applied else source
    if not applied:
        shim = emit_logical_translation_unit('gemmini', shims, kernel_symbol_for=lambda s: s + '_kernel')
    record = dict(schema='authored.host_destination_reuse.v1', source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  selected_sha256=hashlib.sha256(selected.encode()).hexdigest(), qualifications=rows,
                  rewrites=report, scope='closed raw movement; stock bufferization owns destination placement')
    paths = [Path(__file__), Path(__file__).with_name('host_runtime.py'),
             Path(__file__).with_name('host_runtime_worker.py'), *[Path(a.path) for a in artifacts]]
    from . import isa
    import merlin.llvmlower.device_shim as device_shim
    import merlin.llvmlower.descriptor_result_identity as descriptor
    paths += [Path(isa.__file__), Path(device_shim.__file__), Path(descriptor.__file__)]
    record['pins'] = [dict(path=str(p.resolve()), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]
    record['selected_shim_sha256'] = hashlib.sha256(shim.text.encode()).hexdigest()
    record['entry_alias_policy'] = 'disjoint direct; overlap stages every output and copies in original order'
    (work / 'qualification.json').write_text(json.dumps(record, indent=2, default=str))
    (work / 'source.mlir').write_text(source)
    (work / 'selected.mlir').write_text(selected)
    return selected, shim.text, record, applied


def edit_stock_pipeline():
    """Use existing declared edits in the isolated stock-lowering worker."""
    from merlin.llvmlower import pipeline
    from merlin.llvmlower.tensor_helper_inlining import edit_pipeline as inline
    from merlin.llvmlower.descriptor_result_identity import edit_pipeline as identities
    pipeline._UPSTREAM_PASSES = identities(inline(pipeline._UPSTREAM_PASSES))


def alias_safe_entry(abi, entry, symbol, typedefs, declarations, locals_, addresses):
    """Keep the original appended-output ABI when callers overlap buffers.

    Ordinary disjoint calls go directly to the stock destination-passing entry.
    On overlap, standard runtime allocation/memcpy preserves the original fresh
    results and ordered final copies. No tensor operations are implemented here.
    """
    slots, inputs = abi['slots'], abi['inputs']
    sizes = [prod(s['shape']) * max(1, int(s['dtype'][1:]) // 8) for s in slots]
    pairs = [(a, b) for a in range(len(slots)) for b in range(max(a+1, inputs), len(slots))]
    overlap = ' || '.join(f'overlap(p{a}, {sizes[a]}, p{b}, {sizes[b]})' for a, b in pairs) or '0'
    prefix = '\n'.join(typedefs) + '''
_Static_assert(sizeof(long) == 8 && sizeof(void *) == 8, "ranked ABI requires LP64");
extern void *malloc(__SIZE_TYPE__);
extern void free(void *);
extern void *memcpy(void *, const void *, __SIZE_TYPE__);
static int overlap(void *a, __SIZE_TYPE__ as, void *b, __SIZE_TYPE__ bs) {
  __UINTPTR_TYPE__ x = (__UINTPTR_TYPE__)a, y = (__UINTPTR_TYPE__)b;
  return x <= y ? y-x < as : x-y < bs;
}
'''
    prefix += f'extern void _mlir_ciface_{entry}(' + ', '.join(declarations) + ');\n'
    prefix += f'void {symbol}(' + ', '.join(f'void *p{i}' for i in range(len(slots))) + ') {\n'
    prefix += '\n'.join(locals_) + f'\n  int staged = {overlap};\n'
    for i in range(inputs, len(slots)):
        prefix += f'''  void *allocation{i} = 0;
  if (staged) {{
    allocation{i} = malloc({sizes[i] + 63});
    if (!allocation{i}) __builtin_trap();
    d{i}.allocated = allocation{i};
    d{i}.aligned = (void *)(((__UINTPTR_TYPE__)allocation{i} + 63) & ~(__UINTPTR_TYPE__)63);
  }}
'''
    prefix += f'  _mlir_ciface_{entry}(' + ', '.join(addresses) + ');\n'
    for i in range(inputs, len(slots)):
        prefix += f'  if (staged) {{ memcpy(p{i}, d{i}.aligned, {sizes[i]}); free(allocation{i}); }}\n'
    return prefix + '}\n'
