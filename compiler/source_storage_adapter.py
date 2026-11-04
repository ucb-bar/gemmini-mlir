"""Bind actual source full writers and emitted pointer uses to shared storage.

Only the opt-in storage planner uses these boundaries. The target encoder and
shared host tensor store still emit every original operation.
"""
from hashlib import sha256
from math import prod
from pathlib import Path

from xdsl.dialects import llvm
from xdsl.dialects.builtin import IntegerAttr, i64
from merlin.llvmlower.fresh_tensor_writer import PrivateWorkspaceContract
from merlin.llvmlower.host_tensor_codegen import HostLinalg
from merlin.xdsl_dialects._common import text


class StorageEmissionCapture:
    def __init__(self, fn, source_sha256):
        import inspect
        self.fn, self.source_sha256 = fn, source_sha256
        self.active = None
        self.writers = {}
        self.device_writes = []
        self.terminal_completions = {}
        self.terminal_effects = {}
        self.writer_pin = sha256(inspect.getsource(HostLinalg.store_result).encode()).hexdigest()
        self.encoder_pin = sha256(Path(__file__).with_name('encoding.py').read_bytes()).hexdigest()

    def record(self, operation):
        if self.active is not None:
            self.active['operations'].append(operation)

    def _anchor(self, kind):
        op = llvm.ConstantOp(IntegerAttr(0, i64), i64)
        op.attributes['merlin.private_storage_boundary'] = IntegerAttr(kind, i64)
        return self.fn.add(op)

    def begin(self, source_op_index):
        if self.active is not None:
            raise ValueError('unfinished source storage emission boundary')
        begin = self._anchor(0)
        self.active = dict(index=source_op_index, begin=begin, operations=[], writers=[])

    def full_writer(self, value, implementation=None):
        # Called only after an actual writer has emitted all logical stores.
        # Each counted route supplies its own implementation, not an opcode.
        if self.active is None:
            raise ValueError('full writer outside actual emission')
        import inspect
        effect_pin=(self.writer_pin if implementation is None else
            sha256(Path(inspect.getsourcefile(implementation)).read_bytes()).hexdigest())
        self.active['writers'].append((value,effect_pin))

    def device_full_writer(self, value, *, inputs, output_pointer, input_pointers,
                           shape, emitter, receipt):
        """Qualify the actual complete dense raw-i32 writer, after emission.

        This narrow binding uses emit_dense's checked whole-domain schedule and
        its unchanged packed-output Encoder adapter. Fused, sliced, batched and
        nonzero-seed paths must establish their own write contract separately.
        """
        import inspect
        from merlin.llvmlower.host_tensor_codegen import tensor_shape
        if self.active is None:
            raise ValueError('device writer outside actual emission')
        if (tuple(tensor_shape(value.type))!=tuple(shape)
            or len(shape)!=2 or any(type(d) is not int or d<=0 for d in shape)
            or str(value.type.element_type)!='i32'
            or len(inputs)!=2 or len(input_pointers)!=2
            or output_pointer in input_pointers
            or value in inputs
            or not receipt.get('kernel_sha256') or not receipt.get('schedule_plan_sha256')):
            return False
        # The completing instruction must really have been emitted on the
        # current continuation, not merely appear in a schedule description.
        opaque=[op for op in self.active['operations'] if op.name=='llvm.inline_asm']
        if (not opaque or opaque[-1].properties['asm_string'].data!='fence'
            or opaque[-1].parent is not self.fn.blk):
            return False
        emitter_pin=sha256(Path(inspect.getsourcefile(emitter)).read_bytes()).hexdigest()
        effect_pin=sha256((emitter_pin+self.encoder_pin+receipt['kernel_sha256']).encode()).hexdigest()
        self.active['writers'].append((value,effect_pin))
        self.device_writes.append(dict(source_op_index=self.active['index'],
            logical_shape=list(shape),logical_dtype='i32',logical_bytes=prod(shape)*4,
            packed_row_pitch=shape[-1],full_logical_output=True,
            output_disjoint_from_inputs=True,completion='actual_final_fence_on_continuation',
            selected_schedule=receipt['selected_schedule'],
            schedule_kernel_sha256=receipt['kernel_sha256'],
            schedule_plan_sha256=receipt['schedule_plan_sha256'],
            emitter_implementation_sha256=emitter_pin,
            effect_witness_sha256=effect_pin))
        return True

    def convolution_full_writer(self, value, *, inputs, output_pointer,
                                input_pointers, input_shape, weight_shape,
                                strides, dilation, emitter, receipt):
        """Bind the emitted complete raw-i32 NHWC window convolution.

        The actual checked complete-invocation window emitter owns all output
        pixels and column phases. Pointwise/dense and staged readout adapters
        require their separate output-view contracts and remain outside this
        qualification. No padding, bias, scaling or activation is introduced.
        """
        import inspect
        from merlin.llvmlower.host_tensor_codegen import tensor_shape
        from merlin.compile.scheduling.convolution_exploration import emit_convolution_schedule
        if self.active is None:
            raise ValueError('convolution writer outside actual emission')
        if (len(input_shape)!=4 or len(weight_shape)!=4
            or len(strides)!=2 or len(dilation)!=2
            or any(type(d) is not int or d<=0 for d in
                (*input_shape,*weight_shape,*strides,*dilation))):
            return False
        batch,height,width,channels=input_shape
        kh,kw,weight_channels,columns=weight_shape
        shape=(batch,(height-(kh-1)*dilation[0]-1)//strides[0]+1,
            (width-(kw-1)*dilation[1]-1)//strides[1]+1,columns)
        if (weight_channels!=channels or any(d<=0 for d in shape)
            or tuple(tensor_shape(value.type))!=shape
            or str(value.type.element_type)!='i32'
            or len(inputs)!=2 or len(input_pointers)!=2
            or value in inputs or output_pointer in input_pointers
            or receipt.get('schema')!='merlin.convolution_schedule.v1'
            or receipt.get('whole_invocation') is False or 'staged_readout' in receipt
            or not receipt.get('source_sha256') or not receipt.get('plan_sha256')):
            return False
        opaque=[op for op in self.active['operations'] if op.name=='llvm.inline_asm']
        if (not opaque or opaque[-1].properties['asm_string'].data!='fence'
            or opaque[-1].parent is not self.fn.blk):
            return False
        emitter_pin=sha256(Path(inspect.getsourcefile(emitter)).read_bytes()).hexdigest()
        schedule_pin=sha256(Path(inspect.getsourcefile(emit_convolution_schedule)).read_bytes()).hexdigest()
        import json
        receipt_pin=sha256(json.dumps(receipt,sort_keys=True).encode()).hexdigest()
        effect_pin=sha256((emitter_pin+schedule_pin+self.encoder_pin+receipt_pin).encode()).hexdigest()
        self.active['writers'].append((value,effect_pin))
        self.device_writes.append(dict(source_op_index=self.active['index'],
            kind='complete_window_convolution',logical_shape=list(shape),logical_dtype='i32',
            logical_bytes=prod(shape)*4,packed_row_pitch=columns,
            original_input_shape=list(input_shape),original_weight_shape=list(weight_shape),
            original_strides=list(strides),original_dilation=list(dilation),padding=[0,0,0,0],
            full_logical_output=True,output_disjoint_from_inputs=True,
            completion='actual_final_fence_on_continuation',
            selected_schedule=receipt.get('emitted_mode'),
            emitted_schedule_receipt_sha256=receipt_pin,
            schedule_source_sha256=receipt['source_sha256'],
            schedule_plan_sha256=receipt['plan_sha256'],
            realization_sha256=receipt.get('realization_sha256'),
            emitter_implementation_sha256=emitter_pin,
            schedule_implementation_sha256=schedule_pin,effect_witness_sha256=effect_pin))
        return True

    def finish(self):
        if self.active is None:
            return
        active = self.active
        end = self._anchor(1)
        self.active = None
        for value,effect_pin in active['writers']:
            self.writers[value] = (active['index'], active['begin'], end,effect_pin)
        # Selected target adapters end a synchronous source emission with the
        # same full completion fence used by the incumbent. Refuse qualification
        # for opaque pointer terminals if that boundary is absent.
        opaque = [op for op in active['operations'] if op.name == 'llvm.inline_asm']
        complete = bool(opaque and opaque[-1].properties['asm_string'].data == 'fence'
                        and opaque[-1].parent is end.parent)
        for op in active['operations']:
            if op.name in ('llvm.load', 'llvm.store'):
                self.terminal_completions[op] = end
            elif (complete and op.name == 'llvm.inline_asm'
                  and op.properties['asm_string'].data.startswith('.insn r ')
                  and not op.results):
                self.terminal_effects[op] = self.encoder_pin
                self.terminal_completions[op] = end

    def apply(self, kernel, names, tensors, order, source_values):
        import source_storage_reuse
        from merlin.llvmlower.host_tensor_codegen import tensor_shape
        from merlin.llvmlower.source_storage_lifetimes import EmittedStorageRequest
        self.finish()
        entry = kernel.body.blocks.first
        by_name = dict(zip(order, entry.args))
        canonical = {}
        for value, name in names.items():
            canonical.setdefault(name, value)
        bindings = {by_name[name]: canonical[name] for name in order}
        requests = []
        for value, (index, begin, end,effect_pin) in self.writers.items():
            name = names.get(value)
            if name not in tensors or tensors[name]['role'] != 'intermediate':
                continue
            size = prod(tensor_shape(value.type)) * value.type.element_type.bitwidth // 8
            contract = PrivateWorkspaceContract(size, 64, effect_pin, True, True, True)
            requests.append(EmittedStorageRequest(value, by_name[name], index,
                contract, True, begin, end))
        plan = source_storage_reuse.plan_emitted(kernel, source_sha256=self.source_sha256,
            requests=requests, abi_bindings=bindings,
            terminal_effects=self.terminal_effects, terminal_completions=self.terminal_completions)
        argument_names = {argument: name for name, argument in by_name.items()}
        aliases = {argument_names[old]: argument_names[new]
                   for old, new in plan.value_aliases.items()}
        for old, new in plan.value_aliases.items():
            old.replace_all_uses_with(new)
        for name in aliases:
            entry.erase_arg(by_name[name])
            del tensors[name]
        order[:] = [name for name in order if name not in aliases]
        for value, name in list(names.items()):
            names[value] = aliases.get(name, name)
        for binding in source_values:
            binding['tensor'] = aliases.get(binding['tensor'], binding['tensor'])
        kernel.properties['function_type'] = llvm.LLVMFunctionType([arg.type for arg in entry.args])
        record = plan.record()
        record.update(abi_allocation_aliases=aliases,
            realized_llvm_sha256=sha256(text(kernel).encode()).hexdigest(),
            full_writer_implementation_sha256=self.writer_pin,
            counted_writer_implementation_sha256s=sorted({row[3] for row in self.writers.values()}),
            actual_full_writer_count=len(self.writers),
            private_full_writer_requests=len(requests),
            device_output_writes=self.device_writes,
            encoder_implementation_sha256=self.encoder_pin)
        return record
