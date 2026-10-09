"""Source-bound native convolution descriptors, not an im2col/library dispatch.

This target edge emits the pinned header's actual LOOP_CONV_WS register program.
It intentionally supports only one complete, capacity-fitting NHWC integer tile.
It does not change a compiler's source semantics, insert completion instructions,
claim numerical certification, or estimate cycles. Unsupported cases raise.
"""

from __future__ import annotations

import ast
import hashlib
import json
import struct
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Mapping

from merlin.perf.hw_counters import _module_lines, _operand_refs
from merlin.targetgen.address_space import derive_address_space
from merlin.targetgen.capability_discovery import (
    _balanced_end,
    _split_top_level,
    parse_c_header,
)

from .rocc_semantics import derived_readout_bits


class UnsupportedNativeConv(ValueError):
    """No proven native descriptor for this source operation/target revision."""


def _require(condition, message):
    if not condition:
        raise UnsupportedNativeConv(message)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _integer(model, name):
    macro = model.macro(name)
    _require(macro is not None and macro.int_value is not None, f"UNKNOWN header integer {name}")
    return macro.int_value


def _expression(source, values):
    """Restricted C integer packing expression, never Python eval."""
    cleaned = source
    for cast in ("(uint64_t)", "(uint32_t)", "(acc_scale_t)"):
        cleaned = cleaned.replace(cast, "")
    node = ast.parse(cleaned, mode="eval").body

    def visit(item):
        if isinstance(item, ast.Constant) and type(item.value) is int:
            return item.value
        if isinstance(item, ast.Name) and item.id in values:
            return values[item.id]
        if (
            isinstance(item, ast.Call)
            and isinstance(item.func, ast.Name)
            and item.func.id == "acc_scale_t_to_acc_scale_t_bits"
            and len(item.args) == 1
            and not item.keywords
        ):
            return int.from_bytes(struct.pack("<f", visit(item.args[0])), "little")
        if isinstance(item, ast.BinOp):
            a, b = visit(item.left), visit(item.right)
            if isinstance(item.op, ast.BitOr):
                return a | b
            if isinstance(item.op, ast.LShift):
                return a << b
            if isinstance(item.op, ast.Mult):
                return a * b
            if isinstance(item.op, ast.Div):
                _require(b > 0 and a % b == 0, "nonintegral capacity partition")
                return a // b
        raise UnsupportedNativeConv(f"UNKNOWN header expression {source}")

    # Reject fields spilling into their neighbor instead of silently truncating.
    def fields(item):
        if isinstance(item, ast.BinOp) and isinstance(item.op, ast.BitOr):
            return fields(item.left) + fields(item.right)
        if isinstance(item, ast.BinOp) and isinstance(item.op, ast.LShift):
            return [(visit(item.right), visit(item.left))]
        return [(0, visit(item))]

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        packed = sorted(fields(node))
        for index, (shift, value) in enumerate(packed):
            end = packed[index + 1][0] if index + 1 < len(packed) else 64
            _require(
                0 <= shift < end <= 64 and 0 <= value < 1 << (end - shift), f"descriptor field overflow in {source}"
            )
    return visit(node)


def _narrow_store_proof(hw_text, *, full_bit, store_funct):
    """Trace the actual store output mux and prove its read-full bit is zero."""
    lines, error = _module_lines(hw_text, "LoopConvSt")
    _require(lines is not None, error)
    definitions = dict(line.strip().split(" = ", 1) for line in lines if " = " in line and line.strip().startswith("%"))
    # Bind by the module's output port order, not a guessed SSA name/namehint.
    header = lines[0]
    ports = [
        part.strip().split(":", 1)[0][4:].strip()
        for part in _split_top_level(header[header.index("(") + 1 : header.rfind(")")], ",")
        if part.strip().startswith("out ")
    ]
    output = next((line.strip() for line in lines if line.strip().startswith("hw.output ")), "")
    refs = output.removeprefix("hw.output ").split(" :", 1)[0].split(",")
    _require("io_cmd_bits_rs2" in ports and "io_cmd_bits_inst_funct" in ports, "UNKNOWN convolution store output ports")
    rs2 = refs[ports.index("io_cmd_bits_rs2")].strip()
    funct = refs[ports.index("io_cmd_bits_inst_funct")].strip()

    def constant(ref):
        body = definitions.get(ref, "")
        _require(body.startswith("hw.constant "), "UNKNOWN store predicate constant")
        return int(body.split()[1])

    def known_bit(ref, bit):
        body = definitions.get(ref, "")
        if body.startswith("hw.constant "):
            return (constant(ref) >> bit) & 1
        operands = _operand_refs(body, "comb.mux")
        if operands:
            left, right = known_bit(operands[1], bit), known_bit(operands[2], bit)
            return left if left == right else None
        operands = _operand_refs(body, "comb.concat")
        if operands:
            widths = [int(t.strip().removeprefix("i")) for t in body.rsplit(" : ", 1)[1].split(",")]
            for operand, width in reversed(list(zip(operands, widths, strict=True))):
                if bit < width:
                    return known_bit(operand, bit)
                bit -= width
        return None

    mux = _operand_refs(definitions.get(rs2, ""), "comb.mux")
    _require(mux is not None, "UNKNOWN convolution store output mux")
    predicate = definitions.get(mux[0], "")
    _require(predicate.startswith("comb.icmp bin eq "), "UNKNOWN store output predicate")
    compared = predicate.removeprefix("comb.icmp bin eq ").split(" :", 1)[0].split(",")
    compared = [ref.strip() for ref in compared]
    _require(funct in compared, "store predicate not bound to output funct")
    _require(
        constant(compared[1 - compared.index(funct)]) == store_funct, "store predicate does not match pinned opcode"
    )
    _require(
        known_bit(mux[1], full_bit.bit_length() - 1) == 0, "UNKNOWN native convolution full-width readout semantics"
    )
    return {
        "module": "LoopConvSt",
        "output_ssa": rs2,
        "store_branch_ssa": mux[1],
        "full_width_mask": full_bit,
        "full_width_bit": 0,
    }


#: How this RTL spells a descriptor register whose software parameter has another name. The register
#: names are read off the elaborated module; only the renaming is stated here.
_DESCRIPTOR_ALIASES = {
    "weights_dram_addr": "weights",
    "output_dram_addr": "output",
    "bias_dram_addr": "bias",
    "input_dram_addr": "input",
    "a_ex_spad_id": "a_spad_id",
    "b_ex_spad_id": "b_spad_id",
}


def _descriptor_registers(hw_text):
    """(definitions, [(ssa, next_value, field, width)]) for this RTL's convolution descriptor state.

    This is the set of fields the decode proof can REACH -- the `%loops_*` registers the elaborated
    module keeps. It is not the set of fields the device honors, and reading it as one is a mistake
    with a measured cost: `padding` has no register here, and a padded convolution emitted with that
    field zeroed computes the wrong answer on hardware while the proof still reports success.
    """
    lines, error = _module_lines(hw_text, "LoopConv")
    _require(lines is not None, error)
    definitions = dict(line.strip().split(" = ", 1) for line in lines if " = " in line and line.strip().startswith("%"))
    registers = []
    for ref, body in definitions.items():
        if not ref.startswith("%loops_") or not body.startswith("seq.firreg "):
            continue
        suffix = ref.removeprefix("%loops_").split("_", 1)[1]
        field = suffix.removeprefix("outer_bounds_").removeprefix("inner_bounds_")
        registers.append((ref, body.split()[1], _DESCRIPTOR_ALIASES.get(field, field), int(body.rsplit(" : i", 1)[1])))
    return definitions, registers


def _descriptor_decode_proof(hw_text, instructions, parameters, *, pointer_fields):
    """Overapproximate decoder mux paths for these exact descriptor values.

    Every possible non-hold update must equal the requested value. Unknown
    control inputs explore both branches; unsupported data paths refuse. This
    proves descriptor register values conditional on command acceptance, not
    sequencer progress or convolution arithmetic. Pointer truncation becomes an
    explicit physical-address-width obligation, never presumed identity.
    """
    definitions, registers = _descriptor_registers(hw_text)
    registers = [row for row in registers if row[2] in parameters]
    _require(registers, "UNKNOWN convolution descriptor state registers")
    pointers = {parameters[name] for name in pointer_fields if isinstance(parameters.get(name), str)}
    covered = set()
    address_widths = set()
    proof_rows = []
    for instruction in instructions:
        command = {
            "%cmd_q.io_deq_bits_cmd_inst_funct": instruction["funct"],
            "%cmd_q.io_deq_bits_cmd_rs1": instruction["rs1"],
            "%cmd_q.io_deq_bits_cmd_rs2": instruction["rs2"],
        }

        @lru_cache(None)
        def values(ref):
            if ref in command:
                return frozenset([command[ref]])
            body = definitions.get(ref, "")
            if body.startswith("hw.constant "):
                if body.split()[1] in ("true", "false"):
                    return frozenset([int(body.split()[1] == "true")])
                width = int(body.rsplit(" : i", 1)[1])
                return frozenset([int(body.split()[1]) % (1 << width)])
            if body.startswith("seq.firreg ") or not body:
                return frozenset([ref])
            mux = _operand_refs(body, "comb.mux")
            if mux:
                conditions = values(mux[0])
                choices = set()
                if any(value != 0 for value in conditions):
                    choices.update(values(mux[1]))
                if any(value != 1 for value in conditions):
                    choices.update(values(mux[2]))
                return frozenset(choices)
            if body.startswith("comb.extract "):
                head, tail = body.removeprefix("comb.extract ").split(" from ", 1)
                offset = int(tail.split()[0])
                width = int(body.rsplit("-> i", 1)[1])
                result = set()
                for value in values(head):
                    if type(value) is int:
                        result.add((value >> offset) & ((1 << width) - 1))
                    elif value in pointers:
                        _require(offset == 0, "unsupported pointer address transformation")
                        result.add(("pointer", value, width))
                    else:
                        result.add(ref)
                return frozenset(result)
            if body.startswith("comb.icmp "):
                tail = body.removeprefix("comb.icmp ").removeprefix("bin ")
                predicate, operands = tail.split(" ", 1)
                # An operand list may be followed by an attribute dictionary before the type, and
                # cutting only at " :" carries that dictionary into the second operand's name. The
                # name then resolves to nothing and an EXACT comparison degrades to "either" -- which
                # is sound but useless: it is what made a descriptor field the header sets to 3 read
                # back as "1 or 3" and refuse a convolution the device runs correctly.
                operands = operands.split(" {", 1)[0].split(" :", 1)[0]
                a, b = [s.strip() for s in operands.split(",")]
                result = set()
                for left, right in product(values(a), values(b)):
                    if type(left) is int and type(right) is int and predicate in ("eq", "ne"):
                        result.add(int((left == right) == (predicate == "eq")))
                    else:
                        result.update((0, 1))
                return frozenset(result)
            for op in ("and", "or", "xor", "concat"):
                refs = _operand_refs(body, "comb." + op)
                if refs is None:
                    continue
                widths = [int(t.strip().removeprefix("i")) for t in body.rsplit(" : ", 1)[1].split(",")]
                result = set()
                for operands in product(*(values(item) for item in refs)):
                    if op == "and" and 0 in operands:
                        result.add(0)
                    elif op == "or" and widths == [1] and 1 in operands:
                        result.add(1)
                    elif all(type(value) is int for value in operands):
                        value = operands[0]
                        for at, operand in enumerate(operands[1:], 1):
                            if op == "and":
                                value &= operand
                            elif op == "or":
                                value |= operand
                            elif op == "xor":
                                value ^= operand
                            else:
                                value = (value << widths[at]) | operand
                        result.add(value)
                    elif widths == [1]:
                        result.update((0, 1))
                    else:
                        result.add(ref)
                return frozenset(result)
            return frozenset([ref])

        for ref, next_value, field, width in registers:
            possible = set(values(next_value)) - {ref}
            if not possible:
                continue
            expected = parameters[field]
            if isinstance(expected, str):
                expected = ("pointer", expected, width)
                address_widths.add(width)
            _require(
                possible == {expected},
                f"pinned RTL descriptor mismatch: {instruction['name']} -> {ref}: {possible!r} != {expected!r}",
            )
            covered.add(field)
            proof_rows.append(
                {"instruction": instruction["name"], "register": ref, "parameter": field, "register_width": width}
            )
    required = {field for _, _, field, _ in registers}
    _require(covered == required, f"UNKNOWN descriptor fields {sorted(required - covered)}")
    # Fields with no `%loops_*` register. This USED to refuse them at anything but zero, on the
    # reading that a field the descriptor state does not hold is a field the device ignores. That
    # reading is wrong and was measured wrong: the header's own tiler passes `padding` on every call,
    # and a padded convolution emitted with it zeroed computes the wrong answer on hardware while
    # this proof still reports success. What the enumeration actually establishes is which fields the
    # proof can REACH; the rest are carried as the reference tiler carries them and reported here as
    # outside the proof, which is the same scope the sequencer's own arithmetic already sits in.
    absent = sorted(set(parameters) - covered)
    return {
        "status": "verified_exact_descriptor_register_updates",
        "module": "LoopConv",
        "scope": "all_possible_non_hold_mux_paths_for_actual_descriptor_values",
        "conditional_on": "descriptor command accepted by sequencer",
        "register_updates": proof_rows,
        "fields_outside_this_proof": {field: parameters[field] for field in absent},
        "required_physical_address_bits": sorted(address_widths),
        "sequencer_progress_and_arithmetic": "UNPROVEN",
    }


class NativeConvContract:
    """Load explicit sources; verify hardware bytes against the promoted facts."""

    def __init__(self, *, header: Path, params: Path, facts: Path, core_hw: Path):
        self.header_text = header.read_text()
        self.header = parse_c_header(header)
        self.params = parse_c_header(params)
        record = json.loads(facts.read_text())
        hw_bytes = core_hw.read_bytes()
        self.hw_text = hw_bytes.decode()
        _require(_sha(hw_bytes) == record["inputs"]["core_hw_sha256"], "stale core hardware identity")
        self.provenance = {
            "header_sha256": _sha(header.read_bytes()),
            "params_sha256": _sha(params.read_bytes()),
            "facts_sha256": _sha(facts.read_bytes()),
            "core_hw_sha256": _sha(hw_bytes),
        }
        tables = [entry for entry in record["facts"]["interfaces"] if entry.get("name") == "funct_decode_table"]
        _require(len(tables) == 1, "UNKNOWN instruction decode table")
        self.opcodes = {name: int(code) for code, name in tables[0]["names"].items()}
        self.custom_opcode = tables[0]["custom_opcode"]
        self.funct3 = tables[0]["funct3"]
        space = derive_address_space("gemmini", facts=record)
        stores = {store.name: store for store in space.stores}
        self.spad, self.acc = stores["scratchpad"], stores["accumulator"]
        self.dim = _integer(self.params, "DIM")
        _require(self.dim == self.spad.row_elems == self.acc.row_elems, "header/RTL mesh mismatch")
        _require(
            _integer(self.params, "BANK_NUM") * _integer(self.params, "BANK_ROWS") == self.spad.total_rows
            and _integer(self.params, "ACC_ROWS") == self.acc.total_rows,
            "header/RTL capacity mismatch",
        )
        types = {alias: underlying for alias, underlying, _ in self.params.typedefs}
        _require(
            types.get("elem_t") == f"int{self.spad.element_bits}_t"
            and types.get("acc_t") == f"int{self.acc.element_bits}_t",
            "unsupported signed element ABI",
        )
        identity = self.params.macro("ACC_SCALE_IDENTITY")
        _require(
            identity is not None and identity.body.strip() in ("1.0", "1.0f"), "UNKNOWN identity accumulator scale"
        )
        _require(
            types.get("acc_scale_t") == "float" and types.get("acc_scale_t_bits") == "uint32_t",
            "unsupported accumulator scale representation",
        )
        converter = "static acc_scale_t_bits acc_scale_t_to_acc_scale_t_bits(acc_scale_t x)"
        _require(converter in self.header_text, "UNKNOWN scale conversion")
        start = self.header_text.index("{", self.header_text.index(converter))
        end = _balanced_end(self.header_text, start)
        actual = " ".join(self.header_text[start:end].split())
        expected = "{ union { acc_scale_t_bits b; acc_scale_t f; } un; un.f = x; return un.b; }"
        _require(actual == expected, "unsupported scale conversion implementation")
        #: Every descriptor field this ELABORATED revision holds a `%loops_*` register for. Read as
        #: "which fields the decode proof can reach" -- NOT as "which fields the device ignores". The
        #: two were conflated once and it cost a wrong answer: `padding` has no such register, so the
        #: emitter passed 0 for it, and every padded convolution then computed the wrong result on
        #: hardware while the proof reported success. Absence of a register here is absence of PROOF.
        self.descriptor_fields = frozenset(field for _, _, field, _ in _descriptor_registers(self.hw_text)[1])
        #: The address the header's own tiler substitutes for a bias it is not going to load. It is a
        #: sentinel, not a pointer, and it is not zero -- read it off the header rather than guess.
        marker = "bias = (acc_t*)"
        _require(marker in self.header_text, "UNKNOWN absent-bias sentinel")
        sentinel = self.header_text[self.header_text.index(marker) + len(marker) :].split(";", 1)[0].strip()
        _require(sentinel.isdigit(), f"UNKNOWN absent-bias sentinel {sentinel!r}")
        self.absent_bias_address = int(sentinel)
        #: Whether this header packs several pixels into one staged row. The formula that follows from
        #: it lives in `sp_tiled_conv`; its presence is a compile-time switch in the params header.
        self.packs_pixels_per_row = self.params.macro("HAS_FIRST_LAYER_OPTIMIZATIONS") is not None
        self.readout = _narrow_store_proof(
            hw_bytes.decode(),
            full_bit=derived_readout_bits(_integer(self.params, "ADDR_LEN"))["full_c_bit"],
            store_funct=self.opcodes["STORE_CMD"],
        )
        self.capacity = {}
        constants = {name: _integer(self.params, name) for name in ("BANK_NUM", "BANK_ROWS", "ACC_ROWS")}
        for name in ("max_spad_rows", "max_acc_rows"):
            assignments = [
                line.split("=", 1)[1].split(";", 1)[0].strip()
                for line in self.header_text.splitlines()
                if line.strip().startswith(f"const int {name} =")
            ]
            # Same expressions may occur in other tilers; require agreement.
            values = {_expression(value, constants) for value in assignments}
            _require(len(values) == 1, f"UNKNOWN {name} partition")
            self.capacity[name] = values.pop()


def native_entry_instructions(contract, *, output_channels, activation, a_stride=1, c_stride=1, acc_scale=1.0):
    """Exact header expansion of the library's no-pool native-conv entry configs.

    ``a_stride`` and ``c_stride`` are the two execute-unit strides the header's own tiler passes for a
    convolution -- ``stride >> downsample`` and ``input_dilation``. They were pinned at 1 here, which
    is the ONE spelling that cannot express a strided convolution: the sequencer walks the output
    image and the mvin row stride is what turns that walk into the strided input walk. ``acc_scale``
    is the readout multiplier, a per-tensor float the store path applies before it clips.
    """
    ex = dict(
        dataflow=_integer(contract.header, "WEIGHT_STATIONARY"),
        sys_act=0,
        sys_shift=0,
        sys_acc_scale=0,
        C_stride=c_stride,
        A_stride=a_stride,
        A_transpose=0,
        B_transpose=0,
        set_only_strides=0,
        act_mx_fmt=0,
        wgt_mx_fmt=0,
        out_mx_fmt=0,
        uselut=0,
    )
    st = dict(
        stride=output_channels * contract.spad.element_bits // 8,
        acc_act=activation,
        acc_scale=acc_scale,
        pool_stride=0,
        pool_size=0,
        pool_out_dim=0,
        porows=0,
        pocols=0,
        orows=0,
        ocols=0,
        upad=0,
        lpad=0,
    )
    rows = []
    for name, parameters in (("gemmini_extended2_config_st", st), ("gemmini_extended3_config_ex", ex)):
        macro = contract.header.macro(name)
        _require(macro is not None and set(macro.params) == set(parameters), "UNKNOWN native entry config parameters")
        values = dict(
            parameters,
            CONFIG_ST=_integer(contract.header, "CONFIG_ST"),
            CONFIG_EX=_integer(contract.header, "CONFIG_EX"),
        )
        marker = "ROCC_INSTRUCTION_RS1_RS2("
        _require(macro.body.count(marker) == 1, "UNKNOWN native entry config instruction")
        start = macro.body.index(marker) + len(marker) - 1
        end = _balanced_end(macro.body, start)
        args = _split_top_level(macro.body[start + 1 : end - 1], ",")
        _require(len(args) == 4 and args[0].strip() == "XCUSTOM_ACC", "unsupported config encoding")
        funct = _integer(contract.header, args[3].strip())
        _require(funct == contract.opcodes["CONFIG_CMD"], "config opcode differs from RTL")
        rows.append(
            {
                "name": name,
                "funct": funct,
                "rs1": _expression(args[1].strip(), values),
                "rs2": _expression(args[2].strip(), values),
            }
        )
    return rows


def native_conv_single_tile(*, batch, in_rows, in_cols, kernel, stride, padding, kernel_dilation) -> dict[str, int]:
    """The ONE tile the pinned header's tiler would build to cover this whole convolution.

    Every expression is ``tiled_conv``'s loop body specialized to the single iteration that takes the
    whole image -- ``batches = batch_size``, ``porows/pocols = pool_out_*_dim``, ``pochs =
    out_channels``, ``krows = kcols = kernel_dim``, ``kchs = in_channels`` -- with pooling disabled.
    With pooling off the header itself sets ``pool_size = pool_stride = 1`` and ``pool_padding = 0``,
    so ``pool_out_*_dim`` collapse onto ``out_*_dim``, the tile origin ``(orow, ocol)`` is ``(0, 0)``,
    and the four pool pads are zero. What is left is the padded input window the sequencer walks:

        irow = orow * stride + krow * kernel_dilation - padding    ->  -padding
        irows = orows * stride + dilated_krows - 1
        lpad/upad = -icol/-irow when negative;  rpad/dpad = overhang past the input extent

    Returning them as data (rather than inlining them at the one call site) is what lets a test state
    the geometry a convolution implies without constructing a hardware contract.
    """
    _require(stride >= 1 and kernel_dilation >= 1 and kernel >= 1, "nonpositive convolution geometry")
    _require(padding >= 0, "negative padding is a different operation")
    dilated = kernel + (kernel_dilation - 1) * (kernel - 1)
    # The header's own assertion: `if (kernel_dim <= padding) { "kernel_dim must be larger than
    # padding" }`. It is stated against the UNdilated kernel, so it is read back that way.
    _require(padding < kernel, "native_loop_conv_requires_0_le_padding_lt_kernel")
    out_rows = (in_rows + 2 * padding - dilated) // stride + 1
    out_cols = (in_cols + 2 * padding - dilated) // stride + 1
    _require(out_rows > 0 and out_cols > 0, "convolution window does not fit the padded input")
    irow = icol = -padding
    irows = out_rows * stride + dilated - 1
    icols = out_cols * stride + dilated - 1
    return {
        "out_rows": out_rows,
        "out_cols": out_cols,
        "orows": out_rows,
        "ocols": out_cols,
        "irows": irows,
        "icols": icols,
        "lpad": -icol if icol < 0 else 0,
        "upad": -irow if irow < 0 else 0,
        "rpad": icol + icols - in_cols if icol + icols > in_cols else 0,
        "dpad": irow + irows - in_rows if irow + irows > in_rows else 0,
        # The header's own predicate, verbatim minus the terms this route already pins (no pooling,
        # no input dilation, no transposed input). It halves the rows moved in for a 1x1 stride-2
        # convolution -- ResNet-50's projection shortcuts -- and is not a free choice: with it set the
        # execute unit's A_stride becomes stride >> 1, so the two must be derived together.
        "downsample": int(stride == 2 and kernel == 1 and in_rows % 2 == 0 and in_cols % 2 == 0 and padding == 0),
        "batches": batch,
    }


def _address(base, element_offset, element_bytes):
    """The address the header's tiler would pass for an operand slice of a tile.

    ``None`` is the header's own null: a tile that is not the last contributor to its output does not
    store, and only the first loads the bias. It is not "offset zero", and conflating the two would
    have every tile write its partial sums over the finished result.
    """
    if element_offset is None:
        return 0
    byte = element_offset * element_bytes
    if isinstance(base, int):
        return base + byte
    return base if byte == 0 else f"{base}+{byte}"


def _expand_conv_macro(contract, macro, values):
    """The macro body's ROCC invocations, with these descriptor values substituted."""
    instructions = []
    rest = macro.body
    marker = "ROCC_INSTRUCTION_RS1_RS2("
    while marker in rest:
        start = rest.index(marker) + len(marker) - 1
        end = _balanced_end(rest, start)
        args = _split_top_level(rest[start + 1 : end - 1], ",")
        _require(len(args) == 4 and args[0].strip() == "XCUSTOM_ACC", "unsupported descriptor invocation")
        name = args[3].strip()
        funct = _integer(contract.header, name)
        _require(contract.opcodes.get(name.removeprefix("k_")) == funct, "header/RTL opcode mismatch")
        instructions.append(
            {
                "funct": funct,
                "name": name.removeprefix("k_"),
                "rs1": _expression(args[1].strip(), values),
                "rs2": _expression(args[2].strip(), values),
            }
        )
        rest = rest[end:]
    _require(len(instructions) == 7 and instructions[-1]["name"] == "LOOP_CONV_WS", "incomplete convolution program")
    return instructions


def _ceil_div(a, b):
    return -(-a // b)


def native_conv_staged_rows(
    *, dim, stride, kernel_dilation, downsample, batches, porows, pocols, ochs, krows, kcols, kchs
) -> dict[str, int]:
    """`tiled_conv_total_spad_rows` for one tile, with pooling off and nothing transposed.

    What occupies the scratchpad is the PADDED window the sequencer walks, which is a function of the
    tile's output extent and the stride -- not of the source tensor. Pricing the source tensor instead
    understates a padded convolution and badly overstates a strided one.
    """
    orows, ocols = porows, pocols
    krows_dilated = krows + (kernel_dilation - 1) * (krows - 1)
    kcols_dilated = kcols + (kernel_dilation - 1) * (kcols - 1)
    irows = orows * stride + krows_dilated - 1
    icols = ocols * stride + kcols_dilated - 1
    return {
        "input_rows": _ceil_div(kchs, dim) * batches * (irows >> downsample) * (icols >> downsample),
        "weight_rows": _ceil_div(ochs, dim) * kcols * krows * kchs,
        "accumulator_rows": _ceil_div(ochs, dim) * batches * orows * ocols,
    }


#: The tile extents, in the order `tiled_conv_stride_auto` holds them. Named because the search
#: treats three of them specially and an index-keyed transcription hides which.
_TILE_AXES = ("batches", "porows", "pocols", "pochs", "krows", "kcols", "kchs")


def native_conv_tile_shape(
    *,
    dim,
    max_spad_rows,
    max_acc_rows,
    batch_size,
    out_rows,
    out_cols,
    out_channels,
    kernel_dim,
    in_channels,
    stride,
    kernel_dilation,
    downsample,
) -> dict[str, int]:
    """The tile `tiled_conv_stride_auto` would choose, by its own search.

    Start from the whole convolution; while it does not fit, shrink the largest extent -- channel
    counts by a whole row width, because taking one channel off a tile that is already a row wide
    buys nothing -- while avoiding shrinking the output columns below one row width, which is what
    keeps the spatial array busy. Then grow the columns back, then grow anything else that still
    fits. A convolution that does not fit even one output pixel of one channel is refused.
    """
    args = {
        "batches": batch_size,
        "porows": out_rows,
        "pocols": out_cols,
        "pochs": out_channels,
        "krows": kernel_dim,
        "kcols": kernel_dim,
        "kchs": in_channels,
    }
    ceiling = dict(args)
    channel_axes = ("pochs", "kchs")

    def fits(candidate):
        rows = native_conv_staged_rows(
            dim=dim,
            stride=stride,
            kernel_dilation=kernel_dilation,
            downsample=downsample,
            batches=candidate["batches"],
            porows=candidate["porows"],
            pocols=candidate["pocols"],
            ochs=candidate["pochs"],
            krows=candidate["krows"],
            kcols=candidate["kcols"],
            kchs=candidate["kchs"],
        )
        return rows["input_rows"] + rows["weight_rows"] <= max_spad_rows and rows["accumulator_rows"] <= max_acc_rows

    while not fits(args):
        # The column axis is held back only while it is still narrower than one row of the array AND
        # there is another output row to give up instead.
        candidates = [
            axis for axis in _TILE_AXES if not (axis == "pocols" and args[axis] <= dim and args["porows"] > 1)
        ]
        axis = max(candidates, key=lambda name: args[name])
        _require(args[axis] > 1, "single native tile exceeds derived double-buffer capacity")
        if axis in channel_axes:
            args[axis] = (args[axis] // dim) * dim if args[axis] % dim else args[axis] - dim
            args[axis] = args[axis] or 1
        else:
            args[axis] -= 1
    for axes in (("pocols",), _TILE_AXES):
        growing = True
        while growing:
            growing = False
            for axis in axes:
                if args[axis] < ceiling[axis] and fits({**args, axis: args[axis] + 1}):
                    args[axis] += 1
                    growing = True
    return args


def native_conv_tiles(
    *,
    tile,
    batch_size,
    in_rows,
    in_cols,
    in_channels,
    out_channels,
    out_rows,
    out_cols,
    kernel_dim,
    stride,
    padding,
    kernel_dilation,
    downsample,
    packs_pixels_per_row,
    dim,
) -> list[dict]:
    """`tiled_conv`'s loop nest: one descriptor per tile, in the order the header issues them.

    Each entry carries the tile's inner bounds, its four input pads, and the ELEMENT offsets of the
    four operands from their tensor bases. ``None`` for an operand means the header passes a null
    there -- a tile that is not the last contributor to its output does not store, and only the first
    contributor loads the bias -- and that is load-bearing: a tile that stored every pass would write
    partial sums over finished outputs.
    """
    tiles: list[dict] = []
    num = {
        "kch": _ceil_div(in_channels, tile["kchs"]),
        "poch": _ceil_div(out_channels, tile["pochs"]),
        "b": _ceil_div(batch_size, tile["batches"]),
        "porow": _ceil_div(out_rows, tile["porows"]),
        "pocol": _ceil_div(out_cols, tile["pocols"]),
        "krow": _ceil_div(kernel_dim, tile["krows"]),
        "kcol": _ceil_div(kernel_dim, tile["kcols"]),
    }
    b_reuse = num["kch"] * num["poch"] * num["krow"] * num["kcol"] <= 2
    a_reuse = num["kch"] * num["krow"] * num["kcol"] * num["b"] * num["porow"] * num["pocol"] <= 2
    a_spad_id = b_spad_id = 0
    for b in range(0, batch_size, tile["batches"]):
        for porow in range(0, out_rows, tile["porows"]):
            for pocol in range(0, out_cols, tile["pocols"]):
                for poch in range(0, out_channels, tile["pochs"]):
                    for krow in range(0, kernel_dim, tile["krows"]):
                        irow = porow * stride + krow * kernel_dilation - padding
                        for kcol in range(0, kernel_dim, tile["kcols"]):
                            icol = pocol * stride + kcol * kernel_dilation - padding
                            for kch in range(0, in_channels, tile["kchs"]):
                                if a_reuse:
                                    a_spad_id = 1 if (kch + krow + kcol + b + porow + pocol) == 0 else 2
                                if b_reuse:
                                    b_spad_id = 1 if (kch + poch + krow + kcol) == 0 else 2
                                batches_ = min(tile["batches"], batch_size - b)
                                porows_ = min(tile["porows"], out_rows - porow)
                                pocols_ = min(tile["pocols"], out_cols - pocol)
                                pochs_ = min(tile["pochs"], out_channels - poch)
                                krows_ = min(tile["krows"], kernel_dim - krow)
                                kcols_ = min(tile["kcols"], kernel_dim - kcol)
                                kchs_ = min(tile["kchs"], in_channels - kch)
                                # With pooling off the tile's output extent IS its pool extent, so
                                # the four pool pads stay zero however the image is cut.
                                orows_, ocols_ = porows_, pocols_
                                dilated_krows = krows_ + (kernel_dilation - 1) * (krows_ - 1)
                                dilated_kcols = kcols_ + (kernel_dilation - 1) * (kcols_ - 1)
                                irows_ = orows_ * stride + dilated_krows - 1
                                icols_ = ocols_ * stride + dilated_kcols - 1
                                lpad = -icol if icol < 0 else 0
                                upad = -irow if irow < 0 else 0
                                rpad = icol + icols_ - in_cols if icol + icols_ > in_cols else 0
                                dpad = irow + irows_ - in_rows if irow + irows_ > in_rows else 0
                                last = (
                                    krow + tile["krows"] >= kernel_dim
                                    and kcol + tile["kcols"] >= kernel_dim
                                    and kch + tile["kchs"] >= in_channels
                                )
                                first = krow == 0 and kcol == 0 and kch == 0
                                if packs_pixels_per_row and not (downsample or kernel_dilation > 1 or kchs_ > dim):
                                    max_pixels_per_row = min(dim // kchs_, kcols_)
                                else:
                                    max_pixels_per_row = 1
                                tiles.append(
                                    {
                                        "batches": batches_,
                                        "porows": porows_,
                                        "pocols": pocols_,
                                        "pochs": pochs_,
                                        "krows": krows_,
                                        "kcols": kcols_,
                                        "kchs": kchs_,
                                        "orows": orows_,
                                        "ocols": ocols_,
                                        "lpad": lpad,
                                        "rpad": rpad,
                                        "upad": upad,
                                        "dpad": dpad,
                                        "plpad": 0,
                                        "prpad": 0,
                                        "pupad": 0,
                                        "pdpad": 0,
                                        "max_pixels_per_row": max_pixels_per_row,
                                        "a_spad_id": a_spad_id,
                                        "b_spad_id": b_spad_id,
                                        "weight_offset": (krow * kernel_dim * in_channels + kcol * in_channels + kch)
                                        * out_channels
                                        + poch,
                                        "output_offset": (
                                            None
                                            if not last
                                            else (b * out_rows * out_cols + porow * out_cols + pocol) * out_channels
                                            + poch
                                        ),
                                        "bias_offset": None if not first else poch,
                                        "input_offset": (
                                            None
                                            if (a_reuse and poch > 0)
                                            else (b * in_rows * in_cols + (irow + upad) * in_cols + (icol + lpad))
                                            * in_channels
                                            + kch
                                        ),
                                        "weight_null": b_reuse and (pocol + porow + b > 0),
                                    }
                                )
    return tiles


#: The epilogue stages the device readout performs, in the ONLY order it performs them: the
#: accumulator holds the bias before the contraction accumulates onto it, the store path scales by a
#: per-tensor float, clips into the narrow element type, and then activates (`scale_and_sat` in the
#: pinned header). An epilogue naming these stages in another order is a different computation, so it
#: is refused rather than quietly reassociated.
NATIVE_CONV_EPILOGUE_ORDER: tuple[str, ...] = ("bias_add", "acc_scale", "relu")

#: Spellings of the accumulator-bias stage this tree uses interchangeably (`runtime.commandbuffer`).
_BIAS_STAGE_SPELLINGS = ("bias_add", "bias")


def _epilogue_plan(epilogue, attributes, *, operands):
    """Map a declared epilogue onto the readout's fixed stage order, or refuse by name.

    Fails closed on three separate things a silent mapping would lose: a stage the readout has no
    seat for (a pooled or integer-shift requantized readout is NOT this readout), the same stage twice
    (the device applies each once), and an order the device cannot reproduce.
    """
    stages = [("bias_add" if stage in _BIAS_STAGE_SPELLINGS else stage) for stage in (epilogue or [])]
    unsupported = [stage for stage in stages if stage not in NATIVE_CONV_EPILOGUE_ORDER]
    _require(not unsupported, f"unsupported rich epilogue {unsupported}; cannot silently drop stages")
    _require(len(set(stages)) == len(stages), "epilogue repeats a readout stage the device applies once")
    _require(
        stages == [stage for stage in NATIVE_CONV_EPILOGUE_ORDER if stage in stages],
        f"epilogue order {stages} is not the order the device readout applies",
    )
    if "bias_add" in stages:
        _require("bias" in operands, "a bias stage is declared and the command names no bias operand")
    else:
        _require("bias" not in operands, "a bias operand is supplied and no bias stage is declared")
    if "acc_scale" in stages:
        scale = attributes.get("acc_scale")
        # An acc_scale stage with no multiplier is not the identity; reading it as one makes every
        # engine agree on a saturating cast nobody asked for.
        _require(isinstance(scale, (int, float)) and not isinstance(scale, bool), "UNKNOWN acc_scale multiplier")
        scale = float(scale)
    else:
        scale = None
    return {"stages": tuple(stages), "acc_scale": scale, "relu": "relu" in stages}


def emit_native_conv(
    command: Mapping,
    tensors: Mapping,
    *,
    contract: NativeConvContract,
    pointers: Mapping[str, str],
    row_strides: Mapping[str, int],
) -> dict:
    """Emit native C plus exact register descriptors for an eligible schema CONV2D.

    Physical pointers/row strides are supplied by the compiler ABI. Entry must be
    drained, operands disjoint, and no unrelated commands concurrent; completion
    and host visibility remain the caller's explicit target-bound obligations.
    """
    _require(command.get("opcode") == "CONV2D", "not schema CONV2D")
    a = command.get("attributes", {})
    allowed = {"kernel", "stride", "padding", "dilation", "layout", "epilogue", "output_dtype", "acc_scale"}
    _require(set(a) <= allowed, f"unsupported convolution attributes {sorted(set(a) - allowed)}")
    _require(a.get("layout") == NATIVE_CONV_LAYOUT, f"unsupported layout {a.get('layout')!r}")
    stride_attr = list(a.get("stride") or [])
    dilation_attr = list(a.get("dilation") or [])
    padding_attr = list(a.get("padding") or [])
    _require(len(stride_attr) == 2 and len(dilation_attr) == 2, "unsupported stride/dilation rank")
    _require(len(set(stride_attr)) == 1 and len(set(dilation_attr)) == 1, "loop_conv_requires_uniform_2d_geometry")
    # The descriptor carries ONE padding amount for the whole window, so a per-edge padding is a
    # geometry this device cannot express -- not a geometry to average or to take the first of.
    _require(len(padding_attr) == 4 and len(set(padding_attr)) == 1, "loop_conv_requires_uniform_padding")
    stride, kernel_dilation, padding = stride_attr[0], dilation_attr[0], padding_attr[0]
    operands = command.get("operands", {})
    _require(set(operands) <= {"ifm", "weight", "dst", "bias"}, f"unsupported extra operand {sorted(set(operands))}")
    _require({"ifm", "weight", "dst"} <= set(operands), "missing convolution operand")
    plan = _epilogue_plan(a.get("epilogue"), a, operands=operands)
    _require(all(name in tensors for name in operands.values()), "unresolved resident or tensor operand")
    x, w, y = [tensors[operands[name]] for name in ("ifm", "weight", "dst")]
    dtype = contract.spad.element_dtype
    _require(
        x.get("dtype") == w.get("dtype") == y.get("dtype") == a.get("output_dtype") == dtype,
        "native convolution requires signed narrow saturating output; full-width/modular/floating output unsupported",
    )
    _require(len(x["shape"]) == 4 and len(a.get("kernel", [])) == 4, "unsupported tensor/kernel rank")
    n, h, width, ci = x["shape"]
    kh, kw, kci, co = a["kernel"]
    _require(all(type(v) is int and v > 0 for v in (n, h, width, ci, kh, kw, kci, co)), "nonpositive/static shape")
    _require(kh == kw and ci == kci, "unsupported kernel geometry")
    geometry = native_conv_single_tile(
        batch=n, in_rows=h, in_cols=width, kernel=kh, stride=stride, padding=padding, kernel_dilation=kernel_dilation
    )
    oh, ow = geometry["out_rows"], geometry["out_cols"]
    _require(w["shape"] == [kh * kw * ci, co] and y["shape"] == [n * oh * ow, co], "CONV2D physical shape mismatch")
    if "bias" in operands:
        bias = tensors[operands["bias"]]
        _require(bias.get("dtype") == contract.acc.element_dtype, "bias is not the accumulator's element type")
        _require(list(bias.get("shape") or []) == [co], "bias is not one accumulator value per output channel")
    _require(
        kh * kw * ci * (1 << (2 * (contract.spad.element_bits - 1))) <= (1 << (contract.acc.element_bits - 1)) - 1,
        "possible accumulator overflow",
    )
    _require(set(pointers) == set(operands), "physical ABI bindings missing")
    _require(set(row_strides) == {"ifm", "weight", "dst"}, "physical ABI row strides missing")
    _require(
        all(isinstance(p, str) and p.isascii() and p.isidentifier() for p in pointers.values()),
        "pointer must be a C identifier",
    )
    _require(len(set(pointers.values())) == len(pointers), "aliased ABI pointer identity")
    _require(
        row_strides == {"ifm": ci, "weight": co, "dst": co},
        "first native route requires explicit dense channel strides",
    )
    dim = contract.dim
    downsample = geometry["downsample"]
    # The convolution is cut into tiles that fit the derived double-buffer halves, by the header's own
    # search. A whole ResNet-50 convolution never fits one: pricing only the single tile is what made
    # every one of its 53 convolutions report a capacity refusal after the geometry clauses cleared.
    tile = native_conv_tile_shape(
        dim=dim,
        max_spad_rows=contract.capacity["max_spad_rows"],
        max_acc_rows=contract.capacity["max_acc_rows"],
        batch_size=n,
        out_rows=oh,
        out_cols=ow,
        out_channels=co,
        kernel_dim=kh,
        in_channels=ci,
        stride=stride,
        kernel_dilation=kernel_dilation,
        downsample=downsample,
    )
    tiles = native_conv_tiles(
        tile=tile,
        batch_size=n,
        in_rows=h,
        in_cols=width,
        in_channels=ci,
        out_channels=co,
        out_rows=oh,
        out_cols=ow,
        kernel_dim=kh,
        stride=stride,
        padding=padding,
        kernel_dilation=kernel_dilation,
        downsample=downsample,
        packs_pixels_per_row=contract.packs_pixels_per_row,
        dim=dim,
    )
    staged = native_conv_staged_rows(
        dim=dim,
        stride=stride,
        kernel_dilation=kernel_dilation,
        downsample=downsample,
        batches=tile["batches"],
        porows=tile["porows"],
        pocols=tile["pocols"],
        ochs=tile["pochs"],
        krows=tile["krows"],
        kcols=tile["kcols"],
        kchs=tile["kchs"],
    )
    macro = contract.header.macro("gemmini_loop_conv_ws")
    _require(macro is not None, "missing native convolution macro")
    # Everything the descriptor states about the WHOLE convolution. The tile loop below overrides
    # only the inner bounds, the four pads and the four operand addresses -- exactly the fields
    # `tiled_conv` recomputes per iteration.
    element_bytes = {
        "weights": contract.spad.element_bits // 8,
        "output": contract.spad.element_bits // 8,
        "input": contract.spad.element_bits // 8,
        "bias": contract.acc.element_bits // 8,
    }
    bases = {
        "weights": pointers["weight"],
        "output": pointers["dst"],
        "input": pointers["ifm"],
        # An absent bias is not the null pointer: the header's tiler substitutes its own sentinel and
        # sets no_bias, and the emitter carries the same one rather than a plausible zero.
        "bias": pointers.get("bias", contract.absent_bias_address),
    }
    outer = dict(
        batch_size=n,
        in_row_dim=h,
        in_col_dim=width,
        in_channels=ci,
        out_channels=co,
        out_row_dim=oh,
        out_col_dim=ow,
        pool_out_row_dim=oh,
        pool_out_col_dim=ow,
        stride=stride,
        padding=padding,
        kernel_dim=kh,
        kernel_dilation=kernel_dilation,
        pool_size=1,
        pool_stride=1,
        pool_padding=0,
        no_bias=int("bias" not in operands),
        no_pool=1,
        downsample=downsample,
        wrot180=0,
        input_dilated=0,
        activation=_integer(contract.header, "RELU" if plan["relu"] else "NO_ACTIVATION"),
        trans_output_1203=0,
        trans_weight_1203=0,
        trans_weight_0132=0,
        trans_input_3120=0,
        in_stride=ci,
        weight_stride=co,
        out_stride=co,
        dw=0,
    )
    instructions: list[dict] = []
    parameter_sets: list[dict] = []
    proven: set[tuple] = set()
    decode_proof = None
    for entry in tiles:
        v = dict(outer)
        for name in (
            "batches",
            "porows",
            "pocols",
            "pochs",
            "krows",
            "kcols",
            "kchs",
            "orows",
            "ocols",
            "lpad",
            "rpad",
            "upad",
            "dpad",
            "plpad",
            "prpad",
            "pupad",
            "pdpad",
            "max_pixels_per_row",
            "a_spad_id",
            "b_spad_id",
        ):
            v[name] = entry[name]
        for field, key in (
            ("weights", "weight_offset"),
            ("output", "output_offset"),
            ("bias", "bias_offset"),
            ("input", "input_offset"),
        ):
            offset = entry[key]
            if field == "weights" and entry["weight_null"]:
                offset = None
            v[field] = _address(bases[field], offset, element_bytes[field])
        # Every macro parameter is named above deliberately. A parameter this header grows that
        # nothing here assigns must REFUSE, never inherit a zero that reads as "feature off".
        _require(set(macro.params) == set(v), f"UNKNOWN native macro parameter {sorted(set(macro.params) ^ set(v))}")
        tile_instructions = _expand_conv_macro(contract, macro, v)
        # The decode proof walks the whole mux graph, so proving every tile of a 120-tile convolution
        # separately costs minutes for no new information: tiles differing only in their operand
        # ADDRESSES exercise the same paths. Prove one of each distinct non-address descriptor.
        signature = tuple(sorted((k, val) for k, val in v.items() if k not in bases))
        if signature not in proven:
            proven.add(signature)
            decode_proof = _descriptor_decode_proof(contract.hw_text, tile_instructions, v, pointer_fields=tuple(bases))
        instructions.extend(tile_instructions)
        parameter_sets.append(v)
    _require(instructions, "convolution produced no descriptor")
    v = parameter_sets[0]
    call = "\n".join(
        "gemmini_loop_conv_ws(" + ", ".join(str(each[name]) for name in macro.params) + ");" for each in parameter_sets
    )
    for name in ("gemmini_extended_config_st", "gemmini_extended3_config_ex"):
        _require(contract.header.macro(name) is not None, f"missing entry config {name}")
    # The two execute/store strides the header's own tiler derives for this convolution. They are
    # part of the emitted program, not decoration: A_stride is what makes the sequencer's output walk
    # a STRIDED input walk, and acc_scale is the readout multiplier the store path applies.
    a_stride = stride >> downsample
    acc_scale = 1.0 if plan["acc_scale"] is None else plan["acc_scale"]
    scale_literal = "ACC_SCALE_IDENTITY" if plan["acc_scale"] is None else repr(acc_scale)
    code = (
        f"gemmini_extended_config_st({co} * sizeof(elem_t), {v['activation']}, {scale_literal});\n"
        f"gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, 0, 1, {a_stride}, "
        "false, false, false, 0, 0, 0, 0);\n" + call
    )
    return {
        "schema": "native_conv_descriptor_v1",
        "status": "emitted_not_runtime_qualified",
        "instructions": instructions,
        "c_source": code,
        "parameters": v,
        "geometry": geometry,
        "tile": tile,
        "tiles": len(parameter_sets),
        "tile_parameters": parameter_sets,
        "epilogue": {"stages": list(plan["stages"]), "acc_scale": plan["acc_scale"]},
        "entry_strides": {"a_stride": a_stride, "c_stride": 1, "acc_scale": acc_scale},
        "source_command_sha256": _sha(json.dumps(command, sort_keys=True, separators=(",", ":")).encode()),
        "source_tensor_abi_sha256": _sha(
            json.dumps(
                {
                    "tensors": {name: tensors[name] for name in operands.values()},
                    "pointers": dict(pointers),
                    "row_strides": dict(row_strides),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ),
        "provenance": contract.provenance,
        "readout_proof": contract.readout,
        "capacity": {**staged, **contract.capacity},
        "required_entry_contract": ["drained", "no_external_concurrent_commands", "nonaliasing_operand_allocations"],
        "required_exit_contract": ["target_completion", "host_visibility_before_output_use"],
        "descriptor_to_rtl_field_qualification": decode_proof,
        "numerical_runtime_qualification": "UNPROVEN",
        "cycles": "UNKNOWN",
    }


#: Every clause that can refuse a convolution the device sequencer. Reason codes are DATA: the census
#: (`merlin.perf.capability_refusal`) groups by them, and a renamed clause silently re-partitions a
#: ledger, so they are listed once here and never spelled inline.
NATIVE_CONV_CLAUSES: tuple[str, ...] = (
    "target_has_no_loop_conv",
    "loop_conv_store_is_narrow_only",
    "native_layout_contract_not_satisfied",
    "buffer_shape_does_not_match_native_layout",
    "loop_conv_requires_uniform_2d_geometry",
    "loop_conv_requires_uniform_padding",
    "native_loop_conv_requires_0_le_padding_lt_kernel",
    "loop_conv_requires_positive_static_geometry",
    "native_loop_conv_scale_granularity_not_admitted",
    "native_loop_conv_scale_granularity_unknown",
)

#: The operand/weight/destination layouts the sequencer's descriptor addresses.
NATIVE_CONV_LAYOUT = "nhwc"


def native_conv_refusals(
    attributes,
    operands,
    *,
    operand_dtype: str,
    facet=None,
    scale_granularity: str | None = None,
    has_loop_conv: bool = True,
) -> tuple[str, ...]:
    """EVERY clause this convolution fails, not the first one.

    A guard sequence returns on its first failure, so a census built from one is a census of first
    refusals: it counts what was reached and is silent about what was not. Measured on the recorded
    whole-model emission, all 53 convolutions reported exactly one blocker (the narrow-store dtype
    clause) and were failing three -- the layout contract and the buffer shapes underneath it were
    never evaluated. `perf.capability_refusal` carries that caveat as `first_refusal_only`; this
    returns the whole set so the caveat stops being needed.

    ``attributes`` and ``operands`` are the emission's own record of the operation (its kernel,
    stride, padding, dilation, layout, output dtype; and each operand's shape and dtype). Nothing
    about the target is spelled here: ``operand_dtype`` is the narrow dtype the design's own stores
    declare, and the scale granularity is judged by the derived readout facet rather than by a
    literal, so a design whose readout holds a per-column scale answers differently without an edit.
    """
    out: list[str] = []
    if not has_loop_conv:
        out.append("target_has_no_loop_conv")

    dtypes = [attributes.get("output_dtype")] + [op.get("dtype") for op in operands.values()]
    if any(d is not None and d != operand_dtype for d in dtypes):
        # The store path narrows or it does not; a full-width readout would saturate silently.
        out.append("loop_conv_store_is_narrow_only")

    if str(attributes.get("layout") or "").lower() != NATIVE_CONV_LAYOUT:
        out.append("native_layout_contract_not_satisfied")

    kernel = list(attributes.get("kernel") or [])
    ci = kernel[2] if len(kernel) > 3 else None
    co = kernel[3] if len(kernel) > 3 else None
    act, dst = operands.get("ifm") or {}, operands.get("dst") or {}
    a_shape, d_shape = list(act.get("shape") or []), list(dst.get("shape") or [])
    if (len(a_shape) == 4 and ci is not None and a_shape[-1] != ci) or (
        len(d_shape) == 4 and co is not None and d_shape[-1] != co
    ):
        # NHWC puts the channel last; an activation whose trailing extent is not the input channel
        # count is laid out some other way whatever its layout string claims.
        out.append("buffer_shape_does_not_match_native_layout")

    stride = list(attributes.get("stride") or [])
    dilation = list(attributes.get("dilation") or [])
    padding = list(attributes.get("padding") or [])
    if len(kernel) > 1 and kernel[0] != kernel[1]:
        out.append("loop_conv_requires_uniform_2d_geometry")
    elif len(set(stride)) > 1 or len(set(dilation)) > 1:
        out.append("loop_conv_requires_uniform_2d_geometry")
    if len(padding) != 4 or len(set(padding)) != 1:
        out.append("loop_conv_requires_uniform_padding")
    elif kernel and not (0 <= padding[0] < kernel[0]):
        out.append("native_loop_conv_requires_0_le_padding_lt_kernel")
    if any(v is not None and v < 1 for v in (*kernel, *stride, *dilation)):
        out.append("loop_conv_requires_positive_static_geometry")

    if scale_granularity is not None and facet is not None:
        admits = facet.admits_granularity(scale_granularity)
        if admits is False:
            out.append("native_loop_conv_scale_granularity_not_admitted")
        elif admits is None:
            # Unknown never widens: a readout whose granularity could not be derived does not thereby
            # admit one. This is the clause whose absence produced an unsound "32 of 53 admitted".
            out.append("native_loop_conv_scale_granularity_unknown")

    unknown = [c for c in out if c not in NATIVE_CONV_CLAUSES]
    if unknown:
        raise UnsupportedNativeConv(f"refusal clause not declared in NATIVE_CONV_CLAUSES: {unknown}")
    return tuple(out)


#: Where this target's four contract sources live, relative to the repo root. They are spelled here
#: rather than passed in because every production caller passed ``None`` and got the host im2col
#: rewrite instead -- a native route that only runs when a caller remembers to construct a contract is
#: a native route that never runs. This is a per-target edge: the paths are gemmini's own, and the
#: function is reached through gemmini's package.


@lru_cache(maxsize=1)
def derive_native_conv_contract() -> NativeConvContract:
    """This target's native-convolution contract, resolved from its own sources.

    Raises :class:`UnsupportedNativeConv` naming the missing source rather than returning ``None``, so
    an absent input is a stated refusal that reaches the receipt instead of a silent fall-through to
    the host rewrite. ``facts.json`` in particular is a REGENERATED artifact and is gitignored, so it
    is routinely absent in a fresh worktree -- which must read as "this checkout cannot derive the
    contract", never as "this convolution is not native".
    """
    from merlin.targetgen.rtl.facts import ensure_facts
    from .gemmini import rocc_tests_dir

    include = rocc_tests_dir() / "include"
    try:
        facts = ensure_facts("gemmini")
    except (FileNotFoundError, ValueError) as exc:
        raise UnsupportedNativeConv(f"UNKNOWN native-conv contract: {exc}") from exc
    header, params = include / "gemmini.h", include / "gemmini_params.h"
    for label, path in (("header", header), ("params", params), ("rtl facts", facts)):
        _require(path.is_file(), f"UNKNOWN native-conv contract: no {label} at {path}")
    record = json.loads(facts.read_text())
    tables = [row for row in record["facts"]["interfaces"] if row.get("name") == "funct_decode_table"]
    _require(len(tables) == 1, "UNKNOWN instruction decode table")
    core_hw = Path(tables[0]["hw_source"])
    # The elaborated RTL the facts were derived FROM. Its absence is the case the contract's own
    # staleness check cannot reach, and substituting anything for it would fabricate a hardware claim.
    _require(core_hw.is_file(), f"UNKNOWN native-conv contract: pinned RTL bytes absent at {core_hw}")
    return NativeConvContract(header=header, params=params, facts=facts, core_hw=core_hw)
