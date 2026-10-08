"""Opt-in schema-CONV2D selection and native LLVM artifact emission.

The default Gemmini compiler remains unchanged. The opt-in requires an explicit
source-bound contract, retains refusal reasons, and does not certify runtime
numerics. Its dense NHWC/HWIO pointer ABI is NOT the legacy padded im2col ABI.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from .gemmini_loop_conv import UnsupportedNativeConv, _require, emit_native_conv, native_entry_instructions


def emit_selected_native_conv(cb, *, contract):
    """Return (target LLVM MLIR, argument names, selection receipt), or refuse."""
    commands = cb.get("commands", [])
    convolutions = [command for command in commands if command.get("opcode") == "CONV2D"]
    _require(len(convolutions) == 1, "native route requires one schema convolution region")
    conv = deepcopy(convolutions[0])
    weight = conv.get("operands", {}).get("weight")
    packs = [command for command in commands if command.get("opcode") == "RES_PACK"]
    evicts = [command for command in commands if command.get("opcode") == "EVICT"]
    _require(
        len(commands) == 1 + len(packs) + len(evicts) and len(packs) <= 1 and len(evicts) <= 1,
        "native route cannot discard other commands",
    )
    if packs:
        pack = packs[0]
        # Split into its three terms. As one conjunction they shared a message that named the wrong
        # cause: every convolution capsule in the corpus failed here, and the reason it reported was
        # "packing or ordering" when the packing and the ordering were both already correct.
        _require(
            pack.get("operands", {}).get("dst") == weight,
            "resident pack does not produce the convolution's weight",
        )
        _require(
            commands.index(pack) < commands.index(convolutions[0]),
            "resident pack is ordered after the convolution that consumes it",
        )
        # A CONVOLUTION weight pack is spelled `packed_conv_rhs`; `packed_rhs` is the MATMUL spelling.
        # This route required the matmul one, so it rejected every conv capsule the corpus produces
        # (`corpus_spec` emits `packed_conv_rhs` for a convolution) before any convolution property was
        # examined. Both are accepted here because both are what the interface dialect admits for a
        # resident pack; which one is correct for this route is decided by the op that consumes it,
        # and the op consuming it is a CONV2D.
        _require(
            (pack.get("attributes") or {}).get("layout") in ("packed_conv_rhs", "packed_rhs"),
            f"unsupported resident pack layout {(pack.get('attributes') or {}).get('layout')!r}",
        )
        _require(
            set(pack.get("attributes") or {}) == {"layout"},
            "resident pack carries attributes beyond its layout",
        )
        weight = pack["operands"]["src"]
        conv["operands"]["weight"] = weight
    if evicts:
        _require(
            packs
            and evicts[0].get("operands") == {"handle": packs[0]["operands"]["dst"]}
            and not evicts[0].get("attributes")
            and commands.index(evicts[0]) > commands.index(convolutions[0]),
            "unsupported eviction",
        )
    tensors = cb.get("tensors", {})
    operands = conv.get("operands", {})
    _require(set(operands) <= {"ifm", "weight", "dst", "bias"}, "unsupported convolution operands")
    _require({"ifm", "weight", "dst"} <= set(operands), "missing convolution operand")
    _require(all(name in tensors for name in operands.values()), "missing convolution tensor ABI")
    ci, co = tensors[operands["ifm"]]["shape"][-1], tensors[weight]["shape"][-1]
    # Pointer-argument order. A bias is a fourth kernel argument, appended rather than interleaved, so
    # a convolution WITHOUT one keeps the argument order it already had.
    arg_order = [weight, operands["ifm"], operands["dst"]] + ([operands["bias"]] if "bias" in operands else [])
    _require(len(set(arg_order)) == len(arg_order), "aliased convolution tensor ABI")
    pointers = {"weight": "weight_ptr", "ifm": "input_ptr", "dst": "output_ptr"}
    if "bias" in operands:
        pointers["bias"] = "bias_ptr"
    receipt = emit_native_conv(
        conv, tensors, contract=contract, pointers=pointers, row_strides={"ifm": ci, "weight": co, "dst": co}
    )
    strides = receipt["entry_strides"]
    entry = native_entry_instructions(
        contract,
        output_channels=co,
        activation=receipt["parameters"]["activation"],
        a_stride=strides["a_stride"],
        c_stride=strides["c_stride"],
        acc_scale=strides["acc_scale"],
    )
    completion = contract.header.macro("gemmini_fence")
    _require(
        completion is not None and completion.body.startswith('asm volatile("') and completion.body.endswith('")'),
        "unsupported target completion ABI",
    )
    assembly = completion.body[len('asm volatile("') : -len('")')]
    _require(assembly and '"' not in assembly and "\\" not in assembly, "unsupported completion assembly")
    fence = f'    llvm.inline_asm has_side_effects "{assembly}", "~{{memory}}" : () -> ()'
    signature = ", ".join(f"%a{index}: !llvm.ptr" for index in range(len(arg_order)))
    lines = ["module {", f"  llvm.func @gemmini_kernel({signature}) {{", fence]
    names = {pointers["weight"]: "%p0", pointers["ifm"]: "%p1", pointers["dst"]: "%p2"}
    if "bias" in pointers:
        names[pointers["bias"]] = "%p3"
    for index in range(len(arg_order)):
        lines.append(f"    %p{index} = llvm.ptrtoint %a{index} : !llvm.ptr to i64")
    counter = 0
    # A tiled convolution addresses a SLICE of each tensor per tile, so most operand values are a
    # base plus a byte offset rather than a base. Materialize each distinct slice once: a ResNet-50
    # convolution issues up to a few hundred tiles and recomputing the same address per tile would
    # triple the emitted kernel for nothing.
    slices: dict[str, str] = {}

    def operand(value):
        nonlocal counter
        if not isinstance(value, str):
            counter += 1
            lines.append(f"    %c{counter} = llvm.mlir.constant({value} : i64) : i64")
            return f"%c{counter}"
        if value in names:
            return names[value]
        if value in slices:
            return slices[value]
        base, plus, offset = value.rpartition("+")
        _require(plus and base in names and offset.isdigit(), f"unresolved operand address {value!r}")
        counter += 1
        lines.append(f"    %c{counter} = llvm.mlir.constant({offset} : i64) : i64")
        counter += 1
        lines.append(f"    %c{counter} = llvm.add {names[base]}, %c{counter - 1} : i64")
        slices[value] = f"%c{counter}"
        return slices[value]

    for instruction in entry + receipt["instructions"]:
        operands_ssa = [operand(instruction[register]) for register in ("rs1", "rs2")]
        lines.append(
            f'    llvm.inline_asm has_side_effects ".insn r {contract.custom_opcode}, '
            f'{contract.funct3}, {instruction["funct"]}, x0, $0, $1", "r,r,~{{memory}}" '
            f"{operands_ssa[0]}, {operands_ssa[1]} : (i64, i64) -> ()"
        )
    lines.extend([fence, "    llvm.return", "  }", "}"])
    text = "\n".join(lines) + "\n"
    receipt.update(
        {
            "selection": "selected_explicit_opt_in",
            "default_enabled": False,
            "entry_instructions": entry,
            "arg_order": arg_order,
            "physical_abi": "dense_NHWC_flattened_HWIO_no_legacy_padding",
            "command_buffer_sha256": hashlib.sha256(
                json.dumps(cb, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "target_artifact_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "compiler_path": "emit_kernel_mlir_before_CONV2D_normalization",
            "full_model_native_selection": False,
        }
    )
    return text, arg_order, receipt
