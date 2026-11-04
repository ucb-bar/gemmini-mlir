"""Contract CLI and structural interface verification for the Gemmini compiler."""

import argparse
import json
from pathlib import Path
import sys

# The reviewed, editable compiler library travels with the submission.
sys.path.insert(0, str(Path(__file__).resolve().parent / "compiler_library"))


def _interface_dialect():
    """Register the frozen residency surface with real SSA and typed operands."""
    from xdsl.ir import Dialect, ParametrizedAttribute, TypeAttribute
    from xdsl.irdl import IRDLOperation, irdl_attr_definition, irdl_op_definition, var_operand_def, var_result_def
    from merlin.targetgen.contract.interface_emit import defined_mnemonics

    @irdl_attr_definition
    class ResidentType(ParametrizedAttribute, TypeAttribute):
        name = "merlin_iface.resident"

    @irdl_attr_definition
    class AccType(ParametrizedAttribute, TypeAttribute):
        name = "merlin_iface.acc"
        element: TypeAttribute

    def parse(cls, parser):
        operands = []
        position = parser.pos
        if cls.name != "merlin_iface.tensor":
            operands = parser.parse_comma_separated_list(
                parser.Delimiter.NONE, parser.parse_unresolved_operand
            )
        attributes = parser.parse_optional_attr_dict()
        parser.parse_punctuation(":")
        if cls.name == "merlin_iface.tensor":
            inputs, outputs = [], [parser.parse_type()]
        else:
            signature = parser.parse_function_type()
            inputs, outputs = list(signature.inputs), list(signature.outputs)
        return cls.build(
            operands=[parser.resolve_operands(operands, inputs, position)],
            result_types=[outputs], attributes=attributes,
        )

    operations = []
    for mnemonic in sorted(defined_mnemonics()):
        operation = type(
            "Iface_" + mnemonic,
            (IRDLOperation,),
            {"name": "merlin_iface." + mnemonic,
             "arguments": var_operand_def(), "results_": var_result_def(),
             "parse": classmethod(parse)},
        )
        operations.append(irdl_op_definition(operation))
    return Dialect("merlin_iface", operations, [ResidentType, AccType])


def parse_source(source: str):
    """Parse and verify source IR without importing any execution implementation."""
    from xdsl.parser import Parser
    from xdsl.dialects.builtin import StringAttr
    from merlin.targetgen.contract.linalg_iface import make_linalg_context
    from merlin.targetgen.contract.interface_emit import parse_interface_mlir

    context = make_linalg_context()
    context.load_dialect(_interface_dialect())
    module = Parser(context, source).parse_module()
    module.verify()
    version = module.attributes.get("merlin_iface.version")
    if version is not None:
        if not isinstance(version, StringAttr) or version.data != "0.1":
            raise ValueError("unsupported merlin_iface.version; expected 0.1")
        parsed = parse_interface_mlir(source)
        if parsed["target"] != "gemmini":
            raise ValueError("interface target must be gemmini")
    elif not any(op.name == "func.func" for op in module.walk()):
        raise ValueError("interface requires a merlin_iface module or a function definition")
    return module


def main(argv=None) -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--verify-diagnostics", action="store_true")
    cli.add_argument("--convert-iface-to-gemmini", action="store_true")
    cli.add_argument("--emit-command-buffer", metavar="PATH")
    cli.add_argument("--emit-target-artifact", action="store_true")
    cli.add_argument("--emit-llvm", action="store_true")
    cli.add_argument("input_mlir")
    args = cli.parse_args(argv)
    try:
        source = (sys.stdin.read() if args.input_mlir == "-"
                  else Path(args.input_mlir).read_text(encoding="utf-8"))
        parse_source(source)
        if args.convert_iface_to_gemmini or args.emit_command_buffer or args.emit_target_artifact or args.emit_llvm:
            from backend import compile_source
            command_buffer, llvm_module = compile_source(source)
            selection_path = Path(__file__).parent / "LIBRARY_PASS_SELECTION.json"
            if selection_path.is_file():
                from tools.library_pass_adapter import prepare
                from merlin.transforms.pipeline_policy import execute_policy
                selection = json.loads(selection_path.read_text())
                llvm_module = execute_policy(**prepare(
                    Path(args.input_mlir), package=Path(__file__).parent,
                    mechanism=selection["mechanism"],
                )).artifact
            if args.emit_command_buffer:
                Path(args.emit_command_buffer).write_text(
                    json.dumps(command_buffer, indent=2) + "\n", encoding="utf-8"
                )
            if args.emit_target_artifact or args.emit_llvm or not args.emit_command_buffer:
                sys.stdout.write(llvm_module)
                if not llvm_module.endswith("\n"):
                    sys.stdout.write("\n")
        return 0
    except Exception as error:
        print(f"gemmini-opt: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
