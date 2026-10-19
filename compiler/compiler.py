#!/usr/bin/env python3
"""Four-entrypoint typed xDSL/Gemmini compiler."""
import json
import os
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode = True
from frontend import parse


def main():
    args = sys.argv[1:]
    inputs = [arg for arg in args if not arg.startswith('-')]
    if len(inputs) != 1:
        raise ValueError('Expected one interface MLIR file')
    text = Path(inputs[0]).read_text()
    try:
        buffer = parse(text)
    except ValueError:
        if 'merlin_iface.version' in text:
            raise
        from mixed_frontend import parse_mixed
        buffer = parse_mixed(text)
    if '--verify-diagnostics' in args:
        return
    if any('requant' in command.get('attributes',{}).get('epilogue',[])
           for command in buffer['commands']):
        buffer['commands'] = []
        buffer['declined'] = {'reason': 'The selected target has no integer round-half-up requant readout route',
                              'op': 'requant'}
    llvm_ir = None
    if buffer.get('kernel_abi',{}).get('kind') == 'whole_program':
        from mixed_codegen import emit as emit_mixed
        from task_tags import finalize_plan
        llvm_ir = emit_mixed(text,buffer)
        finalize_plan(buffer)
    for arg in args:
        if arg.startswith('--emit-command-buffer='):
            Path(arg.split('=',1)[1]).write_text(json.dumps(buffer,indent=2))
    if '--emit-target-artifact' in args or '--emit-llvm' in args:
        from backend import emit
        llvm_ir = emit(buffer) if llvm_ir is None else llvm_ir
        install = os.environ.get('MERLIN_MLIR_INSTALL','/work/merlin/toolchains/llvm-a47bddcc/install')
        translator = str(Path(install)/'bin'/'mlir-translate')
        result = subprocess.run([translator,'--import-llvm'],input=llvm_ir,text=True,capture_output=True)
        if result.returncode:
            raise ValueError(result.stderr)
        lowered = result.stdout
        if buffer.get('kernel_abi',{}).get('kind') == 'whole_program':
            from mixed_frontend import parsed_source
            from task_tags import attach
            lowered = attach(lowered,buffer,parsed_source(text))
        sys.stdout.write(lowered)
    elif not any(arg.startswith('--emit-command-buffer=') for arg in args):
        from xdsl.dialects.builtin import ModuleOp, StringAttr, UnregisteredOp
        from xdsl.printer import Printer
        ops = []
        for command in buffer['commands']:
            cls = UnregisteredOp.with_name('gemmini.'+command['opcode'].lower())
            ops.append(cls.create(attributes={'command':StringAttr(json.dumps(command))}))
        Printer().print_op(ModuleOp(ops))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(str(error),file=sys.stderr)
        raise SystemExit(1)
