"""Reclose resident capsules and the complete actual primitive/pointer trace."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from captured_schedule_closure_probe import check as check_capsules
from test_resident_conv_command_loops import executed_commands
from xdsl.context import Context
from xdsl.dialects.builtin import Builtin
from xdsl.dialects.llvm import LLVM
from xdsl.parser import Parser

from mlir_oot.ir.gemmini_dialect import GEMMINI


def check(receipt: dict) -> None:
    check_capsules(receipt)
    proof = receipt["original_actual_module_command_identity"]
    # The pinned xDSL retains upstream LLVM loop_annotation as an opaque
    # attribute. Load all actual operations; no operation or annotation is
    # stripped from either module before verification and static CFG tracing.
    context = Context(allow_unregistered=True)
    for dialect in (Builtin, LLVM, GEMMINI):
        context.load_dialect(dialect)
    streams = []
    for pathkey, hashkey in (
        ("source_target_module", "source_target_ir_sha256"),
        ("candidate_target_module", "candidate_target_ir_sha256"),
    ):
        path = Path(proof[pathkey])
        if hashlib.sha256(path.read_bytes()).hexdigest() != proof[hashkey]:
            raise ValueError("source-bound target module changed")
        module = Parser(context, path.read_text()).parse_module()
        streams.append(executed_commands(module))
    if streams[0] != streams[1]:
        raise ValueError("primitive order, operands or DMA pointers changed")
    stream = streams[0]
    if (
        len(stream) != proof["command_count"]
        or dict(Counter(x[0] for x in stream)) != proof["command_counts"]
    ):
        raise ValueError("command census changed")
    if (
        hashlib.sha256(repr(stream).encode()).hexdigest()
        != proof["encoded_sequence_repr_sha256"]
    ):
        raise ValueError("complete command/pointer fingerprint changed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    check(receipt)
    print(f"RESIDENT_RETAINED_COMMAND_CLOSURE_PASS {len(receipt['pins'])} pins")


if __name__ == "__main__":
    main()
