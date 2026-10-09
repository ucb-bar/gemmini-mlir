"""Reclose both source command proofs and conserved CPU/primitive features."""

import argparse
import json
from pathlib import Path

from resident_retained_command_closure_probe import check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    check(receipt)
    check(
        {
            **receipt,
            "original_actual_module_command_identity": receipt[
                "independent_actual_module_command_identity"
            ],
        }
    )
    for path in receipt["paired_feature_exports"]:
        value = json.loads(Path(path).read_text())
        source = value["source_command_trace"]["command_classes"]
        actual = {
            key: count
            for key, count in value["features"]["primitive_commands"].items()
            if count
        }
        if source != actual:
            raise ValueError("emitted and executed primitive class counts disagree")
        if (
            sum(value["features"]["cpu_opcode_classes"].values())
            != value["features"]["executed_instructions"]
        ):
            raise ValueError(
                "CPU opcode census does not conserve observed instructions"
            )
    print("RESIDENT_REDUCTION_CLOSURE_PASS", len(receipt["pins"]))


if __name__ == "__main__":
    main()
