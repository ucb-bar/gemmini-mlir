"""Independent successor screen; preserve the first exact-observer experiment."""

import json
from pathlib import Path

from merlin.llvmlower.prepared_scaled_integer_observer import emit_prepared_scaled_integer_observer
import source_integer_pv_native as base


def main():
    prior = base.OUT / "qualification.json"
    base.OUT = base.OUT.parent / "source-integer-pv-prepared-native-20261007"
    base.emit_scaled_integer_observer = emit_prepared_scaled_integer_observer
    base.main()
    receipt = base.OUT / "qualification.json"
    record = json.loads(receipt.read_text())
    record["parent_immutable_receipt"] = {"path": str(prior), "sha256": base.sha(prior)}
    record["explicit_alternative"] = "Prepare immutable binary64 scale/error/prefix coefficients once per source column; exact source observer unchanged"
    record["pins"][str(Path(__file__))] = base.sha(Path(__file__))
    base.save(receipt, record)
    print("PREPARED_ALL22_EXACT", base.sha(receipt), flush=True)


if __name__ == "__main__":
    main()
