"""Fixed four-panel storage and completion contract for exact rectifiers.

DDR store-to-reload is ordered by a retained Rocket completion fence. The
reservation station supplies only local-resource dependencies; it is never
used as an external-memory alias proof.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .spad_fence_coalescing import OrderingContract
from .tables import rtl_facts as F

_SOURCE_HASHES = {
    "LoadController.scala": "780ee0a292d8653fa8a822aee28e2fb2cffbc25301431b4332182008d34fcf6a",
    "StoreController.scala": "a0b556d109196c3c9d9ac5ed4dffc1bcad334c0f1f87780d948b84bde7402c6d",
    "Scratchpad.scala": "e4d3a12c700fe9edaf5c2ce2f283ec5233a98bd103fb04054160e8097cb89a4f",
    "Controller.scala": "2ce940248a43b46e726e98e72e3258c03bd354535a560393aeb1a7cf7d85b72e",
    "DMACommandTracker.scala": "a0fbc60a1cdd4f58636a3aafed630f5bec106176fa3e5b57ce028321ac73e0b6",
}
_ROCKET_HASH = "2f88a01afa8da2a4dd9706e19a1707aeac4f1edbfc0d3659e8252de76b5ce243"


@dataclass(frozen=True)
class PanelBatch:
    factor: int = 4

    def require(self, plan, ordering_contract: OrderingContract):
        plan.__post_init__()
        if type(self.factor) is not int or self.factor != 4:
            raise ValueError("only the explicit proved four-panel batch is supported")
        if plan.relation is None:
            raise ValueError("panel batching requires the exact sparse rectifier chain")
        if not isinstance(ordering_contract, OrderingContract):
            raise ValueError("panel batching requires a pinned ordering contract")  # noqa: TRY004
        pins = ordering_contract.require()
        directory = Path(ordering_contract.source_directory).resolve()
        paths = [(directory / name, digest) for name, digest in _SOURCE_HASHES.items()]
        chipyard = directory.parents[5]
        rocket = (
            chipyard / "generators/rocket-chip/src/main/scala/rocket/RocketCore.scala"
        )
        paths.append((rocket, _ROCKET_HASH))
        for path, digest in paths:
            if (
                not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != digest
            ):
                raise ValueError("unsupported panel completion implementation")
            pins.append({"path": str(path), "sha256": digest})
        reservations = self.reservations(plan)
        intervals = [(base, base + rows) for base, rows in reservations.values()]
        if any(begin < 0 or end > F.SPAD_ROWS for begin, end in intervals):
            raise ValueError("four-panel SPAD reservation exceeds capacity")
        if any(
            a < d and c < b
            for i, (a, b) in enumerate(intervals)
            for c, d in intervals[i + 1 :]
        ):
            raise ValueError("four-panel live SPAD reservations overlap")
        if self.factor * plan.panel_rows > F.ACC_ROWS:
            raise ValueError("four-panel ACC reservation exceeds capacity")
        return {
            "schema": "exact_rectifier_four_panel_completion_v1",
            "factor": self.factor,
            "source_pins": pins,
            "spad_reservations": reservations,
            "acc_reservation": [0, self.factor * plan.panel_rows],
            "panel_stride_rows": plan.panel_rows,
            "partial_batch_slots": [1, 2, 3],
            "retained_completion_boundaries": [
                "all predictor DDR stores before any same-output DDR reload",
                "all final DDR stores before any next-batch resource reuse",
                "setup and function return",
            ],
            "ordering": "SPAD mvin completion -> real-D execute read -> ACC store; distinct panel local extents throughout each phase",
            "DDR_alias_from_reservation_station": False,
            "store_scale": "constant within each phase; changed only after retained completion fence",
            "numeric": "identical per-output fourteen products, increasing coefficient chunks and exact finite correction",
            "physical_bitstream_support": "UNKNOWN until complete engine qualification",
        }

    def reservations(self, plan):
        extent = self.factor * plan.panel_rows
        bank = F.SPAD_BANK_ROWS
        return {
            "lhs": (0, extent),
            "temporary": (extent, extent),
            "prediction": (2 * extent, extent),
            "rhs": (bank, extent),
            "indicator0": (bank + extent, extent),
            "indicator1": (bank + 2 * extent, extent),
            "weights": plan.spad_intervals["weights"],
            "seeds": plan.spad_intervals["seeds"],
        }
