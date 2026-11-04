"""One explicit wall deadline shared across ordinary execution stages.

This is process-budget plumbing, not compiler, numerical or timing authority.
It supplies no default budget and never resets when passed to another stage.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionDeadline:
    started_at: float
    budget_s: float

    def __post_init__(self):
        self._validate()

    def _validate(self):
        for value in (self.started_at, self.budget_s):
            try:
                finite = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError("execution deadline requires explicit finite wall-clock values")
        if self.budget_s <= 0 or not math.isfinite(self.started_at + self.budget_s):
            raise ValueError("execution deadline requires a positive explicit budget")

    @classmethod
    def start(cls, budget_s):
        return cls(time.monotonic(), budget_s)

    def remaining(self, detail="shared execution wall budget expired"):
        self._validate()
        left = self.started_at + self.budget_s - time.monotonic()
        if left <= 0:
            raise TimeoutError(detail)
        return left


def selected_deadline(*, seconds, parent, build_service):
    """Preserve absent/legacy behavior and the existing integer build option."""
    if seconds is not None and (build_service is None or type(seconds) is not int or seconds <= 0):
        raise ValueError("explicit build timeout requires a pure build service and positive seconds")
    if parent is not None and (type(parent) is not ExecutionDeadline or build_service is None or seconds is not None):
        raise ValueError("shared build deadline requires explicit pure build support and no second budget")
    deadline = parent if parent is not None else ExecutionDeadline.start(seconds) if seconds is not None else None
    if deadline is not None:
        deadline.remaining()
    return deadline
