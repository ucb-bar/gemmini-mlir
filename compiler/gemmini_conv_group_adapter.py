"""Explicit isolated experiment adapter for full-resident spatial row bounds.

Only experiment packages statically import this entrypoint. The ordinary owner
still proves hardware ordering, exact input union, RHS placement and readout.
"""
import json
from pathlib import Path


def plan_array_retention(schedule, rhs, input_rows, *, multirow=False):
    from gemmini_conv_array import plan_array_retention as plan
    settings = json.loads((Path(__file__).resolve().parent /
                           'CONV_GROUP_EXPERIMENT.json').read_text())
    limit = settings['full_resident_output_rows']
    if type(limit) is not int or limit not in (2, 4):
        raise ValueError('full-resident group experiment requires bound 2 or 4')
    return plan(schedule, rhs, input_rows, multirow=multirow,
                full_resident_output_rows=limit)
