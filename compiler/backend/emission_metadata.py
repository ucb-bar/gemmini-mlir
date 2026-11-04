"""Deterministic projection of compiler metadata for executable emission.

Pass invocation timings describe the compiling process, not its selected program.
Process timings are omitted from emission; the full local input retains that
evidence for callers that need compiler performance feedback.
"""

from copy import deepcopy


# PipelineSession.run emits these fields with scope=host_pass_invocation;
# evaluate_variants also retains the same rows as prefix_host_pass_profile.
_PROFILE_KEYS = frozenset(("host_pass_profile", "prefix_host_pass_profile"))
_PROCESS_TIMINGS = frozenset((
    "action_wall_ns", "framework_wall_ns", "total_wall_ns",
))


def deterministic_compiler_metadata(value):
    """Copy metadata, omitting only known host pass invocation timings.

    Pass order, names, status, options, capabilities, identities, and proofs are
    preserved. Unknown profile schemas and timing-like fields elsewhere are
    preserved as well; this is deliberately not a generic volatility filter.
    """
    result = deepcopy(value)

    def visit(node):
        if isinstance(node, dict):
            for key, child in node.items():
                if key in _PROFILE_KEYS and isinstance(child, (list, tuple)):
                    for row in child:
                        if (isinstance(row, dict)
                                and row.get("scope") == "host_pass_invocation"
                                and isinstance(row.get("name"), str)
                                and row.get("status") in ("completed", "failed")
                                and isinstance(row.get("action_invoked"), bool)):
                            for timing in _PROCESS_TIMINGS:
                                if type(row.get(timing)) is int and row[timing] >= 0:
                                    del row[timing]
                visit(child)
        elif isinstance(node, (list, tuple)):
            for child in node:
                visit(child)

    visit(result)
    return result
