# How `801eec3f` was composed

Composed by three-way merge of cell winners g1_stem 27901e2e, conv3x3 4184392d, mm1x1 036e1d96 onto efaed4dc.

| Role | Package digest |
|---|---|
| merge base | `efaed4dcd64a195423c6b79bdbe4a2c558c769fc79fbadf0698c8b6ff778e5ed` |
| g1 stem winner | `27901e2e16babdbe04d507d1adc294fa2c23fe34a9c62ad88442e25ec4b314e1` |
| 3×3 conv winner | `4184392da7a2a93add3b6d91d3c328664fea2cbd4ae397f0e1b53ecbc57224d5` |
| 1×1 matmul winner | `036e1d96d82677f590b8dfa14404475807f1b3965a35e0b09f62d03b9f90f2f6` |

## Recipe

Each winner was produced by a separate Codex (gpt-6-sol) cell run, seeded from the base. The merge followed three
rules:

- Start from a copy of the base.
- For a file changed by exactly one winner, copy that winner's version.
- For a file changed by two winners, take `git merge-file -p <winner A> <base> <winner B>`.

Results:

- **Conflicts:** 0. **Hand edits:** none.
- **Merged files:** the merge reproduces three files byte for byte:
  - `mlir_oot/lowering/contraction.py`: 3×3 + 1×1.
  - `mlir_oot/lowering/conv.py`: g1 + 3×3.
  - `mlir_oot/lowering/iface_to_gemmini.py`: g1 + 1×1.
- **Copied files:** `mlir_oot/codegen/llvm_emit.py`, `mlir_oot/codegen/reroll.py` and `mlir_oot/driver.py` equal the
  1×1 winner's.
- **Other differences:** the only non-code difference from the base is `HANDOFF.md`.
- **Composed by:** an operator-side Claude Code subagent (claude-opus-5-5) on the coordinator's instruction; no agent session authored the merge.
- **Known interaction:** the 3×3 edit (`group_loads_by_row=True` in `conv_a_operand`) reaches every convolution's
  moving operand, the g1 band path included.
- **Board check:** job 1436 (lean): 37,455,758 against the best single winner's 37,751,010 in the same batch.

The winners' bytes are kept in Merlin's measurement store (`artifacts/perf-studies/whole-model/gemmini/
contract_row_layout/a6581020d64a0a3d/<digest>/package`). They are not part of this branch.
