# Prepared source-integer PV: exact observations, unfavorable complete cost

The preceding fixed P14 approximation failed the original whole-model gate.
That failure and its four first-context quantizer crossings remain preserved.
This separate successor uses the same center with source-valid row/column bounds
and original point PV replay whenever the complete integer observation is
ambiguous. No precision variant was selected from golden outputs.

## Qualification

- All 22 original probability, projected i32 value, scale and floating-value
  inputs are unchanged. The native stand-in preserves all 360,448 compiled
  attention integer observations and every one of the 256,000 original whole
  outputs, including the unchanged Torch gate (`atol=.03125`, `rtol=.02`).
- Of 315,392 non-one-hot observations, 310,604 certify and 4,788 replay source
  point contractions. The 704 one-hot rows retain original source evaluation.
- The complete first target passes all 16,384 original integer observations,
  every one of 65,536 real device readout words, dirty destinations/input guards,
  five rounding modes with two sticky input contexts, and the executable noFSM
  audit. Non-RNE modes refuse before writing and execute original source.
- Floating flags are explicitly unobserved on the admitted path. These mode
  tests do not claim source/candidate sticky-flag equality.
- Fifty independent numeric, radix-plan and compiled executor tests pass.

## Complete cost and decision

The first source capsule retires **635,401** instructions. The candidate retires
**2,414,031** instructions, **3.799 times** as many. Both windows include their
complete output computation and stores. Candidate work includes probability and
value encoding, four grouped product callbacks / 16 physical KV matrix calls,
262,144 readback bytes, reconstruction, bound preparation, certificates and
source replay. Private workspace is 246,016 bytes. Its allocation and poisoning
are outside both windows; the prototype accepts an explicit reusable owner.

Actual hardware cycles and whole-model performance remain unknown. No cost
model currently prices the new binary64 certificate, packing and readout
dependency regime sufficiently to rank a win. The family remains a local,
unpromoted prototype with no normal target route or whole hardware request.
The first code bound fits three value digits; other contexts can require four.

The portable numerical/grouping/execution mechanism is in isolated Merlin
commit `6de060cd9`. Target callbacks, resource facts and ISA capabilities are
in OOT. [receipt.json](receipt.json) closes 533 source, tool, input and artifact
pins. Original compiler/object/executable identities and the earlier failure
are retained. Per-agent token allocation is unavailable; root records shared
campaign snapshots.
