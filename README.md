# gemmini-mlir: phase-1 champion `1ba5615f` (cand_07)

> **Certified phase-1 champion.** Merlin's champion export (`merlin.targetgen.champions`, Merlin
> `ad999e05c`) checked this package against its phase-1 evidence profile, with no check relaxed,
> and it passed:
> - a capsule certification at L3: public 121/180 and hidden
>   14/14, every pass executed in this certification (none carried);
> - a clean whole-ELF scan of all 175 capsule ELFs for the hardware-loop instructions.
>
> The gate's record is `.merlin/records/export_gate.json`. A phase-1 compiler is certified on capsules, so the
> profile asks for no whole-model number.
>
> **Unsealed legacy lineage.** This lineage predates the sealed phase 0. It has no corpus seal and no phase-0
> evidence digest, and no stand-in digest is substituted for them.

## What this is

This is a Gemmini compiler. `gemmini-opt` is an out-of-tree MLIR backend written in Python on xDSL (package id
`gemmini_xdsl_oot_backend`). It lowers Merlin's interface dialect to Gemmini RoCC command streams and an LLVM
artifact.

- **Package:** `1ba5615f0d32c54b162703871cea3a2b7faf7ced393ee3d787802899d0e41adc`, 54 files. The digest is Merlin's `hash_tree`.
- **What the branch holds:** the exact bytes that were certified, unmodified. `sha256sum -c .merlin/records/SHA256SUMS`
  checks them.
- **Phase-2 starting point:** this is the compiler phase 2 started from. The phase-2 champion `801eec3f`
  (branch `champions/phase2/801eec3f`) descends from it, and both branches' reconstructed histories share the
  frozen commit `08239bcfedc3`.
- **Not `stable/gemmini_xdsl_rtl_v0`:** that branch is an unrelated compiler from another run.

## How it was produced

- **Run:** the capsule-bench run `merlincirct_p1froz`, round 07.
  - Arm `merlin_assisted`, bundle `merlin_assisted_rtlchecks_public_v0`.
  - Driver: Claude Code with claude-opus-5 at high effort.
  - Schedule: one continuous session.
  - Corpus: 173 public/dev capsules plus 15 held out.
- **Seed:** the run started from `806a1f32`, the frozen submission of `gem_p1_seeded_0923_0628` (same driver and
  model). That submission was formally graded: public 101/103, hidden 14/14.
- **How cand_07 was chosen:**
  - The QA loop scored rounds 01–11 at 161/173 each.
  - cand_07 was picked by hand from the 36 tied candidates; the pipeline records `chosen_by: caller` and no reason.
  - The run itself never froze or graded a submission. The certification below was made afterwards, on these exact
    bytes.
- **Reconstructed history:** no harness git history was kept. Merlin rebuilt one from the stored bytes, checking
  every hop's digest, and recorded it as `reconstructed: true`:
  `806a1f32` → `1ba5615f`.
  The champion is the `frozen` commit `08239bcfedc3`.

## How it was certified

The certification of record is `grade_l3_fresh`. It ran fresh: tier-certificate cache OFF, ELF build cache OFF (every L3 executed now), from 2026-10-06T07:08:46Z to 2026-10-06T09:03:18Z.

- **Grader:** the old-line formal post-freeze grader at Merlin `7013aff4b`. Merlin main cannot
  consume this bundle.
- **Required tier:** L3.
- **L3 engine:** `gsim`, running `GemminiGsimSerialClkConfig` as elaborated RTL.
  - Emulator sha256 `1a3de02a19ddcb4704443e2b565fd6eb0299a940afdcc7c037e2604bbcbb52ea`, registry artifact `gemmini_gsim_emulator`.
  - 135 capsule runs used it; 40 capsules have no
    L3 engine.
- **Fresh versus carried:** every L3 pass was executed in this certification; none was carried from an earlier
  grade. This was counted from each capsule's own result (`tiers.L3`, `tier_reuse`).
- **Toolchain pins declared by the grade:**
  - `gemmini_rtl` and `gemmini_rocket_rtl`: `63f0b68a`.
  - `gemmini_rocket_chipyard`: `009e85b0`.
  - `gemmini_isa_headers`: `6b477a8b`.
- **Corpus identity:**
  - Input bundle manifest sha256 `5d540284de06193b2af67ffdb03d35892913141ede6ffdcfefe3514afa5a6f1b`.
  - Public contract sha256 `44d695d9c22b1c5a77932768d5dc2ec8d413783903d9d7342c9175b6b3272f21`.
  - Hidden capsule snapshot `770b4a350d82417a3f4ecb6cf1509e75eeb14ae4a516bccb6341fc34a09d4247`.
- **No-FSM check:** main's whole-ELF scanner (`merlin.perf.isa_prohibition.scan_elf`) found none of the
  14 loop-descriptor instructions (selectors 8–13, 15–21, 24) in any of the
  175 capsule ELFs. It derived that set from the target's facts.

## The numbers

| Set | Graded | At L3 | Executed now | Carried | Pass at any tier |
|---|---:|---:|---:|---:|---:|
| Public | 180 | **121** | 121 | 0 | 160 |
| Hidden | 14 | **14** | 14 | 0 | 14 |

- **Public status counts:** error 9, fail 1, incomplete 10, not_graded 11, pass 160.
  That is 191 admitted capsules; the
  11 not graded leave 180 graded.
- **L2-only passes:** 39 public capsules passed only at L2. These are never L3 claims.
- **Formal completion of the grade is unreachable** for this bundle with this grader, whatever the compiler does:
  - 39 public capsules declare max_oracle_tier L2; they can never reach L3;
  - 11 bf16 capsules are outside the declared operand dtype capability (not graded);
  - 9 host-lane capsules crash the grader's golden (unsupported operation layernorm/gelu/softmax/reduce_sum/depthwise_conv2d);
  - 10 model capsules lack their weights asset in the grader tree;
  - 1 capsule (SY_host_only_attention) timed out at elaborated_rtl (900 s): unmeasured, not wrong.
- **No whole-model ratios:** a phase-1 compiler is certified on capsules, and there is no valid whole-model cycle
  count for this package, so there is no ratio to the vendor library, to Exo or to the RTL-fact roofline.
  - The only whole-model board reading is job 1015 on the lean board, `MEASURED_INVALID`.
  - That build predates the row-layout fix and was not held to the no-FSM rule.
  - The phase-2 numbers belong to `801eec3f`.

## What this does not claim

- **Not a formally complete grade.** 121/180 public capsules pass at L3, not 180.
- **No whole-model result.** There is no valid FireSim or GSIM ResNet-50 number for these bytes.
- **Not a sealed result.** See the legacy lineage in `MERLIN_PUBLICATION.md`.
- **The payload gate is a separate record.** `.merlin/certification.yaml` belongs to the publish layer's own,
  optional gate: `oot_runner.certify` rungs bound to the payload bytes. No such rungs were run for this package,
  so that file reads `unverified` in its `package-payload` scope. The champion's certification is the capsule
  certification above.
- **Pin drift.** Today's pin registry reports drift for `gemmini_isa_headers` (the grade's revision `6b477a8b`
  against the pinned `7c540b3a`) and for the GSIM compiler checkout. The emulator binary is fixed by digest.

## Using it

The tool needs Python 3 and `xdsl` 0.68.0, which is not vendored. It was verified with Python 3.12 and 3.13; other
`xdsl` versions are untested.

```sh
git clone -b champions/phase1/1ba5615f https://github.com/ucb-bar/gemmini-mlir.git gemmini-mlir-1ba5615f
cd gemmini-mlir-1ba5615f
sha256sum -c --quiet .merlin/records/SHA256SUMS      # the certified bytes
python3 -m venv .venv && .venv/bin/pip install xdsl==0.68.0
. .venv/bin/activate
./gemmini-opt --verify-diagnostics input.mlir                                   # parse
./gemmini-opt --convert-iface-to-gemmini input.mlir                             # lower to the gemmini dialect
./gemmini-opt --convert-iface-to-gemmini --emit-command-buffer=cb.json input.mlir
./gemmini-opt --convert-iface-to-gemmini --emit-target-artifact input.mlir      # LLVM artifact
```

- **Input:** `input.mlir` is a module in Merlin's interface dialect (`merlin_iface`), the form Merlin's frontend
  produces for each capsule.
- **Commands:** `manifest.yaml` declares the exact argv of each command Merlin's experiment ABI uses.

## Files

- `manifest.yaml`, `gemmini-opt`, `mlir_oot/`, `devtools/`, `docs/`, `REPORT.md`: the package, unchanged. Layout
  2.0 keeps the payload in place, unlike the layout-1.0 `stable/*` branches.
- `MERLIN_PUBLICATION.md`: Merlin's generated landing page, with the evidence summary, the legacy lineage and the
  reconstructed history.
- `.merlin/provenance.json`, `certification.json`, `measurements.json`, `isa_prohibition.json`: the champion records
  written by Merlin's export. The ISA record lists every scanned capsule ELF.
- `.merlin/manifest.yaml`, `provenance.yaml`, `certification.yaml`, `CHAMPION`: the publish layer's records.
- `.merlin/records/`:
  - `export_gate.json`: the gate's record (passed, nothing relaxed).
  - `SHA256SUMS`: hashes of the package files.

Paths inside the records are relative to Merlin's `out/` directory on the host that produced them.
