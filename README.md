# Merlin paper compiler versions

Start with the experiment branches below. Every preserved compiler also has a permanent annotated snapshot tag. Original run IDs and paths remain searchable in [runs.csv](runs.csv) and [runs.json](runs.json). Names are aliases for existing runs, not new experiments.

| Experiment | Branch | Original result ID |
|---|---|---|
| Merlin construction without expert kernels | [paper/construction/no-expert-kernels-4h](https://github.com/ucb-bar/gemmini-mlir/tree/paper/construction/no-expert-kernels-4h) | `F10-FULL-NOREF-NOCCA` |
| E1: raw RTL and C++ | [paper/scaffolding/e1-raw-rtl-4h](https://github.com/ucb-bar/gemmini-mlir/tree/paper/scaffolding/e1-raw-rtl-4h) | `EL1-r5` |
| E2: C++ interface support | [paper/scaffolding/e2-cpp-support-4h](https://github.com/ucb-bar/gemmini-mlir/tree/paper/scaffolding/e2-cpp-support-4h) | `EL2-r5` |
| E3: Merlin compiler support | [paper/scaffolding/e3-merlin-support-4h](https://github.com/ucb-bar/gemmini-mlir/tree/paper/scaffolding/e3-merlin-support-4h) | `EL3-r5` |
| CCA study with TinyLlama kernels, revision 2 | [paper/cca/tinyllama-kernels-4plus4h](https://github.com/ucb-bar/gemmini-mlir/tree/paper/cca/tinyllama-kernels-4plus4h) | `CCA-TINYLLAMA-R2` |
| Merlin refinement with Exo kernels | [paper/refinement/exo-kernels-4plus4h](https://github.com/ucb-bar/gemmini-mlir/tree/paper/refinement/exo-kernels-4plus4h) | `CCA-EXO-R3` |
| CCA study with ResNet-50 kernels, revision 2 | [paper/cca/resnet50-kernels-4plus4h](https://github.com/ucb-bar/gemmini-mlir/tree/paper/cca/resnet50-kernels-4plus4h) | `CCA-RESNET50-R2` |
| Checked scheduling, main thirty-minute study: seed scheduler | [paper/scheduling/main-30min/seed-scheduler](https://github.com/ucb-bar/gemmini-mlir/tree/paper/scheduling/main-30min/seed-scheduler) | `followup-scheduling-h3a-30min-v1-control` |
| Checked scheduling, main thirty-minute study: checked scheduler | [paper/scheduling/main-30min/checked-scheduler](https://github.com/ucb-bar/gemmini-mlir/tree/paper/scheduling/main-30min/checked-scheduler) | `followup-scheduling-h3a-30min-v1-treatment` |
| Convolution transformation-library study: seed library | [paper/transformations/convolution-30min/seed-library](https://github.com/ucb-bar/gemmini-mlir/tree/paper/transformations/convolution-30min/seed-library) | `followup-library-oracle-h2conv-30min-v3-control` |
| Convolution transformation-library study: extended library | [paper/transformations/convolution-30min/extended-library](https://github.com/ucb-bar/gemmini-mlir/tree/paper/transformations/convolution-30min/extended-library) | `followup-library-oracle-h2conv-30min-v3-treatment` |

## Find a compiler from a run

After checking out this index branch, use the original result ID, a full run path, an old branch name, or a source digest:

```sh
python3 lookup_run.py CCA-EXO-R3
python3 lookup_run.py F10-FULL-NOREF-NOCCA
python3 lookup_run.py followup-cca-h1d-p2r-control
```

The lookup returns the snapshot tag, commit, source digest, and experiment branch when one is maintained. To retrieve a snapshot without relying on a moving branch:

```sh
git fetch origin tag mlsys-paper-v1/refinement/exo-kernels-4plus4h
git switch --detach mlsys-paper-v1/refinement/exo-kernels-4plus4h
python3 experiment/verify_snapshot.py
```

## Naming and budgets

- `paper/` branches are entry points for the principal construction, refinement, scaffolding, scheduling and transformation conditions.
- `mlsys-paper-v1/` tags preserve all 38 original complete commits, including supporting studies and historical checkpoints.
- `4plus4h` means four hours of construction and four additional hours of refinement, not eight new hours of refinement.
- Thirty- and sixty-minute study budgets exclude the prior work that produced their seed compiler.
- Revision and study numbers identify distinct recorded trials. They do not imply matched conditions or independent replications.
- `seed-scheduler` retains the inherited scheduling support. It does not mean an absence of compiler libraries.

## All preserved snapshots

| Experiment | Original result ID | Permanent tag | Original commit |
|---|---|---|---|
| Earlier CCA campaign, four-hour checkpoint | `fresh10cca-live` | [mlsys-paper-v1/historical/cca-campaign-checkpoint-4h](https://github.com/ucb-bar/gemmini-mlir/tree/mlsys-paper-v1/historical/cca-campaign-checkpoint-4h) | [`4022f07c4fa8`](https://github.com/ucb-bar/gemmini-mlir/tree/4022f07c4fa8950e03b920cd015e438ff9959243) |
| Earlier seed campaign, four-hour checkpoint | `fresh10seed-live` | [mlsys-paper-v1/historical/seed-campaign-checkpoint-4h](https://github.com/ucb-bar/gemmini-mlir/tree/mlsys-paper-v1/historical/seed-campaign-checkpoint-4h) | [`ee46ba4f90f4`](https://github.com/ucb-bar/gemmini-mlir/tree/ee46ba4f90f4de8ce7a5cc10c2badbf2cc63fc99) |
| Earlier CCA campaign, final twelve-hour snapshot | `fresh10cca-final12h` | [mlsys-paper-v1/historical/cca-campaign-final-12h](https://github.com/ucb-bar/gemmini-mlir/tree/mlsys-paper-v1/historical/cca-campaign-final-12h) | [`f7ae619495aa`](https://github.com/ucb-bar/gemmini-mlir/tree/f7ae619495aaa683648a6349aa20bc8ec371a5b8) |
| Earlier seed campaign, final twelve-hour snapshot | `fresh10seed-final12h` | [mlsys-paper-v1/historical/seed-campaign-final-12h](https://github.com/ucb-bar/gemmini-mlir/tree/mlsys-paper-v1/historical/seed-campaign-final-12h) | [`5f8b1a283714`](https://github.com/ucb-bar/gemmini-mlir/tree/5f8b1a2837144d94018bf6354057336c983bcb0f) |
| Compiler-support study with references | `F10-INFRA-REF-NOCCA` | [mlsys-paper-v1/tooling/compiler-support-with-references-4h](https://github.com/ucb-bar/gemmini-mlir/tree/mlsys-paper-v1/tooling/compiler-support-with-references-4h) | [`bb20d8e2d7f8`](https://github.com/ucb-bar/gemmini-mlir/tree/bb20d8e2d7f82243729117f9bd7c8c859a8d8061) |
| Scheduling-support study with references | `F10-SCHED-REF-NOCCA` | [mlsys-paper-v1/tooling/scheduling-support-with-references-4h](https://github.com/ucb-bar/gemmini-mlir/tree/mlsys-paper-v1/tooling/scheduling-support-with-references-4h) | [`4202a912b894`](https://github.com/ucb-bar/gemmini-mlir/tree/4202a912b89421e9a18bac8af63cab9fa6f3df66) |
| Merlin construction without expert kernels | `F10-FULL-NOREF-NOCCA` | [mlsys-paper-v1/construction/no-expert-kernels-4h](https://github.com/ucb-bar/gemmini-mlir/tree/mlsys-paper-v1/construction/no-expert-kernels-4h) | [`8aebaf251eb2`](https://github.com/ucb-bar/gemmini-mlir/tree/8aebaf251eb239c52a33bf85559dbc1a48bdcf0e) |
| E1: raw RTL and C++ | `EL1-r5` | [mlsys-paper-v1/scaffolding/e1-raw-rtl-4h](https://github.com/ucb-bar/gemmini-mlir/tree/mlsys-paper-v1/scaffolding/e1-raw-rtl-4h) | [`32e1d3784d01`](https://github.com/ucb-bar/gemmini-mlir/tree/32e1d3784d018cd121f784a0b872cb2125b4e8c6) |
| E2: C++ interface support | `EL2-r5` | [mlsys-paper-v1/scaffolding/e2-cpp-support-4h](https://github.com/ucb-bar/gemmini-mlir/tree/mlsys-paper-v1/scaffolding/e2-cpp-support-4h) | [`e300b7f47fef`](https://github.com/ucb-bar/gemmini-mlir/tree/e300b7f47fef08bfb8a4e629515cd4d862cbed83) |
| E3: Merlin compiler support | `EL3-r5` | [mlsys-paper-v1/scaffolding/e3-merlin-support-4h](https://github.com/ucb-bar/gemmini-mlir/tree/mlsys-paper-v1/scaffolding/e3-merlin-support-4h) | [`b0b21a359e19`](https://github.com/ucb-bar/gemmini-mlir/tree/b0b21a359e19b46582db47761db2277968c0b917) |
| Raw expert-file continuation | `CCA-RAW` | [mlsys-paper-v1/cca/raw-expert-files-4plus4h](https://github.com/ucb-bar/gemmini-mlir/tree/mlsys-paper-v1/cca/raw-expert-files-4plus4h) | [`05b2b2897cb0`](https://github.com/ucb-bar/gemmini-mlir/tree/05b2b2897cb04d770320e5fe10c54eed2c60c306) |
| Exo continuation, revision 1 | `CCA-EXO` | [mlsys-paper-v1/cca/exo-kernels-revision1-4plus4h](https://github.com/ucb-bar/gemmini-mlir/tree/mlsys-paper-v1/cca/exo-kernels-revision1-4plus4h) | [`13b1aa8d163f`](https://github.com/ucb-bar/gemmini-mlir/tree/13b1aa8d163f3d333adb5c900eb984bcfc381693) |
| TinyLlama continuation, revision 1 | `CCA-TINYLLAMA` | [mlsys-paper-v1/cca/tinyllama-kernels-revision1-4plus4h](https://github.com/ucb-bar/gemmini-mlir/tree/mlsys-paper-v1/cca/tinyllama-kernels-revision1-4plus4h) | [`b06f8a528865`](https://github.com/ucb-bar/gemmini-mlir/tree/b06f8a52886529505cba3290ace378a288db3659) |
| ResNet-50 continuation, revision 1 | `CCA-RESNET50` | [mlsys-paper-v1/cca/resnet50-kernels-revision1-4plus4h](https://github.com/ucb-bar/gemmini-mlir/tree/mlsys-paper-v1/cca/resnet50-kernels-revision1-4plus4h) | [`551d36bd7f0f`](https://github.com/ucb-bar/gemmini-mlir/tree/551d36bd7f0f7931d23277c3630392b4e3f7d90c) |
| All-source continuation, revision 1 | `CCA-ALL` | [mlsys-paper-v1/cca/all-sources-revision1-4plus4h](https://github.com/ucb-bar/gemmini-mlir/tree/mlsys-paper-v1/cca/all-sources-revision1-4plus4h) | [`baaafedf75f2`](https://github.com/ucb-bar/gemmini-mlir/tree/baaafedf75f28237d707bbf6fc5654948f65cf37) |
| Curated-reference construction, four-hour checkpoint | `FULL-REF-4H` | [mlsys-paper-v1/historical/curated-references-checkpoint-4h](https://github.com/ucb-bar/gemmini-mlir/tree/mlsys-paper-v1/historical/curated-references-checkpoint-4h) | [`cdbfa200cb77`](https://github.com/ucb-bar/gemmini-mlir/tree/cdbfa200cb770688b9f1d54e2dce48d40afb783c) |
| Curated-reference construction, final twelve-hour snapshot | `FULL-REF` | [mlsys-paper-v1/historical/curated-references-final-12h](https://github.com/ucb-bar/gemmini-mlir/tree/mlsys-paper-v1/historical/curated-references-final-12h) | [`8b6cbe401575`](https://github.com/ucb-bar/gemmini-mlir/tree/8b6cbe401575d53d46f14e0f4bbea7f274db9dd1) |
| All-source continuation, revision 2 | `CCA-ALL-R2` | [mlsys-paper-v1/cca/all-sources-revision2-4plus4h](https://github.com/ucb-bar/gemmini-mlir/tree/mlsys-paper-v1/cca/all-sources-revision2-4plus4h) | [`cd49f04a8f22`](https://github.com/ucb-bar/gemmini-mlir/tree/cd49f04a8f22a8546f51505d182017b06981d968) |
| CCA study with TinyLlama kernels, revision 2 | `CCA-TINYLLAMA-R2` | [mlsys-paper-v1/cca/tinyllama-kernels-4plus4h](https://github.com/ucb-bar/gemmini-mlir/tree/mlsys-paper-v1/cca/tinyllama-kernels-4plus4h) | [`812d0fa016ed`](https://github.com/ucb-bar/gemmini-mlir/tree/812d0fa016edca86017fe3b594bce1a48688e3df) |
| Merlin refinement with Exo kernels | `CCA-EXO-R3` | [mlsys-paper-v1/refinement/exo-kernels-4plus4h](https://github.com/ucb-bar/gemmini-mlir/tree/mlsys-paper-v1/refinement/exo-kernels-4plus4h) | [`aad7a9b21acf`](https://github.com/ucb-bar/gemmini-mlir/tree/aad7a9b21acfb9695db7d20eee90fab51d273b35) |
| CCA study with ResNet-50 kernels, revision 2 | `CCA-RESNET50-R2` | [mlsys-paper-v1/cca/resnet50-kernels-4plus4h](https://github.com/ucb-bar/gemmini-mlir/tree/mlsys-paper-v1/cca/resnet50-kernels-4plus4h) | [`0cf80344f981`](https://github.com/ucb-bar/gemmini-mlir/tree/0cf80344f9810414aea5d5c134c06e5792e83960) |
| Exo continuation, revision 2 | `CCA-EXO-R2` | [mlsys-paper-v1/cca/exo-kernels-revision2-4plus4h](https://github.com/ucb-bar/gemmini-mlir/tree/mlsys-paper-v1/cca/exo-kernels-revision2-4plus4h) | [`bfd1da28a192`](https://github.com/ucb-bar/gemmini-mlir/tree/bfd1da28a192e57f91b36324b38b1b63fd1f8061) |
| Checked scheduling, main thirty-minute study: seed scheduler | `followup-scheduling-h3a-30min-v1-control` | [mlsys-paper-v1/scheduling/main-30min/seed-scheduler](https://github.com/ucb-bar/gemmini-mlir/tree/mlsys-paper-v1/scheduling/main-30min/seed-scheduler) | [`0c1aad41ec0e`](https://github.com/ucb-bar/gemmini-mlir/tree/0c1aad41ec0e722c00d5eca190ea552d9ccea39e) |
| Checked scheduling, main thirty-minute study: checked scheduler | `followup-scheduling-h3a-30min-v1-treatment` | [mlsys-paper-v1/scheduling/main-30min/checked-scheduler](https://github.com/ucb-bar/gemmini-mlir/tree/mlsys-paper-v1/scheduling/main-30min/checked-scheduler) | [`87f0f47ca08a`](https://github.com/ucb-bar/gemmini-mlir/tree/87f0f47ca08a77967a7927fead97a1cf46bb0dd6) |
| Checked scheduling, sixty-minute study 2: seed scheduler | `scheduling-completion-authors-control` | [mlsys-paper-v1/scheduling/study02-60min/seed-scheduler](https://github.com/ucb-bar/gemmini-mlir/tree/mlsys-paper-v1/scheduling/study02-60min/seed-scheduler) | [`1c8236332cab`](https://github.com/ucb-bar/gemmini-mlir/tree/1c8236332cab090947dcc9daf167e62b028f6868) |
| Checked scheduling, sixty-minute study 2: checked scheduler | `scheduling-completion-authors-treatment` | [mlsys-paper-v1/scheduling/study02-60min/checked-scheduler](https://github.com/ucb-bar/gemmini-mlir/tree/mlsys-paper-v1/scheduling/study02-60min/checked-scheduler) | [`2931d3f242d3`](https://github.com/ucb-bar/gemmini-mlir/tree/2931d3f242d3df35350456fcd70ecd2bfe7889f8) |
| Checked scheduling, sixty-minute study 3: seed scheduler | `scheduling-completion-matched-60-r1-control` | [mlsys-paper-v1/scheduling/study03-60min/seed-scheduler](https://github.com/ucb-bar/gemmini-mlir/tree/mlsys-paper-v1/scheduling/study03-60min/seed-scheduler) | [`1af651fe24fd`](https://github.com/ucb-bar/gemmini-mlir/tree/1af651fe24fda3a736becdeb0ec852457f9b5702) |
| Checked scheduling, sixty-minute study 3: checked scheduler | `scheduling-completion-matched-60-r1-treatment` | [mlsys-paper-v1/scheduling/study03-60min/checked-scheduler](https://github.com/ucb-bar/gemmini-mlir/tree/mlsys-paper-v1/scheduling/study03-60min/checked-scheduler) | [`8e9daa7e3638`](https://github.com/ucb-bar/gemmini-mlir/tree/8e9daa7e3638d9e0cb6eb789fc6a06045f75df48) |
| Checked scheduling, sixty-minute study 4: seed scheduler | `followup-scheduling-reference-free-recovery-r1-control` | [mlsys-paper-v1/scheduling/study04-60min/seed-scheduler](https://github.com/ucb-bar/gemmini-mlir/tree/mlsys-paper-v1/scheduling/study04-60min/seed-scheduler) | [`b7294534993a`](https://github.com/ucb-bar/gemmini-mlir/tree/b7294534993aa5fb30a006df67c7f61aa2bfdd3d) |
| Checked scheduling, sixty-minute study 4: checked scheduler | `followup-scheduling-reference-free-recovery-r1-treatment` | [mlsys-paper-v1/scheduling/study04-60min/checked-scheduler](https://github.com/ucb-bar/gemmini-mlir/tree/mlsys-paper-v1/scheduling/study04-60min/checked-scheduler) | [`012ca7567fd1`](https://github.com/ucb-bar/gemmini-mlir/tree/012ca7567fd1649c13daddad029c98dd299c3007) |
| Checked scheduling, capability-extension study: seed scheduler | `followup-scheduling-extension-v1-control` | [mlsys-paper-v1/scheduling/extension-60min/seed-scheduler](https://github.com/ucb-bar/gemmini-mlir/tree/mlsys-paper-v1/scheduling/extension-60min/seed-scheduler) | [`00eb9cc42731`](https://github.com/ucb-bar/gemmini-mlir/tree/00eb9cc42731605bda1e480252d14fd6fa81f10f) |
| Checked scheduling, capability-extension study: checked scheduler | `followup-scheduling-extension-v1-treatment` | [mlsys-paper-v1/scheduling/extension-60min/checked-scheduler](https://github.com/ucb-bar/gemmini-mlir/tree/mlsys-paper-v1/scheduling/extension-60min/checked-scheduler) | [`80fd2f201fcf`](https://github.com/ucb-bar/gemmini-mlir/tree/80fd2f201fcf5206b17290c17eb6111da3fab2cc) |
| Convolution transformation-library study: seed library | `followup-library-oracle-h2conv-30min-v3-control` | [mlsys-paper-v1/transformations/convolution-30min/seed-library](https://github.com/ucb-bar/gemmini-mlir/tree/mlsys-paper-v1/transformations/convolution-30min/seed-library) | [`58f007262071`](https://github.com/ucb-bar/gemmini-mlir/tree/58f0072620713ae6543a8caefa83439e9cc30388) |
| Convolution transformation-library study: extended library | `followup-library-oracle-h2conv-30min-v3-treatment` | [mlsys-paper-v1/transformations/convolution-30min/extended-library](https://github.com/ucb-bar/gemmini-mlir/tree/mlsys-paper-v1/transformations/convolution-30min/extended-library) | [`766ca91656e9`](https://github.com/ucb-bar/gemmini-mlir/tree/766ca91656e917a3ede13dc8ef0d90a2efc99e03) |
| Controlled CCA pair 1: cca off | `followup-cca-h1d-p1r-control` | [mlsys-paper-v1/cca-controlled/pair1-30min/cca-off](https://github.com/ucb-bar/gemmini-mlir/tree/mlsys-paper-v1/cca-controlled/pair1-30min/cca-off) | [`2c2fdb2c3cf3`](https://github.com/ucb-bar/gemmini-mlir/tree/2c2fdb2c3cf38294566ce21fae05886d647ddb18) |
| Controlled CCA pair 1: cca on | `followup-cca-h1d-p1r-treatment` | [mlsys-paper-v1/cca-controlled/pair1-30min/cca-on](https://github.com/ucb-bar/gemmini-mlir/tree/mlsys-paper-v1/cca-controlled/pair1-30min/cca-on) | [`a5c73bfcbb6a`](https://github.com/ucb-bar/gemmini-mlir/tree/a5c73bfcbb6ae644ad7907b4b50ffd44e3b506d5) |
| Controlled CCA pair 2: cca on | `followup-cca-h1d-p2r-control` | [mlsys-paper-v1/cca-controlled/pair2-30min/cca-on](https://github.com/ucb-bar/gemmini-mlir/tree/mlsys-paper-v1/cca-controlled/pair2-30min/cca-on) | [`9130e99b7c01`](https://github.com/ucb-bar/gemmini-mlir/tree/9130e99b7c014eb2ee6098e13bf930e47dcd3ab6) |
| Controlled CCA pair 2: cca off | `followup-cca-h1d-p2r-treatment` | [mlsys-paper-v1/cca-controlled/pair2-30min/cca-off](https://github.com/ucb-bar/gemmini-mlir/tree/mlsys-paper-v1/cca-controlled/pair2-30min/cca-off) | [`ed9b3f45dcd9`](https://github.com/ucb-bar/gemmini-mlir/tree/ed9b3f45dcd92f00372cc3c04f7e5f4ec8675eef) |

## Evidence and package contents

Each snapshot contains the original compiler in `compiler/`, setup in `experiment/setup.json`, results in `experiment/metrics.json`, and a source-hash verifier in `experiment/verify_snapshot.py`. Where captured, shared dependencies are retained in `dependencies/`. Principal branches add only navigation documentation and identity metadata. Compiler, dependency and measurement bytes match their tagged snapshots.

The 89-case hardware panel, construction tests and small paired studies have different evaluation scopes. Measurements from different images and the common plotted FSM reference remain separate. The later current-library and adapted-FSM controls are attached to the Exo-refined snapshot under `experiment/later-validation/`.

Controlled CCA pair 2 reverses the positional queue labels: the old ID ending in `control` is the CCA-on arm, and the ID ending in `treatment` is CCA-off. The tag names follow the recorded `study_role` and `tooling`, rather than the queue position. Original IDs and observations are unchanged.

This is a verified source-and-record archive, not a fully reconstructed execution environment. Four historical snapshots retain vendored Z3 shared libraries, and two early checkpoint model fields remain unresolved. Framework exports, runtime, toolchains, evaluators and hardware dependencies still require full artifact closure. No compiler, agent, simulator or FPGA experiment was rerun for this organization.

Snapshot commits descend from a repository landing commit for storage. This is not the agent development history. No repository main branch or unrelated branch was modified. The separate `mlsys27/test-generation-coverage` branch is outside this reorganization.
