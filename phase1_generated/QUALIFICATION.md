# R48 qualification

The compiler bundle comes from Merlin EL4 run `el4-proper-20261014-r48`.
Its complete authored submission digest is
`3a8ee6fd85b99934246d178786a03eef43697f71ced852e4e69518791d7fc20d`.
The target files in this directory are copied byte-for-byte from that
submission; diagnostic capsules and experiment machinery remain in Merlin.

The public self-check request
`2_1792006095292728820_a64fab31f1b1deecf5775af3a4561354`
certified all **146/146 public capsules at gSIM L3** for that digest. The
supplemental offload census joined command-buffer artifacts for 133/146
programs; 13 host-lane/source cases had no graded command buffer, so that
census is incomplete. The public certificate does not establish private
whole-model qualification.

The manager's final hidden/private verdict is pending. This record will be
updated before the branch is published.

The copied tool passed a focused parse and target-artifact emission check on
the public `g0_matmul.interface.mlir` example.
