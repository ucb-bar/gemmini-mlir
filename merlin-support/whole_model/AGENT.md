# Gemmini whole-model program driver

The target-owned half of a whole-model build. Shared Merlin (`merlin.perf.whole_model_build`) reaches
these modules only through the backend's `whole_model_driver()` capability and never by name.

- `group_model_program.py`: extracts a captured model's steps, renders the bare-metal C program
  (vendor-library fallback calls, timing brackets, UART protocol, the program's own loop-free host
  routines such as `hr_conv2d`/`hr_acc_matmul`) and links it.
- `group_model_submission_kernels.py`: binds a package's per-group kernel objects into that program.
- `group_model_dispatch.py`: answers an open model's device dispatches.
- `group_model_sched_kernels.py`: this repository's own schedules for the program's CLI.

Host-owned reference code, never a candidate payload. Select this provider with `MERLIN_TARGET_PATH`.
Tests live with Merlin's gemmini bucket and skip when no provider is selected. Imports are absolute
or provider-relative, with no ambient `sys.path` change. Migration hashes are immutable.
