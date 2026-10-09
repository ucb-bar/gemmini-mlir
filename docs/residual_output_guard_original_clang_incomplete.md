# Original Clang residual capsule: incomplete performance run

The control reached terminal source/input/guard checks at 1,996,045 GSIM cycles. The candidate passed strict Spike and all 802,816 raw-predictor checks on GSIM, then reached the 1,800-second wall deadline before its timed interval or terminal source/input/guard marker. There is no candidate cycle result and no qualified performance comparison.

The same-text paired ELFs differ only by their recorded selector byte. The selected executable is preserved as `085b4940894c2b4947ad52a34db2df74ccd25911391aabe31b5783f6c4e861a7`. Compiler, fixture, source certificate, target object, engine, build receipt, and run records are pinned in the incomplete receipt. The earlier complete GCC negative remains a separate result.

The candidate emulator reported about 6M total simulation cycles at 1,777.9 seconds; the terminal control required 11.94M total simulation cycles. Total emulator progress includes harness setup and validation and is distinct from the timed target/correction interval. This shows why the wall deadline was insufficient for execution closure; it does not predict the candidate's ROI cycles.

The parent authorized one targeted replay of this unchanged candidate ELF. It retains the 30M simulation-cycle budget and raises only the wall deadline to 5,400 seconds. The replay runs in `out/residual_output_compose/clang_original_unchanged_replay`, preserving the first incomplete run. It performs no recompilation, control sweep, source mutation, coefficient search, or FireSim submission. Hardware release remains held until terminal checks and the actual ROI result close.
