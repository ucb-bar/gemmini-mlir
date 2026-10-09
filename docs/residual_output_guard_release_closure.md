# Exact residual output guard: terminal capsule and release closure

The unchanged original-capture candidate replay terminated successfully. Its complete 802,816-output GSIM interval takes **1,656,089 cycles**, versus **1,996,045** for the unchanged control: 339,956 fewer cycles (17.03%). The interval includes input/coefficient DMA, prediction, one scaled store and fence, the complete output guard scan, and original ordered binary32 correction. Both raw predictions and all corrected outputs pass, with 4,096 guard bytes and 1,605,632 immutable input bytes checked.

The replay uses the exact previously incomplete candidate ELF `085b4940894c2b4947ad52a34db2df74ccd25911391aabe31b5783f6c4e861a7`. Only its wall deadline changed from 1,800 to 5,400 seconds; its 30M simulation-cycle limit and source, parameters, addresses, compiler, and engine stayed fixed. It finished in 1,805 seconds. The earlier incomplete run remains historical. The complete original GCC arm remains rejected at +2.82%.

The normal compiler route and frozen 1903 controlled route independently pass native and strict Spike against all 1,000 original binary32 words, with bitwise equality and `atol=rtol=0`. Both final ELFs pass fresh zero-FSM audits. The controlled route reproduces the actual 1903 control, changes exactly one residual kernel and its adapter, and preserves every other device object plus all host/runtime/startup/weight/shim objects. The sealed typed source, complete signed-byte certificate, and fresh disjoint three-argument output contract remain bound to emitted objects.

## Stock comparison packet

The controlled candidate is `/scratch/agustin/tmp/gemmini-residual-output-compose-20261006/out/residual_output_compose/controlled1903/model.elf`, SHA256 `11c224650e91080b8fc28e15c7b5bf1747cec74797678492fe783e03c8790a59`. Its control is stock job 1903, **36,102,704 cycles**. The inherited build marker is nonunique; full ELF SHA is authoritative. This candidate is independent of later segmented, spatial-loop, and stem-loop arms.

Whole-model candidate cycles remain **UNKNOWN**. The GSIM capsule establishes admission evidence in its own memory regime; it does not predict stock savings. Defaults remain unchanged. Root release to the sole queue owner precedes one stock comparison.

The combined receipt pins 321 artifacts, including the original incomplete record, complete replay, source proofs, whole validations, component closure, compiler and engine evidence. Reclose with:

```sh
PYTHONPATH=/scratch/agustin/tmp/gemmini-residual-output-compose-20261006:/scratch/agustin/tmp/merlin-residual-output-word-20261005/src \
  /scratch/agustin/projects/oscar-merlin/.venv/bin/python \
  tests/residual_output_release_probe.py \
  docs/perf_records/residual_output_guard_release_closure.json
```

Merlin owns certificate/guard/source replay; OOT owns diagonal decomposition, device lifetimes, ABI, and source-bound catalog integration. Child token attribution is unavailable; root retains session counters.
