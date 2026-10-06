#!/usr/bin/env bash
set -euo pipefail
set -a
source /scratch/agustin/projects/oscar-merlin/.env
set +a
export PYTHONPATH=.:/scratch/agustin/tmp/merlin-source-frontier-i64-20261006/src
export MERLIN_CLANG=/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin/clang
export MERLIN_MLIR_TRANSLATE=/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin/mlir-translate
export MERLIN_RTL_FACTS=/scratch/agustin/tmp/merlin-golden-language-models-20261005/out/artifacts/cache/rtl_introspect/gemmini/facts.json
export PATH=/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin:$PATH
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
/scratch/agustin/projects/oscar-merlin/.venv/bin/python out/normal_attention_provider_i64/validate_native.py
exec /scratch/agustin/projects/oscar-merlin/.venv/bin/python out/normal_attention_provider_i64/check_source_fallback.py
