#!/bin/sh
set -eu
cd "$(dirname "$0")"
prefix=${MERLIN_MLIR_INSTALL:-/work/merlin/toolchains/llvm-a47bddcc/install}
if [ -f build/CMakeCache.txt ] && ! grep -Fq "CMAKE_HOME_DIRECTORY:INTERNAL=$(pwd)" build/CMakeCache.txt; then
  rm -rf build
fi
cmake -S . -B build -G Ninja -DMLIR_DIR="$prefix/lib/cmake/mlir" -DLLVM_DIR="$prefix/lib/cmake/llvm" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 3
