#!/bin/sh
set -eu
cd "$(dirname "$0")"
prefix=${MERLIN_MLIR_INSTALL:-/work/merlin/toolchains/llvm-a47bddcc/install}
build_dir=$(mktemp -d /tmp/merlin-build-XXXXXX)
trap 'rm -rf "$build_dir"' EXIT
cmake -S . -B "$build_dir" -G Ninja -DMLIR_DIR="$prefix/lib/cmake/mlir" -DCMAKE_BUILD_TYPE=Release
cmake --build "$build_dir" -j 3
mkdir -p build
cp "$build_dir/merlin-opt" build/merlin-opt
