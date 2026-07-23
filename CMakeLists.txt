cmake_minimum_required(VERSION 3.13)
project(gemmini_mlir CXX)

# Buildable OOT tree emitted by merlin-target-publish (WS-E). The thin driver below makes
# the repo contract-shaped so oot_runner can build build/bin/{tool} on a fresh clone; a
# real champion with a full mlir_oot/ tree overwrites this file with its own CMakeLists.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
add_executable(gemmini-opt tools/gemmini-opt/main.cpp)
