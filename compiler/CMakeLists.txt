cmake_minimum_required(VERSION 3.20)
project(GeneralGemminiCompiler LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddLLVM)
include(AddMLIR)
include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
add_executable(gemmini-compile main.cpp frontend.cpp commands.cpp device.cpp blocked.cpp conv.cpp pool.cpp pool_hw.cpp conv_pool_hw.cpp residual.cpp codegen.cpp settings.cpp)
target_compile_options(gemmini-compile PRIVATE -O1 -fno-rtti)
target_compile_definitions(gemmini-compile PRIVATE DEFAULT_MLIR_INSTALL="${LLVM_INSTALL_PREFIX}")
target_compile_definitions(gemmini-compile PRIVATE DEFAULT_COMPILER_SETTINGS="${CMAKE_CURRENT_SOURCE_DIR}/settings.json")
target_link_libraries(gemmini-compile PRIVATE
 MLIRIR MLIRParser MLIRSupport MLIRFuncDialect MLIRArithDialect MLIRTensorDialect
 MLIRLinalgDialect MLIRMathDialect MLIRAffineDialect MLIRSCFDialect
 MLIRLLVMDialect MLIRLLVMIRToLLVMTranslation MLIRTargetLLVMIRImport
 MLIRDLTIDialect MLIRControlFlowDialect
 LLVMIRReader LLVMCore LLVMSupport)
