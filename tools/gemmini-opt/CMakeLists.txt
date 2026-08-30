set(LIBS
    MLIRIR
    MLIRParser
    MLIRSupport
    MLIRFuncDialect
    MLIRLLVMDialect
    MLIRMerlinIface
    MLIRGemminiDialect
    MLIRInterfaceToGemmini
    MLIRGemminiToCommandBuffer
    MLIRGemminiToLLVM
  )
add_llvm_executable(gemmini-opt gemmini-opt.cpp)

llvm_update_compile_flags(gemmini-opt)
target_link_libraries(gemmini-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(gemmini-opt)
