add_mlir_library(MLIRGemminiToLLVM
        GemminiToLLVM.cpp

        DEPENDS
        MLIRGemminiOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRFuncDialect
        MLIRLLVMDialect
        MLIRGemminiDialect
        )
