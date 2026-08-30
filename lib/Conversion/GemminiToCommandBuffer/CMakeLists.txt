add_mlir_library(MLIRGemminiToCommandBuffer
        GemminiToCommandBuffer.cpp

        DEPENDS
        MLIRGemminiOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRFuncDialect
        MLIRGemminiDialect
        )
