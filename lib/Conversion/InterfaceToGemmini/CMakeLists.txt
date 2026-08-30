add_mlir_library(MLIRInterfaceToGemmini
        InterfaceToGemmini.cpp

        DEPENDS
        MLIRMerlinIfaceOpsIncGen
        MLIRGemminiOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRFuncDialect
        MLIRMerlinIface
        MLIRGemminiDialect
        )
