add_mlir_dialect_library(MLIRGemminiDialect
        GemminiDialect.cpp
        GemminiOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Gemmini

        DEPENDS
        MLIRGemminiOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        )
