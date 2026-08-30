add_mlir_dialect_library(MLIRMerlinIface
        MerlinIfaceDialect.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/MerlinIface

        DEPENDS
        MLIRMerlinIfaceOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        )
