
#
# Sources depending on Precision 
#
include(RulesPrecisions)
set(generated_files "")
include_directories(".")

set( precision_files
)
precisions_rules(generated_files "+;s;d;c;z" "${precision_files}")

if (!MPI_FOUND)
    add_library(dague_distribution_sparse_matrix STATIC 
        sparse-shm-matrix.c 
        ${generated_files})
    install(TARGETS dague_distribution_sparse_matrix ARCHIVE DESTINATION lib)
endif()

