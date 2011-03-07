
#
# Sources depending on Precision 
#
if( PASTIX_FOUND AND SCOTCH_FOUND)

include(RulesPrecisions)
set(generated_files "")
include_directories(".")

set( precision_files
  zread.c
)
precisions_rules(generated_files "+;s;d;c;z" "${precision_files}")

set( dague_sparse_SRC
  sparse-shm-matrix.c 
  si-to-tssm.c
  ${generated_files}
  )

include_directories(${PASTIX_INCLUDE_DIRS}) 
include_directories(${SCOTCH_INCLUDE_DIRS}) 

if (MPI_FOUND)
    add_library(dague_distribution_sparse_matrix-mpi STATIC
        ${dague_sparse_SRC})
    install(TARGETS dague_distribution_sparse_matrix-mpi ARCHIVE DESTINATION lib)
else()
    add_library(dague_distribution_sparse_matrix STATIC 
        ${dague_sparse_SRC})
    install(TARGETS dague_distribution_sparse_matrix ARCHIVE DESTINATION lib)
endif()

endif( PASTIX_FOUND AND SCOTCH_FOUND)
