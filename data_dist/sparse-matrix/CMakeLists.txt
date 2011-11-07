#
# Sources depending on Precision 
#

# Included only if( PASTIX_FOUND AND SCOTCH_FOUND)

include(RulesPrecisions)
set(generated_files "")

include_directories(".")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/pastix_internal)
include_directories(${PASTIX_INCLUDE_DIRS}) 
include_directories(${SCOTCH_INCLUDE_DIRS}) 

set( precision_files
  zread.c
  )
precisions_rules(generated_files
                 "${precision_files}"
                 PRECISIONS "+;s;d;c;z;")

set( dague_sparse_SRC
  sparse-matrix.c 
  ${generated_files}
  )

if (MPI_FOUND)
  add_library(dague_distribution_sparse_matrix-mpi STATIC
    ${dague_sparse_SRC})
  install(TARGETS dague_distribution_sparse_matrix-mpi ARCHIVE DESTINATION lib)
else()
  add_library(dague_distribution_sparse_matrix STATIC 
    ${dague_sparse_SRC})
  install(TARGETS dague_distribution_sparse_matrix ARCHIVE DESTINATION lib)
endif()
