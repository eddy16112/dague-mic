add_subdirectory(matrix)

if( DAGUE_WITH_SPARSE )
  add_subdirectory(sparse-matrix)
endif()
