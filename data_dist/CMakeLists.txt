add_subdirectory(matrix)

if( PASTIX_FOUND AND SCOTCH_FOUND)
  add_subdirectory(sparse-matrix)
endif()
