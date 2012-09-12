add_subdirectory(matrix)

if( BUILD_DSPARSE )
  add_subdirectory(sparse-matrix)
endif()
