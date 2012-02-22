if ( DAGUE_WITH_SPARSE )

  if(NOT DSPARSE_PRECISIONS)
    set(DSPARSE_PRECISIONS "z;c;d;s" CACHE STRING "The precisions to compile in dsparse (accepts a colon separated list of z;c;d;s)" FORCE)
  endif()

  include_directories(${PLASMA_INCLUDE_DIRS})
  include_directories(${PASTIX_INCLUDE_DIRS})
  include_directories(${SCOTCH_INCLUDE_DIRS})
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

  link_directories(${PASTIX_LIBRARY_DIRS})
  link_directories(${SCOTCH_LIBRARY_DIRS})

  if( ${CMAKE_CURRENT_BINARY_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR} )
  else( ${CMAKE_CURRENT_BINARY_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR} )
    include_directories(${CMAKE_CURRENT_BINARY_DIR})
    include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
  endif( ${CMAKE_CURRENT_BINARY_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR} )
  
  add_subdirectory(include)
  if(CUDA_FOUND)
        add_subdirectory(cores)
  endif()
  add_subdirectory(lib)
  add_subdirectory(testing)

endif()
