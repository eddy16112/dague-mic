IF( NOT BUILD_DSPARSE )
  message(FATAL "The CMakeLists.txt in the dsparse directory is not supposed to be included if support for DSPARSE has been disabled")
ENDIF( NOT BUILD_DSPARSE )

if(NOT DSPARSE_PRECISIONS)
  set(DSPARSE_PRECISIONS "z;c;d;s" CACHE STRING "The precisions to compile in dsparse (accepts a colon separated list of z;c;d;s)" FORCE)
else()
  set(DSPARSE_PRECISIONS "${DSPARSE_PRECISIONS}" CACHE STRING "The precisions to compile in dsparse (accepts a colon separated list of s;d;c;z)" FORCE)
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
add_subdirectory(cores)
add_subdirectory(lib)

if( BUILD_TESTING )
  add_subdirectory(testing)
endif( BUILD_TESTING )

