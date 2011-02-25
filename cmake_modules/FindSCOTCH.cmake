# - Find SCOTCH library
# This module finds an installed  library that implements the SCOTCH
# linear-algebra interface (see http://icl.cs.utk.edu/plasma/).
# The list of libraries searched for is taken
# from the autoconf macro file, acx_blas.m4 (distributed at
# http://ac-archive.sourceforge.net/ac-archive/acx_blas.html).
#
# This module sets the following variables:
#  SCOTCH_FOUND - set to true if a library implementing the SCOTCH interface
#    is found
#  SCOTCH_PKG_DIR - Directory where the SCOTCH pkg file is stored
#  SCOTCH_LIBRARIES - uncached list of libraries (using full path name) to
#    link against to use SCOTCH
#  SCOTCH_INCLUDE_DIRS - Directory where the SCOTCH include files are located
#  SCOTCH_LINKER_FLAGS - uncached list of required linker flags (excluding -l
#    and -L).
#  SCOTCH_STATIC  if set on this determines what kind of linkage we do (static)
#  SCOTCH_VENDOR  if set checks only the specified vendor, if not set checks
#     all the possibilities
##########

get_property(_LANGUAGES_ GLOBAL PROPERTY ENABLED_LANGUAGES)
if(NOT _LANGUAGES_ MATCHES Fortran)
  if(SCOTCH_FIND_REQUIRED)
    message(FATAL_ERROR "Find SCOTCH requires Fortran support so Fortran must be enabled.")
  else(SCOTCH_FIND_REQUIRED)
    message(STATUS "Looking for SCOTCH... - NOT found (Fortran not enabled)") #
    return()
  endif(SCOTCH_FIND_REQUIRED)
endif(NOT _LANGUAGES_ MATCHES Fortran)

if(NOT SCOTCH_DIR)
  set(SCOTCH_FOUND FALSE)
else(NOT SCOTCH_DIR)
  set(SCOTCH_FOUND TRUE)
  set(SCOTCH_INCLUDE_DIRS "${SCOTCH_DIR}/include")
  set(SCOTCH_LIBRARY_DIRS "${SCOTCH_DIR}/lib")
  set(SCOTCH_LIBRARIES    "-lscotch -lscotcherrexit")
endif(NOT SCOTCH_DIR)

if ( SCOTCH_FOUND )
  message(STATUS "Looking for Scotch - found")
else( SCOTCH_FOUND )
  if ( SCOTCH_FIND_REQUIRED )
    message(FATAL_ERROR "Looking for Scotch - not found")
  else( SCOTCH_FIND_REQUIRED )
    message(STATUS "Looking for Scotch - not found")
  endif( SCOTCH_FIND_REQUIRED )
endif( SCOTCH_FOUND )

mark_as_advanced(SCOTCH_LIBRARIES SCOTCH_INCLUDE_DIRS)
set(SCOTCH_DIR          "${SCOTCH_DIR}"          CACHE PATH   "Location of the SCOTCH library" FORCE)
set(SCOTCH_INCLUDE_DIRS "${SCOTCH_INCLUDE_DIRS}" CACHE PATH   "SCOTCH include directories" FORCE)
set(SCOTCH_LIBRARY_DIRS "${SCOTCH_LIBRARY_DIRS}" CACHE PATH   "SCOTCH libraries directory" FORCE)
set(SCOTCH_LIBRARIES    "${SCOTCH_LIBRARIES}"    CACHE STRING "libraries to link with SCOTCH" FORCE)

