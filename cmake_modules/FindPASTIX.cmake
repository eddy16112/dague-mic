# - Find PASTIX library
# This module finds an installed  library that implements the PASTIX
# linear-algebra interface (see http://icl.cs.utk.edu/plasma/).
# The list of libraries searched for is taken
# from the autoconf macro file, acx_blas.m4 (distributed at
# http://ac-archive.sourceforge.net/ac-archive/acx_blas.html).
#
# This module sets the following variables:
#  PASTIX_FOUND - set to true if a library implementing the PASTIX interface
#    is found
#  PASTIX_PKG_DIR - Directory where the PASTIX pkg file is stored
#  PASTIX_LIBRARIES - uncached list of libraries (using full path name) to
#    link against to use PASTIX
#  PASTIX_INCLUDE_DIRS - Directory where the PASTIX include files are located
#  PASTIX_LINKER_FLAGS - uncached list of required linker flags (excluding -l
#    and -L).
#  PASTIX_STATIC  if set on this determines what kind of linkage we do (static)
#  PASTIX_VENDOR  if set checks only the specified vendor, if not set checks
#     all the possibilities
##########

get_property(_LANGUAGES_ GLOBAL PROPERTY ENABLED_LANGUAGES)
if(NOT _LANGUAGES_ MATCHES Fortran)
  if(PASTIX_FIND_REQUIRED)
    message(FATAL_ERROR "Find PASTIX requires Fortran support so Fortran must be enabled.")
  else(PASTIX_FIND_REQUIRED)
    message(STATUS "Looking for PASTIX... - NOT found (Fortran not enabled)") #
    return()
  endif(PASTIX_FIND_REQUIRED)
endif(NOT _LANGUAGES_ MATCHES Fortran)

if(NOT PASTIX_DIR)
  set(PASTIX_FOUND FALSE)
else(NOT PASTIX_DIR)
  set(PASTIX_FOUND TRUE)
  set(PASTIX_INCLUDE_DIRS "${PASTIX_DIR}")
  set(PASTIX_LIBRARIES pastix matrix_driver)
endif(NOT PASTIX_DIR)

if ( PASTIX_FOUND )
  message(STATUS "Looking for PaStiX - found")
else( PASTIX_FOUND )
  if ( PASTIX_FIND_REQUIRED )
    message(FATAL_ERROR "Looking for Pastix - not found")
  else( PASTIX_FIND_REQUIRED )
    message(STATUS "Looking for Pastix - not found")
  endif( PASTIX_FIND_REQUIRED )
endif( PASTIX_FOUND )

mark_as_advanced(PASTIX_LIBRARIES PASTIX_INCLUDE_DIRS)
set(PASTIX_DIR          "${PASTIX_DIR}"          CACHE PATH   "Location of the PASTIX library" FORCE)
set(PASTIX_INCLUDE_DIRS "${PASTIX_INCLUDE_DIRS}" CACHE PATH   "PASTIX include directories" FORCE)
set(PASTIX_LIBRARIES    "${PASTIX_LIBRARIES}"    CACHE STRING "libraries to link with PASTIX" FORCE)

