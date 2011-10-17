if(MPI_FOUND)
  add_library(common-mpi common.c common_timing.h flops.h)
  set_target_properties(common-mpi PROPERTIES COMPILE_FLAGS
      "${MPI_COMPILE_FLAGS}")
else()
  add_library(common common.c common_timing.h flops.h)
endif()
  
include(RulesTestingsSP.cmake)

set(TESTS
  testing_zpotrf_sp.c
)
testingssp_addexec(targets "+;${DSPARSE_PRECISIONS}" "${TESTS}")
