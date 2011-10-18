include(RulesPrecisions)
include(RulesJDF)
# reset variables
set(generated_files "")
set(generated_jdf "")

set(EXTRA_SOURCES 
)

### Generate .c files from .jdf for all required precisions
set(JDF
  zpotrf_sp1dplus.jdf
)
precisions_rules(generated_jdf
                 "${JDF}"
                 PRECISIONS "+;${DSPARSE_PRECISIONS}")

jdf_rules(generated_files "${generated_jdf}")

### Generate the dplasma wrappers for all required precisions 
set(SOURCES
#  zpotrf_sp1dplus_wrapper.c
)
precisions_rules(generated_files
                 "${SOURCES}"
                 PRECISIONS "+;${DSPARSE_PRECISIONS}")

### Generate the lib 
if (MPI_FOUND)
  add_library(dsparse-mpi
    ${generated_files} 
    ${EXTRA_SOURCES})
  set_target_properties(dsparse-mpi PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
#  add_dependencies(dsparse-mpi dsparse_includes)
  install(TARGETS dsparse-mpi ARCHIVE DESTINATION lib)
else (MPI_FOUND)
  add_library(dsparse 
    ${generated_files}
    ${EXTRA_SOURCES})
#  add_dependencies(dsparse dsparse_includes)
  install(TARGETS dsparse ARCHIVE DESTINATION lib)
endif (MPI_FOUND)


