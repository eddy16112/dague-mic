include(RulesPrecisions)
include(RulesJDF)
# reset variables
set(generated_files "")
set(generated_jdf "")
set(generated_headers "")

set(EXTRA_SOURCES 
)

set(HEADERS
)

# precisions_rules_py(generated_headers
#                  "${HEADERS}"
#                  PRECISIONS "z;c;d;s")

### Generate .c files from .jdf for all required precisions
set(JDF
  zcsc2cblk.jdf
  zpotrf_sp1dplus.jdf
  zgetrf_sp1dplus.jdf
  zhetrf_sp1dplus.jdf
  zsytrf_sp1dplus.jdf
  # Solvers
  zpotrs_sp1dplus.jdf
)
precisions_rules_py(generated_jdf
                 "${JDF}"
                 PRECISIONS "${DSPARSE_PRECISIONS}")

jdf_rules(generated_files "${generated_jdf}")

### Generate the dsparse wrappers for all required precisions 
set(SOURCES
  zcsc2cblk_wrapper.c
  zsytrf_sp1dplus_wrapper.c
  zhetrf_sp1dplus_wrapper.c
  zpotrf_sp1dplus_wrapper.c
  zgetrf_sp1dplus_wrapper.c
  zpotrs_sp1dplus_wrapper.c
)

precisions_rules_py(generated_files
                 "${SOURCES}"
                 PRECISIONS "${DSPARSE_PRECISIONS}")

### Generate the lib 
if (MPI_FOUND)
  add_library(dsparse-mpi
    ${generated_files} 
    ${EXTRA_SOURCES})
  set_target_properties(dsparse-mpi PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
  add_dependencies(dsparse-mpi dsparse_includes)
  install(TARGETS dsparse-mpi ARCHIVE DESTINATION lib)
else (MPI_FOUND)
  add_library(dsparse 
    ${generated_headers}
    ${generated_files}
    ${EXTRA_SOURCES})
  add_dependencies(dsparse dsparse_includes)
  install(TARGETS dsparse ARCHIVE DESTINATION lib)
endif (MPI_FOUND)


