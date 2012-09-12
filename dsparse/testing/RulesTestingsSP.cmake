include(RulesPrecisions)

macro(testingssp_addexec OUTPUTLIST PRECISIONS ZSOURCES)
  include_directories(.)

  set(testingssp_addexec_CFLAGS "-DADD_")
  foreach(arg ${PLASMA_CFLAGS})
    set(testingssp_addexec_CFLAGS "${testingssp_addexec_CFLAGS} ${arg}")
  endforeach(arg ${PLASMA_CFLAGS})

  set(testingssp_addexec_LDFLAGS "${LOCAL_FORTRAN_LINK_FLAGS} -L${PASTIX_DIR} -L${SCOTCH_DIR}/lib")
  set(testingssp_addexec_LIBS    "${EXTRA_LIBS}")
  list(APPEND testingssp_addexec_LIBS "${PASTIX_LIBRARIES}")
  list(APPEND testingssp_addexec_LIBS "${SCOTCH_LIBRARIES}")
  # Set flags for compilation
  if( MPI_FOUND )
#    set(testingssp_addexec_CFLAGS  "${MPI_COMPILE_FLAGS} ${testingssp_addexec_CFLAGS} -DUSE_MPI")
#    set(testingssp_addexec_LDFLAGS "${MPI_LINK_FLAGS} ${testingssp_addexec_LDFLAGS}")
#    set(testingssp_addexec_LIBS   
#      commonsp-mpi dsparse-mpi dague-mpi dague_distribution_sparse_matrix-mpi 
#      ${testingssp_addexec_LIBS} ${MPI_LIBRARIES} 
#      )
  else ( MPI_FOUND )
    set(testingssp_addexec_LIBS   
      commonsp dsparse dsparse_cores dague dague_distribution_sparse_matrix
      ${testingssp_addexec_LIBS}
      )
  endif()

  set(testingssp_addexec_GENFILES "")
  precisions_rules_py(testingssp_addexec_GENFILES 
    "${ZSOURCES}"
    PRECISIONS "${PRECISIONS}")
  foreach(testingssp_addexec_GENFILE ${testingssp_addexec_GENFILES})
    string(REGEX REPLACE "\\.[scdz]" "" testingssp_addexec_EXEC ${testingssp_addexec_GENFILE})

    add_executable(${testingssp_addexec_EXEC} ${testingssp_addexec_GENFILE})
    set_target_properties(${testingssp_addexec_EXEC} PROPERTIES
                            LINKER_LANGUAGE Fortran
                            COMPILE_FLAGS "${testingssp_addexec_CFLAGS}"
                            LINK_FLAGS "${testingssp_addexec_LDFLAGS}")
    target_link_libraries(${testingssp_addexec_EXEC} ${testingssp_addexec_LIBS} ${PLASMA_LDFLAGS} ${PLASMA_LIBRARIES})
    install(TARGETS ${testingssp_addexec_EXEC} RUNTIME DESTINATION bin)
    list(APPEND ${OUTPUTLIST} ${testingssp_addexec_EXEC})
  endforeach()

endmacro(testingssp_addexec)

