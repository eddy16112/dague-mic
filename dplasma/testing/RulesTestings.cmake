include(RulesPrecisions)

macro(testings_addexec OUTPUTLIST PRECISIONS ZSOURCES)
  include_directories(.)

  set(testings_addexec_CFLAGS "-DADD_")
  foreach(arg ${PLASMA_CFLAGS})
    set(testings_addexec_CFLAGS "${testings_addexec_CFLAGS} ${arg}")
  endforeach(arg ${PLASMA_CFLAGS})

  set(testings_addexec_LDFLAGS "${LOCAL_FORTRAN_LINK_FLAGS}")
  set(testings_addexec_LIBS    "${EXTRA_LIBS}")
  # Set flags for compilation
  if( MPI_FOUND )
    set(testings_addexec_CFLAGS  "${MPI_COMPILE_FLAGS} ${testings_addexec_CFLAGS} -DUSE_MPI")
    set(testings_addexec_LDFLAGS "${MPI_LINK_FLAGS} ${testings_addexec_LDFLAGS}")
    set(testings_addexec_LIBS   
      common-mpi dplasma-mpi dague-mpi  dague_distribution_matrix-mpi 
      ${testings_addexec_LIBS} ${MPI_LIBRARIES} 
      )
  else ( MPI_FOUND )
    set(testings_addexec_LIBS   
      common dplasma dague dague_distribution_matrix 
      ${testings_addexec_LIBS}
      )
  endif()

  if( DAGUE_SPARSE )
    if( MPI_FOUND )
      set(testings_addexec_LDFLAGS "${testings_addexec_LDFLAGS} -L${PASTIX_INCLUDE_DIRS} -L${SCOTCH_LIBRARY_DIRS}")
      set(testings_addexec_LIBS   
        ${testings_addexec_LIBS} dague_distribution_sparse_matrix-mpi ${PASTIX_LIBRARIES} ${SCOTCH_LIBRARIES}
        )
    else( MPI_FOUND )
      set(testings_addexec_LDFLAGS "${testings_addexec_LDFLAGS} -L${PASTIX_INCLUDE_DIRS} -L${SCOTCH_LIBRARY_DIRS}")
      set(testings_addexec_LIBS   
        ${testings_addexec_LIBS} dague_distribution_sparse_matrix ${PASTIX_LIBRARIES} ${SCOTCH_LIBRARIES}
        )
    endif( MPI_FOUND )
  endif( DAGUE_SPARSE )

  set(testings_addexec_GENFILES "")
  precisions_rules(testings_addexec_GENFILES "${PRECISIONS}" "${ZSOURCES}")
  foreach(testings_addexec_GENFILE ${testings_addexec_GENFILES})
    string(REGEX REPLACE "\\.[scdz]" "" testings_addexec_EXEC ${testings_addexec_GENFILE})
    string(REGEX REPLACE "generated/" "" testings_addexec_EXEC ${testings_addexec_EXEC})

    add_executable(${testings_addexec_EXEC} ${testings_addexec_GENFILE})
    set_target_properties(${testings_addexec_EXEC} PROPERTIES
                            LINKER_LANGUAGE Fortran
                            COMPILE_FLAGS "${testings_addexec_CFLAGS}"
                            LINK_FLAGS "${testings_addexec_LDFLAGS}")
    target_link_libraries(${testings_addexec_EXEC} ${testings_addexec_LIBS} ${PLASMA_LDFLAGS} ${PLASMA_LIBRARIES})
    install(TARGETS ${testings_addexec_EXEC} RUNTIME DESTINATION bin)
    list(APPEND ${OUTPUTLIST} ${testings_addexec_EXEC})
  endforeach()

endmacro(testings_addexec)

