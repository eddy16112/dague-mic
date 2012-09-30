include(RulesPrecisions)
# reset variables
set(generated_files "")
set(generated_headers "")

include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR})

### Generate the CUDA kernels if necessary
if(CUDA_FOUND)
    set(CUDA_SM_TARGETS 20)
    set(CUDA_NVCC_FLAGS_11 -maxrregcount 32 -arch sm_11 -DCUDA_SM_VERSION=11)
    set(CUDA_NVCC_FLAGS_13 -maxrregcount 32 -arch sm_13 -DCUDA_SM_VERSION=13)
    set(CUDA_NVCC_FLAGS_20                  -arch sm_20 -DCUDA_SM_VERSION=20)

### CUDA .cu sources
    set( CUDA11_SOURCES
         zgemm_sparse_11_12_13.cu
    )
    set( CUDA13_SOURCES 
         ${CUDA11_SOURCES} )
    set( CUDA20_SOURCES 
         zgemm_sparse_20_30.cu
    )

    foreach( _smtarget ${CUDA_SM_TARGETS} )
        set(cuda_generated_files "")
        precisions_rules_py(cuda_generated_files
                            "${CUDA${_smtarget}_SOURCES}" 
                            PRECISIONS "${DSPARSE_PRECISIONS}")
        foreach( _cudafile ${cuda_generated_files} )
            get_source_file_property(_IsInBinaryDir ${_cudafile} IS_IN_BINARY_DIR )
            if( _IsInBinaryDir )
                string(REGEX REPLACE 
                        ${_cudafile}
                        ${CMAKE_CURRENT_BINARY_DIR}/${_cudafile}
                        cuda_generated_files "${cuda_generated_files}")
            endif( _IsInBinaryDir )
        endforeach()

        set(CUDA_NVCC_FLAGS_BACKUP ${CUDA_NVCC_FLAGS})
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS_${_smtarget}} ${CUDA_NVCC_FLAGS})
        cuda_add_library(dsparse_cucores_sm${_smtarget} ${cuda_generated_files} )
        target_link_libraries( dsparse_cucores_sm${_smtarget}
                                 ${CUDA_cublas_LIBRARY}
                                 ${CUDA_CUDART_LIBRARY}
                                 ${CUDA_CUDA_LIBRARY} )
        install(TARGETS dsparse_cucores_sm${_smtarget} LIBRARY DESTINATION lib ARCHIVE DESTINATION lib )
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS_BACKUP})
        if( NOT CMAKE_BUILD_SHARED_LIBS )
            list(APPEND GPU_KERNEL_LIBS dsparse_cucores_sm${_smtarget})
        endif()
    endforeach()

    #cuda_build_clean_target()

### Now the gpu wrappers to the .cu files
    set(GPU_KERNEL_SOURCES_GEN
         cuda_zpotrfsp_gemm.c
         cuda_zgetrfsp_gemm.c
#        cuda_zgemmsp.c
    )
    set(GPU_KERNEL_HEADERS_GEN
         cuda_zpotrfsp_gemm.h
         cuda_zgetrfsp_gemm.h
#        cuda_zgemmsp.h
    )

    set(GPU_KERNEL_SOURCES
         cuda_sparse.c
    )
else()
    set(GPU_KERNEL_SOURCES_GEN "")
    set(GPU_KERNEL_HEADERS_GEN "")
    set(GPU_KERNEL_SOURCES     "")
    set(GPU_KERNEL_LIBS        "")
endif()

set(HEADERS
    dsparse_zcores.h
    ${GPU_KERNEL_HEADERS_GEN}
)
### generate the dsparse_cores headers for all possible precisions
precisions_rules_py(generated_headers
                    "${HEADERS}"
                    PRECISIONS "s;d;c;z")

### Generate the dsparse wrappers for all required precisions 
set(SOURCES
  # extra kernel
  core_zgemdm.c
  core_zgetro.c
  # Kernels
  core_zsytrfsp.c
  core_zhetrfsp.c
  core_zgetrfsp.c
  core_zpotrfsp.c
  ${GPU_KERNEL_SOURCES_GEN}
)
precisions_rules_py(generated_files
                 "${SOURCES}"
                 PRECISIONS "${DSPARSE_PRECISIONS}")

### Generate the lib
add_library(dsparse_cores
  ${generated_headers}
  ${generated_files}
  ${GPU_KERNEL_SOURCES}
)
target_link_libraries(dsparse_cores ${GPU_KERNEL_LIBS})
install(TARGETS dsparse_cores ARCHIVE DESTINATION lib)

foreach(generated_header ${generated_headers})
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${generated_header} DESTINATION include/cores)
endforeach()
