include(RulesPrecisions)
# reset variables
set(generated_files "")
set(generated_headers "")

### generate the dplasma_cores headers for all possible precisions
#precisions_rules_py(generated_headers
#                 "dplasma_zcores.h"
#                 PRECISIONS "s;d;c;z")
### Generate the dplasma wrappers for all required precisions 

set(CUDA_SOURCES
    zgemm_fermi.cu
)
precisions_rules_py(cuda_generated_files
                    "${CUDA_SOURCES}"
                    PRECISIONS "${DPLASMA_PRECISIONS}")

set( CUDA_SOURCES
     ${CUDA_SOURCES}
     cuda_sparse_sgemm_kernel_bloktab.cu
)

set(CUDA_NVCC_FLAGS_11 -maxrregcount 32 -arch sm_11 -DCUDA_SM_VERSION=11 ${CUDA_NVCC_FLAGS_BACKUP})
set(CUDA_NVCC_FLAGS_13 -maxrregcount 32 -arch sm_13 -DCUDA_SM_VERSION=13 ${CUDA_NVCC_FLAGS_BACKUP})
set(CUDA_NVCC_FLAGS_20 -arch sm_20 -DCUDA_SM_VERSION=20 ${CUDA_NVCC_FLAGS_BACKUP})

### Generate the CUDA kernels if necessary
if(CUDA_FOUND)
    set(CUDA_BUILD_CUBIN ON)
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    set(CUDA_NVCC_FLAGS_BACKUP ${CUDA_NVCC_FLAGS})

    foreach( _cudafile ${cuda_generated_files} )
        string(REGEX REPLACE "\\.cu" "" _cudafilename ${_cudafile})

        set(SM11_FILES ${SM11_FILES} ${_cudafile})
        set(SM13_FILES ${SM11_FILES} ${_cudafile})
        set(SM20_FILES ${SM11_FILES} ${_cudafile})

    endforeach()

#     set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS_11})
#     cuda_add_library(dsparse-sm_11 ${SM11_FILES})
#     install(TARGETS dsparse-sm_11 ARCHIVE DESTINATION lib)

#     set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS_13})
#     cuda_add_library(dsparse-sm_13 ${SM13_FILES})
#     install(TARGETS dsparse-sm_13 ARCHIVE DESTINATION lib)

    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS_20})
    cuda_add_library(dsparse-sm_20 ${SM20_FILES})
    install(TARGETS dsparse-sm_20 ARCHIVE DESTINATION lib)


    # Install cubin files
    foreach( _cudafile ${cuda_generated_files} )
        string(REGEX REPLACE "\\.cu" "" _cudafilename ${_cudafile})

#         install(FILES ${CMAKE_CURRENT_BINARY_DIR}/dsparse-sm_11_generated_${_cudafile}.o.cubin.txt
#                 DESTINATION lib
#                 RENAME dsparse-sm_11_${_cudafilename}.cubin )

#         install(FILES ${CMAKE_CURRENT_BINARY_DIR}/dsparse-sm_13_generated_${_cudafile}.o.cubin.txt
#                 DESTINATION lib
#                 RENAME dsparse-sm_13_${_cudafilename}.cubin )

        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/dsparse-sm_20_generated_${_cudafile}.o.cubin.txt
                DESTINATION lib
                RENAME dsparse-sm_20_${_cudafilename}.cubin )
    endforeach()

#    cuda_build_clean_target()

    precisions_rules_py(gpu_generated_headers
                        cuda_zpotrf1d_gemm.h
                        PRECISIONS "${DPLASMA_PRECISIONS}")

    precisions_rules_py(gpu_generated_sources
                        cuda_zpotrf1d_gemm.c
                        PRECISIONS "${DPLASMA_PRECISIONS}")

    set(GPU_KERNEL_SOURCES 
             cuda_sparse.h
             ${gpu_generated_headers}
             cuda_sparse.c
             ${gpu_generated_sources})

    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS_BACKUP})
else()
    set(GPU_KERNEL_SOURCES "")
endif()

### Generate the lib 
add_library(dsparse_cores 
  ${generated_headers}
  ${generated_files}
  ${GPU_KERNEL_SOURCES})
install(TARGETS dsparse_cores ARCHIVE DESTINATION lib)

foreach(generated_header ${generated_headers})
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${generated_header} DESTINATION include/cores)
endforeach()
