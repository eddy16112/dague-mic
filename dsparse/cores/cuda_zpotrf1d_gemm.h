/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> z c d s
 *
 */

#ifndef _cuda_zpotrfsp1d_gemm_h
#define _cuda_zpotrfsp1d_gemm_h

#include "dague_config.h"
#include "gpu_data.h"
#include "dague.h"
#include "execution_unit.h"
#include "scheduling.h"
#include "list.h"
#include "fifo.h"
#include "data_distribution.h"
#include "data_dist/matrix/matrix.h"

#define KERNEL_KEY( _cblk_ ) (uint32_t)(NULL == dague_gpu_map.desc ?    \
                                        0 : (_cblk_))

typedef int my_tmp_int_t;

int sparse_register_bloktab( dague_context_t* dague_context, 
                             sparse_matrix_desc_t *tileA );
int sparse_unregister_bloktab( dague_context_t* dague_context, 
                               sparse_matrix_desc_t *tileA );

int gpu_kernel_init_zpotrfsp1d( dague_context_t* dague_context, 
                                sparse_matrix_desc_t *tileA );

int gpu_zpotrfsp1d( dague_execution_unit_t* eu_context,
                    dague_execution_context_t* this_task );


void
magmablas_zgemm_SM20( char TRANSA, char TRANSB, int m , int n , int k ,
                      cuDoubleComplex alpha, cuDoubleComplex *d_A, int lda,
                                             cuDoubleComplex *d_B, int ldb,
                      cuDoubleComplex beta,  cuDoubleComplex *d_C, int ldc,
                      int blocknbr, const int *blocktab, int fblocknbr, const int *fblocktab,
                      CUstream stream );
#endif
