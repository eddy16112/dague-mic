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

#define KERNEL_KEY( _desc_, _cblk_ ) (uint32_t)(NULL == (_desc_) ? 0 : (_cblk_))

typedef int my_tmp_int_t;

int sparse_register_bloktab( dague_context_t* dague_context, 
                             sparse_matrix_desc_t *tileA );
int sparse_unregister_bloktab( dague_context_t* dague_context, 
                               sparse_matrix_desc_t *tileA );

int gpu_kernel_init_zpotrfsp_gemm( dague_context_t* dague_context );

int gpu_zpotrfsp_gemm( dague_execution_unit_t* eu_context,
                       dague_execution_context_t* this_task,
                       int pushout, 
                       my_tmp_int_t cblknum,
                       my_tmp_int_t bloknum,
                       my_tmp_int_t fcblknum,
                       const sparse_matrix_desc_t *ddesc  );

#define symbol_get_cblk_bloknbr( _datacode_, _cblknum_ ) (SYMB_BLOKNUM((_cblknum_)+1)-SYMB_BLOKNUM(_cblknum_))
#define symbol_get_cblk_stride(  _datacode_, _cblknum_ ) (SOLV_STRIDE(_cblknum_))
#define symbol_get_cblk_width(   _datacode_, _cblknum_ ) (SYMB_LCOLNUM(_cblknum_)-SYMB_FCOLNUM(_cblknum_)+1)
#define symbol_get_cblk_fcolnum( _datacode_, _cblknum_ ) (SYMB_FCOLNUM(_cblknum_))
#define symbol_get_blok_coefind( _datacode_, _bloknum_ ) (SOLV_COEFIND(_bloknum_))
#define symbol_get_blok_frownum( _datacode_, _bloknum_ ) (SYMB_FROWNUM(_bloknum_))
#define symbol_get_blok_height(  _datacode_, _bloknum_ ) (SYMB_LROWNUM(_bloknum_)-SYMB_FROWNUM(_bloknum_)+1)

#endif
