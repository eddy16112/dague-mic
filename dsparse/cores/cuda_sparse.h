/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _cuda_sparse_h
#define _cuda_sparse_h

#include "dague_config.h"
#include "gpu_data.h"
#include "dague.h"
#include "execution_unit.h"
#include "scheduling.h"
#include "list.h"
#include "fifo.h"
#include "data_distribution.h"
#include "data_dist/matrix/matrix.h"

int sparse_register_bloktab( dague_context_t* dague_context, 
                             sparse_matrix_desc_t *tileA );
int sparse_unregister_bloktab( dague_context_t* dague_context, 
                               sparse_matrix_desc_t *tileA );

#endif
