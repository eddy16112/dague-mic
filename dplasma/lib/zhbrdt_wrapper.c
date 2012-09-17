/*
 * Copyright (c) 2011-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "dague_internal.h"
#include <plasma.h>
#include "data_dist/matrix/matrix.h"
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"
#include "dplasma/lib/memory_pool.h"

#include "zhbrdt.h"

dague_object_t* dplasma_zhbrdt_New(tiled_matrix_desc_t* A /* data A */)
{
    dague_zhbrdt_object_t *dague_zhbrdt = NULL;

    dague_zhbrdt = dague_zhbrdt_new(A, A->mb-1);

    dplasma_add2arena_rectangle( dague_zhbrdt->arenas[DAGUE_zhbrdt_DEFAULT_ARENA],
                                 (A->nb)*(A->mb)*sizeof(dague_complex64_t), 16,
                                 MPI_DOUBLE_COMPLEX, 
                                 A->mb, A->nb, -1 );
    return (dague_object_t*)dague_zhbrdt;
}

void dplasma_zhbrdt_Destruct( dague_object_t* o )
{
    DAGUE_INTERNAL_OBJECT_DESTRUCT(o);
}

