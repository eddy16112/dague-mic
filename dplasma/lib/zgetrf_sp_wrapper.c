/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#include <plasma.h>
#include <dague.h>
#include <scheduling.h>
#include "dplasma.h"
#include "dplasmatypes.h"
#include "dplasmaaux.h"
#include "memory_pool.h"

#include "data_dist/sparse-matrix/sparse-shm-matrix.h"

#include "generated/zgetrf_sp.h"

dague_object_t* dplasma_zgetrf_sp_New(dague_tssm_desc_t *A, int ib, int *info)
{
    dague_zgetrf_sp_object_t *dague_getrf;

    dague_getrf = dague_zgetrf_sp_new( (dague_ddesc_t*)A, ib, info,
                                       A->super.m, A->super.n, A->super.mb, A->super.nb, 
                                       A->super.mt, A->super.nt);

    /* A */
    dplasma_add2arena_tile( dague_getrf->arenas[DAGUE_zgetrf_sp_DEFAULT_ARENA], 
                            A->super.mb*A->super.nb*sizeof(Dague_Complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->super.mb );
    
    return (dague_object_t*)dague_getrf;
}

void
dplasma_zgetrf_sp_Destruct( dague_object_t *o )
{
    dague_zgetrf_sp_destroy((dague_zgetrf_sp_object_t *)o);
}

int dplasma_zgetrf_sp( dague_context_t *dague, dague_tssm_desc_t *A, int ib ) 
{
    dague_object_t *dague_zgetrf = NULL;

    int info = 0;
    dague_zgetrf = dplasma_zgetrf_sp_New(A, ib, &info);

    dague_enqueue( dague, (dague_object_t*)dague_zgetrf);
    dague_progress(dague);
    
    dplasma_zgetrf_sp_Destruct( (dague_object_t*)dague_zgetrf );

    return info;
}

