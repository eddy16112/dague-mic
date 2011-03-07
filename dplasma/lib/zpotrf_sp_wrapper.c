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

#include "data_dist/sparse-matrix/sparse-shm-matrix.h"

#include "generated/zpotrf_sp.h"

dague_object_t* 
dplasma_zpotrf_sp_New(dague_tssm_desc_t *A, int *info)
{
    dague_object_t *dague_zpotrf = NULL;
    int pri_change = dplasma_aux_get_priority( "POTRF", (tiled_matrix_desc_t *)A );
 
    *info = 0;
    dague_zpotrf = (dague_object_t*)dague_zpotrf_sp_new(
        (dague_ddesc_t*)A, 
        pri_change, info, 
        A->super.m, A->super.n, A->super.mb, A->super.nb, A->super.mt, A->super.nt);
    
    dplasma_add2arena_tile(
        ((dague_zpotrf_sp_object_t*)dague_zpotrf)->arenas[DAGUE_zpotrf_sp_DEFAULT_ARENA], 
        A->super.mb*A->super.nb*sizeof(Dague_Complex64_t),
        DAGUE_ARENA_ALIGNMENT_SSE,
        MPI_DOUBLE_COMPLEX, A->super.mb);
    
    return dague_zpotrf;
}
 
void
dplasma_zpotrf_sp_Destruct( dague_object_t *o )
{
    dague_zpotrf_sp_destroy((dague_zpotrf_sp_object_t *)o);
}

int dplasma_zpotrf_sp( dague_context_t *dague, dague_tssm_desc_t* ddescA) 
{
    dague_object_t *dague_zpotrf = NULL;
    int info = 0;

    dague_zpotrf = dplasma_zpotrf_sp_New(ddescA, &info);
    dague_enqueue( dague, (dague_object_t*)dague_zpotrf);
    dague_progress(dague);
    dplasma_zpotrf_sp_Destruct( dague_zpotrf );

    return info;
}
