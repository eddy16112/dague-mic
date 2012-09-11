/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> c
 *
 */
#include "dague.h"
#include <plasma.h>
#include "data_dist/sparse-matrix/pastix_internal/pastix_internal.h"
#include "dsparse.h"

#include "memory_pool.h"
#include "zsytrf_sp1dplus.h"

dague_object_t* 
dsparse_zsytrf_sp_New(sparse_matrix_desc_t *A)
{
    dague_zsytrf_sp1dplus_object_t *dague_zsytrf_sp = NULL;
 
    dague_zsytrf_sp = dague_zsytrf_sp1dplus_new(A, (dague_ddesc_t *)A, NULL, NULL );

    dague_zsytrf_sp->p_work1 = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( dague_zsytrf_sp->p_work1, (A->pastix_data->solvmatr).coefmax * sizeof(dague_complex64_t) );

    dague_zsytrf_sp->p_work2 = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( dague_zsytrf_sp->p_work2, (A->pastix_data->solvmatr).coefmax * sizeof(dague_complex64_t) );

    /* dsparse_add2arena_tile(((dague_zsytrf_Url_object_t*)dague_zsytrf)->arenas[DAGUE_zsytrf_Url_DEFAULT_ARENA],  */
    /*                        A->mb*A->nb*sizeof(dague_complex64_t), */
    /*                        DAGUE_ARENA_ALIGNMENT_SSE, */
    /*                        MPI_DOUBLE_COMPLEX, A->mb); */
    
    return (dague_object_t*)dague_zsytrf_sp;
}
 
void
dsparse_zsytrf_sp_Destruct( dague_object_t *o )
{
    dague_zsytrf_sp1dplus_object_t *dague_zsytrf_sp = NULL;
    dague_zsytrf_sp = (dague_zsytrf_sp1dplus_object_t *)o;

    /*dsparse_datatype_undefine_type( &(osytrf->arenas[DAGUE_zsytrf_Url_DEFAULT_ARENA]->opaque_dtt) );*/

    dague_private_memory_fini( dague_zsytrf_sp->p_work1 );
    free( dague_zsytrf_sp->p_work1 );
    dague_private_memory_fini( dague_zsytrf_sp->p_work2 );
    free( dague_zsytrf_sp->p_work2 );

    dague_zsytrf_sp1dplus_destroy( dague_zsytrf_sp );
}

int dsparse_zsytrf_sp( dague_context_t *dague, sparse_matrix_desc_t *A) 
{
    dague_object_t *dague_zsytrf_sp = NULL;
    int info = 0;

    dague_zsytrf_sp = dsparse_zsytrf_sp_New( A );

    if ( dague_zsytrf_sp != NULL )
    {
        dague_enqueue( dague, (dague_object_t*)dague_zsytrf_sp);
        dague_progress( dague );
        dsparse_zsytrf_sp_Destruct( dague_zsytrf_sp );
    }
    return info;
}

