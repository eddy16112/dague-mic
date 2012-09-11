/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#include "dague.h"
#include <plasma.h>
#include "data_dist/sparse-matrix/pastix_internal/pastix_internal.h"
#include "dsparse.h"

#include "memory_pool.h"
#include "zhetrf_sp1dplus.h"

dague_object_t* 
dsparse_zhetrf_sp_New(sparse_matrix_desc_t *A)
{
    dague_zhetrf_sp1dplus_object_t *dague_zhetrf_sp = NULL;
 
    dague_zhetrf_sp = dague_zhetrf_sp1dplus_new(A, (dague_ddesc_t *)A, NULL, NULL );

    dague_zhetrf_sp->p_work1 = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( dague_zhetrf_sp->p_work1, (A->pastix_data->solvmatr).coefmax * sizeof(dague_complex64_t) );

    dague_zhetrf_sp->p_work2 = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( dague_zhetrf_sp->p_work2, (A->pastix_data->solvmatr).coefmax * sizeof(dague_complex64_t) );

    /* dsparse_add2arena_tile(((dague_zhetrf_Url_object_t*)dague_zhetrf)->arenas[DAGUE_zhetrf_Url_DEFAULT_ARENA],  */
    /*                        A->mb*A->nb*sizeof(dague_complex64_t), */
    /*                        DAGUE_ARENA_ALIGNMENT_SSE, */
    /*                        MPI_DOUBLE_COMPLEX, A->mb); */
    
    return (dague_object_t*)dague_zhetrf_sp;
}
 
void
dsparse_zhetrf_sp_Destruct( dague_object_t *o )
{
    dague_zhetrf_sp1dplus_object_t *dague_zhetrf_sp = NULL;
    dague_zhetrf_sp = (dague_zhetrf_sp1dplus_object_t *)o;

    /*dsparse_datatype_undefine_type( &(ohetrf->arenas[DAGUE_zhetrf_Url_DEFAULT_ARENA]->opaque_dtt) );*/

    dague_private_memory_fini( dague_zhetrf_sp->p_work1 );
    free( dague_zhetrf_sp->p_work1 );
    dague_private_memory_fini( dague_zhetrf_sp->p_work2 );
    free( dague_zhetrf_sp->p_work2 );

    dague_zhetrf_sp1dplus_destroy( dague_zhetrf_sp );
}

int dsparse_zhetrf_sp( dague_context_t *dague, sparse_matrix_desc_t *A) 
{
    dague_object_t *dague_zhetrf_sp = NULL;
    int info = 0;

    dague_zhetrf_sp = dsparse_zhetrf_sp_New( A );

    if ( dague_zhetrf_sp != NULL )
    {
        dague_enqueue( dague, (dague_object_t*)dague_zhetrf_sp);
        dague_progress( dague );
        dsparse_zhetrf_sp_Destruct( dague_zhetrf_sp );
    }
    return info;
}

