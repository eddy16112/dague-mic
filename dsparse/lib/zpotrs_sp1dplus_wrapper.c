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
#include "zpotrs_sp1dplus.h"

dague_object_t* 
dsparse_zpotrs_sp_New(sparse_matrix_desc_t *A, sparse_vector_desc_t *B)
{
    dague_zpotrs_sp1dplus_object_t *dague_zpotrs_sp = NULL;
 
    dague_zpotrs_sp = dague_zpotrs_sp1dplus_new(A, (dague_ddesc_t *)A, 
                                                B, (dague_ddesc_t *)B, NULL );

    dague_zpotrs_sp->p_work = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( dague_zpotrs_sp->p_work, (A->pastix_data->solvmatr).coefmax * sizeof(dague_complex64_t) );

    /* dsparse_add2arena_tile(((dague_zpotrs_Url_object_t*)dague_zpotrs)->arenas[DAGUE_zpotrs_Url_DEFAULT_ARENA],  */
    /*                        A->mb*A->nb*sizeof(dague_complex64_t), */
    /*                        DAGUE_ARENA_ALIGNMENT_SSE, */
    /*                        MPI_DOUBLE_COMPLEX, A->mb); */
    
    return (dague_object_t*)dague_zpotrs_sp;
}
 
void
dsparse_zpotrs_sp_Destruct( dague_object_t *o )
{
    dague_zpotrs_sp1dplus_object_t *dague_zpotrs_sp = NULL;
    dague_zpotrs_sp = (dague_zpotrs_sp1dplus_object_t *)o;

    /*dsparse_datatype_undefine_type( &(opotrs->arenas[DAGUE_zpotrs_Url_DEFAULT_ARENA]->opaque_dtt) );*/

    dague_private_memory_fini( dague_zpotrs_sp->p_work );
    free( dague_zpotrs_sp->p_work );

    dague_zpotrs_sp1dplus_destroy( dague_zpotrs_sp );
}

int dsparse_zpotrs_sp( dague_context_t *dague, sparse_matrix_desc_t *A, sparse_vector_desc_t *B) 
{
    dague_object_t *dague_zpotrs_sp = NULL;
    int info = 0;

    dague_zpotrs_sp = dsparse_zpotrs_sp_New( A, B );

    if ( dague_zpotrs_sp != NULL )
    {
        dague_enqueue( dague, (dague_object_t*)dague_zpotrs_sp);
        dague_progress( dague );
        dsparse_zpotrs_sp_Destruct( dague_zpotrs_sp );
    }
    return info;
}

