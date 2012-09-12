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
#include "zgetrf_sp1dplus.h"

dague_object_t* 
dsparse_zgetrf_sp_New(sparse_matrix_desc_t *A)
{
    dague_zgetrf_sp1dplus_object_t *dague_zgetrf_sp = NULL;
 
    dague_zgetrf_sp = dague_zgetrf_sp1dplus_new(A, (dague_ddesc_t *)A, NULL );

    dague_zgetrf_sp->p_work = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( dague_zgetrf_sp->p_work, (A->pastix_data->solvmatr).coefmax * sizeof(dague_complex64_t) );

    /* dsparse_add2arena_tile(((dague_zgetrf_Url_object_t*)dague_zgetrf)->arenas[DAGUE_zgetrf_Url_DEFAULT_ARENA],  */
    /*                        A->mb*A->nb*sizeof(dague_complex64_t), */
    /*                        DAGUE_ARENA_ALIGNMENT_SSE, */
    /*                        MPI_DOUBLE_COMPLEX, A->mb); */
    
    return (dague_object_t*)dague_zgetrf_sp;
}
 
void
dsparse_zgetrf_sp_Destruct( dague_object_t *o )
{
    dague_zgetrf_sp1dplus_object_t *dague_zgetrf_sp = NULL;
    dague_zgetrf_sp = (dague_zgetrf_sp1dplus_object_t *)o;

    /*dsparse_datatype_undefine_type( &(ogetrf->arenas[DAGUE_zgetrf_Url_DEFAULT_ARENA]->opaque_dtt) );*/

    dague_private_memory_fini( dague_zgetrf_sp->p_work );
    free( dague_zgetrf_sp->p_work );

    DAGUE_INTERNAL_OBJECT_DESTRUCT(o);
}

int dsparse_zgetrf_sp( dague_context_t *dague, sparse_matrix_desc_t *A) 
{
    dague_object_t *dague_zgetrf_sp = NULL;
    int info = 0;

    dague_zgetrf_sp = dsparse_zgetrf_sp_New( A );

    if ( dague_zgetrf_sp != NULL )
    {
        dague_enqueue( dague, (dague_object_t*)dague_zgetrf_sp);
        dague_progress( dague );
        dsparse_zgetrf_sp_Destruct( dague_zgetrf_sp );
    }
    return info;
}

