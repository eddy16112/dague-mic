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

#include "zcsc2cblk.h"

dague_object_t* 
dsparse_zcsc2cblk_New(sparse_matrix_desc_t *A)
{
    dague_zcsc2cblk_object_t *dague_zcsc2cblk = NULL;
 
    dague_zcsc2cblk = dague_zcsc2cblk_new(A, (dague_ddesc_t *)A );

    /* dsparse_add2arena_tile(((dague_zcsc2cblk_Url_object_t*)dague_zcsc2cblk)->arenas[DAGUE_zcsc2cblk_Url_DEFAULT_ARENA],  */
    /*                        A->mb*A->nb*sizeof(dague_complex64_t), */
    /*                        DAGUE_ARENA_ALIGNMENT_SSE, */
    /*                        MPI_DOUBLE_COMPLEX, A->mb); */
    
    return (dague_object_t*)dague_zcsc2cblk;
}
 
void
dsparse_zcsc2cblk_Destruct( dague_object_t *o )
{
    dague_zcsc2cblk_object_t *dague_zcsc2cblk = NULL;
    dague_zcsc2cblk = (dague_zcsc2cblk_object_t *)o;

    /*dsparse_datatype_undefine_type( &(opotrf->arenas[DAGUE_zcsc2cblk_Url_DEFAULT_ARENA]->opaque_dtt) );*/

    dague_zcsc2cblk_destroy( dague_zcsc2cblk );
}

int dsparse_zcsc2cblk( dague_context_t *dague, sparse_matrix_desc_t *A) 
{
    dague_object_t *dague_zcsc2cblk = NULL;
    int info = 0;

    dague_zcsc2cblk = dsparse_zcsc2cblk_New( A );

    if ( dague_zcsc2cblk != NULL )
    {
        dague_enqueue( dague, (dague_object_t*)dague_zcsc2cblk);
        dague_progress( dague );
        dsparse_zcsc2cblk_Destruct( dague_zcsc2cblk );
    }
    return info;
}

