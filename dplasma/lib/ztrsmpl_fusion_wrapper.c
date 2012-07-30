/*
 * Copyright (c) 2011      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#include "dague.h"
#include <plasma.h>
#include <core_blas.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"

#include "ztrsmpl_fusion.h"

static inline int dague_imin(int a, int b) { return (a <= b) ? a : b; };

dague_object_t* dplasma_ztrsmpl_fusion_New( const tiled_matrix_desc_t *A,
                                            const tiled_matrix_desc_t *IPIV,
                                            tiled_matrix_desc_t *B )
{
    dague_object_t *dague_ztrsmpl_fusion = NULL;
    int nb = A->nb;
    int P = ((two_dim_block_cyclic_t*)A)->grid.rows;

    dague_ztrsmpl_fusion = (dague_object_t*)dague_ztrsmpl_fusion_new((dague_ddesc_t*)A,
                                                                     (dague_ddesc_t*)IPIV,
                                                                     (dague_ddesc_t*)B,
                                                                     P);

    /* A */
    dplasma_add2arena_tile( ((dague_ztrsmpl_fusion_object_t*)dague_ztrsmpl_fusion)->arenas[DAGUE_ztrsmpl_fusion_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(dague_complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb );

    /* PERMUT */
    dplasma_add2arena_rectangle( ((dague_ztrsmpl_fusion_object_t*)dague_ztrsmpl_fusion)->arenas[DAGUE_ztrsmpl_fusion_PERMUT_ARENA],
                                 2 * nb * sizeof(int),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_INT, 2, nb, -1 );

    return (dague_object_t*)dague_ztrsmpl_fusion;
}

void
dplasma_ztrsmpl_fusion_Destruct( dague_object_t *o )
{
    dague_ztrsmpl_fusion_object_t *dague_ztrsmpl_fusion = (dague_ztrsmpl_fusion_object_t *)o;

    dplasma_datatype_undefine_type( &(dague_ztrsmpl_fusion->arenas[DAGUE_ztrsmpl_fusion_DEFAULT_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_ztrsmpl_fusion->arenas[DAGUE_ztrsmpl_fusion_PERMUT_ARENA ]->opaque_dtt) );

    DAGUE_INTERNAL_OBJECT_DESTRUCT(o);
}

void dplasma_ztrsmpl_fusion( dague_context_t *dague,
                             const tiled_matrix_desc_t *A,
                             const tiled_matrix_desc_t *IPIV,
                             tiled_matrix_desc_t *B )
{
    dague_object_t *dague_ztrsmpl_fusion = NULL;

    dague_ztrsmpl_fusion = dplasma_ztrsmpl_fusion_New(A, IPIV, B );

    if ( dague_ztrsmpl_fusion != NULL )
    {
        dague_enqueue( dague, (dague_object_t*)dague_ztrsmpl_fusion);
        dplasma_progress(dague);
        dplasma_ztrsmpl_fusion_Destruct( dague_ztrsmpl_fusion );
    }

    return;
}

