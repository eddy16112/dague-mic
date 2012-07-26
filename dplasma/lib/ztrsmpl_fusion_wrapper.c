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
    two_dim_block_cyclic_t *BUFFER, *BCOPY, *Wperm;
    int mb = A->mb, nb = A->nb;
    int P = ((two_dim_block_cyclic_t*)A)->grid.rows;

    BUFFER = (two_dim_block_cyclic_t*)malloc(sizeof(two_dim_block_cyclic_t));
    PASTE_CODE_INIT_AND_ALLOCATE_MATRIX(
        (*BUFFER), two_dim_block_cyclic,
        (BUFFER, matrix_ComplexDouble, matrix_Tile,
         A->super.nodes, A->super.cores, A->super.myrank,
         mb,   nb,      /* Dimesions of the tile                */
         mb*P, nb*A->nt,/* Dimensions of the matrix             */
         0,    0,       /* Starting points (not important here) */
         mb*P, nb*A->nt,/* Dimensions of the submatrix          */
         1, 1, P));

    
    /* This can be removed and replace by allocation inside the kernels */
    BCOPY  = (two_dim_block_cyclic_t*)malloc(sizeof(two_dim_block_cyclic_t));
    PASTE_CODE_INIT_AND_ALLOCATE_MATRIX(
        (*BCOPY), two_dim_block_cyclic,
        (BCOPY, matrix_ComplexDouble, matrix_Tile,
         B->super.nodes, B->super.cores, B->super.myrank,
         mb, nb,   /* Dimesions of the tile                */
         mb, B->n, /* Dimensions of the matrix             */
         0,  0,    /* Starting points (not important here) */
         mb, B->n, /* Dimensions of the submatrix          */
         1, 1, P));

    /* This can be removed and replace by allocation inside the kernels */
    Wperm = (two_dim_block_cyclic_t*)malloc(sizeof(two_dim_block_cyclic_t));
    PASTE_CODE_INIT_AND_ALLOCATE_MATRIX( 
        (*Wperm), two_dim_block_cyclic,
        (Wperm, matrix_Integer, matrix_Tile,
         A->super.nodes, A->super.cores, A->super.myrank, 
         2, nb, 2*P, dague_imin(A->m, A->n),
         0, 0,  2*P, dague_imin(A->m, A->n), 1, 1, P));

    dague_ztrsmpl_fusion = (dague_object_t*)dague_ztrsmpl_fusion_new((dague_ddesc_t*)A,
                                                                     (dague_ddesc_t*)IPIV,
                                                                     (dague_ddesc_t*)B,
                                                                     (dague_ddesc_t*)BUFFER,
                                                                     (dague_ddesc_t*)BCOPY,
                                                                     (dague_ddesc_t*)Wperm,
                                                                     P);

    /* A */
    dplasma_add2arena_tile( ((dague_ztrsmpl_fusion_object_t*)dague_ztrsmpl_fusion)->arenas[DAGUE_ztrsmpl_fusion_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(dague_complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb );

    /* PERMUT */
    dplasma_add2arena_rectangle( ((dague_ztrsmpl_fusion_object_t*)dague_ztrsmpl_fusion)->arenas[DAGUE_ztrsmpl_fusion_PERMUT_ARENA],
                                 Wperm->super.mb * Wperm->super.nb * sizeof(int),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_INT, Wperm->super.mb, Wperm->super.nb, -1 );

    return (dague_object_t*)dague_ztrsmpl_fusion;
}

void
dplasma_ztrsmpl_fusion_Destruct( dague_object_t *o )
{
    dague_ztrsmpl_fusion_object_t *dague_ztrsmpl_fusion = (dague_ztrsmpl_fusion_object_t *)o;
    two_dim_block_cyclic_t *desc;

    desc = (two_dim_block_cyclic_t*)(dague_ztrsmpl_fusion->BUFFER);
    dague_data_free( desc->mat );
    dague_ddesc_destroy( dague_ztrsmpl_fusion->BUFFER );
    free( dague_ztrsmpl_fusion->BUFFER );

    desc= (two_dim_block_cyclic_t*)(dague_ztrsmpl_fusion->BCOPY);
    dague_data_free( desc->mat );
    dague_ddesc_destroy( dague_ztrsmpl_fusion->BCOPY );
    free( dague_ztrsmpl_fusion->BCOPY );

    desc= (two_dim_block_cyclic_t*)(dague_ztrsmpl_fusion->Wperm);
    dague_data_free( desc->mat );
    dague_ddesc_destroy( dague_ztrsmpl_fusion->Wperm );
    free( dague_ztrsmpl_fusion->Wperm );

    dplasma_datatype_undefine_type( &(dague_ztrsmpl_fusion->arenas[DAGUE_ztrsmpl_fusion_DEFAULT_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_ztrsmpl_fusion->arenas[DAGUE_ztrsmpl_fusion_PERMUT_ARENA]->opaque_dtt) );

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

