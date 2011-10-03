/*
 * Copyright (c) 2009-2011 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "common.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"

int  dplasma_qr_check(         tiled_matrix_desc_t *A, qr_piv_t *qrpiv );
void dplasma_qr_print_type(    tiled_matrix_desc_t *A, qr_piv_t *qrpiv );
void dplasma_qr_print_pivot(   tiled_matrix_desc_t *A, qr_piv_t *qrpiv );
void dplasma_qr_print_nbgeqrt( tiled_matrix_desc_t *A, qr_piv_t *qrpiv );
void dplasma_qr_print_next_k(  tiled_matrix_desc_t *A, qr_piv_t *qrpiv, int k );
void dplasma_qr_print_prev_k(  tiled_matrix_desc_t *A, qr_piv_t *qrpiv, int k );
void dplasma_qr_print_geqrt_k( tiled_matrix_desc_t *A, qr_piv_t *qrpiv, int k );

int main(int argc, char ** argv)
{
    dague_context_t* dague;
    qr_piv_t *qrpiv;
    int ret;
    int iparam[IPARAM_SIZEOF];

    /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 1, 1, 1);
    iparam[IPARAM_LDA] = -'m';
    iparam[IPARAM_LDB] = -'m';

    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam)
      
    LDA = max(M, LDA);
    /* initializing matrix structure */
    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1, 
        two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, 
                               nodes, cores, rank, MB, NB, LDA, N, 0, 0, 
                               M, N, SMB, SNB, P));
 
    qrpiv = dplasma_pivgen_init( (tiled_matrix_desc_t*)&ddescA, 
                                 iparam[IPARAM_LOWLVL_TREE], iparam[IPARAM_HIGHLVL_TREE],
                                 iparam[IPARAM_QR_TS_SZE], iparam[IPARAM_QR_HLVL_SZE],
                                 iparam[IPARAM_QR_DOMINO]);

    /*dplasma_qr_print_dag( (tiled_matrix_desc_t*)&ddescA, qrpiv);*/
    /* dplasma_qr_print_pivot( (tiled_matrix_desc_t*)&ddescA, qrpiv); */
    /* dplasma_qr_print_next_k( (tiled_matrix_desc_t*)&ddescA, qrpiv, 0); */
    ret = dplasma_qr_check( (tiled_matrix_desc_t*)&ddescA, qrpiv );
    
    dplasma_pivgen_finalize( qrpiv );

    dague_data_free(ddescA.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescA);

    cleanup_dague(dague, iparam);

    if ( ret == 0 )
      return EXIT_SUCCESS;
    else {
      fprintf(stderr, "ret = %d\n", ret);
      return EXIT_FAILURE;
    }
}
