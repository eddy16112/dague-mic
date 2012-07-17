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

static int check_solution( dague_context_t *dague, int loud,
                           tiled_matrix_desc_t *ddescA,
                           tiled_matrix_desc_t *ddescB,
                           tiled_matrix_desc_t *ddescX );

int main(int argc, char ** argv)
{
    dague_context_t* dague;
    int iparam[IPARAM_SIZEOF];
    int info = 0;
    int info_solution = 0;
    int ret = 0;

    /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 40, 200, 200);
    iparam[IPARAM_LDA] = -'m';
    iparam[IPARAM_LDB] = -'m';

    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);

    /* initializing matrix structure */
    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
        two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, matrix_Tile,
                               nodes, cores, rank, MB, NB, LDA, N, 0, 0,
                               M, N, SMB, SNB, P));

    PASTE_CODE_ALLOCATE_MATRIX(ddescA0, 1,
        two_dim_block_cyclic, (&ddescA0, matrix_ComplexDouble, matrix_Tile,
                               nodes, cores, rank, MB, NB, LDA, N, 0, 0,
                               M, N, SMB, SNB, P));

    PASTE_CODE_ALLOCATE_MATRIX(ddescB, 1,
        two_dim_block_cyclic, (&ddescB, matrix_ComplexDouble, matrix_Tile,
                               nodes, cores, rank, MB, NB, LDB, NRHS, 0, 0,
                               N, NRHS, SMB, SNB, P));

    PASTE_CODE_ALLOCATE_MATRIX(ddescX, 1,
        two_dim_block_cyclic, (&ddescX, matrix_ComplexDouble, matrix_Tile,
                               nodes, cores, rank, MB, NB, LDB, NRHS, 0, 0,
                               N, NRHS, SMB, SNB, P));

    PASTE_CODE_ALLOCATE_MATRIX(ddescL, 1,
        two_dim_block_cyclic, (&ddescL, matrix_ComplexDouble, matrix_Tile,
                               nodes, cores, rank, IB, NB, MT*IB, N, 0, 0,
                               MT*IB, N, SMB, SNB, P));

    PASTE_CODE_ALLOCATE_MATRIX(ddescIPIV, 1,
        two_dim_block_cyclic, (&ddescIPIV, matrix_Integer, matrix_Tile,
                               nodes, cores, rank, MB, 1, M, NT, 0, 0,
                               M, NT, SMB, SNB, P));

    /* matrix generation */
    if(loud > 2) printf("+++ Generate matrices ... ");
    dplasma_zplrnt( dague, (tiled_matrix_desc_t *)&ddescA0, 1358);
    if(loud > 2) printf("Done\n");

    /*********************************************************************
     *               First Check
     */
    if ( rank == 0 ) {
        printf("***************************************************\n");
    }

    /* matrix generation */
    if( loud > 2 ) printf("Generate matrices ... ");
    dplasma_zlacpy( dague, PlasmaUpperLower,
                    (tiled_matrix_desc_t *)&ddescA0, (tiled_matrix_desc_t *)&ddescA );
    dplasma_zplrnt( dague, (tiled_matrix_desc_t *)&ddescB, 3872);
    dplasma_zlacpy( dague, PlasmaUpperLower,
                    (tiled_matrix_desc_t *)&ddescB, (tiled_matrix_desc_t *)&ddescX );
    if( loud > 2 ) printf("Done\n");

    /* Compute */
    if( loud > 2 ) printf("Compute ... ... ");
    info = dplasma_zgesv_incpiv(dague,
                                (tiled_matrix_desc_t *)&ddescA,
                                (tiled_matrix_desc_t *)&ddescL,
                                (tiled_matrix_desc_t *)&ddescIPIV,
                                (tiled_matrix_desc_t *)&ddescX );
    if( loud > 2 ) printf("Done\n");
    if ( info != 0 ) printf("%d: Info = %d\n", rank, info);

    /* Check the solution */
    if ( info == 0 ) {
        info_solution = check_solution( dague, rank ? 0 : loud,
                                        (tiled_matrix_desc_t *)&ddescA0,
                                        (tiled_matrix_desc_t *)&ddescB,
                                        (tiled_matrix_desc_t *)&ddescX );
    }
    if ( rank == 0 ) {
        if ( info || info_solution) {
            printf(" ----- TESTING ZGESV ... FAILED !\n");
            ret |= 1;
        }
        else {
            printf(" ----- TESTING ZGESV ....... PASSED !\n");
        }
        printf("***************************************************\n");
    }

    /*********************************************************************
     *               Second Check
     */
    if ( rank == 0 ) {
        printf("***************************************************\n");
    }
    
    /* matrix generation */
    if( loud > 2 ) printf("Generate matrices ... ");
    dplasma_zlacpy( dague, PlasmaUpperLower,
                    (tiled_matrix_desc_t *)&ddescA0, (tiled_matrix_desc_t *)&ddescA );
    dplasma_zplrnt( dague, (tiled_matrix_desc_t *)&ddescB, 3872);
    dplasma_zlacpy( dague, PlasmaUpperLower,
                    (tiled_matrix_desc_t *)&ddescB, (tiled_matrix_desc_t *)&ddescX );
    if( loud > 2 ) printf("Done\n");

    /* Compute */
    if( loud > 2 ) printf("Compute ... ... ");
    info = dplasma_zgetrf_incpiv(dague,
                                 (tiled_matrix_desc_t *)&ddescA,
                                 (tiled_matrix_desc_t *)&ddescL,
                                 (tiled_matrix_desc_t *)&ddescIPIV );
    if ( info == 0 ) {
        dplasma_zgetrs_incpiv(dague, PlasmaNoTrans,
                              (tiled_matrix_desc_t *)&ddescA,
                              (tiled_matrix_desc_t *)&ddescL,
                              (tiled_matrix_desc_t *)&ddescIPIV,
                              (tiled_matrix_desc_t *)&ddescX );
    }
    if ( loud > 2 ) printf("Done\n");
    if ( info != 0 ) printf("%d: Info = %d\n", rank, info);

    /* Check the solution */
    if ( info == 0 ) {
        info_solution = check_solution( dague, rank ? 0 : loud,
                                        (tiled_matrix_desc_t *)&ddescA0,
                                        (tiled_matrix_desc_t *)&ddescB,
                                        (tiled_matrix_desc_t *)&ddescX );
    }
    if ( rank == 0 ) {
        if ( info || info_solution) {
            printf(" ----- TESTING ZGETRF + ZGETRS ... FAILED !\n");
            ret |= 1;
        }
        else {
            printf(" ----- TESTING ZGETRF + ZGETRS ....... PASSED !\n");
        }
        printf("***************************************************\n");
    }

    /*********************************************************************
     *               Third Check
     */
    if ( rank == 0 ) {
        printf("***************************************************\n");
    }
    
    /* matrix generation */
    if( loud > 2 ) printf("Generate matrices ... ");
    dplasma_zlacpy( dague, PlasmaUpperLower,
                    (tiled_matrix_desc_t *)&ddescA0, (tiled_matrix_desc_t *)&ddescA );
    dplasma_zplrnt( dague, (tiled_matrix_desc_t *)&ddescB, 3872);
    dplasma_zlacpy( dague, PlasmaUpperLower,
                    (tiled_matrix_desc_t *)&ddescB, (tiled_matrix_desc_t *)&ddescX );
    if( loud > 2 ) printf("Done\n");
    
    /* Compute */
    if( loud > 2 )printf("Compute ... ... ");
    info = dplasma_zgetrf_incpiv(dague,
                                 (tiled_matrix_desc_t *)&ddescA,
                                 (tiled_matrix_desc_t *)&ddescL,
                                 (tiled_matrix_desc_t *)&ddescIPIV );

    if ( info == 0 ) {
        dplasma_ztrsmpl(dague,
                        (tiled_matrix_desc_t *)&ddescA,
                        (tiled_matrix_desc_t *)&ddescL,
                        (tiled_matrix_desc_t *)&ddescIPIV,
                        (tiled_matrix_desc_t *)&ddescX);
        
        dplasma_ztrsm(dague, PlasmaLeft, PlasmaUpper,
                      PlasmaNoTrans, PlasmaNonUnit, 1.0,
                      (tiled_matrix_desc_t *)&ddescA,
                      (tiled_matrix_desc_t *)&ddescX);
    }
    if ( loud > 2 ) printf("Done\n");
    if ( info != 0 ) printf("%d: Info = %d\n", rank, info);

    /* Check the solution */
    if ( info == 0 ) {
            info_solution = check_solution( dague, rank ? 0 : loud,
                                            (tiled_matrix_desc_t *)&ddescA0,
                                            (tiled_matrix_desc_t *)&ddescB,
                                            (tiled_matrix_desc_t *)&ddescX );
    }
    if ( rank == 0 ) {
        if ( info || info_solution ) {
            printf(" ----- TESTING ZGETRF + ZTRSMPL + ZTRSM ... FAILED !\n");
            ret |= 1;
        }
        else {
            printf(" ----- TESTING ZGETRF + ZTRSMPL + ZTRSM ....... PASSED !\n");
        }
        printf("***************************************************\n");
    }

    cleanup_dague(dague, iparam);

    dague_data_free(ddescA.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescA);
    dague_data_free(ddescA0.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescA0);
    dague_data_free(ddescB.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescB);
    dague_data_free(ddescX.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescX);
    dague_data_free(ddescL.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescL);
    dague_data_free(ddescIPIV.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescIPIV);

    return ret;
}



static int check_solution( dague_context_t *dague, int loud,
                           tiled_matrix_desc_t *ddescA,
                           tiled_matrix_desc_t *ddescB,
                           tiled_matrix_desc_t *ddescX )
{
    int info_solution;
    int N = ddescB->m;
    double Rnorm, Anorm, Bnorm, Xnorm, result;
    double eps = LAPACKE_dlamch_work('e');
    
    Anorm = dplasma_zlange(dague, PlasmaInfNorm, ddescA);
    Bnorm = dplasma_zlange(dague, PlasmaInfNorm, ddescB);
    Xnorm = dplasma_zlange(dague, PlasmaInfNorm, ddescX);

    /* Compute b - A*x */
    dplasma_zgemm( dague, PlasmaNoTrans, PlasmaNoTrans, -1.0, ddescA, ddescX, 1.0, ddescB);
    Rnorm = dplasma_zlange(dague, PlasmaInfNorm, ddescB);
    result = Rnorm / ( ( Anorm * Xnorm + Bnorm ) * N * eps ) ;

    if( loud ) { 
        if( loud > 2 )
            printf( "||A||_oo = %e, ||X||_oo = %e, ||B||_oo= %e, ||A X - B||_oo = %e\n",
                    Anorm, Xnorm, Bnorm, Rnorm );

        printf("============\n");
        printf("Checking the Residual of the solution \n");
        printf("-- ||Ax-B||_oo/((||A||_oo||x||_oo+||B||_oo).N.eps) = %e \n", result);
    }
    if (  isnan(Xnorm) || isinf(Xnorm) || isnan(result) || isinf(result) || (result > 60.0) ) {
        info_solution = 1;
    }
    else{
        info_solution = 0;
    }

    return info_solution;
}
