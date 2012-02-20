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

static int check_solution( dague_context_t *dague, tiled_matrix_desc_t *ddescA,
                           tiled_matrix_desc_t *ddescB, tiled_matrix_desc_t *ddescX );

int main(int argc, char ** argv)
{
    dague_context_t* dague;
    int iparam[IPARAM_SIZEOF];

    /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 40, 200, 200);
    iparam[IPARAM_LDA] = -'m';
    iparam[IPARAM_LDB] = -'m';
#if defined(HAVE_CUDA) && defined(PRECISION_s) && 0
    iparam[IPARAM_NGPUS] = 0;
#endif
    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam)
    PASTE_CODE_FLOPS(FLOPS_ZGETRF, ((DagDouble_t)M,(DagDouble_t)N))
    /* initializing matrix structure */
    int info = 0;
    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
        two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, matrix_Tile,
                                    nodes, cores, rank, MB, NB, M, N, 0, 0,
                                    LDA, N, SMB, SNB, P))
    PASTE_CODE_ALLOCATE_MATRIX(ddescL, 1,
        two_dim_block_cyclic, (&ddescL, matrix_ComplexDouble, matrix_Tile,
                                    nodes, cores, rank, IB, NB, MT*IB, N, 0, 0,
                                    MT*IB, N, SMB, SNB, P))
    PASTE_CODE_ALLOCATE_MATRIX(ddescIPIV, 1,
        two_dim_block_cyclic, (&ddescIPIV, matrix_Integer, matrix_Tile,
                                    nodes, cores, rank, MB, 1, M, NT, 0, 0,
                                    M, NT, SMB, SNB, P))

#if defined(DAGUE_PROF_TRACE)
    ddescA.super.super.key = strdup("A");
    ddescL.super.super.key = strdup("L");
    ddescIPIV.super.super.key = strdup("IPIV");
#endif

    if(!check)
    {
        /* matrix generation */
        if(loud > 2) printf("+++ Generate matrices ... ");
        dplasma_zplrnt( dague, (tiled_matrix_desc_t *)&ddescA, 3872);
        if(loud > 2) printf("Done\n");
        /* Create DAGuE */
        PASTE_CODE_ENQUEUE_KERNEL(dague, zgetrf_incpiv,
                                            ((tiled_matrix_desc_t*)&ddescA,
                                             (tiled_matrix_desc_t*)&ddescL,
                                             (tiled_matrix_desc_t*)&ddescIPIV,
                                             &info));
        /* lets rock! */
        PASTE_CODE_PROGRESS_KERNEL(dague, zgetrf_incpiv);

        /* dplasma_zgetrf_incpiv_Destruct( DAGUE_zgetrf_incpiv ); */
    }
    else
    {
        if ( iparam[IPARAM_NNODES] > 1 ) {
            fprintf(stderr, "Checking doesn't work in distributed\n");
            return EXIT_FAILURE;
        }

        int info_solution;

        PASTE_CODE_ALLOCATE_MATRIX(ddescA0, 1,
            two_dim_block_cyclic, (&ddescA0, matrix_ComplexDouble, matrix_Tile,
                                   nodes, cores, rank, MB, NB, LDA, N, 0, 0,
                                   N, N, SMB, SNB, P));

        PASTE_CODE_ALLOCATE_MATRIX(ddescB, 1,
            two_dim_block_cyclic, (&ddescB, matrix_ComplexDouble, matrix_Tile,
                                   nodes, cores, rank, MB, NB, LDB, NRHS, 0, 0,
                                   N, NRHS, SMB, SNB, P));

        PASTE_CODE_ALLOCATE_MATRIX(ddescX, 1,
            two_dim_block_cyclic, (&ddescX, matrix_ComplexDouble, matrix_Tile,
                                   nodes, cores, rank, MB, NB, LDB, NRHS, 0, 0,
                                   N, NRHS, SMB, SNB, P));

        dplasma_zplrnt( dague, (tiled_matrix_desc_t *)&ddescA0, 3872);

        /*********************************************************************
         *               First Check
         */
        if ( rank == 0 ) {
            printf("***************************************************\n");
        }

        /* matrix generation */
        printf("Generate matrices ... ");
        dplasma_zlacpy( dague, PlasmaUpperLower,
                        (tiled_matrix_desc_t *)&ddescA0, (tiled_matrix_desc_t *)&ddescA );
        dplasma_zplrnt( dague, (tiled_matrix_desc_t *)&ddescB, 3872);
        dplasma_zlacpy( dague, PlasmaUpperLower,
                        (tiled_matrix_desc_t *)&ddescB, (tiled_matrix_desc_t *)&ddescX );
        printf("Done\n");

        /* Compute */
        printf("Compute ... ... ");
        info = dplasma_zgesv_incpiv(dague, (tiled_matrix_desc_t *)&ddescA, (tiled_matrix_desc_t *)&ddescL,
                             (tiled_matrix_desc_t *)&ddescIPIV, (tiled_matrix_desc_t *)&ddescX );
        printf("Done\n");
        printf("Info = %d\n", info);

        /* Check the solution */
        info_solution = check_solution( dague, (tiled_matrix_desc_t *)&ddescA0, (tiled_matrix_desc_t *)&ddescB, (tiled_matrix_desc_t *)&ddescX);

        if ( rank == 0 ) {
            if (info_solution == 0) {
                printf(" ----- TESTING ZGESV ....... PASSED !\n");
            }
            else {
                printf(" ----- TESTING ZGESV ... FAILED !\n");
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
        printf("Generate matrices ... ");
        dplasma_zlacpy( dague, PlasmaUpperLower,
                        (tiled_matrix_desc_t *)&ddescA0, (tiled_matrix_desc_t *)&ddescA );
        dplasma_zplrnt( dague, (tiled_matrix_desc_t *)&ddescB, 3872);
        dplasma_zlacpy( dague, PlasmaUpperLower,
                        (tiled_matrix_desc_t *)&ddescB, (tiled_matrix_desc_t *)&ddescX );
        printf("Done\n");

        /* Compute */
        printf("Compute ... ... ");
        info = dplasma_zgetrf_incpiv(dague, (tiled_matrix_desc_t *)&ddescA, (tiled_matrix_desc_t *)&ddescL,
                              (tiled_matrix_desc_t *)&ddescIPIV );
        if ( info == 0 ) {
            dplasma_zgetrs_incpiv(dague, PlasmaNoTrans, (tiled_matrix_desc_t *)&ddescA, (tiled_matrix_desc_t *)&ddescL,
                           (tiled_matrix_desc_t *)&ddescIPIV, (tiled_matrix_desc_t *)&ddescX );
        }
        printf("Done\n");
        printf("Info = %d\n", info);

        /* Check the solution */
        info_solution = check_solution( dague, (tiled_matrix_desc_t *)&ddescA0, (tiled_matrix_desc_t *)&ddescB, (tiled_matrix_desc_t *)&ddescX);

        if ( rank == 0 ) {
            if (info_solution == 0) {
                printf(" ----- TESTING ZGETRF + ZGETRS ....... PASSED !\n");
            }
            else {
                printf(" ----- TESTING ZGETRF + ZGETRS ... FAILED !\n");
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
        printf("Generate matrices ... ");
        dplasma_zlacpy( dague, PlasmaUpperLower,
                        (tiled_matrix_desc_t *)&ddescA0, (tiled_matrix_desc_t *)&ddescA );
        dplasma_zplrnt( dague, (tiled_matrix_desc_t *)&ddescB, 3872);
        dplasma_zlacpy( dague, PlasmaUpperLower,
                        (tiled_matrix_desc_t *)&ddescB, (tiled_matrix_desc_t *)&ddescX );
        printf("Done\n");

        /* Compute */
        printf("Compute ... ... ");
        info = dplasma_zgetrf_incpiv(dague, (tiled_matrix_desc_t *)&ddescA, (tiled_matrix_desc_t *)&ddescL,
                              (tiled_matrix_desc_t *)&ddescIPIV );
        if ( info == 0 ) {
            dplasma_ztrsmpl(dague, (tiled_matrix_desc_t *)&ddescA, (tiled_matrix_desc_t *)&ddescL,
                            (tiled_matrix_desc_t *)&ddescIPIV, (tiled_matrix_desc_t *)&ddescX);
            dplasma_ztrsm(dague, PlasmaLeft, PlasmaUpper, PlasmaNoTrans, PlasmaNonUnit, 1.0,
                          (tiled_matrix_desc_t *)&ddescA, (tiled_matrix_desc_t *)&ddescX);
        }
        printf("Done\n");
        printf("Info = %d\n", info);

        /* Check the solution */
        info_solution = check_solution( dague, (tiled_matrix_desc_t *)&ddescA0, (tiled_matrix_desc_t *)&ddescB, (tiled_matrix_desc_t *)&ddescX);

        if ( rank == 0 ) {
            if (info_solution == 0) {
                printf(" ----- TESTING ZGETRF + ZTRSMPL + ZTRSM ....... PASSED !\n");
            }
            else {
                printf(" ----- TESTING ZGETRF + ZTRSMPL + ZTRSM ... FAILED !\n");
            }
            printf("***************************************************\n");
        }

        dague_data_free(ddescA0.mat);
        dague_ddesc_destroy( (dague_ddesc_t*)&ddescA0);
        dague_data_free(ddescB.mat);
        dague_ddesc_destroy( (dague_ddesc_t*)&ddescB);
        dague_data_free(ddescX.mat);
        dague_ddesc_destroy( (dague_ddesc_t*)&ddescX);
    }

    dague_data_free(ddescA.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescA);
    dague_data_free(ddescL.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescL);
    dague_data_free(ddescIPIV.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescIPIV);

    cleanup_dague(dague, iparam);

    return EXIT_SUCCESS;
}



static int check_solution( dague_context_t *dague,
                           tiled_matrix_desc_t *ddescA, tiled_matrix_desc_t *ddescB, tiled_matrix_desc_t *ddescX )
{
    int info_solution;
    double Rnorm, Anorm, Bnorm, Xnorm, result;
    int N = ddescB->m;
    int NRHS = ddescB->n;
    double *work = (double *)malloc(N*sizeof(double));
    double eps = LAPACKE_dlamch_work('e');
    Dague_Complex64_t *W;

    W = (Dague_Complex64_t *)malloc( N*max(N,NRHS)*sizeof(Dague_Complex64_t));

    twoDBC_ztolapack( (two_dim_block_cyclic_t *)ddescA, W, N );
    Anorm = LAPACKE_zlange_work( LAPACK_COL_MAJOR, 'i', N, NRHS, W, N, work );

    twoDBC_ztolapack( (two_dim_block_cyclic_t *)ddescB, W, N );
    Bnorm = LAPACKE_zlange_work( LAPACK_COL_MAJOR, 'i', N, NRHS, W, N, work );

    twoDBC_ztolapack( (two_dim_block_cyclic_t *)ddescX, W, N );
    Xnorm = LAPACKE_zlange_work( LAPACK_COL_MAJOR, 'i', N, NRHS, W, N, work );

    dplasma_zgemm( dague, PlasmaNoTrans, PlasmaNoTrans, -1.0, ddescA, ddescX, 1.0, ddescB);

    twoDBC_ztolapack( (two_dim_block_cyclic_t *)ddescB, W, N );
    Rnorm = LAPACKE_zlange_work( LAPACK_COL_MAJOR, 'i', N, NRHS, W, N, work );

    if (getenv("DPLASMA_TESTING_VERBOSE"))
        printf( "||A||_oo = %e, ||X||_oo = %e, ||B||_oo= %e, ||A X - B||_oo = %e\n",
                Anorm, Xnorm, Bnorm, Rnorm );

    result = Rnorm / ( ( Anorm * Xnorm + Bnorm ) * N * eps ) ;
    printf("============\n");
    printf("Checking the Residual of the solution \n");
    printf("-- ||Ax-B||_oo/((||A||_oo||x||_oo+||B||_oo).N.eps) = %e \n", result);

    if (  isnan(Xnorm) || isinf(Xnorm) || isnan(result) || isinf(result) || (result > 60.0) ) {
        info_solution = 1;
     }
    else{
        info_solution = 0;
    }

    free(work); free(W);
    return info_solution;
}
