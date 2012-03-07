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

static int check_solution(int loud, PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum trans, PLASMA_enum diag,
                          Dague_Complex64_t alpha, two_dim_block_cyclic_t *ddescA,
                          two_dim_block_cyclic_t *ddescB, two_dim_block_cyclic_t *ddescX );

int main(int argc, char ** argv)
{
    dague_context_t* dague;
    int iparam[IPARAM_SIZEOF];
    int ret = 0;

    /* Set defaults for non argv iparams */
    iparam_default_gemm(iparam);
    iparam_default_ibnbmb(iparam, 0, 200, 200);
#if defined(HAVE_CUDA) && defined(PRECISION_s) && 0
    iparam[IPARAM_NGPUS] = 0;
#endif
    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);
    /* initializing matrix structure */
    int Am = max(M, NRHS);
    LDA = max(LDA, Am);
    LDB = max(LDB, M);
    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
        two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, matrix_Tile,
                               nodes, cores, rank, MB, NB, LDA, LDA, 0, 0,
                               Am, Am, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescB, 1,
        two_dim_block_cyclic, (&ddescB, matrix_ComplexDouble, matrix_Tile,
                               nodes, cores, rank, MB, NB, LDB, NRHS, 0, 0,
                               M, NRHS, SMB, SNB, P));

    if(!check)
    {
        PLASMA_enum side  = PlasmaLeft;
        PLASMA_enum uplo  = PlasmaLower;
        PLASMA_enum trans = PlasmaNoTrans;
        PLASMA_enum diag  = PlasmaUnit;

        PASTE_CODE_FLOPS(FLOPS_ZTRSM, (side, (DagDouble_t)M, (DagDouble_t)NRHS));

        /* matrix generation */
        if(loud > 2) printf("+++ Generate matrices ... ");
        dplasma_zplghe( dague, 2., uplo, (tiled_matrix_desc_t *)&ddescA, 1358);
        dplasma_zplrnt( dague,            (tiled_matrix_desc_t *)&ddescB, 5676);
        if(loud > 2) printf("Done\n");

        /* Create DAGuE */
        PASTE_CODE_ENQUEUE_KERNEL(dague, ztrsm,
                                  (side, uplo, trans, diag,
                                   (Dague_Complex64_t)1.0,
                                   (tiled_matrix_desc_t *)&ddescA,
                                   (tiled_matrix_desc_t *)&ddescB));

        /* lets rock! */
        PASTE_CODE_PROGRESS_KERNEL(dague, ztrsm);

        dplasma_ztrsm_Destruct( DAGUE_ztrsm );
    }
    else
    {
        if ( iparam[IPARAM_NNODES] > 1 ) {
            fprintf(stderr, "Checking doesn't work in distributed\n");
            return EXIT_FAILURE;
        }

        int s, u, t, d;
        int info_solution;
        Dague_Complex64_t alpha = 3.5;

        PASTE_CODE_ALLOCATE_MATRIX(ddescX, 1,
            two_dim_block_cyclic, (&ddescX, matrix_ComplexDouble, matrix_Tile,
                                   nodes, cores, rank, MB, NB, LDB, NRHS, 0, 0,
                                   M, NRHS, SMB, SNB, P));

        dplasma_zplghe( dague, 2., PlasmaUpperLower, (tiled_matrix_desc_t *)&ddescA, 1358);
        dplasma_zplrnt( dague, (tiled_matrix_desc_t *)&ddescB, 5676);

        for (s=0; s<2; s++) {
            for (u=0; u<2; u++) {
#if defined(PRECISION_z) || defined(PRECISION_c)
                for (t=0; t<3; t++) {
#else
                for (t=0; t<2; t++) {
#endif
                    for (d=0; d<2; d++) {

                        if ( rank == 0 ) {
                            printf("***************************************************\n");
                            printf(" ----- TESTING ZTRSM (%s, %s, %s, %s) -------- \n",
                                   sidestr[s], uplostr[u], transstr[t], diagstr[d]);
                        }

                        /* matrix generation */
                        printf("Generate matrices ... ");
                        dplasma_zlacpy( dague, PlasmaUpperLower,
                                        (tiled_matrix_desc_t *)&ddescB, (tiled_matrix_desc_t *)&ddescX );
                        printf("Done\n");

                        /* Compute */
                        printf("Compute ... ... ");
                        dplasma_ztrsm(dague, side[s], uplo[u], trans[t], diag[d], (Dague_Complex64_t)alpha,
                                      (tiled_matrix_desc_t *)&ddescA, (tiled_matrix_desc_t *)&ddescX);
                        printf("Done\n");

                        /* Check the solution */
                        info_solution = check_solution(rank == 0 ? loud : 0,
                                                       side[s], uplo[u], trans[t], diag[d],
                                                       alpha, &ddescA, &ddescB, &ddescX);
                        if ( rank == 0 ) {
                            if (info_solution == 0) {
                                printf(" ---- TESTING ZTRSM (%s, %s, %s, %s) ...... PASSED !\n",
                                       sidestr[s], uplostr[u], transstr[t], diagstr[d]);
                            }
                            else {
                                printf(" ---- TESTING ZTRSM (%s, %s, %s, %s) ... FAILED !\n",
                                       sidestr[s], uplostr[u], transstr[t], diagstr[d]);
                                ret |= 1;
                            }
                            printf("***************************************************\n");
                        }
                    }
                }
#ifdef __UNUSED__
                }
#endif
            }
        }
        dague_data_free(ddescX.mat);
        dague_ddesc_destroy((dague_ddesc_t*)&ddescX);
    }

    cleanup_dague(dague, iparam);

    dague_data_free(ddescA.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescA);
    dague_data_free(ddescB.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescB);

    return ret;
}


/**********************************
 * static functions
 **********************************/

/*------------------------------------------------------------------------
 *  Check the accuracy of the solution
 */
static int check_solution(int loud, PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum trans, PLASMA_enum diag,
                          Dague_Complex64_t alpha, two_dim_block_cyclic_t *ddescA, two_dim_block_cyclic_t *ddescB, two_dim_block_cyclic_t *ddescX )
{
    int info_solution;
    double Anorm, Binitnorm, Bdaguenorm, Blapacknorm, Rnorm, result;
    Dague_Complex64_t *A, *B, *X;
    int M   = ddescB->super.m;
    int N   = ddescB->super.n;
    int LDA = ddescA->super.lm;
    int LDB = ddescB->super.lm;
    double eps = LAPACKE_dlamch_work('e');
    double *work = (double *)malloc(max(M, N)* sizeof(double));
    int Am;
    Dague_Complex64_t mzone = (Dague_Complex64_t)-1.0;

    if (side == PlasmaLeft) {
        Am = M;
    } else {
        Am = N;
    }

    A = (Dague_Complex64_t *)malloc((ddescA->super.lm)*(ddescA->super.n)*sizeof(Dague_Complex64_t));
    B = (Dague_Complex64_t *)malloc((ddescB->super.lm)*(ddescB->super.n)*sizeof(Dague_Complex64_t));
    X = (Dague_Complex64_t *)malloc((ddescX->super.lm)*(ddescX->super.n)*sizeof(Dague_Complex64_t));

    twoDBC_ztolapack( ddescA, A, LDA );
    twoDBC_ztolapack( ddescB, B, LDB );
    twoDBC_ztolapack( ddescX, X, LDB );

    /* TODO: check lantr because it returns 0.0, it looks like a parameter is wrong */
    //Anorm      = LAPACKE_zlantr_work( LAPACK_COL_MAJOR, 'i', lapack_const(uplo), lapack_const(diag), Am, Am, A, LDA, work );
    Anorm      = LAPACKE_zlanhe_work( LAPACK_COL_MAJOR, 'i', lapack_const(uplo), Am, A, LDA, work );
    Binitnorm  = LAPACKE_zlange_work( LAPACK_COL_MAJOR, 'i', M,  N,  B, LDB, work );
    Bdaguenorm = LAPACKE_zlange_work( LAPACK_COL_MAJOR, 'i', M,  N,  X, LDB, work );

    cblas_ztrsm(CblasColMajor,
                (CBLAS_SIDE)side, (CBLAS_UPLO)uplo, (CBLAS_TRANSPOSE)trans, (CBLAS_DIAG)diag,
                M, N, CBLAS_SADDR(alpha), A, LDA, B, LDB);

    Blapacknorm = LAPACKE_zlange_work(LAPACK_COL_MAJOR, 'i', M, N, B, LDB, work);

    cblas_zaxpy(LDB * N, CBLAS_SADDR(mzone), X, 1, B, 1);
    Rnorm = LAPACKE_zlange_work(LAPACK_COL_MAJOR, 'm', M, N, B, LDB, work);

    if ( loud > 2 )
        printf("Rnorm %e, Anorm %e, Binitnorm %e, Bdaguenorm %e, Blapacknorm %e\n",
               Rnorm, Anorm, Binitnorm, Bdaguenorm, Blapacknorm);

    result = Rnorm / ((Anorm + Blapacknorm) * max(M,N) * eps);
    if (  isinf(Blapacknorm) || isinf(Bdaguenorm) || isnan(result) || isinf(result) || (result > 10.0) ) {
        info_solution = 1;
    }
    else{
        info_solution = 0;
    }

    free(work);
    free(A);
    free(B);
    free(X);

    return info_solution;
}
