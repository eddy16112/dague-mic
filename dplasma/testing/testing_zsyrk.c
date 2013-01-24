/*
 * Copyright (c) 2009-2011 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> z c
 *
 */

#include "common.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "data_dist/matrix/sym_two_dim_rectangle_cyclic.h"

static int check_solution( dague_context_t *dague, int loud,
                           PLASMA_enum uplo, PLASMA_enum trans,
                           dague_complex64_t alpha, int Am, int An, int Aseed,
                           dague_complex64_t beta,  int M,  int N,  int Cseed,
                           sym_two_dim_block_cyclic_t *ddescCfinal );

int main(int argc, char ** argv)
{
    dague_context_t* dague;
    int iparam[IPARAM_SIZEOF];
    int ret = 0;
    int Aseed = 3872;
    int Cseed = 2873;
    dague_complex64_t alpha =  3.5 - I * 4.2;
    dague_complex64_t beta  = -2.8 + I * 0.7;

    /* Set defaults for non argv iparams */
    iparam_default_gemm(iparam);
    iparam_default_ibnbmb(iparam, 0, 200, 200);
    iparam[IPARAM_NGPUS] = 0;

    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);

    M = N;
    LDC = max(LDC, N);

    PASTE_CODE_ALLOCATE_MATRIX(ddescC2, check,
        two_dim_block_cyclic, (&ddescC2, matrix_ComplexDouble, matrix_Tile,
                               nodes, cores, rank, MB, NB, LDC, N, 0, 0,
                               N, N, SMB, SNB, P));

    if (loud > 2) printf("Generate matrices ... ");
    dplasma_zplrnt( dague, (tiled_matrix_desc_t *)&ddescC2, Cseed);
    if (loud > 2) printf("Done\n");

    if(!check)
    {
        PLASMA_enum uplo  = PlasmaLower;
        PLASMA_enum trans = PlasmaNoTrans;
        int Am = ( trans == PlasmaNoTrans ? N : K );
        int An = ( trans == PlasmaNoTrans ? K : N );
        LDA = max(LDA, Am);

        PASTE_CODE_FLOPS(FLOPS_ZSYRK, ((DagDouble_t)K, (DagDouble_t)N));

        PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
            two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, matrix_Tile,
                                   nodes, cores, rank, MB, NB, LDA, An, 0, 0,
                                   Am, An, SMB, SNB, P));

        PASTE_CODE_ALLOCATE_MATRIX(ddescC, 1,
            sym_two_dim_block_cyclic, (&ddescC, matrix_ComplexDouble,
                                       nodes, cores, rank, MB, NB, LDC, N, 0, 0,
                                       N, N, P, uplo));

        /* matrix generation */
        if(loud > 2) printf("+++ Generate matrices ... ");
        dplasma_zplrnt( dague, (tiled_matrix_desc_t *)&ddescA,  Aseed);
        dplasma_zplghe( dague, 0., uplo, (tiled_matrix_desc_t *)&ddescC, Cseed);
        if(loud > 2) printf("Done\n");

        /* Create DAGuE */
        PASTE_CODE_ENQUEUE_KERNEL(dague, zsyrk,
                                  (uplo, trans,
                                   alpha, (tiled_matrix_desc_t *)&ddescA,
                                   beta,  (tiled_matrix_desc_t *)&ddescC));

        /* lets rock! */
        PASTE_CODE_PROGRESS_KERNEL(dague, zsyrk);

        dplasma_zsyrk_Destruct( DAGUE_zsyrk );

        dague_data_free(ddescA.mat);
        dague_ddesc_destroy((dague_ddesc_t*)&ddescA);
        dague_data_free(ddescC.mat);
        dague_ddesc_destroy((dague_ddesc_t*)&ddescC);
    }
    else
    {
        int u, t;
        int info_solution;

        for (u=0; u<2; u++) {

            PASTE_CODE_ALLOCATE_MATRIX(ddescC, 1,
                sym_two_dim_block_cyclic, (&ddescC, matrix_ComplexDouble,
                                           nodes, cores, rank, MB, NB, LDC, N, 0, 0,
                                           N, N, P, uplo[u]));

            for (t=0; t<2; t++) {

                /* initializing matrix structure */
                int Am = ( trans[t] == PlasmaNoTrans ? N : K );
                int An = ( trans[t] == PlasmaNoTrans ? K : N );
                LDA = max(LDA, Am);

                PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
                    two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, matrix_Tile,
                                           nodes, cores, rank, MB, NB, LDA, An, 0, 0,
                                           Am, An, SMB, SNB, P));

                if (loud > 2) printf("Generate matrices ... ");
                dplasma_zplrnt( dague, (tiled_matrix_desc_t *)&ddescA, Aseed);
                dplasma_zlacpy( dague, uplo[u],
                                (tiled_matrix_desc_t *)&ddescC2, (tiled_matrix_desc_t *)&ddescC );
                if (loud > 2) printf("Done\n");

                /* Compute */
                if (loud > 2) printf("Compute ... ... ");
                dplasma_zsyrk(dague, uplo[u], trans[t],
                              alpha, (tiled_matrix_desc_t *)&ddescA,
                              beta,  (tiled_matrix_desc_t *)&ddescC);
                if (loud > 2) printf("Done\n");

                /* Check the solution */
                info_solution = check_solution(dague, rank == 0 ? loud : 0,
                                               uplo[u], trans[t],
                                               alpha, Am, An, Aseed,
                                               beta,  N,  N,  Cseed,
                                               &ddescC);

                if ( rank == 0 ) {
                    if (info_solution == 0) {
                        printf(" ---- TESTING ZSYRK (%s, %s) ...... PASSED !\n",
                               uplostr[u], transstr[t]);
                    }
                    else {
                        printf(" ---- TESTING ZSYRK (%s, %s) ... FAILED !\n",
                               uplostr[u], transstr[t]);
                        ret |= 1;
                    }
                    printf("***************************************************\n");
                }

                dague_data_free(ddescA.mat);
                dague_ddesc_destroy((dague_ddesc_t*)&ddescA);
            }
            dague_data_free(ddescC.mat);
            dague_ddesc_destroy((dague_ddesc_t*)&ddescC);
        }

        dague_data_free(ddescC2.mat);
        dague_ddesc_destroy((dague_ddesc_t*)&ddescC2);
    }

    cleanup_dague(dague, iparam);

    return ret;
}


/**********************************
 * static functions
 **********************************/

/*------------------------------------------------------------------------
 *  Check the accuracy of the solution
 */
static int check_solution( dague_context_t *dague, int loud,
                           PLASMA_enum uplo, PLASMA_enum trans,
                           dague_complex64_t alpha, int Am, int An, int Aseed,
                           dague_complex64_t beta,  int M,  int N,  int Cseed,
                           sym_two_dim_block_cyclic_t *ddescCfinal )
{
    int info_solution;
    double Anorm, Cinitnorm, Cdplasmanorm, Clapacknorm, Rnorm;
    double eps, result;
    int MB = ddescCfinal->super.mb;
    int NB = ddescCfinal->super.nb;
    int LDA = (Am%MB==0) ? Am : (Am/MB+1) * MB;
    int LDC = ( M%MB==0) ? M  : ( M/MB+1) * MB;
    int cores = ddescCfinal->super.super.cores;
    int rank  = ddescCfinal->super.super.myrank;

    eps = LAPACKE_dlamch_work('e');

    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
        two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, matrix_Lapack,
                               1, cores, rank, MB, NB, LDA, An, 0, 0,
                               Am, An, 1, 1, 1));
    PASTE_CODE_ALLOCATE_MATRIX(ddescC, 1,
        two_dim_block_cyclic, (&ddescC, matrix_ComplexDouble, matrix_Lapack,
                               1, cores, rank, MB, NB, LDC, N, 0, 0,
                               M, N, 1, 1, 1));

    dplasma_zplrnt( dague, (tiled_matrix_desc_t *)&ddescA, Aseed);
    dplasma_zplrnt( dague, (tiled_matrix_desc_t *)&ddescC, Cseed );

    Anorm        = dplasma_zlange( dague, PlasmaInfNorm, (tiled_matrix_desc_t*)&ddescA );
    Cinitnorm    = dplasma_zlanhe( dague, PlasmaInfNorm, uplo, (tiled_matrix_desc_t*)&ddescC     );
    Cdplasmanorm = dplasma_zlanhe( dague, PlasmaInfNorm, uplo, (tiled_matrix_desc_t*)ddescCfinal );

    if ( rank == 0 ) {
        cblas_zsyrk(CblasColMajor,
                    (CBLAS_UPLO)uplo, (CBLAS_TRANSPOSE)trans,
                    N, (trans == PlasmaNoTrans) ? An : Am,
                    CBLAS_SADDR(alpha), ddescA.mat, LDA,
                    CBLAS_SADDR(beta),  ddescC.mat, LDC);
    }

    Clapacknorm = dplasma_zlanhe( dague, PlasmaInfNorm, uplo, (tiled_matrix_desc_t*)&ddescC );

    dplasma_zgeadd( dague, uplo, -1.0, (tiled_matrix_desc_t*)ddescCfinal,
                                       (tiled_matrix_desc_t*)&ddescC );

    Rnorm = dplasma_zlanhe( dague, PlasmaMaxNorm, uplo, (tiled_matrix_desc_t*)&ddescC );

    result = Rnorm / (Clapacknorm * max(M,N) * eps);

    if ( rank == 0 ) {
        if ( loud > 2 ) {
            printf("  ||A||_inf = %e, ||C||_inf = %e\n"
                   "  ||lapack(a*A*At+b*C)||_inf = %e, ||dplasma(a*A*At+b*C)||_inf = %e, ||R||_m = %e, res = %e\n",
                   Anorm, Cinitnorm, Clapacknorm, Cdplasmanorm, Rnorm, result);
        }

        if (  isinf(Clapacknorm) || isinf(Cdplasmanorm) ||
              isnan(result) || isinf(result) || (result > 10.0) ) {
            info_solution = 1;
        }
        else {
            info_solution = 0;
        }
    }

#if defined(HAVE_MPI)
    MPI_Bcast(&info_solution, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif

    dague_data_free(ddescA.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescA);
    dague_data_free(ddescC.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescC);

    return info_solution;
}
