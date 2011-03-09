/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "common.h"
#include "data_dist/matrix/precision.h"
#include "data_dist/sparse-matrix/sparse-shm-matrix.h"
#include "data_dist/sparse-matrix/sparse-input.h"
#include <pastix.h>
#include <read_matrix.h>

/* #if defined(HAVE_CUDA) && defined(PRECISION_s) */
/* #include "cuda_sgemm.h" */
/* #endif */

/* #define FMULS_POTRF(N) ((N) * (1.0 / 6.0 * (N) + 0.5) * (N)) */
/* #define FADDS_POTRF(N) ((N) * (1.0 / 6.0 * (N)      ) * (N)) */

/* #define FMULS_POTRS(N, NRHS) ( (NRHS) * ( (N) * ((N) + 1.) ) ) */
/* #define FADDS_POTRS(N, NRHS) ( (NRHS) * ( (N) * ((N) - 1.) ) ) */

/* static int check_solution( dague_context_t *dague, PLASMA_enum uplo,  */
/*                            tiled_matrix_desc_t *ddescA, tiled_matrix_desc_t *ddescB, tiled_matrix_desc_t *ddescX ); */

int main(int argc, char ** argv)
{
    dague_context_t* dague;
    dsp_context_t    dspctxt;
    int iparam[IPARAM_SIZEOF];

    /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 0, 180, 180);

    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);

    /* initializing matrix structure */
    //int info = 0;

    dspctxt.format = RSA;     /* Matrix file format                         */
    dspctxt.factotype = SPARSE_LLT;
    dspctxt.matrixname = "/tmp/rsaname"; /* Filename to get the matrix                 */
    dspctxt.ordername = "/tmp/ordername";  /* Filename where the ordering is stored      */
    dspctxt.symbname = "/tmp/symbname";   /* Filename where the symbol matrix is stored */
    dspctxt.type    = NULL;       /* Type of the matrix                         */
    dspctxt.rhstype = NULL;    /* Type of the RHS                            */
    dspctxt.n       = 0;          /* Number of unknowns/columns/rows            */
    dspctxt.nnz     = 0;        /* Number of non-zero values in the input matrix */
    dspctxt.colptr  = NULL; /* Vector of size N+1 storing the starting point of each column in the array rows */
    dspctxt.rows    = NULL; /* Indices of the rows present in each column */
    dspctxt.values  = NULL; /* Values of the matrix                       */
    dspctxt.rhs     = NULL; /* Right Hand Side                            */ 
    dspctxt.permtab = NULL; /* vector of permutation                      */
    dspctxt.peritab = NULL; /* vector of inverse permutation              */
    dspctxt.symbmtx = NULL; /* Pointer to symbol matrix structure, filled in by zrdmtx  */

    dague_sparse_zrdmtx( &dspctxt );
    dague_tssm_init(cores, MB*NB*sizeof(Dague_Complex64_t), 64);

    /* Initialize the descriptor */
    dague_tssm_desc_t ddescA;
    dague_tssm_zmatrix_init(&ddescA, matrix_ComplexDouble, 
                            cores, MB, NB, dspctxt.n, dspctxt.n, 0, 0, 
                            dspctxt.n, dspctxt.n, dspctxt.symbmtx);

    dplasma_zpotrf_sp(dague, &ddescA);

    dague_tssm_flush_matrix((dague_ddesc_t *)&ddescA);

#if 0
    if(!check) 
    {
        PLASMA_enum uplo = PlasmaLower;
        PASTE_CODE_FLOPS_COUNT(FADDS_POTRF, FMULS_POTRF, ((DagDouble_t)N));

        /* matrix generation */
        if(loud > 2) printf("+++ Generate matrices ... ");
        generate_tiled_random_sym_pos_mat((tiled_matrix_desc_t *) &ddescA, 100);
        if(loud > 2) printf("Done\n");

#if defined(LLT_LL)
        PASTE_CODE_ENQUEUE_KERNEL(dague, zpotrf_ll, 
                                  (uplo, (tiled_matrix_desc_t*)&ddescA, &info));
        PASTE_CODE_PROGRESS_KERNEL(dague, zpotrf_ll);
#else
        PASTE_CODE_ENQUEUE_KERNEL(dague, zpotrf, 
                                  (uplo, (tiled_matrix_desc_t*)&ddescA, &info));
        PASTE_CODE_PROGRESS_KERNEL(dague, zpotrf);

        dplasma_zpotrf_Destruct( DAGUE_zpotrf );
#endif
    }
    else 
    {
        int u, t1, t2;
        int info_solution;

       PASTE_CODE_ALLOCATE_MATRIX(ddescA0, 1, 
          two_dim_block_cyclic, (&ddescA0, matrix_ComplexDouble, 
                                 nodes, cores, rank, MB, NB, LDA, N, 0, 0, 
                                 N, N, SMB, SNB, P));
       
       PASTE_CODE_ALLOCATE_MATRIX(ddescB, 1, 
            two_dim_block_cyclic, (&ddescB, matrix_ComplexDouble, 
                                   nodes, cores, rank, MB, NB, LDB, NRHS, 0, 0, 
                                   N, NRHS, SMB, SNB, P));

        PASTE_CODE_ALLOCATE_MATRIX(ddescX, 1, 
            two_dim_block_cyclic, (&ddescX, matrix_ComplexDouble, 
                                   nodes, cores, rank, MB, NB, LDB, NRHS, 0, 0, 
                                   N, NRHS, SMB, SNB, P));
        
        for ( u=0; u<2; u++) {
            if ( uplo[u] == PlasmaUpper ) {
                t1 = PlasmaConjTrans; t2 = PlasmaNoTrans;
            } else {
                t1 = PlasmaNoTrans; t2 = PlasmaconjTrans;
            }   

            /*********************************************************************
             *               First Check
             */
            if ( rank == 0 ) {
                printf("***************************************************\n");
            }

            /* matrix generation */
            printf("Generate matrices ... ");
            generate_tiled_random_sym_pos_mat((tiled_matrix_desc_t *) &ddescA,  400);
            generate_tiled_random_sym_pos_mat((tiled_matrix_desc_t *) &ddescA0, 400);
            generate_tiled_random_mat((tiled_matrix_desc_t *) &ddescB, 200);
            generate_tiled_random_mat((tiled_matrix_desc_t *) &ddescX, 200);
            printf("Done\n");


            /* Compute */
            printf("Compute ... ... ");
            info = dplasma_zposv(dague, uplo[u], (tiled_matrix_desc_t *)&ddescA, (tiled_matrix_desc_t *)&ddescB );
            printf("Done\n");
            printf("Info = %d\n", info);

            /* Check the solution */
            info_solution = check_solution( dague, uplo[u], (tiled_matrix_desc_t *)&ddescA0, (tiled_matrix_desc_t *)&ddescB, (tiled_matrix_desc_t *)&ddescX);

            if ( rank == 0 ) {
                if (info_solution == 0) {
                    printf(" ----- TESTING ZPOSV (%s) ....... PASSED !\n", uplostr[u]);
                }
                else {
                    printf(" ----- TESTING ZPOSV (%s) ... FAILED !\n", uplostr[u]);
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
            generate_tiled_random_sym_pos_mat((tiled_matrix_desc_t *) &ddescA,  400);
            generate_tiled_random_sym_pos_mat((tiled_matrix_desc_t *) &ddescA0, 400);
            generate_tiled_random_mat((tiled_matrix_desc_t *) &ddescB, 200);
            generate_tiled_random_mat((tiled_matrix_desc_t *) &ddescX, 200);
            printf("Done\n");


            /* Compute */
            printf("Compute ... ... ");
            info = dplasma_zpotrf(dague, uplo[u], (tiled_matrix_desc_t *)&ddescA );
            if ( info == 0 ) {
                dplasma_zpotrs(dague, uplo[u], (tiled_matrix_desc_t *)&ddescA, (tiled_matrix_desc_t *)&ddescX );
            }
            printf("Done\n");
            printf("Info = %d\n", info);

            /* Check the solution */
            info_solution = check_solution( dague, uplo[u], (tiled_matrix_desc_t *)&ddescA0, (tiled_matrix_desc_t *)&ddescB, (tiled_matrix_desc_t *)&ddescX);

            if ( rank == 0 ) {
                if (info_solution == 0) {
                    printf(" ----- TESTING ZPOTRF + ZPOTRS (%s) ....... PASSED !\n", uplostr[u]);
                }
                else {
                    printf(" ----- TESTING ZPOTRF + ZPOTRS (%s) ... FAILED !\n", uplostr[u]);
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
            generate_tiled_random_sym_pos_mat((tiled_matrix_desc_t *) &ddescA,  400);
            generate_tiled_random_sym_pos_mat((tiled_matrix_desc_t *) &ddescA0, 400);
            generate_tiled_random_mat((tiled_matrix_desc_t *) &ddescB, 200);
            generate_tiled_random_mat((tiled_matrix_desc_t *) &ddescX, 200);
            printf("Done\n");


            /* Compute */
            printf("Compute ... ... ");
            info = dplasma_zpotrf(dague, uplo[u], (tiled_matrix_desc_t *)&ddescA );
            if ( info == 0 ) {
                dplasma_ztrsm(dague, PlasmaLeft, uplo[u], t1, PlasmaNonUnit, 1.0, (tiled_matrix_desc_t *)&ddescA, (tiled_matrix_desc_t *)&ddescX);
                dplasma_ztrsm(dague, PlasmaLeft, uplo[u], t2, PlasmaNonUnit, 1.0, (tiled_matrix_desc_t *)&ddescA, (tiled_matrix_desc_t *)&ddescX);
            }
            printf("Done\n");
            printf("Info = %d\n", info);

            /* Check the solution */
            info_solution = check_solution( dague, uplo[u], (tiled_matrix_desc_t *)&ddescA0, (tiled_matrix_desc_t *)&ddescB, (tiled_matrix_desc_t *)&ddescX);

            if ( rank == 0 ) {
                if (info_solution == 0) {
                    printf(" ----- TESTING ZPOTRF + ZTRSM + ZTRSM (%s) ....... PASSED !\n", uplostr[u]);
                }
                else {
                    printf(" ----- TESTING ZPOTRF + ZTRSM + ZTRSM (%s) ... FAILED !\n", uplostr[u]);
                }
                printf("***************************************************\n");
            }

        }

        dague_data_free(ddescA0.mat);
        dague_ddesc_destroy( (dague_ddesc_t*)&ddescA0);
        dague_data_free(ddescB.mat);
        dague_ddesc_destroy( (dague_ddesc_t*)&ddescB);
        dague_data_free(ddescX.mat);
        dague_ddesc_destroy( (dague_ddesc_t*)&ddescX);
    }

#endif

    cleanup_dague(dague);

    //dague_data_free(ddescA.mat);
    //dague_ddesc_destroy( (dague_ddesc_t*)&ddescA);

    return EXIT_SUCCESS;
}

#if 0

static int check_solution( dague_context_t *dague, PLASMA_enum uplo, 
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
    Anorm = LAPACKE_zlanhe_work( LAPACK_COL_MAJOR, 'i', lapack_const(uplo), N, W, N, work );

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
#endif
