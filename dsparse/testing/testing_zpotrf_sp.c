/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "common.h"
#include "data_dist/sparse-matrix/sparse-matrix.h"

#if defined(HAVE_CUDA)
#include <cublas.h>
#include "dsparse/cores/cuda_sparse.h"
#include "dsparse/cores/cuda_zpotrfsp_gemm.h"
#endif

int main(int argc, char ** argv)
{
    dague_context_t* dague;
    sparse_context_t dspctxt;
    int   iparam[IPARAM_SIZEOF];
    char *sparam[SPARAM_SIZEOF];
    DagDouble_t flops, gflops;
    dague_complex64_t *rhssaved = NULL;
#if defined(HAVE_CUDA) && defined(PRECISION_s)
    iparam[IPARAM_NGPUS] = 0;
#endif

    /* Set defaults for non argv iparams/sparam */
    param_default(iparam, sparam); /* setup default params (e.g., nodes = 1) *
								    * see dsparse/testing/common.c           */

    /* Initialize DAGuE */
	/* setup DAGue (e.g., dague_init(iparam[IPARAM_NCORES], &argc, &argv)) *
     * see dsparse/testing/common.c                                        */
    dague = setup_dague(argc, argv, iparam, sparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);

    if( nodes != 1 ) {
        fprintf(stderr, 
                "Check that the absorbant property of the JDF refer only local tiles\n"
                "and remove this warning in %s at line %d...\n",
                __FILE__, __LINE__);
        cleanup_dague(dague, iparam, sparam);
        return 1;
    }
    /* initializing matrix structure */
    PASTE_CODE_INIT_CONTEXT( dspctxt, factotype ); /* init matrix data structure (e.g., _dspctxt.colptr = NULL; ) *
													* see dsparse/testing/common.h                                */

    /* Initialize the descriptor */
    sparse_matrix_desc_t ddescA;
    sparse_matrix_init( &ddescA, spmtx_ComplexDouble, nodes, cores, rank );
    dspctxt.desc = &ddescA;      /* setup matrix parameters (e.g., desc->super.cores   = cores) *
								  * see data_dist/sparse-matrix/sparse-matrix.c                 */

    sparse_vector_desc_t ddescB;
    sparse_vector_init( &ddescB, spmtx_ComplexDouble, nodes, cores, rank );
    dspctxt.rhsdesc = &ddescB;

    /* Read the matrix files */
	/* 1) call PaStIx for API_TASK_ORDERING, ..,  API_TASK_ANALYSE; *
     * 2) allocate space to store factors                           */
    flops = sparse_matrix_zrdmtx( &dspctxt );
    
    if ( check ) {
        rhssaved = malloc(dspctxt.n * sizeof(dague_complex64_t));
        memcpy(rhssaved, dspctxt.rhs, dspctxt.n * sizeof(dague_complex64_t));
    }

    /* load the GPU kernel */
#if defined(HAVE_CUDA)
    if(iparam[IPARAM_NGPUS] > 0) {
        if(loud) printf("+++ Load GPU kernel ... ");
        if(0 != gpu_kernel_init_zpotrfsp_gemm(dague)) {
            fprintf(stderr, "XXX Unable to load GPU kernel.\n");
            exit(3);
        }
        int cblknbr = sparse_register_bloktab( dague, 
                                               (sparse_matrix_desc_t*)&ddescA);
        dague_gpu_data_register(dague,
                                (dague_ddesc_t*)&ddescA,
                                cblknbr, 
                                sizeof(dague_complex64_t) );
        if(loud) printf("Done\n");
    }
#endif

    /* Initialize the matrix */
	/* call sparse_matrix_zcsc2cblk for each block column *
	 * DAGUe is used to setup columns in parallel         *
	 * solvmatr->cscmtx -> descA->pastix_data->solvmatr   */
    dsparse_zcsc2cblk( dague, &ddescA );

    if ( loud && rank == 0 ) {
      printf("Number of floating points operations: %g GFLOPs\n", flops/1.e9);
    }

    switch ( factotype ) {
    case DSPARSE_LLT:
      if(loud > 2) printf("+++ Computing potrf ... ");
      PASTE_CODE_ENQUEUE_KERNEL( dague, zpotrf_sp, (&ddescA) );
      PASTE_CODE_PROGRESS_KERNEL( dague, zpotrf_sp );
      
      dsparse_zpotrf_sp_Destruct( DAGUE_zpotrf_sp );

      break;

    case DSPARSE_LDLT:
#if defined(PRECISION_z) || defined(PRECISION_c)
      if(loud > 2) printf("+++ Computing sytrf ... ");
      PASTE_CODE_ENQUEUE_KERNEL( dague, zsytrf_sp, (&ddescA) );
      PASTE_CODE_PROGRESS_KERNEL( dague, zsytrf_sp );
      
      dsparse_zsytrf_sp_Destruct( DAGUE_zsytrf_sp );
      break;
#endif

    case DSPARSE_LDLTH:
      if(loud > 2) printf("+++ Computing hetrf ... ");
      PASTE_CODE_ENQUEUE_KERNEL( dague, zhetrf_sp, (&ddescA) );
      PASTE_CODE_PROGRESS_KERNEL( dague, zhetrf_sp );
      
      dsparse_zhetrf_sp_Destruct( DAGUE_zhetrf_sp );
      break;

    case DSPARSE_LU:
    default:
      if(loud > 2) printf("+++ Computing getrf ... ");
      PASTE_CODE_ENQUEUE_KERNEL( dague, zgetrf_sp, (&ddescA) );
      PASTE_CODE_PROGRESS_KERNEL( dague, zgetrf_sp );
      
      dsparse_zgetrf_sp_Destruct( DAGUE_zgetrf_sp );
    }
    
    if(loud > 2) printf("Done.\n");
        
    /* dsparse_zdumpmat( dague, &ddescA ); */

    if( check ) {
#if defined(DSPARSE_WITH_SOLVE)
        switch ( factotype ) {
        case DSPARSE_LLT:
            if(loud > 2) printf("+++ Computing potrs ... ");
            sparse_vector_zinit( &dspctxt );
            dsparse_zpotrs_sp( dague, &ddescA, &ddescB);
            sparse_vector_zfinalize( &dspctxt );
        default:
            (void)0;
        }
#endif
        sparse_matrix_zcheck( &dspctxt );
        
        {
            int i, j, ncol = dspctxt.n;
            dague_complex64_t *ax     = malloc(ncol*sizeof(dague_complex64_t));
            dague_complex64_t *values = (dague_complex64_t*)dspctxt.values;
            dague_complex64_t *rhs    = (dague_complex64_t*)dspctxt.rhs;
            double norm1, norm2;

            memset(ax, 0, ncol*sizeof(dague_complex64_t));
            for (i= 0; i < ncol; i++)
                {
                    for (j = dspctxt.colptr[i]-1; j < dspctxt.colptr[i+1] - 1; j++)
                        {
                            ax[ dspctxt.rows[j]-1 ] += values[j] * rhs[i];
                            if ((MTX_ISSYM(dspctxt.type) == 1) && (i != (dspctxt.rows[j]-1)))
                                {
                                    ax[i] += values[j] * rhs[ dspctxt.rows[j] - 1 ];
                                }
                        }
                }
            norm1 = 0.;
            norm2 = 0.;
            for (i= 0; i < ncol; i++)
                {
                    norm1 += (double)( (ax[i] - rhssaved[i]) * conj(ax[i] - rhssaved[i]) );
                    norm2 += (double)( rhssaved[i] * conj(rhssaved[i]) );
                }
            
            fprintf(stdout, "Precision : ||ax-b||/||b|| = %.20lg\n", sqrt(norm1/norm2));
            free(ax);
            free(rhssaved);
        }
    }
    sparse_matrix_zclean( &dspctxt );
    
#if defined(HAVE_CUDA)
    if(iparam[IPARAM_NGPUS] > 0) {
        dague_gpu_data_unregister((dague_ddesc_t*)&ddescA);
        sparse_unregister_bloktab(dague, 
                                  (sparse_matrix_desc_t*)&ddescA);
        dague_gpu_kernel_fini(dague, "gemm");
    }
#endif

    sparse_matrix_destroy( &ddescA );
    sparse_vector_destroy( &ddescB );

    cleanup_dague(dague, iparam, sparam);
    
    return EXIT_SUCCESS;
}
