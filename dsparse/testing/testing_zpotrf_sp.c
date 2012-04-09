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

int sparse_sgemm_cuda_init( dague_context_t* context, sparse_matrix_desc_t *tileA );
int sparse_sgemm_cuda_fini( dague_context_t* dague_context );

int main(int argc, char ** argv)
{
    dague_context_t* dague;
    sparse_context_t dspctxt;
    int   iparam[IPARAM_SIZEOF];
    char *sparam[SPARAM_SIZEOF];
    DagDouble_t flops, gflops;
    Dague_Complex64_t *rhssaved = NULL;
#if defined(HAVE_CUDA) && defined(PRECISION_s)
    iparam[IPARAM_NGPUS] = 0;
#endif

    /* Set defaults for non argv iparams/sparam */
    param_default(iparam, sparam);

    /* Initialize DAGuE */
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
    PASTE_CODE_INIT_CONTEXT( dspctxt, factotype );

    /* Initialize the descriptor */
    sparse_matrix_desc_t ddescA;
    sparse_matrix_init( &ddescA, spmtx_ComplexDouble, nodes, cores, rank );
    dspctxt.desc = &ddescA;

    sparse_vector_desc_t ddescB;
    sparse_vector_init( &ddescB, spmtx_ComplexDouble, nodes, cores, rank );
    dspctxt.rhsdesc = &ddescB;

    /* Read the matrix files */
    flops = sparse_matrix_zrdmtx( &dspctxt );
    
    if ( check ) {
        rhssaved = malloc(dspctxt.n * sizeof(Dague_Complex64_t));
        memcpy(rhssaved, dspctxt.rhs, dspctxt.n * sizeof(Dague_Complex64_t));
    }

    /* load the GPU kernel */
#if defined(HAVE_CUDA) && defined(PRECISION_s)
    if(iparam[IPARAM_NGPUS] > 0)
        {
            if(loud) printf("+++ Load GPU kernel ... ");
            if(0 != sparse_sgemm_cuda_init(dague, &ddescA))
                {
                    fprintf(stderr, "XXX Unable to load GPU kernel.\n");
                    exit(3);
                }
            if(loud) printf("Done\n");
        }
#endif

    /* Initialize the matrix */
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
            Dague_Complex64_t *ax     = malloc(ncol*sizeof(Dague_Complex64_t));
            Dague_Complex64_t *values = (Dague_Complex64_t*)dspctxt.values;
            Dague_Complex64_t *rhs    = (Dague_Complex64_t*)dspctxt.rhs;
            double norm1, norm2;

            memset(ax, 0, ncol*sizeof(Dague_Complex64_t));
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
    
    sparse_matrix_destroy( &ddescA );
    sparse_vector_destroy( &ddescB );

#if defined(HAVE_CUDA) && defined(PRECISION_s)
    if(iparam[IPARAM_NGPUS] > 0) {
        sparse_sgemm_cuda_fini(dague);
    }
#endif

    cleanup_dague(dague, iparam, sparam);
    
    return EXIT_SUCCESS;
}
