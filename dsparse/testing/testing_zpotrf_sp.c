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
//#include "data_dist/sparse-matrix/pastix_internal/pastix_internal.h"

//#define DUMP_SOLV 0x2

int main(int argc, char ** argv)
{
    dague_context_t* dague;
    sparse_context_t dspctxt;
    int   iparam[IPARAM_SIZEOF];
    char *sparam[SPARAM_SIZEOF];
    DagDouble_t flops, gflops;
    struct timeval start, bench, realend, realtime, benchtime;

    /* Set defaults for non argv iparams/sparam */
    param_default(iparam, sparam);

    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam, sparam);
    if( dague->nb_nodes != 1 ) {
        fprintf(stderr, 
                "Check that the absorbant property of the JDF refer only local tiles\n"
                "and remove this warning in %s at line %d...\n",
                __FILE__, __LINE__);
        cleanup_dague(dague, iparam, sparam);
        return 1;
    }
    PASTE_CODE_IPARAM_LOCALS(iparam);

    /* initializing matrix structure */
    PASTE_CODE_INIT_CONTEXT( dspctxt, factotype );

    /* Initialize the descriptor */
    sparse_matrix_desc_t ddescA;
    sparse_matrix_init( &ddescA, spmtx_ComplexDouble, nodes, cores, rank );
    dspctxt.desc = &ddescA;

    /* Read the matrix files */
    flops = sparse_matrix_zrdmtx( &dspctxt );
    
    /* Initialize the matrix */
    dsparse_zcsc2cblk( dague, &ddescA );
/*     D_Udump_all( &(ddescA.pastix_data->solvmatr), DUMP_SOLV ); */

    if ( loud && rank == 0 ) {
      printf("Number of floating points operations: %g GFLOPs\n", flops/1.e9);
    }

    switch ( factotype ) {
    case DSPARSE_LLT:
      if(loud > 2) printf("+++ Computing potrf ... ");
      PASTE_CODE_ENQUEUE_KERNEL( dague, zpotrf_sp,
                                 ((sparse_matrix_desc_t*)&ddescA) );
      PASTE_CODE_PROGRESS_KERNEL( dague, zpotrf_sp );
      
      dsparse_zpotrf_sp_Destruct( DAGUE_zpotrf_sp );
      break;

    case DSPARSE_LDLT:
      if(loud > 2) printf("+++ Computing sytrf ... ");
      PASTE_CODE_ENQUEUE_KERNEL( dague, zsytrf_sp,
                                 ((sparse_matrix_desc_t*)&ddescA) );
      PASTE_CODE_PROGRESS_KERNEL( dague, zsytrf_sp );
      
      dsparse_zsytrf_sp_Destruct( DAGUE_zsytrf_sp );
      break;

    case DSPARSE_LDLTH:
      if(loud > 2) printf("+++ Computing hetrf ... ");
      PASTE_CODE_ENQUEUE_KERNEL( dague, zhetrf_sp,
                                 ((sparse_matrix_desc_t*)&ddescA) );
      PASTE_CODE_PROGRESS_KERNEL( dague, zhetrf_sp );
      
      dsparse_zhetrf_sp_Destruct( DAGUE_zhetrf_sp );
      break;

    case DSPARSE_LU:
    default:
      if(loud > 2) printf("+++ Computing getrf ... ");
      PASTE_CODE_ENQUEUE_KERNEL( dague, zgetrf_sp,
                                 ((sparse_matrix_desc_t*)&ddescA) );
      PASTE_CODE_PROGRESS_KERNEL( dague, zgetrf_sp );
      
      dsparse_zgetrf_sp_Destruct( DAGUE_zgetrf_sp );
    }
    
    if(loud > 2) printf("Done.\n");
        
    cleanup_dague(dague, iparam, sparam);
    
/*     D_Udump_all( &(ddescA.pastix_data->solvmatr), DUMP_SOLV ); */
    if( check )
        sparse_matrix_zcheck( &dspctxt );
    sparse_matrix_zclean( &dspctxt );
    
    sparse_matrix_destroy( (sparse_matrix_desc_t*)&ddescA );

    return EXIT_SUCCESS;
}
