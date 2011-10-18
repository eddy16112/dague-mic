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

int main(int argc, char ** argv)
{
    dague_context_t* dague;
    sparse_context_t dspctxt;
    int   iparam[IPARAM_SIZEOF];
    char *sparam[SPARAM_SIZEOF];
    double flops;
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
    //int info = 0;
    PASTE_CODE_INIT_CONTEXT( dspctxt, DSPARSE_LDLT );

    /* Initialize the descriptor */
    sparse_matrix_desc_t ddescA;
    sparse_matrix_init( &ddescA, spmtx_ComplexDouble, nodes, cores, rank );

    /* Read the matrix files */
    sparse_matrix_zrdmtx( &dspctxt );

    /* compute the number of flops */
    /* flops = dsparse_zpotrf_sp_flops_count( &ddescA ); */
    /* if ( loud && rank == 0 ) */
    /*   printf("Number of floating points operations: %g GFLOPs\n", flops/1.e9); */
        
    /* if(loud > 2) printf("+++ Computing potrf ... "); */
    /* PASTE_CODE_ENQUEUE_KERNEL(dague, zpotrf_sp,  */
    /*                           ((sparse_matrix_desc_t*)&ddescA)); */
    /* PASTE_CODE_PROGRESS_KERNEL(dague, zpotrf_sp); */

    /* dsparse_zpotrf_sp_Destruct( DAGUE_zpotrf_sp ); */
    /* if(loud > 2) printf("Done.\n"); */

    cleanup_dague(dague, iparam, sparam);

    sparse_matrix_destroy( (sparse_matrix_desc_t*)&ddescA );

    return EXIT_SUCCESS;
}
