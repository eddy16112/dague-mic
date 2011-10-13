/**
 *
 * @file sparse-matrix.c
 *
 * @author Mathieu Faverge
 * @date 2011-03-01
 * @precisions normal z -> c d s
 *
 **/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <stdarg.h>
#include <stdint.h>

#ifdef HAVE_MPI
#include <mpi.h>
#endif /* HAVE_MPI */

#include "dague_config.h"
#include "dague.h"
#include "data_dist/matrix/precision.h"
#include "data_dist/sparse-matrix/sparse-matrix.h"

static int sparse_matrix_size_of(enum spmtx_type type)
{
    switch ( type ) {
    case spmtx_RealFloat:
        return sizeof(float);
    case spmtx_RealDouble:
        return sizeof(double);
    case spmtx_ComplexFloat:
        return sizeof(Dague_Complex32_t);
    case spmtx_ComplexDouble:
        return sizeof(Dague_Complex64_t);
    default:
        return sizeof(float);
    }
}

uint32_t sparse_matrix_rank_of(struct dague_ddesc *mat, ... )
{
    return 0;
}

void *sparse_matrix_data_of(struct dague_ddesc *mat, ... )
{
    va_list ap;
    dague_int_t cblknum, bloknum;
    sparse_matrix_desc_t *spmtx = (sparse_matrix_desc_t*)mat;

    va_start(ap, mat);
    cblknum = va_arg(ap, unsigned int);
    bloknum = va_arg(ap, unsigned int);
    va_end(ap);

    return (char*)spmtx->symbmtx.cblktab[cblknum].cblkptr 
      + (size_t)(spmtx->typesze) * (size_t)(spmtx->symbmtx.bloktab[bloknum].coefind);
}

#ifdef DAGUE_PROF_TRACE
uint32_t sparse_matrix_data_key(struct dague_ddesc *mat, ... )
{
    va_list ap;
    dague_int_t cblknum, bloknum;
    
    va_start(ap, mat);
    cblknum = va_arg(ap, unsigned int);
    bloknum = va_arg(ap, unsigned int);
    va_end(ap);
    (void)cblknum;
    return (uint32_t)bloknum;
}

int sparse_matrix_key_to_string(struct dague_ddesc *mat, uint32_t datakey, char *buffer, uint32_t buffer_size)
{
    sparse_matrix_desc_t *spmtx = (sparse_matrix_desc_t*)mat;
    dague_int_t bloknum, cblknum;
    dague_int_t first, last, middle;
    
    first   = 0;
    last    = spmtx.symbmtx.cblknbr;
    middle  = (last+first) / 2;
    bloknum = (dague_int_t)datakey;
    cblknum = -1;

    while( last - first > 0 ) {
        if ( bloknum >= spmtx.symbmtx.cblktab[middle].bloknum ) {
            if ( bloknum < spmtx.symbmtx.cblktab[middle+1].bloknum ) {
                cblknum = middle;
                break;
            }
            first = middle;
        }
        else {
            last = middle;
        }
        middle = (last+first) / 2;
    }

    res = snprintf(buffer, buffer_size, "(%ld, %ld)", 
                   (long int)cblknum, (long int)bloknum);
    if (res < 0)
    {
        printf("error in key_to_string for tile (%ld, %ld) key: %u\n", 
               (long int)cblknum, (long int)bloknum, datakey);
    }
    return res;
}
#endif

void sparse_matrix_init( sparse_matrix_desc_t *desc, 
                         enum spmtx_type mtype, 
                         int nodes, int cores, int myrank)
{
    /* dague_ddesc structure */
    desc->super.nodes   = nodes;
    desc->super.cores   = cores;
    desc->super.myrank  = myrank;
    desc->super.rank_of = sparse_matrix_rank_of;
    desc->super.data_of = sparse_matrix_data_of;
#ifdef DAGUE_PROF_TRACE
    desc->super.data_key      = sparse_matrix_data_key;
    desc->super.key_to_string = sparse_matrix_key_to_string;
    desc->super.key    = NULL; /* Initialized when the matrix is read */
    desc->super.keydim = NULL; /* Initialized when the matrix is read */
#endif /* DAGUE_PROF_TRACE */

    desc->mtype   = mtype;
    desc->typesze = sparse_matrix_size_of( mtype );

    DEBUG(("sparse_matrix_init: desc = %p, mtype = %zu, \n"
           "\tnodes = %u, cores = %u, myrank = %u\n",
           desc, (size_t) desc->super.mtype, 
           desc->super.super.nodes, 
           desc->super.super.cores,
           desc->super.super.myrank));
}
