/**
 *
 * @file sparse-matrix.c
 *
 * @author Mathieu Faverge 
 * @author Pierre Ramet
 * @date 2011-10-17
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

//typedef void FLOAT;
#include "pastix_internal.h"
#include "data_dist/sparse-matrix/sparse-matrix.h"

static int sparse_vector_size_of(enum spmtx_type type)
{
    switch ( type ) {
    case spmtx_RealFloat:
        return sizeof(float);
    case spmtx_RealDouble:
        return sizeof(double);
    case spmtx_ComplexFloat:
        return sizeof(dague_complex32_t);
    case spmtx_ComplexDouble:
        return sizeof(dague_complex64_t);
    default:
        return sizeof(float);
    }
}

uint32_t sparse_vector_rank_of(struct dague_ddesc *mat, ... )
{
    (void)mat;
    return 0;
}

void *sparse_vector_data_of(struct dague_ddesc *mat, ... )
{
    va_list ap;
    dague_int_t cblknum, bloknum;
    sparse_vector_desc_t *spmtx = (sparse_vector_desc_t*)mat;

    va_start(ap, mat);
    cblknum = va_arg(ap, unsigned int);
    va_end(ap);
    bloknum = spmtx->pastix_data->solvmatr.symbmtx.cblktab[cblknum].bloknum;

    char *ptr = (char*)(spmtx->pastix_data->solvmatr.updovct.sm2xtab);

    return ptr + (size_t)(spmtx->pastix_data->solvmatr.updovct.cblktab[cblknum].sm2xind)
        *        (size_t)(spmtx->typesze);
}

#ifdef DAGUE_PROF_TRACE
uint32_t sparse_vector_data_key(struct dague_ddesc *mat, ... )
{
    sparse_vector_desc_t *spmtx = (sparse_vector_desc_t*)mat;
    va_list ap;
    dague_int_t cblknum, bloknum;
    
    va_start(ap, mat);
    cblknum = va_arg(ap, unsigned int);
    va_end(ap);
    bloknum = spmtx->pastix_data->solvmatr.symbmtx.cblktab[cblknum].bloknum;

    return (uint32_t)bloknum;
}

int sparse_vector_key_to_string(struct dague_ddesc *mat, uint32_t datakey, char *buffer, uint32_t buffer_size)
{
    sparse_vector_desc_t *spmtx = (sparse_vector_desc_t*)mat;
    dague_int_t bloknum, cblknum;
    dague_int_t first, last, middle;
    int res;
    
    first   = 0;
    last    = spmtx->pastix_data->solvmatr.symbmtx.cblknbr;
    middle  = (last+first) / 2;
    bloknum = (dague_int_t)datakey;
    cblknum = -1;

    while( last - first > 0 ) {
        if ( bloknum >= spmtx->pastix_data->solvmatr.symbmtx.cblktab[middle].bloknum ) {
            if ( bloknum < spmtx->pastix_data->solvmatr.symbmtx.cblktab[middle+1].bloknum ) {
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

void sparse_vector_init( sparse_vector_desc_t *desc, 
                         enum spmtx_type mtype, 
                         int nodes, int cores, int myrank)
{
    /* dague_ddesc structure */
    desc->super.nodes   = nodes;
    desc->super.cores   = cores;
    desc->super.myrank  = myrank;
    desc->super.rank_of = sparse_vector_rank_of;
    desc->super.data_of = sparse_vector_data_of;
#ifdef DAGUE_PROF_TRACE
    desc->super.data_key      = sparse_vector_data_key;
    desc->super.key_to_string = sparse_vector_key_to_string;
    desc->super.key_dim = NULL; /* Initialized when the matrix is read */
    desc->super.key     = NULL; /* Initialized when the matrix is read */
#endif /* DAGUE_PROF_TRACE */

    desc->mtype       = mtype;
    desc->typesze     = sparse_vector_size_of( mtype );
    desc->pastix_data = NULL;

/*     DEBUG(("sparse_vector_init: desc = %p, mtype = %zu, \n" */
/*            "\tnodes = %u, cores = %u, myrank = %u\n", */
/*            desc, (size_t) desc->super.mtype,  */
/*            desc->super.super.nodes,  */
/*            desc->super.super.cores, */
/*            desc->super.super.myrank)); */
}

void sparse_vector_destroy( sparse_vector_desc_t *desc )
{
    (void)desc;
}
