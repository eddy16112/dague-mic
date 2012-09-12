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

dague_int_t sparse_matrix_get_lcblknum(sparse_matrix_desc_t *spmtx, dague_int_t bloknum )
{
    dague_int_t cblknum;
    dague_int_t first, last, middle;
    
    first   = 0;
    last    = spmtx->pastix_data->solvmatr.symbmtx.cblknbr;
    middle  = (last+first) / 2;
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
    assert( spmtx->pastix_data->solvmatr.symbmtx.cblktab[cblknum].bloknum <= bloknum 
	    && spmtx->pastix_data->solvmatr.symbmtx.cblktab[cblknum+1].bloknum > bloknum);
    return cblknum;
}

dague_int_t sparse_matrix_get_listptr_prev(sparse_matrix_desc_t *spmtx, dague_int_t bloknum, dague_int_t fcblknum ) 
{
    dague_int_t count, browfirst, browlast;
    SolverMatrix *datacode=&(spmtx->pastix_data->solvmatr);
    
    /* fprintf(stderr, "TOTO cblknum=%d, browfirst=%d, browlast=%d\n", */
    /*         cblknum,  */
    /*         UPDOWN_LISTPTR( UPDOWN_GCBLK2LIST(UPDOWN_LOC2GLOB(cblknum))  ),  */
    /*         UPDOWN_LISTPTR( UPDOWN_GCBLK2LIST(UPDOWN_LOC2GLOB(cblknum))+1)); */

    browfirst = UPDOWN_LISTPTR( UPDOWN_GCBLK2LIST(UPDOWN_LOC2GLOB(fcblknum))  );
    browlast  = UPDOWN_LISTPTR( UPDOWN_GCBLK2LIST(UPDOWN_LOC2GLOB(fcblknum))+1);
    if ( bloknum == UPDOWN_LISTBLOK( browfirst ) )
        return 0; 
    for (count=browfirst; count<browlast-1; count++)
        if (bloknum==UPDOWN_LISTBLOK(count+1)) return UPDOWN_LISTBLOK(count);
    assert(0);
    return -1;
}

dague_int_t sparse_matrix_get_listptr_next(sparse_matrix_desc_t *spmtx, dague_int_t bloknum, dague_int_t fcblknum )
{
    dague_int_t count, browfirst, browlast;
    SolverMatrix *datacode=&(spmtx->pastix_data->solvmatr);
    
    browfirst = UPDOWN_LISTPTR( UPDOWN_GCBLK2LIST(UPDOWN_LOC2GLOB(fcblknum))   );
    browlast  = UPDOWN_LISTPTR( UPDOWN_GCBLK2LIST(UPDOWN_LOC2GLOB(fcblknum))+1 );
    
    if ( bloknum == UPDOWN_LISTBLOK( browlast-1 ) )
        return 0; 
    for (count=browlast-1; count>browfirst; count--)
        if (bloknum==UPDOWN_LISTBLOK(count-1)) return UPDOWN_LISTBLOK(count);
    assert(0);
    return -1;
}

uint32_t sparse_matrix_rank_of(struct dague_ddesc *mat, ... )
{
    (void)mat;
    return 0;
}

void *sparse_matrix_data_of(struct dague_ddesc *mat, ... )
{
    sparse_matrix_desc_t *spmtx = (sparse_matrix_desc_t*)mat;
    va_list ap;
    dague_int_t cblknum, bloknum;

    va_start(ap, mat);
    cblknum = va_arg(ap, unsigned int);
    va_end(ap);
    bloknum = spmtx->pastix_data->solvmatr.symbmtx.cblktab[cblknum].bloknum;

    return (char*)(spmtx->pastix_data->solvmatr.coeftab[cblknum])
        + (size_t)(spmtx->typesze) * (size_t)(spmtx->pastix_data->solvmatr.bloktab[bloknum].coefind);
}

#ifdef DAGUE_PROF_TRACE
uint32_t sparse_matrix_data_key(struct dague_ddesc *mat, ... )
{
    va_list ap;
    dague_int_t cblknum, bloknum;
    sparse_matrix_desc_t *spmtx = (sparse_matrix_desc_t*)mat;
    
    va_start(ap, mat);
    cblknum = va_arg(ap, unsigned int);
    va_end(ap);
    bloknum = spmtx->pastix_data->solvmatr.symbmtx.cblktab[cblknum].bloknum;

    return (uint32_t)bloknum;
}

int sparse_matrix_key_to_string(struct dague_ddesc *mat, uint32_t datakey, char *buffer, uint32_t buffer_size)
{
    sparse_matrix_desc_t *spmtx = (sparse_matrix_desc_t*)mat;
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
    desc->super.key_dim = NULL; /* Initialized when the matrix is read */
    desc->super.key     = NULL; /* Initialized when the matrix is read */
#endif /* DAGUE_PROF_TRACE */

    desc->mtype       = mtype;
    desc->typesze     = sparse_matrix_size_of( mtype );
    desc->pastix_data = NULL;

/*     DEBUG(("sparse_matrix_init: desc = %p, mtype = %zu, \n" */
/*            "\tnodes = %u, cores = %u, myrank = %u\n", */
/*            desc, (size_t) desc->super.mtype,  */
/*            desc->super.super.nodes,  */
/*            desc->super.super.cores, */
/*            desc->super.super.myrank)); */
}

void sparse_matrix_destroy( sparse_matrix_desc_t *desc )
{
    (void)desc;
}
