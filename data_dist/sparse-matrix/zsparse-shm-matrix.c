/**
 *
 * @file zsparse-shm-matrix.c
 *
 * @author Thomas Herault
 * @date 2011-03-01
 * @precisions normal z -> c d s
 *
 **/
#include "dague_config.h"

#include <stdarg.h>
#include <assert.h>
#include <stdio.h>
#include <signal.h>
#include <pthread.h>

#include "lifo.h"
#include "linked_list.h"
#include "bindthread.h"

#include "data_dist/sparse-matrix/sparse-shm-matrix.h"
#include "data_dist/sparse-matrix/si-to-tssm.h"

#undef MAX
#define MAX(a, b) (((a) < (b))?(b):(a))

extern uint32_t dague_tssm_nbthreads;
uint32_t dague_tssm_rank_of(struct dague_ddesc *desc, ...);
void    *dague_tssm_data_of(struct dague_ddesc *desc, ...);
void     dague_tssm_data_release(struct dague_ddesc *desc, ...);

void dague_tssm_zmatrix_init(dague_tssm_desc_t * desc, enum matrix_type mtype, unsigned int cores, 
                             unsigned int mb, unsigned int nb, 
                             unsigned int lm, unsigned int ln, unsigned int i, unsigned int j, 
                             unsigned int m, unsigned int n,
                             dague_sparse_input_symbol_matrix_t *sm)
{
    tiled_matrix_desc_t *mat;
    dague_ddesc_t *res;

    assert( dague_tssm_nbthreads != 0 );
    assert( dague_tssm_nbthreads == cores );

    mat = (tiled_matrix_desc_t  *)desc;
    res = (dague_ddesc_t *)desc;

    mat->mtype = mtype;
    mat->mb = mb;
    mat->nb = nb;
    mat->bsiz = mb * nb;
    mat->lm = lm;
    mat->ln = ln;
    mat->lmt = (lm + mb - 1) / mb;
    mat->lnt = (ln + nb - 1) / nb;
    assert(i == 0);
    mat->i = i;
    assert(j == 0);
    mat->j = j;
    assert(m == lm);
    mat->m = m;
    assert(n == ln);
    mat->n = n;
    mat->mt = (m + mb - 1) / mb;
    mat->nt = (n + nb - 1) / nb;
    mat->nb_local_tiles = mat->mt * mat->nt;
    
    res->myrank = 0;
    res->cores = cores;
    res->nodes = 1;
    res->rank_of      = dague_tssm_rank_of;
    res->data_of      = dague_tssm_data_of;
    desc->mesh = (dague_tssm_tile_entry_t **)calloc( mat->nb_local_tiles, sizeof(dague_tssm_tile_entry_t*));

    desc->unpack = dague_tssm_ztile_unpack;
    desc->pack   = dague_tssm_ztile_pack;

    /* Init mat->mesh using Anthony load function around here */
    dague_sparse_input_to_tiles_load(desc, mat->mt, mat->nt, mat->mb, mat->nb, sm);
}
