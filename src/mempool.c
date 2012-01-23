/*
 * Copyright (c) 2010-2011 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "mempool.h"
#ifdef HAVE_STRING_H
#include <string.h>
#endif

/** dague_thread_mempool_construct
 *    constructs the thread-specific memory pool.
 */
static void dague_thread_mempool_construct( dague_thread_mempool_t *thread_mempool,
                                            dague_mempool_t *mempool )
{
    thread_mempool->parent = mempool;
    dague_atomic_lifo_construct( &thread_mempool->mempool );
    thread_mempool->nb_elt = 0;
}

static void dague_thread_mempool_destruct( dague_thread_mempool_t *thread_mempool )
{
    void *elt;
    while( NULL != (elt = dague_atomic_lifo_pop( &thread_mempool->mempool ) ) ) 
        free(elt);
    dague_atomic_lifo_destruct( &thread_mempool->mempool );
}

void dague_mempool_construct( dague_mempool_t *mempool, size_t elt_size, size_t pool_offset, unsigned int nbthreads )
{
    uint32_t tid;

    mempool->nb_thread_mempools = nbthreads;
    mempool->elt_size = elt_size;
    mempool->pool_owner_offset = pool_offset;
    mempool->nb_max_elt = 0;
    mempool->thread_mempools = (dague_thread_mempool_t *)malloc( sizeof(dague_thread_mempool_t) * nbthreads );

    for(tid = 0; tid < mempool->nb_thread_mempools; tid++)
        dague_thread_mempool_construct( &mempool->thread_mempools[tid], mempool );
}

void dague_mempool_destruct( dague_mempool_t *mempool )
{
    uint32_t tid;

    for(tid = 0; tid < mempool->nb_thread_mempools; tid++)
        dague_thread_mempool_destruct( &mempool->thread_mempools[tid] );

    free( mempool->thread_mempools );
    mempool->thread_mempools = NULL;
    mempool->nb_thread_mempools = 0;
}

void *dague_thread_mempool_allocate_when_empty( dague_thread_mempool_t *thread_mempool )
{
    /*
     * Simple heuristic: don't try to balance things between threads,
     * just allocate within this thread
     */
    void *elt;
    dague_thread_mempool_t **owner;

    DAGUE_LIFO_ELT_ALLOC( elt, thread_mempool->parent->elt_size );
    owner = (dague_thread_mempool_t **)((char*)elt + thread_mempool->parent->pool_owner_offset);
    *owner = thread_mempool;
    return elt;
}
