#include "dague_config.h"

#include <stdarg.h>
#include <assert.h>
#include <stdio.h>
#include <signal.h>
#include <pthread.h>

#include "debug.h"
#include "linked_list.h"
#include "bindthread.h"

#include "data_dist/sparse-matrix/sparse-shm-matrix.h"
#include "data_dist/sparse-matrix/si-to-tssm.h"

/*
#undef DEBUG
#define DEBUG(a) do { printf a; fflush(stdout); } while(0)
*/
#undef MAX
#define MAX(a, b) (((a) < (b))?(b):(a))

#if defined(DAGUE_DEBUG)
#define CHECK_AND_RETURN(v) do { \
    assert( (v) != NULL );       \
    return (v);                  \
    } while(0)
#else
#define CHECK_AND_RETURN(v) return (v)
#endif

static dague_linked_list_t *dirty_tiles = NULL;
static dague_linked_list_t *clean_tiles = NULL;
static dague_linked_list_t *free_tiles = NULL;
static size_t all_tiles_size = 0;
uint32_t dague_tssm_nbthreads = 0;

static void *dague_tssm_thread_init(void *_p)
{
    uint64_t *p = (uint64_t*)_p;
    uint64_t i;
    dague_list_item_t *tile;
    uint64_t threadid = p[0];
    uint64_t nbtilesperthread = p[1];
    uint64_t tile_size = p[2];

    dague_bindthread(threadid);

    dague_linked_list_construct( &dirty_tiles[threadid] );
    dague_linked_list_construct( &clean_tiles[threadid] );
    dague_linked_list_construct( &free_tiles[threadid] );
    for(i = 0; i < nbtilesperthread; i++) {
        tile = (dague_list_item_t*)malloc( MAX(sizeof(dague_list_item_t), tile_size) );
        dague_linked_list_add_tail(&free_tiles[threadid], tile);
    }

    return NULL;
}

void dague_tssm_init(uint32_t nbthreads, size_t tile_size, uint32_t nbtilesperthread)
{
    uint32_t   i;
    pthread_t *tid;
    uint64_t **tp;

    tid = (pthread_t*)malloc(nbthreads*sizeof(pthread_t));
    assert( 0 == dague_tssm_nbthreads );
    assert( NULL == dirty_tiles );
    assert( NULL == clean_tiles );
    assert( NULL == free_tiles );

    dague_tssm_nbthreads = nbthreads;

    dirty_tiles = (dague_linked_list_t*)malloc( nbthreads * sizeof(dague_linked_list_t) );
    clean_tiles = (dague_linked_list_t*)malloc( nbthreads * sizeof(dague_linked_list_t) );
    free_tiles = (dague_linked_list_t*)malloc( nbthreads * sizeof(dague_linked_list_t) );

    all_tiles_size = tile_size;

    tp = (uint64_t**)malloc( nbthreads * 3 * sizeof(uint64_t) );
    for(i = 0; i < nbthreads; i++) {
        tp[i] = (uint64_t*)malloc(3 * sizeof(uint64_t));
        tp[i][0] = i;
        tp[i][1] = nbtilesperthread;
        tp[i][2] = (uint64_t)tile_size;
        if( i != 0 )
            pthread_create(&tid[i], NULL, dague_tssm_thread_init, tp[i]);
    }
    (void)dague_tssm_thread_init(tp[0]);
    free(tp[0]);
    for(i = 1; i < nbthreads; i++) {
        pthread_join(tid[i], NULL);
        free(tp[i]);
    }

    free(tid);
    free(tp);
}

static void dague_tssm_move_tiles_locked(dague_linked_list_t *dst, dague_tssm_tile_entry_t *tptr)
{
    assert( NULL == tptr->current_list );
    dague_linked_list_add_tail( dst, (dague_list_item_t*)tptr );
    tptr->current_list = dst;
}

static int dague_tssm_reclaim_free_tile(int this_thread, int find_in_other_threads, dague_tssm_tile_entry_t *tptr)
{
    void *tile;
    int thid;

    tile = (void*)dague_linked_list_remove_head( &free_tiles[this_thread] );
    if(NULL != tile) {
        tptr->tile = tile;
        tptr->tile_owner = this_thread;
        return 1;
    }

    if( find_in_other_threads ) {
        for(thid = ((this_thread + 1) % dague_tssm_nbthreads);
            thid != this_thread;
            thid = ((thid + 1) % dague_tssm_nbthreads) ) {
            if( dague_tssm_reclaim_free_tile(thid, 0, tptr) )
                return 1;
        }
    }

    return 0;
}

static int dague_tssm_reclaim_clean_tile(int this_thread, int find_in_other_threads, dague_tssm_tile_entry_t *tptr)
{
    dague_tssm_tile_entry_t *victim;
    int thid;

    do {
        victim = (dague_tssm_tile_entry_t *)dague_linked_list_remove_head( &clean_tiles[this_thread] );
        if( NULL != victim ) {
            dague_atomic_lock(&victim->lock);
            
            if( (victim->current_list != &clean_tiles[this_thread]) ||
                (victim->nbreaders > 0) || (victim->writer != -1) || 
                (NULL == victim->tile) ) {
                /* Nope: between the instant where this was poped from clean
                 * and now, somebody started using this tile again, or claimed it!
                 */
                dague_atomic_unlock(&victim->lock);
                continue;
            }
            victim->current_list = NULL;

            assert( tptr->status & TILE_STATUS_UNPACKING );

            /* Yes! I can claim the tile of this entry */
            tptr->tile_owner = victim->tile_owner;
            victim->tile_owner = -1;

            tptr->tile = victim->tile;
            victim->tile = NULL;
            DEBUG(("      Clean tile %p at %lu, %lu found on thread %d for tile %p at %lu, %lu: [%p] is now free to be used\n",
                   victim, victim->m, victim->n, this_thread, tptr, tptr->m, tptr->n, tptr->tile));
            dague_atomic_unlock(&victim->lock);
            return 1;
        }
    } while( NULL != victim );

    if( find_in_other_threads ) {
        for(thid = ((this_thread + 1) % dague_tssm_nbthreads);
            thid != this_thread;
            thid = ((thid + 1) % dague_tssm_nbthreads) ) {
            if( dague_tssm_reclaim_clean_tile(thid, 0, tptr) ) {
                return 1;
            }
        }
    }

    return 0;
}

static int dague_tssm_cleanup_some_tile(int thid, dague_tssm_tile_entry_t *tptr)
{
    dague_tssm_tile_entry_t *victim;
    void *tile;
    int tile_owner;
    do {
        victim = (dague_tssm_tile_entry_t *)dague_linked_list_remove_head( &dirty_tiles[thid] );
        if( NULL != victim ) {
            dague_atomic_lock(&victim->lock);
            if( (victim->nbreaders > 0) || (victim->writer != -1) || 
                (victim->status & TILE_STATUS_PACKING) || (victim->tile == NULL) ||
                (victim->current_list != &dirty_tiles[thid]) ) {
                /* Nope: between the instant where this was poped from dirty
                 * and now, somebody started using this tile again, or packing it.
                 */
                dague_atomic_unlock(&victim->lock);
                continue;
            }
            victim->current_list = NULL;

            /* Possible candidate: at least right now, nobody is using this tile
             * Let everybody know that I'm packing it. Hopefully, nobody will be dumb
             * enough to write into it while I do that
             */
            victim->worker_id = thid;
            victim->status = TILE_STATUS_PACKING;
            DEBUG(("       Packing of tile %p at %lu, %lu, memory [%p] by thread %d\n", 
                   victim, victim->m, victim->n, victim->tile, thid));
            dague_atomic_unlock(&victim->lock);

            /* Without the lock, pack it */
            victim->desc->pack(victim->tile, victim->m, victim->n, 
                               ((tiled_matrix_desc_t*)victim->desc)->mb, 
                               ((tiled_matrix_desc_t*)victim->desc)->nb, 
                               victim->packed_ptr);

            dague_atomic_lock(&victim->lock);
            /* Assuming nobody wrote in it, which should be the case,
             * mark it is now clean. And if nobody is using it, move
             * it to the clean tiles, and since this is a succesfull clean
             * return */
            assert(victim->status == TILE_STATUS_PACKING);
            victim->status = 0;
            victim->worker_id = -1;
            if( (victim->nbreaders == 0) && (victim->writer == -1) ) {
                tile = victim->tile;
                victim->tile = NULL;
                if( victim->current_list != NULL ) {
                    assert( victim->tile_owner != -1 );
                    assert( victim->current_list == &clean_tiles[victim->tile_owner] );
                    dague_linked_list_remove_item( victim->current_list, (dague_list_item_t*)victim );
                    victim->current_list = NULL;
                }
                tile_owner = victim->tile_owner;
                victim->tile_owner = -1;
                dague_atomic_unlock(&victim->lock);
                dague_atomic_lock( &tptr->lock );
                tptr->tile = tile;
                tptr->tile_owner = tile_owner;;
                dague_atomic_unlock( &tptr->lock );
                return 1;
            }
            /* This tile has been cleaned, but it's now in use should be by a reader */
            /* Unlock it, and find another victim: it's in the used list, not the clean list */
            assert( victim->writer == -1 );
            dague_atomic_unlock(&victim->lock);
        }
    } while( NULL != victim );

    /* Nothing else to clean */
    return 0;
}

static void dague_tssm_reclaim_tile(int this_thread, int find_in_other_threads, dague_tssm_tile_entry_t *tptr)
{
    int thid;
    void *tile;

    for(;;) {
        if( dague_tssm_reclaim_free_tile(this_thread, 0, tptr) ) 
            return;

        /* Could not find a free tile in this thread.
         * Ok, let see if there is a clean tile for this thread that we can preempt */
        if( dague_tssm_reclaim_clean_tile(this_thread, 0, tptr) )
            return;

        /* Arf, no clean tile either.
         * Ok, if I can look in another thread, let's try to see if there is a free tile 
         * in another thread, it's less costly than packing one of my dirty tiles
         */
        if( dague_tssm_reclaim_free_tile(this_thread, find_in_other_threads, tptr) )
            return;

        /* A clean tile somewhere else, then? */
        if( dague_tssm_reclaim_clean_tile(this_thread, find_in_other_threads, tptr) )
            return;
        
        /* Nope: I really need to reclaim a tile...
         * Let's try to move some from dirty to clean
         */
        if( dague_tssm_cleanup_some_tile(this_thread, tptr) ) {
            return;
        }

        if( find_in_other_threads ) {
            for(thid = ((this_thread + 1) % dague_tssm_nbthreads);
                thid != this_thread;
                thid = ((thid + 1) % dague_tssm_nbthreads) ) {
                if( dague_tssm_cleanup_some_tile(thid, tptr)  ) {                    
                    return;
                }
            }
        }
    }
}


uint32_t dague_tssm_rank_of(struct dague_ddesc *desc, ...)
{
    uint32_t m, n;
    va_list ap;
    va_start(ap, desc);
    m = va_arg(ap, uint32_t);
    n = va_arg(ap, uint32_t);
    va_end(ap);

    (void)m;
    (void)n;

    return 0;
}

void *dague_tssm_data_of(struct dague_ddesc *desc, ...)
{
    uint32_t m, n;
    va_list ap;
    dague_tssm_desc_t *mat = (dague_tssm_desc_t *)desc;

    va_start(ap, desc);
    m = va_arg(ap, uint32_t);
    n = va_arg(ap, uint32_t);
    va_end(ap);

    assert( NULL != mat->mesh );
    assert( (m < mat->super.mt) && (n < mat->super.nt) );

    return (void*)mat->mesh[ n * mat->super.mt + m ];
}

void *dague_tssm_data_expand(void *metadata, int write_access, int this_thread)
{
    dague_tssm_tile_entry_t *tptr = (dague_tssm_tile_entry_t *)metadata;
    dague_tssm_desc_t *mat;

    /* If we receive NULL (Zero tile), or an odd pointer (network-related / in-arenas pointer )
     * return the pointer (cleaned from its potential in-arenas flag)
     */
    if( (NULL == metadata) || (1 == ((intptr_t)metadata & 0x1)) ) {
        return (void*)( (intptr_t)metadata & (~0x1) );
    }
    mat = tptr->desc;
    (void)mat;

    dague_atomic_lock( &tptr->lock );
    /** If somebody else is already working on getting up this tile,
     *  give the scheduler an opportunity to select another task
     *  on this thread.
     */
    if( tptr->status & TILE_STATUS_UNPACKING ) {
        dague_atomic_unlock( &tptr->lock );
        DEBUG(("Thread %d is already unpacking: expansion of tile %p at %lu, %lu for thread %d in %s mode delayed\n",
               tptr->worker_id, tptr, tptr->m, tptr->n, this_thread, write_access ? "write" : "read" ));
        return (void*)1;
    }

    /** Nobody is getting this tile up, and I still own the lock.
     *  Is the tile there?
     */
    if( NULL != tptr->tile ) {
        if( NULL != tptr->current_list ) {
            dague_linked_list_remove_item( tptr->current_list, (dague_list_item_t*)tptr );
            tptr->current_list = NULL;
        }
        if( write_access ) {
            assert( tptr->writer == -1 );
            assert( tptr->nbreaders == 0 );
            if( tptr->status & TILE_STATUS_PACKING ) {
                /** Oops: somebody decided to pack this tile...
                 *  can't write while this is going on
                 */
                DEBUG(("Thread %d decided to pack while trying to access tile for writing: expansion of tile %p at %lu, %lu for thread %d in %s mode delayed\n",
                       tptr->worker_id, tptr, tptr->m, tptr->n, this_thread, write_access ? "write" : "read" ));
                dague_atomic_unlock( &tptr->lock );
                return (void*)1;
            }
            /* Reclaim the exclusive access to the tile */
            tptr->status = TILE_STATUS_DIRTY;
            tptr->worker_id = -1;
            tptr->writer = this_thread;
            tptr->current_list = NULL;
        } else {
            assert( tptr->writer == -1 );
            /** We don't care if the tile is being packed right now.
             *  We just ensure that nobody is going to claim the
             *  tile out */
            if( tptr->nbreaders == 0 ) {
                tptr->current_list = NULL;
            }
            tptr->nbreaders++;
        }
        DEBUG(("Expanding tile %p at %lu, %lu, in memory [%p] for thread %d in %s mode\n", 
               tptr, tptr->m, tptr->n, tptr->tile, this_thread, write_access ? "write" : "read" ));
    } else {
        assert( NULL == tptr->current_list );
        assert( (tptr->status & TILE_STATUS_PACKING) == 0 );
        /** This tile is not here, and nobody is trying to pick it up yet.
         *  Mark that we are unpacking it, note that this tile is in use,
         *  find a tile space to unpack, and unpack it
         */
        tptr->status = TILE_STATUS_UNPACKING;
        tptr->worker_id = this_thread;
        assert( tptr->writer == -1 );
        assert( tptr->nbreaders == 0 );
        DEBUG(("Expanding tile %p at %lu, %lu, in memory [%p] for thread %d in %s mode (unpacking)\n", 
               tptr, tptr->m, tptr->n, tptr->tile, this_thread, write_access ? "write" : "read" ));
        dague_atomic_unlock( &tptr->lock );
        
        dague_tssm_reclaim_tile(this_thread, 1, tptr);
        memset(tptr->tile, 0, all_tiles_size);
        mat->unpack(tptr->tile, tptr->m, tptr->n, mat->super.mb, mat->super.nb, tptr->packed_ptr);
        DEBUG(("       Unpacking of tile %p at %lu, %lu, in memory [%p] done by thread %d in %s mode\n", 
               tptr, tptr->m, tptr->n, tptr->tile, this_thread, write_access ? "write" : "read" ));

        dague_atomic_lock( &tptr->lock );
        DEBUG(("       Update of the status of tile %p at %lu, %lu, done by thread %d in %s mode\n", 
               tptr, tptr->m, tptr->n, this_thread, write_access ? "write" : "read" ));

        /* Done with the unpacking */
        if( write_access ) {
            tptr->status = TILE_STATUS_DIRTY;
            tptr->worker_id = -1;
            tptr->writer    = this_thread;
            tptr->nbreaders = 0;
        } else {
            tptr->status    = 0;
            tptr->worker_id = -1;
            tptr->writer    = -1;
            tptr->nbreaders = 1;
        }
    }

    assert( tptr->current_list == NULL );
    dague_atomic_unlock( &tptr->lock );
    CHECK_AND_RETURN( tptr->tile );
}

void dague_tssm_data_release(void *metadata, int write_access, int this_thread)
{
    dague_tssm_tile_entry_t *tptr = (dague_tssm_tile_entry_t *)metadata;

    /* If we receive NULL (Zero tile), or an odd pointer (network-related / in-arenas pointer ),
     * ignore.
     */
    if( (NULL == metadata) || (1 == ((intptr_t)metadata & 0x1)) ) {
        return;
    }
    
    dague_atomic_lock( &tptr->lock );
    if( write_access ) {
        assert( tptr->writer == this_thread );
        assert( tptr->nbreaders == 0 );
        tptr->writer = -1;
    } else {
        tptr->nbreaders--;
    }

    DEBUG(("Tile %p at %lu, %lu in memory [%p] released by thread %d in %s mode. After release, writer = %d, nbreaders = %u\n",
           tptr, tptr->m, tptr->n, tptr->tile, this_thread, write_access ? "write" : "read",
           tptr->writer, tptr->nbreaders));

    if( (tptr->nbreaders == 0) && (tptr->writer == -1) ) {
        assert( tptr->current_list == NULL );
        if( tptr->status & TILE_STATUS_DIRTY ) {
            DEBUG(("    Tile %p at %lu, %lu in memory [%p] released by thread %d in %s mode. Pushed in dirty_tiles[%d]\n",
                   tptr, tptr->m, tptr->n, tptr->tile, this_thread, write_access ? "write" : "read",
                   tptr->tile_owner));
            dague_tssm_move_tiles_locked( &dirty_tiles[tptr->tile_owner], tptr );
        } else {
            DEBUG(("    Tile %p at %lu, %lu in memory [%p] released by thread %d in %s mode. Pushed in clean_tiles[%d]\n",
                   tptr, tptr->m, tptr->n, tptr->tile, this_thread, write_access ? "write" : "read",
                   tptr->tile_owner));
            dague_tssm_move_tiles_locked( &clean_tiles[tptr->tile_owner], tptr );
        }
    }
    dague_atomic_unlock( &tptr->lock );
}

void dague_tssm_mesh_create_tile(dague_tssm_desc_t *mesh, 
                                 uint64_t m, uint64_t n, 
                                 uint32_t mb, uint32_t nb, 
                                 dague_tssm_data_map_t *packed_ptr)
{
    dague_tssm_tile_entry_t *e;
    assert( (m < mesh->super.mt) && (n < mesh->super.nt) );
    if( NULL != packed_ptr ) {
        e = (dague_tssm_tile_entry_t*)calloc(1, sizeof(dague_tssm_tile_entry_t));
        e->current_list = NULL;
        e->packed_ptr = packed_ptr;
        e->m = m;
        e->n = n;
        e->lock = 0;
        e->writer = -1;
        e->nbreaders = 0;
        e->status = 0;
        e->worker_id = -1;
        e->tile = NULL;
        e->tile_owner = -1;
        e->desc = mesh;
        assert( mesh->super.mb == mb &&
                mesh->super.nb == nb );
    } else {
        e = NULL;
    }
    mesh->mesh[ m * mesh->super.nt + n ] = e;
}

#if defined(DAGUE_DEBUG)
int dague_tssm_flush_matrix(dague_ddesc_t *_mat)
{
    dague_tssm_desc_t *mat = (dague_tssm_desc_t*)_mat;
    uint64_t m, n, errors;
    dague_tssm_tile_entry_t *tptr;
    uint32_t thid;

    errors = 0;
    for(m = 0; m < mat->super.mt; m++) {
        for(n = 0; n < mat->super.nt; n++) {
            tptr = mat->mesh[ m * mat->super.nt + n ];
            if( NULL == tptr )
                continue;
            dague_atomic_lock( &tptr->lock );
            if( tptr->status != 0 ||
                tptr->writer != -1 ||
                tptr->nbreaders != 0 ||
                tptr->current_list == NULL ) {
                dague_atomic_unlock( &tptr->lock );
                fprintf(stderr, 
                        "dague:tssm:flush_matrix failed: [%lu, %lu] is not available for flushing:\n"
                        "  status = %x, writer = %d, nbreaders = %u, current_list = NULL\n",
                        m, n,
                        tptr->status, tptr->writer, tptr->nbreaders);
                errors++;
                continue;
            }
            if( NULL == tptr->tile )
                continue;
            for( thid = 0; thid < dague_tssm_nbthreads; thid++ ) {
                if( tptr->current_list == &clean_tiles[thid] ) {
                    dague_linked_list_remove_item(tptr->current_list, &tptr->super);
                    tptr->current_list = NULL;
                    dague_linked_list_add_tail( &free_tiles[ tptr->tile_owner ], (dague_list_item_t*)tptr->tile );
                    tptr->tile = NULL;
                    tptr->tile_owner = -1;
                    dague_atomic_unlock( &tptr->lock );
                    break;
                }
            }
            if( thid == dague_tssm_nbthreads ) {
                /* Not in clean, not in used... It must be in dirty */
                for( thid = 0; thid < dague_tssm_nbthreads; thid++ ) {
                    if( tptr->current_list == &dirty_tiles[thid] ) {
                        dague_linked_list_remove_item(tptr->current_list, &tptr->super);
                        tptr->current_list = NULL;

                        DEBUG(("       Packing of tile %p at %lu, %lu\n", 
                               tptr, tptr->m, tptr->n));

                        mat->pack(tptr->tile, tptr->m, tptr->n, 
                                  mat->super.mb, mat->super.nb, 
                                  tptr->packed_ptr);
                        
                        dague_linked_list_add_tail( &free_tiles[ tptr->tile_owner ], (dague_list_item_t*)tptr->tile );
                        tptr->tile = NULL;
                        tptr->tile_owner = -1;
                        break;
                    }
                }

                dague_atomic_unlock( &tptr->lock );

                if( thid == dague_tssm_nbthreads ) {
                    /* That's bad: the tile is not NULL, but the entry does not belong to
                     * used, any of the clean or any of the dirty... Internal Error 42. */
                    fprintf(stderr, "dague:tssm:flush_matrix: internal error 42 - entry [%lu, %lu] points to a tile, but does not belong to any list\n",
                            m, n);
                    errors++;
                    continue;
                }
            }
        }
    }
    return errors;
}
#else
/* This version assumes that there is absolutely no error in tracking,
 * and nobody else is working on the matrix
 * But as a consequence, it's *so much* faster than the other...*/
static void *dague_tssm_flush_matrix_thread(void *_param)
{
    uintptr_t *param = (uintptr_t*)_param;
    dague_tssm_desc_t *mat = (dague_tssm_desc_t *)param[0];
    uintptr_t thid = param[1];
    dague_linked_list_t todo, tonotdo;
    dague_tssm_tile_entry_t *tptr;

    dague_linked_list_construct( &todo );
    dague_linked_list_construct( &tonotdo );

    while( (tptr = (dague_tssm_tile_entry_t *)dague_linked_list_remove_head( &clean_tiles[thid] )) ) {
        if( tptr->desc == mat )
            dague_linked_list_add_tail( &todo, (dague_list_item_t*)tptr );
        else 
            dague_linked_list_add_tail( &tonotdo, (dague_list_item_t*)tptr );
    }
    while( (tptr = (dague_tssm_tile_entry_t *)dague_linked_list_remove_head( &tonotdo )) ) {
        dague_linked_list_add_tail( &clean_tiles[thid], (dague_list_item_t*)tptr );
    }
    while( (tptr = (dague_tssm_tile_entry_t *)dague_linked_list_remove_head( &todo )) ) {
        tptr->current_list = NULL;
        dague_linked_list_add_tail( &free_tiles[ tptr->tile_owner ], (dague_list_item_t*)tptr->tile );
        tptr->tile = NULL;
        tptr->tile_owner = -1;
    }
    
    while( (tptr = (dague_tssm_tile_entry_t *)dague_linked_list_remove_head( &dirty_tiles[thid] )) ) {
        if( tptr->desc == mat )
            dague_linked_list_add_tail( &todo, (dague_list_item_t*)tptr );
        else 
            dague_linked_list_add_tail( &tonotdo, (dague_list_item_t*)tptr );
    }
    while( (tptr = (dague_tssm_tile_entry_t *)dague_linked_list_remove_head( &tonotdo )) ) {
        dague_linked_list_add_tail( &dirty_tiles[thid], (dague_list_item_t*)tptr );
    }
    while( (tptr = (dague_tssm_tile_entry_t *)dague_linked_list_remove_head( &todo )) ) {
        tptr->current_list = NULL;
        
        DEBUG(("       Packing of tile %p at %lu, %lu\n", 
               tptr, tptr->m, tptr->n));
        mat->pack(tptr->tile, tptr->m, tptr->n, 
                  mat->super.mb, mat->super.nb, 
                  tptr->packed_ptr);

        dague_linked_list_add_tail( &free_tiles[ tptr->tile_owner ], (dague_list_item_t*)tptr->tile );
        tptr->tile = NULL;
        tptr->tile_owner = -1;
    }

    return NULL;
}

int dague_tssm_flush_matrix(dague_ddesc_t *_mat)
{
    dague_tssm_desc_t *mat = (dague_tssm_desc_t*)_mat;
    uint32_t thid;
    uintptr_t **tp;
    pthread_t *tids;
    
    tp = (uintptr_t**)malloc(sizeof(uintptr_t) * dague_tssm_nbthreads);
    tids = (pthread_t *)malloc(sizeof(pthread_t) * dague_tssm_nbthreads);
    for(thid = 0; thid < dague_tssm_nbthreads; thid++) {
        tp[thid] = (uintptr_t*)malloc(2 * sizeof(uintptr_t));
        tp[thid][0] = (uintptr_t)mat;
        tp[thid][1] = thid;
        if( thid != 0 )
            pthread_create(&tids[thid], NULL, dague_tssm_flush_matrix_thread, tp[thid]);
    }
    dague_tssm_flush_matrix_thread( tp[0] );
    free( tp[0] );
    for(thid = 1; thid < dague_tssm_nbthreads; thid++) {
        pthread_join(tids[thid], NULL);
        free(tp[thid]);
    }
    free(tp);
    free(tids);
    return 0;
}
#endif /* defined(DAGUE_DEBUG) */
