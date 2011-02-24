#include "dague_config.h"

#include <stdarg.h>
#include <assert.h>

#include "lifo.h"
#include "linked_list.h"

#include "data_dist/sparse-matrix/sparse-shm-matrix.h"
#include "data_dist/sparse-matrix/ssm-cpulru.h"

static dague_linked_list_t  used_tiles;
static dague_linked_list_t *dirty_tiles;
static dague_linked_list_t *clean_tiles;
static dague_atomic_lifo_t *free_tiles;
static int dague_tssm_nbthread;

void dague_tssm_init(int nbthreads)
{
    int i;

    dague_tssm_nbthread = nbthreads;

    dague_linked_list_construct( &used_tiles );
    dirty_tiles = (dague_linked_list_t*)malloc( nbthreads * sizeof(dague_linked_list_t) );
    clean_tiles = (dague_linked_list_t*)malloc( nbthreads * sizeof(dague_linked_list_t) );
    free_tiles = (dague_linked_list_t*)malloc( nbthreads * sizeof(dague_linked_list_t) );
    for(i = 0; i < nbthreads; i++) {
        dague_linked_list_construct( &dirty_tiles[i] );
        dague_linked_list_construct( &clean_tiles[i] );
        dague_atomic_lifo_construct( &free_tiles[i] );
    }    
}

void dague_tssm_thread_init(int threadid, int nbtilesperthread, size_t tile_size)
{
    int i;
    dague_list_item_t *tile;

    for(i = 0; i < nbtilesperthread; i++) {
        DAGUE_LIFO_ELT_ALLOC(tile, MAX(sizeof(dague_list_item_t), tile_size));
        DAGUE_LIST_ITEM_SINGLETON(tile);
        dague_atomic_lifo_push(&free_tiles[threadid], tile);
    }
}

static void dague_tssm_move_tiles_locked(dague_linked_list_t *dst, dague_tssm_tile_entry_t *tptr)
{
    if( NULL != tptr->current_list ) {
        dague_linked_list_remove_item( tptr->current_list, (dague_list_item_t*)tptr );
    }
    dague_linked_list_add_tail( dst, (dague_list_item_t*)tptr );
    tptr->current_list = dst;
}

static int dague_tssm_reclaim_free_tile(int this_thread, int find_in_other_threads, dague_tssm_tile_entry_t *tptr)
{
    void *tile;
    int thid;

    tile = (void*)dague_atomic_lifo_pop( &free_tiles[this_thread] );
    if(NULL != tile) {
        tptr->tile = tile;
        tptr->tile_owner = this_thread;
        return 1;
    }

    if( find_in_other_threads ) {
        for(thid = ((this_thread + 1) % dague_tssm_nbthread);
            thid != this_thread;
            thid = ((thid + 1) % dague_tssm_nbthread) ) {
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
            if( (victim->nbreaders > 0) || (victim->writer != -1) ) {
                /* Nope: between the instant where this was poped from clean
                 * and now, somebody started using this tile again
                 */
                dague_atomic_unlock(&victim->lock);
                continue;
            }
            /* Yes! I can claim the tile of this entry */
            victim->tile_owner = -1;
            tptr->tile = victim->tile;
            tptr->tile_owner = this_thread;
            victim->tile = NULL;
            dague_atomic_unlock(&victim->lock);
            return 1;
        }
    } while( NULL != victim );

    if( find_in_other_threads ) {
        for(thid = ((this_thread + 1) % dague_tssm_nbthread);
            thid != this_thread;
            thid = ((thid + 1) % dague_tssm_nbthread) ) {
            if( dague_tssm_reclaim_clean_tile(thid, 0, tptr) ) {
                return 1;
            }
        }
    }
}

static int dague_tssm_cleanup_some_tile(int thid)
{
    dague_tssm_tile_entry_t *victim;
    do {
        victim = (dague_tssm_tile_entry_t *)dague_linked_list_remove_head( &dirty_tiles[this_thread] );
        if( NULL != victim ) {
            dague_atomic_lock(&victim->lock);
            if( (victim->nbreaders > 0) || (victim->writer != -1) ) {
                /* Nope: between the instant where this was poped from dirty
                 * and now, somebody started using this tile again
                 */
                dague_atomic_unlock(&victim->lock);
                continue;
            }
            /* Possible candidate: at least right now, nobody is using this tile
             * Let everybody know that I'm packing it. Hopefully, nobody will be dumb
             * enough to write into it while I do that
             */
            victim->status |= TILE_STATUS_PACKING;
            dague_atomic_unlock(&victim->lock);

            /* Without the lock, pack it */
            dague_anthony_unpack(victim->tile, victim->n, victim->m, 
                                 victim->tile_n, victim->tile_m, 
                                 victim->packed_ptr);

            dague_atomic_lock(&victim->lock);
            /* Assuming nobody wrote in it, which should be the case,
             * mark it is now clean. And if nobody is using it, move
             * it to the clean tiles, and since this is a succesfull clean
             * return */
            victim->status &= ~(TILE_STATUS_DIRTY|TILE_STATUS_PACKING);
            if( (victim->nbreaders == 0) && (victim->writer == -1) ) {
                dague_tssm_move_tiles_locked( &clean_tiles[victim->tile_owner], victim );
                dague_atomic_unlock(&victim->lock);
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
    int thid, cleaned;

    do {
        cleaned = 0;

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
        if( dague_tssm_cleanup_some_tile(this_thread) ) {
            cleaned = 1;
            continue; /* hopefully, some tile is now clean, try to find it */
        }

        if( find_in_other_threads ) {
            for(thid = ((this_thread + 1) % dague_tssm_nbthread);
                thid != this_thread;
                thid = ((thid + 1) % dague_tssm_nbthread) ) {
                if( dague_tssm_cleanup_some_tile(thid) ) {
                    cleaned = 1;
                    break; 
                }
            }

            if( cleaned ) {
                continue; /* hopefully, some tile is now clean, try to find it */
            }
        }
    } while( cleaned );
    
    /* Really? I couldn't find a free tile, clean tile, or clean a single
     * dirty tile? Man, I should have more tiles than that...
     */
    fprintf(stderr, 
            "Sparse Shared Memory Tiled Matrix Data Storage Fatal Error:\n"
            "  Out-of-memory -- Unable to find any tile cleanable, clean, or free.\n");
    raise(SIGABRT);
}


static uint32_t rank_of(struct dague_ddesc *desc, ...)
{
    va_list ap;
    va_start(ap, desc);
    va_end(ap);

    return 0;
}

static void *data_of(struct dague_ddesc *desc, ...)
{
    unsigned int m, n;
    va_list ap;
    tiled_sparse_shm_matrix_desc_t *mat = (tiled_sparse_shm_matrix_desc_t *)desc;
    void *tptr;
    uint32_t status;
    int write_access;
    int this_tread;

    va_start(ap, desc);
    m = va_arg(ap, unsigned int);
    n = va_arg(ap, unsigned int);
    write_access = va_arg(ap, int);
    this_thread = va_arg(ap, int);
    va_end(ap);

    assert( NULL != mat->mesh );
    assert( (m < mat->mb) && (n < mat->nb) );

    tptr = mat->mesh[ n + m * mat->nb ];
    if( NULL == tptr ) {
        return NULL;
    }

    dague_atomic_lock( &tptr->lock );
    /** If somebody else is already working on getting up this tile,
     *  give the scheduler an opportunity to select another task
     *  on this thread.
     */
    if( tptr->status & TILE_STATUS_UNPACKING ) {
        dague_atomic_unlock( &tptr->lock );
        return (void*)1;
    }

    /** Nobody is getting this tile up, and I still own the lock.
     *  Is the tile there?
     */
    if( NULL != tptr->tile ) {
        if( write_access ) {
            assert( tptr->writer == -1 );
            assert( tptr->nbreaders == 0 );
            if( tptr->status & TILE_STATUS_PACKING ) {
                /** Oops: somebody decided to pack this tile...
                 *  can't write while this is going on
                 */
                dague_atomic_unlock( &tptr->lock );
                return (void*)1;
            }
            dague_tssm_move_tiles_locked( &used_tiles, tptr );

            /* Reclaim the exclusive access to the tile */
            tptr->status = TILE_STATUS_DIRTY;
            tptr->writer = this_thread;
            dague_atomic_unlock( &tptr->lock );
            return tptr->tile;
        } else {
            assert( tptr->writer == -1 );
            /** We don't care if the tile is being packed right now.
             *  We just ensure that nobody is going to claim the
             *  tile out */
            tptr->nbreaders++;
            dague_tssm_move_tiles_locked( &used_tiles, tptr );
            dague_atomic_unlock( &tptr->lock );
            return tptr->tile;
        }
    } else {
        /** This page is not there, and nobody is trying to pick it up 
         *  Mark that we are unpacking it, find a tile to unpack, and unpack it
         */
        tptr->status = TILE_STATUS_UNPACKING;
        dague_atomic_unlock( &tptr->lock );
        
        dague_tssm_reclaim_tile(this_thread, 1, tptr);
        dague_anthony_unpack(tptr->tile, n, m, mat->tile_n, mat->tile_m, tptr->packed_ptr);
        
        dague_atomic_lock( &tptr->lock );
        if( write_access ) {
            tptr->status    = TILE_STATUS_DIRTY;
            tptr->writer    = this_thread;
            tptr->nbreaders = 0;
        } else {
            tptr->status    = 0;
            tptr->writer    = -1;
            tptr->nbreaders = 1;
        }
        dague_tssm_move_tiles_locked( &used_tiles, tptr );
        dague_atomic_unlock( &tptr->lock );
        return tptr->tile;
    }
}

static void data_release(struct dague_ddesc *desc, ...)
{
    unsigned int m, n;
    va_list ap;
    tiled_sparse_shm_matrix_desc_t *mat = (tiled_sparse_shm_matrix_desc_t *)desc;
    void *tptr;
    uint32_t status;
    int write_access;
    int this_thread;

    va_start(ap, desc);
    m = va_arg(ap, unsigned int);
    n = va_arg(ap, unsigned int);
    write_access = va_arg(ap, int);
    this_thread = va_arg(ap, int);
    va_end(ap);

    assert( NULL != mat->mesh );
    assert( (m < mat->mb) && (n < mat->nb) );
    tptr = mat->mesh[ n + m * mat->nb ];
    if( NULL == tptr ) {
        return;
    }
    
    dague_atomic_lock( &tptr->lock );
    if( write_access ) {
        assert( tptr->writer == this_thread );
        assert( tptr->nbreader == 0 );
        tptr->writer = -1;
    } else {
        tptr->nbreaders--;
    }
    if( (tptr->nbreaders == 0) && (tptr->writer == -1) ) {
        if( tptr->status & TILE_STATUS_DIRTY ) {
            dague_tssm_move_tiles_locked( &dirty_tiles[tptr->tile_owner], tptr );
        } else {
            dague_tssm_move_tiles_locked( &clean_tiles[tptr->tile_owner], tptr );
        }
    }
    dague_atomic_unlock( &tptr->lock );
}

int dague_tssm_mesh_create_tile(dague_tssm_desc_t *mesh, unsigned int m, unsigned int n, 
                                unsigned int tile_m, unsigned int tile_n, 
                                void *packed_ptr)
{
    dague_tssm_tile_entry_t *e;
    assert( (m < mesh->mt) && (n < mesh->nt) );
    if( NULL != packed_ptr ) {
        e = (dague_tssm_tile_entry_t*)calloc(1, sizeof(dague_tssm_tile_entry_t));
        DAGUE_LIST_ITEM_SINGLETON( &e->super );
        e->current_list = NULL;
        e->packed_ptr = packed_ptr;
        e->n = n;
        e->m = m;
        e->tile_n = tile_n;
        e->tile_m = tile_m;
        e->lock = 0;
        e->writer = -1;
        e->nbreaders = 0;
        e->status = 0;
        e->tile = NULL;
        e->tile_owner = -1;
    } else {
        e = NULL;
    }
    mesh->mesh[ n + m * mesh->nb ] = e;

    return 0;
}
