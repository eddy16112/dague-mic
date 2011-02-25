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

static dague_linked_list_t  used_tiles;
static dague_linked_list_t *dirty_tiles = NULL;
static dague_linked_list_t *clean_tiles = NULL;
static dague_atomic_lifo_t *free_tiles = NULL;
static int dague_tssm_nbthreads = 0;

static void *dague_tssm_thread_init(void *_p)
{
    uint64_t *p = (uint64_t*)_p;
    int i;
    dague_list_item_t *tile;
    uint64_t threadid = p[0];
    uint64_t nbtilesperthread = p[1];
    uint64_t tile_size = p[2];

    dague_bindthread(threadid);

    dague_linked_list_construct( &dirty_tiles[threadid] );
    dague_linked_list_construct( &clean_tiles[threadid] );
    dague_atomic_lifo_construct( &free_tiles[threadid] );
    for(i = 0; i < nbtilesperthread; i++) {
        DAGUE_LIFO_ELT_ALLOC(tile, MAX(sizeof(dague_list_item_t), tile_size));
        DAGUE_LIST_ITEM_SINGLETON(tile);
        dague_atomic_lifo_push(&free_tiles[threadid], tile);
    }

    return NULL;
}

void dague_tssm_init(uint32_t nbthreads, size_t tile_size, uint32_t nbtilesperthread)
{
    int i;
    pthread_t *tid;
    uint64_t **tp;

    assert( 0 == dague_tssm_nbthreads );
    assert( NULL == dirty_tiles );
    assert( NULL == clean_tiles );
    assert( NULL == free_tiles );

    dague_tssm_nbthreads = nbthreads;

    dague_linked_list_construct( &used_tiles );
    dirty_tiles = (dague_linked_list_t*)malloc( nbthreads * sizeof(dague_linked_list_t) );
    clean_tiles = (dague_linked_list_t*)malloc( nbthreads * sizeof(dague_linked_list_t) );
    free_tiles = (dague_atomic_lifo_t*)malloc( nbthreads * sizeof(dague_atomic_lifo_t) );

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

static int dague_tssm_cleanup_some_tile(int thid)
{
    dague_tssm_tile_entry_t *victim;
    do {
        victim = (dague_tssm_tile_entry_t *)dague_linked_list_remove_head( &dirty_tiles[thid] );
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
            dague_tssm_sparse_tile_pack(victim->tile, victim->m, victim->n, 
                                        victim->desc->mb, victim->desc->nb, 
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
            for(thid = ((this_thread + 1) % dague_tssm_nbthreads);
                thid != this_thread;
                thid = ((thid + 1) % dague_tssm_nbthreads) ) {
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
    uint64_t m, n;
    va_list ap;
    dague_tssm_desc_t *mat = (dague_tssm_desc_t *)desc;
    dague_tssm_tile_entry_t *tptr;
    int write_access;
    int this_thread;

    va_start(ap, desc);
    m = va_arg(ap, uint64_t);
    n = va_arg(ap, uint64_t);
    write_access = va_arg(ap, int);
    this_thread = va_arg(ap, int);
    va_end(ap);

    assert( NULL != mat->mesh );
    assert( (m < mat->mt) && (n < mat->nt) );

    tptr = (dague_tssm_tile_entry_t *)mat->mesh[ m * mat->nt + n ];
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
        
        dague_tssm_sparse_tile_unpack(tptr->tile, m, n, mat->mb, mat->nb, tptr->packed_ptr);
        
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
    uint64_t m, n;
    va_list ap;
    dague_tssm_desc_t *mat = (dague_tssm_desc_t *)desc;
    dague_tssm_tile_entry_t *tptr;
    int write_access;
    int this_thread;

    va_start(ap, desc);
    m = va_arg(ap, uint64_t);
    n = va_arg(ap, uint64_t);
    write_access = va_arg(ap, int);
    this_thread = va_arg(ap, int);
    va_end(ap);

    assert( NULL != mat->mesh );
    assert( (m < mat->mt) && (n < mat->nt) );
    tptr = (dague_tssm_tile_entry_t *)mat->mesh[ m * mat->nt + n ];
    if( NULL == tptr ) {
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
    if( (tptr->nbreaders == 0) && (tptr->writer == -1) ) {
        if( tptr->status & TILE_STATUS_DIRTY ) {
            dague_tssm_move_tiles_locked( &dirty_tiles[tptr->tile_owner], tptr );
        } else {
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
    assert( (m < mesh->mt) && (n < mesh->nt) );
    if( NULL != packed_ptr ) {
        e = (dague_tssm_tile_entry_t*)calloc(1, sizeof(dague_tssm_tile_entry_t));
        DAGUE_LIST_ITEM_SINGLETON( &e->super );
        e->current_list = NULL;
        e->packed_ptr = packed_ptr;
        e->m = m;
        e->n = n;
        e->lock = 0;
        e->writer = -1;
        e->nbreaders = 0;
        e->status = 0;
        e->tile = NULL;
        e->tile_owner = -1;
        e->desc = mesh;
        assert( mesh->mb == mb &&
                mesh->nb == nb );
    } else {
        e = NULL;
    }
    mesh->mesh[ m * mesh->nt + n ] = e;
}

dague_ddesc_t *dague_tssm_create_matrix(uint64_t mt, uint64_t nt, uint32_t mb, uint32_t nb,
                                        dague_sparse_input_symbol_matrix_t *sm)
{
    dague_ddesc_t *res;
    dague_tssm_desc_t *mat;

    assert( dague_tssm_nbthreads != 0 );

    res = (dague_ddesc_t*)calloc(1, sizeof(dague_tssm_desc_t));
    mat = (dague_tssm_desc_t *)res;

    mat->nt = nt;
    mat->mt = mt;
    mat->nb = nb;
    mat->mb = mb;
    mat->data_size = ELEM_SIZE;
    
    res->myrank = 0;
    res->cores = dague_tssm_nbthreads;
    res->nodes = 1;
    res->rank_of = rank_of;
    res->data_of = data_of;
    res->data_release = data_release;

    mat->mesh = (dague_tssm_tile_entry_t **)calloc( nt * mt, sizeof(dague_tssm_tile_entry_t*));
    /* Init mat->mesh using Anthony load function around here */
    dague_sparse_input_to_tiles_load(mat, mt, nt, mb, nb, sm);

    return res;
}

#if defined(DAGUE_DEBUG)
int dague_tssm_flush_matrix(dague_ddesc_t *_mat)
{
    dague_tssm_desc_t *mat = (dague_tssm_desc_t*)_mat;
    uint64_t m, n, errors;
    dague_tssm_tile_entry_t *tptr;
    uint32_t thid;

    errors = 0;
    for(m = 0; m < mat->mt; m++) {
        for(n = 0; n < mat->nt; n++) {
            tptr = mat->mesh[ m * mat->nt + n ];
            if( NULL == tptr )
                continue;
            dague_atomic_lock( &tptr->lock );
            if( tptr->status != 0 ||
                tptr->writer != -1 ||
                tptr->nbreaders != 0 ||
                tptr->current_list == &used_tiles ) {
                dague_atomic_unlock( &tptr->lock );
                fprintf(stderr, "dague:tssm:flush_matrix failed: [%lu, %lu] is not available for flushing\n",
                        m, n);
                errors++;
                continue;
            }
            if( NULL == tptr->tile )
                continue;
            for( thid = 0; thid < dague_tssm_nbthreads; thid++ ) {
                if( tptr->current_list == &clean_tiles[thid] ) {
                    dague_linked_list_remove_item(tptr->current_list, &tptr->super);
                    tptr->current_list = NULL;
                    dague_atomic_lifo_push( &free_tiles[ tptr->tile_owner ], (dague_list_item_t*)tptr->tile );
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
                        
                        dague_tssm_sparse_tile_pack(tptr->tile, tptr->m, tptr->n, 
                                                    mat->mb, mat->nb, 
                                                    tptr->packed_ptr);

                         dague_atomic_lifo_push( &free_tiles[ tptr->tile_owner ], (dague_list_item_t*)tptr->tile );
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
        dague_atomic_lifo_push( &free_tiles[ tptr->tile_owner ], (dague_list_item_t*)tptr->tile );
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
        
        dague_tssm_sparse_tile_pack(tptr->tile, tptr->m, tptr->n, 
                                    mat->mb, mat->nb, 
                                    tptr->packed_ptr);

        dague_atomic_lifo_push( &free_tiles[ tptr->tile_owner ], (dague_list_item_t*)tptr->tile );
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
