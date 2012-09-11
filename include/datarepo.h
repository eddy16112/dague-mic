/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _datarepo_h_
#define _datarepo_h_

#include "dague_config.h"

typedef struct data_repo_entry data_repo_entry_t;
typedef struct data_repo_head  data_repo_head_t;
typedef struct data_repo       data_repo_t;

#include <stdlib.h>
#include "atomic.h"
#include "stats.h"
#include "debug.h"
#include "execution_unit.h"
#include "arena.h"

#define MAX_DATAREPO_HASH 4096

/**
 * Hash table:
 *  Because threads are allowed to use elements deposited in the hash table
 *  while we are still discovering how many times these elements will be used,
 *  it is necessary to
 *     - use a positive reference counting method only.
 *       Think for example if 10 threads decrement the reference counter, while
 *       the thread that pushes the element is still counting. The reference counter
 *       goes negative. Bad.
 *     - use a retaining facility.
 *       Think for example that the thread that pushes the element computed for now that
 *       the limit is going to be 3. While it's still exploring the dependencies, other
 *       threads use this element 3 times. The element is going to be removed while the
 *       pushing thread is still exploring, and SEGFAULT will occur.
 *
 *  An alternative solution consisted in having a function that will compute how many
 *  times the element will be used at creation time, and keep this for the whole life
 *  of the entry without changing it. But this requires to write a specialized function
 *  dumped by the precompiler, that will do a loop on the predicates. This was ruled out.
 *
 *  The element can still be inserted in the table, counted for some data (not all),
 *  used by some tasks (not all), removed from the table, then re-inserted when a
 *  new data arrives that the previous tasks did not depend upon.
 *
 *  Here is how it is used:
 *    the table is created with data_repo_create_nothreadsafe
 *    entries can be looked up with data_repo_lookup_entry (no side effect on the counters)
 *    entries are created using data_repo_lookup_entry_and_create. This turns the retained flag on.
 *    The same thread that called data_repo_lookup_entry_and_create must eventually call
 *    data_repo_entry_addto_usage_limit to set the usage limit and remove the retained flag.
 *    Between the two calls, any thread can call data_repo_lookup_entry and
 *    data_repo_entry_used_once if the entry has been "used". When data_repo_entry_addto_usage_limit
 *    has been called the same number of times as data_repo_lookup_entry_and_create and data_repo_entry_used_once
 *    has been called N times where N is the sum of the usagelmt parameters of data_repo_lookup_entry_and_create,
 *    the entry is garbage collected from the hash table. Notice that the values pointed by the entry
 *    are not collected.
 */

/**
 * data_repo_entries as mempool manageable elements:
 *  a mempool manageable element must be a dague_list_item_t,
 *  and it must have a pointer to it's own mempool_thread_t.
 * Thus, we use the dague_list_item_t to point to the next fields,
 * althgough this is not done atomically at the datarepo level (not
 * needed)
 *
 * The following #define are here to help port the code.
 */

#define data_repo_next_entry     data_repo_next_item.list_next

struct data_repo_entry {
    dague_list_item_t       data_repo_next_item;
    dague_thread_mempool_t* data_repo_mempool_owner;
    volatile uint32_t       usagecnt;
    volatile uint32_t       usagelmt;
    volatile uint32_t       retained;
    long int                key;
#if defined(DAGUE_SIM)
    int                     sim_exec_date;
#endif
    dague_arena_chunk_t    *data[1];
};

struct data_repo_head {
    volatile uint32_t  lock;
    uint32_t           size;
    data_repo_entry_t *first_entry;
};

struct data_repo {
    unsigned int      nbentries;
    unsigned int      nbdata;
    data_repo_head_t  heads[1];
};

static inline data_repo_t *data_repo_create_nothreadsafe(unsigned int hashsize, unsigned int nbdata)
{
    data_repo_t *res = (data_repo_t*)calloc(1, sizeof(data_repo_t) + sizeof(data_repo_head_t) * hashsize);
    res->nbentries = hashsize;
    res->nbdata = nbdata;
    DAGUE_STAT_INCREASE(mem_hashtable, sizeof(data_repo_t) + sizeof(data_repo_head_t) * (hashsize-1) + STAT_MALLOC_OVERHEAD);
    return res;
}

static inline data_repo_entry_t *data_repo_lookup_entry(data_repo_t *repo, long int key)
{
    data_repo_entry_t *e;
    int h = key % repo->nbentries;

    dague_atomic_lock(&repo->heads[h].lock);
    for(e = repo->heads[h].first_entry;
        e != NULL;
        e = (data_repo_entry_t *)e->data_repo_next_entry)
        if( e->key == key ) break;
    dague_atomic_unlock(&repo->heads[h].lock);

    return e;
}

/* If using lookup_and_create, don't forget to call add_to_usage_limit on the same entry when
 * you're done counting the number of references, otherwise the entry is non erasable.
 * See comment near the structure definition.
 */
static inline data_repo_entry_t*
data_repo_lookup_entry_and_create(dague_execution_unit_t *eu, data_repo_t *repo, long int key)
{
    data_repo_entry_t *e, *n;
    int h = key % repo->nbentries;

    dague_atomic_lock(&repo->heads[h].lock);
    for(e = repo->heads[h].first_entry;
        e != NULL;
        e = (data_repo_entry_t *)e->data_repo_next_entry)
        if( e->key == key ) {
            e->retained++; /* Until we update the usage limit */
            dague_atomic_unlock(&repo->heads[h].lock);
            return e;
        }
    dague_atomic_unlock(&repo->heads[h].lock);

    n = (data_repo_entry_t*)dague_thread_mempool_allocate( eu->datarepo_mempools[repo->nbdata] );
    n->data_repo_mempool_owner = eu->datarepo_mempools[repo->nbdata];
    n->key = key;
#if defined(DAGUE_SIM)
    n->sim_exec_date = 0;
#endif
    n->usagelmt = 0;
    n->usagecnt = 0;
    n->retained = 1; /* Until we update the usage limit */

    dague_atomic_lock(&repo->heads[h].lock);
    n->data_repo_next_entry = (volatile dague_list_item_t *)repo->heads[h].first_entry;
    repo->heads[h].first_entry = n;
    repo->heads[h].size++;
    DAGUE_STAT_INCREASE(mem_hashtable, sizeof(data_repo_entry_t)+(repo->nbdata-1)*sizeof(dague_arena_chunk_t*) + STAT_MALLOC_OVERHEAD);
    DAGUE_STATMAX_UPDATE(counter_hashtable_collisions_size, repo->heads[h].size);
    dague_atomic_unlock(&repo->heads[h].lock);

    return n;
}

#if defined(DAGUE_DEBUG_VERBOSE3)
# define data_repo_entry_used_once(eu, repo, key) __data_repo_entry_used_once(eu, repo, key, #repo, __FILE__, __LINE__)
static inline void __data_repo_entry_used_once(dague_execution_unit_t *eu, data_repo_t *repo, long int key, const char *tablename, const char *file, int line)
#else
# define data_repo_entry_used_once(eu, repo, key) __data_repo_entry_used_once(eu, repo, key)
static inline void __data_repo_entry_used_once(dague_execution_unit_t *eu, data_repo_t *repo, long int key)
#endif
{
    data_repo_entry_t *e, *p;
    int h = key % repo->nbentries;
    uint32_t r = 0xffffffff;

    dague_atomic_lock(&repo->heads[h].lock);
    p = NULL;
    for(e = repo->heads[h].first_entry;
        e != NULL;
        p = e, e = (data_repo_entry_t*)e->data_repo_next_entry)
        if( e->key == key ) {
            r = dague_atomic_inc_32b(&e->usagecnt);
            break;
        }

#ifdef DAGUE_DEBUG_VERBOSE3
    if( NULL == e ) {
        DEBUG3(("entry %ld of hash table %s could not be found at %s:%d\n", key, tablename, file, line));
    }
#endif
    assert( NULL != e );

    if( (e->usagelmt == r) && (0 == e->retained) ) {
        DEBUG3(("entry %p/%ld of hash table %s has a usage count of %u/%u and is not retained: freeing it at %s:%d\n",
                e, e->key, tablename, r, r, file, line));
        if( NULL != p ) {
            p->data_repo_next_entry = e->data_repo_next_entry;
        } else {
            repo->heads[h].first_entry = (data_repo_entry_t*)e->data_repo_next_entry;
        }
        repo->heads[h].size--;
        dague_atomic_unlock(&repo->heads[h].lock);

        dague_thread_mempool_free(e->data_repo_mempool_owner, e );
        DAGUE_STAT_DECREASE(mem_hashtable, sizeof(data_repo_entry_t)+(repo->nbdata-1)*sizeof(dague_arena_chunk_t*) + STAT_MALLOC_OVERHEAD);
    } else {
        DEBUG3(("entry %p/%ld of hash table %s has %u/%u usage count and %s retained: not freeing it, even if it's used at %s:%d\n",
                     e, e->key, tablename, r, e->usagelmt, e->retained ? "is" : "is not", file, line));
        dague_atomic_unlock(&repo->heads[h].lock);
    }
    (void)eu;
}

#if defined(DAGUE_DEBUG_VERBOSE3)
# define data_repo_entry_addto_usage_limit(repo, key, usagelmt) __data_repo_entry_addto_usage_limit(repo, key, usagelmt, #repo, __FILE__, __LINE__)
static inline void __data_repo_entry_addto_usage_limit(data_repo_t *repo, long int key, uint32_t usagelmt, const char *tablename, const char *file, int line)
#else
# define data_repo_entry_addto_usage_limit(repo, key, usagelmt) __data_repo_entry_addto_usage_limit(repo, key, usagelmt)
static inline void __data_repo_entry_addto_usage_limit(data_repo_t *repo, long int key, uint32_t usagelmt)
#endif
{
    data_repo_entry_t *e, *p;
    uint32_t ov, nv;
    int h = key % repo->nbentries;

    dague_atomic_lock(&repo->heads[h].lock);
    p = NULL;
    for(e = repo->heads[h].first_entry;
        e != NULL;
        p = e, e = (data_repo_entry_t*)e->data_repo_next_entry)
        if( e->key == key ) {
            assert(e->retained > 0);
            do {
                ov = e->usagelmt;
                nv = ov + usagelmt;
            } while( !dague_atomic_cas_32b( &e->usagelmt, ov, nv) );
            e->retained--;
            break;
        }

    assert( NULL != e );

    if( (e->usagelmt == e->usagecnt) && (0 == e->retained) ) {
        DEBUG3(("entry %p/%ld of hash table %s has a usage count of %u/%u and is not retained: freeing it at %s:%d\n",
                     e, e->key, tablename, e->usagecnt, e->usagelmt, file, line));
        if( NULL != p ) {
            p->data_repo_next_entry = e->data_repo_next_entry;
        } else {
            repo->heads[h].first_entry = (data_repo_entry_t*)e->data_repo_next_entry;
        }
        repo->heads[h].size--;
        dague_atomic_unlock(&repo->heads[h].lock);
        dague_thread_mempool_free(e->data_repo_mempool_owner, e );
        DAGUE_STAT_DECREASE(mem_hashtable, sizeof(data_repo_entry_t)+(repo->nbdata-1)*sizeof(dague_arena_chunk_t*) + STAT_MALLOC_OVERHEAD);
    } else {
        DEBUG3(("entry %p/%ld of hash table %s has a usage count of %u/%u and is %s retained at %s:%d\n",
                     e, e->key, tablename, e->usagecnt, e->usagelmt, e->retained ? "still" : "no more", file, line));
        dague_atomic_unlock(&repo->heads[h].lock);
    }
}

static inline void data_repo_destroy_nothreadsafe(data_repo_t *repo)
{
    DAGUE_STAT_DECREASE(mem_hashtable,  sizeof(data_repo_t) + sizeof(data_repo_head_t) * (repo->nbentries-1) + STAT_MALLOC_OVERHEAD);
    free(repo);
}

#endif /* _datarepo_h_ */
