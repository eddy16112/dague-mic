/* 
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights 
 *                         reserved. 
 */

#ifndef DEQUEUE_H_HAS_BEEN_INCLUDED
#define DEQUEUE_H_HAS_BEEN_INCLUDED

#include "atomic.h"
#include "lifo.h"

typedef struct dplasma_dequeue_t {
    dplasma_list_item_t  ghost_element;
    uint32_t atomic_lock;
} dplasma_dequeue_t;

static inline void dplasma_dequeue_construct( dplasma_dequeue_t* dequeue )
{
    dequeue->ghost_element.list_next = &(dequeue->ghost_element);
    dequeue->ghost_element.list_prev = &(dequeue->ghost_element);
    dequeue->atomic_lock = 0;
}

static inline void dplamsa_dequeue_item_construct( dplasma_list_item_t *item )
{
    item->list_prev = item;
}

static inline int dplasma_dequeue_is_empty( dplasma_dequeue_t * dequeue )
{
    int res;
    
    dplasma_atomic_lock(&(dequeue->atomic_lock));
    
    res = (dequeue->ghost_element.list_prev == &(dequeue->ghost_element)) 
       && (dequeue->ghost_element.list_next == &(dequeue->ghost_element));
    
    dplasma_atomic_unlock(&(dequeue->atomic_lock));
    return res;
}

static inline dplasma_list_item_t* dplasma_dequeue_pop_back( dplasma_dequeue_t* dequeue )
{
    dplasma_list_item_t* item;

    if( !dplasma_atomic_trylock(&(dequeue->atomic_lock)) ) {
        return NULL;
    }

    item = (dplasma_list_item_t*)dequeue->ghost_element.list_prev;
    dequeue->ghost_element.list_prev = item->list_prev;
    item->list_prev->list_next = &(dequeue->ghost_element);
    
    dplasma_atomic_unlock(&(dequeue->atomic_lock));

    if( &(dequeue->ghost_element) == item )
        return NULL;

    return item;
}

static inline dplasma_list_item_t* dplasma_dequeue_pop_front( dplasma_dequeue_t* dequeue )
{
    dplasma_list_item_t* item;

    if( !dplasma_atomic_trylock(&(dequeue->atomic_lock)) ) {
        return NULL;
    }

    item = (dplasma_list_item_t*)dequeue->ghost_element.list_next;
    dequeue->ghost_element.list_next = item->list_next;
    item->list_next->list_prev = &(dequeue->ghost_element);
    
    dplasma_atomic_unlock(&(dequeue->atomic_lock));

    if( &(dequeue->ghost_element) == item )
        return NULL;
    return item;
}

static inline void dplasma_dequeue_push_back(dplasma_dequeue_t* dequeue, dplasma_list_item_t* items )
{
    dplasma_list_item_t* tail = (dplasma_list_item_t*)items->list_prev;

    tail->list_next = &(dequeue->ghost_element);

    dplasma_atomic_lock(&(dequeue->atomic_lock));

    items->list_prev = dequeue->ghost_element.list_prev;
    items->list_prev->list_next = items;
    dequeue->ghost_element.list_prev = tail;

    dplasma_atomic_unlock(&(dequeue->atomic_lock));
}

static inline void dplasma_dequeue_push_front(dplasma_dequeue_t* dequeue, dplasma_list_item_t* items )
{
    dplasma_list_item_t* tail = (dplasma_list_item_t*)items->list_prev;

    items->list_prev = &(dequeue->ghost_element);

    dplasma_atomic_lock(&(dequeue->atomic_lock));

    tail->list_next = dequeue->ghost_element.list_next;
    tail->list_next->list_prev = tail;
    dequeue->ghost_element.list_next = items;

    dplasma_atomic_unlock(&(dequeue->atomic_lock));
}

#endif  /* DEQUEUE_H_HAS_BEEN_INCLUDED */
