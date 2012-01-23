/*
 * Copyright (c) 2009-2011 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "remote_dep.h"
#include "scheduling.h"
#include "execution_unit.h"
#include <stdio.h>
#include <string.h>

#ifdef DISTRIBUTED
/* Clear the already forwarded remote dependency matrix */
static inline void remote_dep_reset_forwarded( dague_execution_unit_t* eu_context, dague_remote_deps_t* rdeps )
{
    /*DEBUG(("fw reset\tcontext %p\n", (void*) eu_context));*/
    memset(rdeps->remote_dep_fw_mask, 0, eu_context->master_context->remote_dep_fw_mask_sizeof);
}

/* Mark a rank as already forwarded the termination of the current task */
static inline void remote_dep_mark_forwarded( dague_execution_unit_t* eu_context, dague_remote_deps_t* rdeps, int rank )
{
    unsigned int boffset;
    uint32_t mask;
    
    /*DEBUG(("fw mark\tREMOTE rank %d\n", rank));*/
    boffset = rank / (8 * sizeof(uint32_t));
    mask = ((uint32_t)1) << (rank % (8 * sizeof(uint32_t)));
    assert(boffset <= eu_context->master_context->remote_dep_fw_mask_sizeof);
    rdeps->remote_dep_fw_mask[boffset] |= mask;
}

/* Check if rank has already been forwarded the termination of the current task */
static inline int remote_dep_is_forwarded( dague_execution_unit_t* eu_context, dague_remote_deps_t* rdeps, int rank )
{
    unsigned int boffset;
    uint32_t mask;
    
    boffset = rank / (8 * sizeof(uint32_t));
    mask = ((uint32_t)1) << (rank % (8 * sizeof(uint32_t)));
    assert(boffset <= eu_context->master_context->remote_dep_fw_mask_sizeof);
    /*DEBUG(("fw test\tREMOTE rank %d (value=%x)\n", rank, (int) (eu_context->remote_dep_fw_mask[boffset] & mask)));*/
    return (int) ((rdeps->remote_dep_fw_mask[boffset] & mask) != 0);
}


/* make sure we don't leave before serving all data deps */
static inline void remote_dep_inc_flying_messages(dague_object_t *dague_object, dague_context_t* ctx)
{
    dague_atomic_inc_32b( &(dague_object->nb_local_tasks) );
    (void)ctx;
}

/* allow for termination when all deps have been served */
static inline void remote_dep_dec_flying_messages(dague_object_t *dague_object, dague_context_t* ctx)
{
    __dague_complete_task(dague_object, ctx);
}

/* Mark that one of the remote deps is finished, and return the remote dep to
 * the free items queue if it is now done */
static void remote_dep_complete_one_and_cleanup(dague_remote_deps_t* deps) {
    deps->output_sent_count++;
    if(deps->output_count == deps->output_sent_count) {
        unsigned int count = 0;
        int k = 0;
        while( count < deps->output_count ) {
            for(uint32_t a = 0; a < (dague_remote_dep_context.max_nodes_number + 31)/32; a++)
                deps->output[k].rank_bits[a] = 0;
            count += deps->output[k].count;
            deps->output[k].count = 0;
#if defined(DAGUE_DEBUG)
            deps->output[k].data = NULL;
            deps->output[k].type = NULL;
#endif
            k++;
            assert(k < MAX_PARAM_COUNT);
        }
        assert(count == deps->output_count);
        deps->output_count = 0;
        deps->output_sent_count = 0;
#if defined(DAGUE_DEBUG)
        memset( &deps->msg, 0, sizeof(remote_dep_wire_activate_t) );
#endif
        dague_atomic_lifo_push(deps->origin,           
             dague_list_item_singleton((dague_list_item_t*) deps));
    }
}                                                                                           

#endif

#ifndef DAGUE_DIST_EAGER_LIMIT 
#define RDEP_MSG_EAGER_LIMIT    0
#else
#define RDEP_MSG_EAGER_LIMIT    (DAGUE_DIST_EAGER_LIMIT*1024)
#endif
#define RDEP_MSG_EAGER_SET(msg) ((msg)->which |= (((remote_dep_datakey_t)1)<<(8*sizeof(remote_dep_datakey_t)-1)))
#define RDEP_MSG_EAGER_CLR(msg) ((msg)->which &= ~(((remote_dep_datakey_t)1)<<(8*sizeof(remote_dep_datakey_t)-1)))
#define RDEP_MSG_EAGER(msg)     ((msg)->which & (((remote_dep_datakey_t)1)<<(8*sizeof(remote_dep_datakey_t)-1)))

#ifdef HAVE_MPI
#include "remote_dep_mpi.c" 

#else 
#endif /* NO TRANSPORT */


#ifdef DISTRIBUTED
int dague_remote_dep_init(dague_context_t* context)
{
    (void)remote_dep_init(context);

    if(context->nb_nodes > 1)
    {
        context->remote_dep_fw_mask_sizeof = ((context->nb_nodes + 31) / 32) * sizeof(uint32_t);
    }
    else 
    {
        context->remote_dep_fw_mask_sizeof = 0; /* hoping memset(0b) is fast */
    }
    return context->nb_nodes;
}

int dague_remote_dep_fini(dague_context_t* context)
{
    int rc = remote_dep_fini(context);
    remote_deps_allocation_fini();
    return rc;
}

int dague_remote_dep_on(dague_context_t* context)
{
    return remote_dep_on(context);
}

int dague_remote_dep_off(dague_context_t* context)
{
    return remote_dep_off(context);
}

int dague_remote_dep_progress(dague_execution_unit_t* eu_context)
{
    return remote_dep_progress(eu_context);
}


#ifdef DAGUE_DIST_COLLECTIVES
#define DAGUE_DIST_COLLECTIVES_TYPE_CHAINPIPELINE
#undef  DAGUE_DIST_COLLECTIVES_TYPE_BINOMIAL

# ifdef DAGUE_DIST_COLLECTIVES_TYPE_CHAINPIPELINE
static inline int remote_dep_bcast_chainpipeline_child(int me, int him)
{
    assert(him >= 0);
    if(me == -1) return 0;
    if(him == me+1) return 1;
    return 0;
}
#  define remote_dep_bcast_child(me, him) remote_dep_bcast_chainpipeline_child(me, him)

# elif defined(DAGUE_DIST_COLLECTIVES_TYPE_BINOMIAL)
static inline int remote_dep_bcast_binonial_child(int me, int him)
{
    int k, mask;
    
    /* flush out the easy cases first */
    assert(him >= 0);
    if(him == 0) return 0; /* root is child to nobody */
    if(me == -1) return 0; /* I don't even know who I am yet... */
    
    /* look for the leftmost 1 bit */
    for(k = sizeof(int) * 8 - 1; k >= 0; k--)
    {
        mask = 0x1<<k;
        if(him & mask)
        {
            him ^= mask;
            break;
        }
    }
    /* is the remainder suffix "me" ? */
    return him == me;
}
#  define remote_dep_bcast_child(me, him) remote_dep_bcast_binonial_child(me, him)

# else
#  error "INVALID COLLECTIVE TYPE. YOU MUST DEFINE ONE COLLECTIVE TYPE WHEN ENABLING COLLECTIVES"
# endif

#else
static inline int remote_dep_bcast_star_child(int me, int him)
{
    (void)him;
    if(me == 0) return 1;
    else return 0;
}
#  define remote_dep_bcast_child(me, him) remote_dep_bcast_star_child(me, him)
#endif

int dague_remote_dep_activate(dague_execution_unit_t* eu_context,
                              const dague_execution_context_t* exec_context,
                              dague_remote_deps_t* remote_deps,
                              uint32_t remote_deps_count )
{
    const dague_function_t* function = exec_context->function;
    int i, me, him, current_mask;
    unsigned int array_index, count, bit_index;
    
#if defined(DAGUE_DEBUG)
    char tmp[128];
    /* make valgrind happy */
    memset(&remote_deps->msg, 0, sizeof(remote_dep_wire_activate_t));
#endif

    remote_dep_reset_forwarded(eu_context, remote_deps);
    remote_deps->dague_object = exec_context->dague_object;
    remote_deps->output_count = remote_deps_count;
    remote_deps->msg.deps = (uintptr_t) remote_deps;
    remote_deps->msg.object_id   = exec_context->dague_object->object_id;
    remote_deps->msg.function_id = function->function_id;
    for(i = 0; i < function->nb_definitions; i++) {
        remote_deps->msg.locals[i] = exec_context->locals[i];
    }
    
    if(remote_deps->root == eu_context->master_context->my_rank) me = 0;
    else me = -1; 
    
    for( i = 0; remote_deps_count; i++) {
        if( 0 == remote_deps->output[i].count ) continue;
        
        him = 0;
        for( array_index = count = 0; count < remote_deps->output[i].count; array_index++ ) {
            current_mask = remote_deps->output[i].rank_bits[array_index];
            if( 0 == current_mask ) continue;  /* no bits here */
            for( bit_index = 0; (bit_index < (8 * sizeof(uint32_t))) && (current_mask != 0); bit_index++ ) {
                if( current_mask & (1 << bit_index) ) {
                    int rank = (array_index * sizeof(uint32_t) * 8) + bit_index;
                    assert(rank >= 0);
                    assert(rank < eu_context->master_context->nb_nodes);

                    current_mask ^= (1 << bit_index);
                    count++;
                    remote_deps_count--;

                    //DEBUG((" TOPO\t%s\troot=%d\t%d (d%d) -? %d (dna)\n", dague_service_to_string(exec_context, tmp, 128), remote_deps->root, eu_context->master_context->my_rank, me, rank));
                    
                    /* root already knows but falsely appear in this bitfield */
                    if(rank == remote_deps->root) continue;

                    if((me == -1) && (rank >= eu_context->master_context->my_rank))
                    {
                        /* the next bit points after me, so I know my dense rank now */
                        me = ++him;
                        if(rank == eu_context->master_context->my_rank) continue;
                    }
                    him++;
                    
                    if(remote_dep_bcast_child(me, him))
                    {
                        DEBUG((" TOPO\t%s\troot=%d\t%d (d%d) -> %d (d%d)\n", dague_service_to_string(exec_context, tmp, 128), remote_deps->root, eu_context->master_context->my_rank, me, rank, him));
                        
                        if(ACCESS_NONE != exec_context->function->out[i]->access_type)
                        {
                            AREF(remote_deps->output[i].data);
                            if((int)(remote_deps->output[i].type->elem_size) < RDEP_MSG_EAGER_LIMIT) {
                                RDEP_MSG_EAGER_SET(&remote_deps->msg);
                            } else {
                                RDEP_MSG_EAGER_CLR(&remote_deps->msg);
                            }
                            DEBUG((" RDEP\t%s\toutput=%d, type size=%d, eager=%lx\n",
                                   dague_service_to_string(exec_context, tmp, 128), i,
                                   (NULL == remote_deps->output[i].type ? 0 : remote_deps->output[i].type->elem_size), RDEP_MSG_EAGER(&remote_deps->msg)));
                        }
                        if(remote_dep_is_forwarded(eu_context, remote_deps, rank))
                        {
                            continue;
                        }
                        remote_dep_inc_flying_messages(exec_context->dague_object, eu_context->master_context);
                        remote_dep_mark_forwarded(eu_context, remote_deps, rank);
                        remote_dep_send(rank, remote_deps);
                    } else {
                        DEBUG((" TOPO\t%s\troot=%d\t%d (d%d) ][ %d (d%d)\n",
                               dague_service_to_string(exec_context, tmp, 128), remote_deps->root,
                               eu_context->master_context->my_rank, me, rank, him));
                    }
                }
            }
        }
    }
    return 0;
}

dague_remote_dep_context_t dague_remote_dep_context;
static int dague_remote_dep_inited = 0;

/* THIS FUNCTION MUST NOT BE CALLED WHILE REMOTE DEP IS ON. 
 * NOT THREAD SAFE (AND SHOULD NOT BE) */
void remote_deps_allocation_init(int np, int max_output_deps)
{
    /* First, if we have already allocated the list but it is now too tight,
     * lets redo it at the right size */
    if( dague_remote_dep_inited && (max_output_deps > (int)dague_remote_dep_context.max_dep_count) )
    {
        remote_deps_allocation_fini();
    }

    if( 0 == dague_remote_dep_inited ) {
        /* compute the maximum size of the dependencies array */
        int rankbits_size = sizeof(uint32_t) * ((np + 31)/32);
        dague_remote_deps_t fake_rdep;

        dague_remote_dep_context.max_dep_count = max_output_deps;
        dague_remote_dep_context.max_nodes_number = np;
        dague_remote_dep_context.elem_size = 
            /* sizeof(dague_remote_deps_t+outputs+padding) */
            ((intptr_t)&fake_rdep.output[dague_remote_dep_context.max_dep_count])-(intptr_t)&fake_rdep +
            /* One rankbits fw array per output param */
            dague_remote_dep_context.max_dep_count * rankbits_size +
            /* One extra rankbit to track the delivery of Activates */
            rankbits_size;
        dague_atomic_lifo_construct(&dague_remote_dep_context.freelist);
        dague_remote_dep_inited = 1;
    }

    assert( (int)dague_remote_dep_context.max_dep_count >= max_output_deps );
    assert( (int)dague_remote_dep_context.max_nodes_number >= np );
}


void remote_deps_allocation_fini(void)
{
    dague_remote_deps_t* rdeps;
        
    if(1 == dague_remote_dep_inited) {
        while(NULL != (rdeps = (dague_remote_deps_t*) dague_atomic_lifo_pop(&dague_remote_dep_context.freelist))) {
            free(rdeps);
        }
        dague_atomic_lifo_destruct(&dague_remote_dep_context.freelist);
    }
    dague_remote_dep_inited = 0;
} 

#endif /* DISTRIBUTED */

