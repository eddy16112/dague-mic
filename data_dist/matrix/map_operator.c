/*
 * Copyright (c) 2011      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague.h"
#include "matrix.h"
#include "dague_prof_grapher.h"
#include <scheduling.h>

#if defined(DAGUE_PROF_TRACE)
int dague_map_operator_profiling_array[2] = {-1};
#define TAKE_TIME(context, key, eid, refdesc, refid) do {   \
   dague_profile_ddesc_info_t info;                         \
   info.desc = (dague_ddesc_t*)refdesc;                     \
   info.id = refid;                                         \
   dague_profiling_trace(context->eu_profile,               \
                         __dague_object->super.super.profiling_array[(key)],\
                         eid, (void*)&info);                \
  } while(0);
#else
#define TAKE_TIME(context, key, id, refdesc, refid)
#endif

typedef struct dague_map_operator_object {
    dague_object_t             super;
    const tiled_matrix_desc_t* src;
          tiled_matrix_desc_t* dest;
    volatile uint32_t          next_k;
    dague_operator_t           op;
    void*                      op_data;
} dague_map_operator_object_t;

typedef struct __dague_map_operator_object {
    dague_map_operator_object_t super;
} __dague_map_operator_object_t;

static const dague_flow_t flow_of_map_operator;
static const dague_function_t dague_map_operator;

#define src(k,n)  (((dague_ddesc_t*)__dague_object->super.src)->data_of((dague_ddesc_t*)__dague_object->super.src, (k), (n)))
#define dest(k,n)  (((dague_ddesc_t*)__dague_object->super.dest)->data_of((dague_ddesc_t*)__dague_object->super.dest, (k), (n)))

#if defined(DAGUE_PROF_TRACE)
static inline uint32_t map_operator_op_hash(const __dague_map_operator_object_t *o, int k, int n )
{
    return o->super.src->mt * k + n;
}
#endif  /* defined(DAGUE_PROF_TRACE) */

static inline int minexpr_of_row_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
    const __dague_map_operator_object_t *__dague_object = (const __dague_map_operator_object_t*)__dague_object_parent;
    (void)assignments;
    return __dague_object->super.src->i;
}
static const expr_t minexpr_of_row = {
    .op = EXPR_OP_INLINE,
    .inline_func = minexpr_of_row_fct
};
static inline int maxexpr_of_row_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
    const __dague_map_operator_object_t *__dague_object = (const __dague_map_operator_object_t*)__dague_object_parent;

    (void)__dague_object;
    (void)assignments;
    return __dague_object->super.src->mt;
}
static const expr_t maxexpr_of_row = {
    .op = EXPR_OP_INLINE,
    .inline_func = maxexpr_of_row_fct
};
static const symbol_t symb_row = {
    .min = &minexpr_of_row,
    .max = &maxexpr_of_row,
    .flags = DAGUE_SYMBOL_IS_STANDALONE
};

static inline int minexpr_of_column_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
    const __dague_map_operator_object_t *__dague_object = (const __dague_map_operator_object_t*)__dague_object_parent;
    (void)assignments;
    return __dague_object->super.src->j;
}

static const expr_t minexpr_of_column = {
    .op = EXPR_OP_INLINE,
    .inline_func = minexpr_of_column_fct
};

static inline int maxexpr_of_column_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
    const __dague_map_operator_object_t *__dague_object = (const __dague_map_operator_object_t*)__dague_object_parent;

    (void)__dague_object;
    (void)assignments;
    return __dague_object->super.src->nt;
}
static const expr_t maxexpr_of_column = {
    .op = EXPR_OP_INLINE,
    .inline_func = maxexpr_of_column_fct
};
static const symbol_t symb_column = {
    .min = &minexpr_of_column,
    .max = &maxexpr_of_column,
    .flags = DAGUE_SYMBOL_IS_STANDALONE
};

static inline int pred_of_map_operator_all_as_expr_fct(const dague_object_t *__dague_object_parent,
                                                const assignment_t *assignments)
{
    const __dague_map_operator_object_t *__dague_object = (const __dague_map_operator_object_t*)__dague_object_parent;

    /* Silent Warnings: should look into predicate to know what variables are usefull */
    (void)__dague_object;
    (void)assignments;
    /* Compute Predicate */
    return 1;
}
static const expr_t pred_of_map_operator_all_as_expr = {
    .op = EXPR_OP_INLINE,
    .inline_func = pred_of_map_operator_all_as_expr_fct
};

static inline int
expr_of_p1_for_flow_of_map_operator_dep_in_fct(const dague_object_t *__dague_object_parent,
                                                const assignment_t *assignments)
{
    (void)__dague_object_parent;
    return assignments[0].value;
}
static const expr_t expr_of_p1_for_flow_of_map_operator_dep_in = {
    .op = EXPR_OP_INLINE,
    .inline_func = expr_of_p1_for_flow_of_map_operator_dep_in_fct
};
static const dep_t flow_of_map_operator_dep_in = {
    .cond = NULL,
    .dague = &dague_map_operator,
    .flow = &flow_of_map_operator,
    .datatype_index = 0,
    .call_params = {
        &expr_of_p1_for_flow_of_map_operator_dep_in
    }
};

static inline int
expr_of_p1_for_flow_of_map_operator_dep_out_fct(const dague_object_t *__dague_object_parent,
                                                 const assignment_t *assignments)
{
    (void)__dague_object_parent;
    return (assignments[0].value + 1);
}
static const expr_t expr_of_p1_for_flow_of_map_operator_dep_out = {
    .op = EXPR_OP_INLINE,
    .inline_func = expr_of_p1_for_flow_of_map_operator_dep_out_fct
};
static const dep_t flow_of_map_operator_dep_out = {
    .cond = NULL,
    .dague = &dague_map_operator,
    .flow = &flow_of_map_operator,
    .datatype_index = 0,
    .call_params = {
        &expr_of_p1_for_flow_of_map_operator_dep_out
    }
};

static const dague_flow_t flow_of_map_operator = {
    .name = "I",
    .sym_type = SYM_INOUT,
    .access_type = ACCESS_RW,
    .flow_index = 0,
    .dep_in  = { &flow_of_map_operator_dep_in },
    .dep_out = { &flow_of_map_operator_dep_out }
};

static dague_ontask_iterate_t
add_task_to_list(struct dague_execution_unit *eu_context,
                 dague_execution_context_t *newcontext,
                 dague_execution_context_t *oldcontext,
                 int flow_index, int outdep_index,
                 int rank_src, int rank_dst,
                 dague_arena_t* arena,
                 void *flow)
{
    dague_execution_context_t** pready_list = (dague_execution_context_t**)flow;
    dague_execution_context_t* new_context = (dague_execution_context_t*)dague_thread_mempool_allocate( eu_context->context_mempool );
    dague_thread_mempool_t* mpool = new_context->mempool_owner;

    memcpy( new_context, newcontext, sizeof(dague_execution_context_t) );
    new_context->mempool_owner = mpool;

    dague_list_add_single_elem_by_priority( pready_list, new_context );
    (void)arena; (void)oldcontext; (void)flow_index; (void)outdep_index; (void)rank_src; (void)rank_dst;
    return DAGUE_ITERATE_STOP;
}

static void iterate_successors(dague_execution_unit_t *eu,
                               dague_execution_context_t *this_task,
                               uint32_t action_mask,
                               dague_ontask_function_t *ontask,
                               void *ontask_arg)
{
    __dague_map_operator_object_t *__dague_object = (__dague_map_operator_object_t*)this_task->dague_object;
    int k = this_task->locals[0].value;
    int n = this_task->locals[1].value+1;
    dague_execution_context_t nc;

    nc.priority = 0;
    nc.data[0].data_repo = NULL;
    nc.data[0].data_repo = NULL;
    /* If this is the last n, try to move to the next k */
    for( ; k < (int)__dague_object->super.src->nt; n = 0) {
        for( ; n < (int)__dague_object->super.src->mt; n++ ) {
            int is_local = (__dague_object->super.src->super.myrank ==
                            ((dague_ddesc_t*)__dague_object->super.src)->rank_of((dague_ddesc_t*)__dague_object->super.src,
                                                                               k, n));
            if( !is_local ) continue;
            /* Here we go, one ready local task */
            nc.locals[0].value = k;
            nc.locals[1].value = n;
            nc.function = &dague_map_operator /*this*/;
            nc.dague_object = this_task->dague_object;
            nc.data[0].data = this_task->data[0].data;
            nc.data[1].data = this_task->data[1].data;
            ontask(eu, &nc, this_task, 0, 0,
                   __dague_object->super.src->super.myrank,
                   __dague_object->super.src->super.myrank, NULL, ontask_arg);
            return;
        }
        /* Go to the next row ... atomically */
        k = dague_atomic_inc_32b( &__dague_object->super.next_k );
    }
    (void)action_mask;
}

static int release_deps(dague_execution_unit_t *eu,
                        dague_execution_context_t *this_task,
                        uint32_t action_mask,
                        dague_remote_deps_t *deps)
{
    dague_execution_context_t* ready_list = NULL;

    iterate_successors(eu, this_task, action_mask, add_task_to_list, &ready_list);

    if(action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS) {
        if( NULL != ready_list ) {
            __dague_schedule(eu, ready_list);
            ready_list = NULL;
        }
    }

    if(action_mask & DAGUE_ACTION_RELEASE_LOCAL_REFS) {
        (void)AUNREF(this_task->data[0].data);
    }

    assert( NULL == ready_list );
    (void)deps;
    return 1;
}

static int hook_of(dague_execution_unit_t *context,
                   dague_execution_context_t *this_task)
{
    const __dague_map_operator_object_t *__dague_object = (const __dague_map_operator_object_t*)this_task->dague_object;
    int k = this_task->locals[0].value;
    int n = this_task->locals[1].value;
    dague_arena_chunk_t *asrc = NULL, *adest;
    const void* src_data = NULL;
    void* dest_data;

    if( NULL != __dague_object->super.src ) {
        asrc = (dague_arena_chunk_t*) src(k,n);
        src_data = ADATA(asrc);
    }
    adest = (dague_arena_chunk_t*) dest(k,n);
    dest_data = ADATA(adest);

    this_task->data[0].data = asrc;
    this_task->data[0].data_repo = NULL;
    this_task->data[1].data = adest;
    this_task->data[1].data_repo = NULL;

#if !defined(DAGUE_PROF_DRY_BODY)
    TAKE_TIME(context, 2*this_task->function->function_id,
              map_operator_op_hash( __dague_object, k, n ), __dague_object->super.src,
              ((dague_ddesc_t*)(__dague_object->super.src))->data_key((dague_ddesc_t*)__dague_object->super.src, k, n) );
    __dague_object->super.op( context, src_data, dest_data, __dague_object->super.op_data, k, n );
#endif
    (void)context;
    return 0;
}

static int complete_hook(dague_execution_unit_t *context,
                         dague_execution_context_t *this_task)
{
    const __dague_map_operator_object_t *__dague_object = (const __dague_map_operator_object_t *)this_task->dague_object;
    int k = this_task->locals[0].value;
    int n = this_task->locals[1].value;
    (void)k; (void)n; (void)__dague_object;

    TAKE_TIME(context, 2*this_task->function->function_id+1, map_operator_op_hash( __dague_object, k, n ), NULL, 0);

    dague_prof_grapher_task(this_task, context->eu_id, k+n);

    release_deps(context, this_task,
                 (DAGUE_ACTION_RELEASE_REMOTE_DEPS |
                  DAGUE_ACTION_RELEASE_LOCAL_DEPS |
                  DAGUE_ACTION_RELEASE_LOCAL_REFS |
                  DAGUE_ACTION_DEPS_MASK),
                 NULL);

    return 0;
}

static const dague_function_t dague_map_operator = {
    .name = "map_operator",
    .deps = 0,
    .flags = 0x0,
    .function_id = 0,
    .dependencies_goal = 0x1,
    .nb_definitions = 2,
    .nb_parameters = 2,
    .params = { &symb_row, &symb_column },
    .locals = { &symb_row, &symb_column },
    .pred = &pred_of_map_operator_all_as_expr,
    .priority = NULL,
    .in = { &flow_of_map_operator },
    .out = { &flow_of_map_operator },
    .iterate_successors = iterate_successors,
    .release_deps = release_deps,
    .hook = hook_of,
    .complete_execution = complete_hook,
};

static void dague_map_operator_startup_fn(dague_context_t *context, 
                                          dague_object_t *dague_object,
                                          dague_execution_context_t** startup_list)
{
    __dague_map_operator_object_t *__dague_object = (__dague_map_operator_object_t*)dague_object;
    dague_execution_context_t fake_context;
    dague_execution_context_t *ready_list;
    int k = 0, n = 0, count = 0;
    dague_execution_unit_t* eu;

    *startup_list = NULL;
    fake_context.function = &dague_map_operator /*this*/;
    fake_context.dague_object = dague_object;
    fake_context.priority = 0;
    fake_context.data[0].data_repo = NULL;
    fake_context.data[0].data      = NULL;
    fake_context.data[1].data_repo = NULL;
    fake_context.data[1].data      = NULL;
    /* If this is the last n, try to move to the next k */
    for( ; k < (int)__dague_object->super.src->nt; n = 0) {
        eu = context->execution_units[count];
        ready_list = NULL;

        for( ; n < (int)__dague_object->super.src->mt; n++ ) {
            int is_local = (__dague_object->super.src->super.myrank ==
                            ((dague_ddesc_t*)__dague_object->super.src)->rank_of((dague_ddesc_t*)__dague_object->super.src,
                                                                               k, n));
            if( !is_local ) continue;
            /* Here we go, one ready local task */
            fake_context.locals[0].value = k;
            fake_context.locals[1].value = n;
            add_task_to_list(eu, &fake_context, NULL, 0, 0,
                             __dague_object->super.src->super.myrank,
                             __dague_object->super.src->super.myrank, NULL, (void*)&ready_list);
            __dague_schedule( eu, ready_list );
            count++;
            if( count == context->nb_cores ) goto done;
            break;
        }
        /* Go to the next row ... atomically */
        k = dague_atomic_inc_32b( &__dague_object->super.next_k );
    }
 done:
    return;
}

/**
 * Apply the operator op on all tiles of the src matrix. The src matrix is const, the
 * result is supposed to be pushed on the dest matrix. However, any of the two matrices
 * can be NULL, and then the data is reported as NULL in the corresponding op
 * floweter.
 */
struct dague_object_t*
dague_map_operator_New(const tiled_matrix_desc_t* src,
                       tiled_matrix_desc_t* dest,
                       dague_operator_t op,
                       void* op_data)
{
    __dague_map_operator_object_t *res = (__dague_map_operator_object_t*)calloc(1, sizeof(__dague_map_operator_object_t));

    if( (NULL == src) && (NULL == dest) )
        return NULL;
    /* src and dest should have similar distributions */
    /* TODO */

    res->super.src     = src;
    res->super.dest    = dest;
    res->super.op      = op;
    res->super.op_data = op_data;

#  if defined(DAGUE_PROF_TRACE)
    res->super.super.profiling_array = dague_map_operator_profiling_array;
    if( -1 == dague_map_operator_profiling_array[0] ) {
        dague_profiling_add_dictionary_keyword("operator", "fill:CC2828",
                                               sizeof(dague_profile_ddesc_info_t), dague_profile_ddesc_key_to_string,
                                               (int*)&res->super.super.profiling_array[0 + 2 * dague_map_operator.function_id],
                                               (int*)&res->super.super.profiling_array[1 + 2 * dague_map_operator.function_id]);
    }
#  endif /* defined(DAGUE_PROF_TRACE) */

    res->super.super.object_id = 1111;
    res->super.super.nb_local_tasks = src->nb_local_tiles;
    res->super.super.startup_hook = dague_map_operator_startup_fn;
    return (struct dague_object_t*)res;
}

void dague_map_operator_Destruct( struct dague_object_t* o )
{
#if defined(DAGUE_PROF_TRACE)
    char* filename = NULL;
#if defined(HAVE_MPI)
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    asprintf(&filename, "%s.%d.profile", "operator", rank);
#else
    asprintf(&filename, "%s.profile", "operator");
#endif
    dague_profiling_dump_xml(filename);
    free(filename);
#endif  /* defined(DAGUE_PROF_TRACE) */
    (void)o;
}
