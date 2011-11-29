/*
 * Copyright (c) 2009-2011 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "dague.h"
#include "stats.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <pthread.h>
#include <errno.h>
#if defined(HAVE_GETOPT_H)
#include <getopt.h>
#endif  /* defined(HAVE_GETOPT_H) */
#include "scheduling.h"
#include "dequeue.h"
#include "barrier.h"
#include "remote_dep.h"
#include "bindthread.h"
#include "dague_prof_grapher.h"
#include "priority_sorted_queue.h"

#ifdef DAGUE_PROF_TRACE
#include "profiling.h"
#endif

#ifdef HAVE_PAPI
#include <papime.h>
#endif

#ifdef HAVE_HWLOC
#include "hbbuffer.h"
#include "dague_hwloc.h"
#endif

#ifdef HAVE_CUDA
#include "cuda.h"
#include "cublas.h"
#include "cuda_runtime_api.h"
#endif

dague_allocate_data_t dague_data_allocate = malloc;
dague_free_data_t     dague_data_free = free;

#ifdef DAGUE_PROF_TRACE
int MEMALLOC_start_key, MEMALLOC_end_key;
int schedule_poll_begin, schedule_poll_end;
int schedule_push_begin, schedule_push_end;
int schedule_sleep_begin, schedule_sleep_end;
#endif  /* DAGUE_PROF_TRACE */

#ifdef HAVE_PAPI
int eventSet = PAPI_NULL;
int num_events = 0;
char* event_names[MAX_EVENTS];
#endif

#if defined(HAVE_GETRUSAGE)
#include <sys/time.h>
#include <sys/resource.h>

static int _dague_rusage_first_call = 1;
static struct rusage _dague_rusage;

static void dague_object_empty_repository(void);

static void dague_statistics(char* str)
{
    struct rusage current;

    getrusage(RUSAGE_SELF, &current);

    if ( !_dague_rusage_first_call ) {
        double usr, sys;

        usr = ((current.ru_utime.tv_sec - _dague_rusage.ru_utime.tv_sec) +
               (current.ru_utime.tv_usec - _dague_rusage.ru_utime.tv_usec) / 1000000.0);
        sys = ((current.ru_stime.tv_sec - _dague_rusage.ru_stime.tv_sec) +
               (current.ru_stime.tv_usec - _dague_rusage.ru_stime.tv_usec) / 1000000.0);

        printf("=============================================================\n");
        printf("%s: Resource Usage Data...\n", str);
        printf("-------------------------------------------------------------\n");
        printf("User Time   (secs)          : %10.3f\n", usr);
        printf("System Time (secs)          : %10.3f\n", sys);
        printf("Total Time  (secs)          : %10.3f\n", usr + sys);
        printf("Minor Page Faults           : %10ld\n", (current.ru_minflt  - _dague_rusage.ru_minflt));
        printf("Major Page Faults           : %10ld\n", (current.ru_majflt  - _dague_rusage.ru_majflt));
        printf("Swap Count                  : %10ld\n", (current.ru_nswap   - _dague_rusage.ru_nswap));
        printf("Voluntary Context Switches  : %10ld\n", (current.ru_nvcsw   - _dague_rusage.ru_nvcsw));
        printf("Involuntary Context Switches: %10ld\n", (current.ru_nivcsw  - _dague_rusage.ru_nivcsw));
        printf("Block Input Operations      : %10ld\n", (current.ru_inblock - _dague_rusage.ru_inblock));
        printf("Block Output Operations     : %10ld\n", (current.ru_oublock - _dague_rusage.ru_oublock));
        printf("=============================================================\n");
    }

    _dague_rusage_first_call = !_dague_rusage_first_call;
    _dague_rusage = current;

    return;
}
#else
static void dague_statistics(char* str) { (void)str; return; }
#endif /* defined(HAVE_GETRUSAGE) */


const dague_function_t* dague_find(const dague_object_t *dague_object, const char *fname)
{
    unsigned int i;
    const dague_function_t* object;

    for( i = 0; i < dague_object->nb_functions; i++ ) {
        object = dague_object->functions_array[i];
        if( 0 == strcmp( object->name, fname ) ) {
            return object;
        }
    }
    return NULL;
}

typedef struct __dague_temporary_thread_initialization_t {
    dague_context_t* master_context;
    int th_id;
    int nb_cores;
    int bindto;
} __dague_temporary_thread_initialization_t;

static void* __dague_thread_init( __dague_temporary_thread_initialization_t* startup )
{
    dague_execution_unit_t* eu;
    int pi;

    /* Bind to the specified CORE */
    dague_bindthread(startup->bindto);

    eu = (dague_execution_unit_t*)malloc(sizeof(dague_execution_unit_t));
    if( NULL == eu ) {
        return NULL;
    }
    eu->eu_id          = startup->th_id;
    eu->master_context = startup->master_context;
    eu->scheduler_object = NULL;
    (startup->master_context)->execution_units[startup->th_id] = eu;

#if defined(DAGUE_SCHED_REPORT_STATISTICS)
    eu->sched_nb_tasks_done = 0;
#endif

    eu->context_mempool = &(eu->master_context->context_mempool.thread_mempools[eu->eu_id]);
    for(pi = 0; pi <= MAX_PARAM_COUNT; pi++)
        eu->datarepo_mempools[pi] = &(eu->master_context->datarepo_mempools[pi].thread_mempools[eu->eu_id]);

#ifdef DAGUE_PROF_TRACE
    eu->eu_profile = dague_profiling_thread_init( 2*1024*1024, "DAGuE Thread %d", eu->eu_id );
#endif

#if defined(DAGUE_SIM)
    eu->largest_simulation_date = 0;
#endif

    /* The main thread will go back to the user level */
    if( 0 == eu->eu_id )
        return NULL;

    return __dague_progress(eu);
}

#ifdef HAVE_PAPI
extern int num_events;
extern char* event_names[];
#endif

dague_context_t* dague_init( int nb_cores, int* pargc, char** pargv[])
{
    int argc = (*pargc), i;
    char** argv = NULL;

#if defined(HAVE_GETOPT_LONG)
    struct option long_options[] =
        {
            {"papi",        required_argument,  NULL, 'p'},
            {"bind",        required_argument,  NULL, 'b'},
            {0, 0, 0, 0}
        };
#endif  /* defined(HAVE_GETOPT_LONG) */

    dague_context_t* context = (dague_context_t*)malloc(sizeof(dague_context_t) +
                                                        nb_cores * sizeof(dague_execution_unit_t*));
    __dague_temporary_thread_initialization_t* startup = 
        (__dague_temporary_thread_initialization_t*)malloc(nb_cores * sizeof(__dague_temporary_thread_initialization_t));
    /* Prepare the temporary storage for each thread startup */
    for( i = 0; i < nb_cores; i++ ) {
        startup[i].th_id = i;
        startup[i].master_context = context;
        startup[i].nb_cores = nb_cores;
        startup[i].bindto = i;
    }

#if defined(HAVE_PAPI)
    papime_start();
#endif

#if defined(HAVE_HWLOC)
    dague_hwloc_init();
#endif  /* defined(HWLOC) */

    context->__dague_internal_finalization_in_progress = 0;
    context->nb_cores       = (int32_t) nb_cores;
    context->__dague_internal_finalization_counter = 0;
    context->nb_nodes       = 1;
    context->active_objects = 0;
    context->my_rank        = 0;

#ifdef HAVE_PAPI
    num_events = 0;
#endif
    
    {
        int index = 0;
        /* Check for the upper level arguments */
        while(1) {
            if( NULL == (*pargv)[index] )
                break;
            if( 0 == strcmp( "--", (*pargv)[index]) ) {
                argv = &(*pargv)[index];
                break;
            }
            index++;
        }
        argc = (*pargc) - index;
    }

    if( argv != NULL ) {
        optind = 1;
        do {
            int ret;
#if defined(HAVE_GETOPT_LONG)
            int option_index = 0;
            
            ret = getopt_long (argc, argv, "p:b:",
                               long_options, &option_index);
#else
            ret = getopt (argc, argv, "p:b:");
#endif  /* defined(HAVE_GETOPT_LONG) */
            if( -1 == ret ) break;  /* we're done */
            
            switch(ret) {
            case 'b':
                {
                    char* option = strdup(optarg);
                    char* position;
                    if( NULL != (position = strchr(option, ':')) ) {
                        /* range expression such as [start]:[end]:[step] */
                        int start = 0, end, step = 1;
                        if( position != option ) {  /* we have a starting position */
                            start = strtol(option, NULL, 10);
                        }
                        end = start + nb_cores;  /* automatically compute the end */
                        position++;  /* skip the : */
                        if( '\0' != position[0] ) {
                            if( ':' != position[0] ) {
                                end = strtol(position, &position, 10);
                                position = strchr(position, ':');  /* find the step */
                            }
                            if( NULL != position ) position++;  /* skip the : directly into the step */
                            if( (NULL != position) && ('\0' != position[0]) ) {
                                step = strtol(position, NULL, 10);
                            }
                        }
                        DEBUG(( "core range [%d:%d:%d]\n", start, end, step));
                        {
                            int where = start, skip = 1;
                            for( i = 0; i < nb_cores; i++ ) {
                                startup[i].bindto = where;
                                where += step;
                                if( where >= end ) {
                                    where = start + skip;
                                    skip++;
                                    if( (skip > step) && (i < (nb_cores - 1))) {
                                        printf( "No more available cores to bind to. The remaining %d threads are not bound\n", nb_cores - i );
                                        break;
                                    }
                                }
                            }
                        }
                    } else {
                        i = 0;
                        /* array of cores c1,c2,... */
                        position = option;
                        while( NULL != position ) {
                            /* We have more information than the number of cores. Ignore it! */
                            if( i == nb_cores ) break;
                            startup[i].bindto = strtol(position, &position, 10);
                            i++;
                            if( (',' != position[0]) || ('\0' == position[0]) ) {
                                break;
                            }
                            position++;
                        }
                        if( i < nb_cores ) {
                            printf( "Based on the information provided to --bind some threads are not binded\n" );
                        }
                    }
                    free(option);
                }
                break;
            }
        } while(1);
    }

    /* Initialize the barriers */
    dague_barrier_init( &(context->barrier), NULL, nb_cores );
#ifdef DAGUE_PROF_TRACE
    dague_profiling_init( "%s", (*pargv)[0] );

    dague_profiling_add_dictionary_keyword( "MEMALLOC", "fill:#FF00FF",
                                            0, NULL,
                                            &MEMALLOC_start_key, &MEMALLOC_end_key);
    dague_profiling_add_dictionary_keyword( "Sched POLL", "fill:#8A0886",
                                            0, NULL,
                                            &schedule_poll_begin, &schedule_poll_end);
    dague_profiling_add_dictionary_keyword( "Sched PUSH", "fill:#F781F3",
                                            0, NULL,
                                            &schedule_push_begin, &schedule_push_end);
    dague_profiling_add_dictionary_keyword( "Sched SLEEP", "fill:#FA58F4",
                                            0, NULL,
                                            &schedule_sleep_begin, &schedule_sleep_end);
#endif  /* DAGUE_PROF_TRACE */

    {
        dague_execution_context_t fake_context;
        dague_mempool_construct( &context->context_mempool, sizeof(dague_execution_context_t),
                                 ((char*)&fake_context.mempool_owner) - ((char*)&fake_context), nb_cores );
    }
    {
        data_repo_entry_t fake_entry;
        int pi;
        for(pi = 0; pi <= MAX_PARAM_COUNT; pi++)
            dague_mempool_construct( &context->datarepo_mempools[pi], 
                                     sizeof(data_repo_entry_t)+(pi-1)*sizeof(dague_arena_chunk_t*),
                                     ((char*)&fake_entry.data_repo_mempool_owner) - ((char*)&fake_entry),
                                     nb_cores);
    }

    if( nb_cores > 1 ) {
        pthread_attr_t thread_attr;

        pthread_attr_init(&thread_attr);
        pthread_attr_setscope(&thread_attr, PTHREAD_SCOPE_SYSTEM);
#ifdef __linux
        pthread_setconcurrency(nb_cores);
#endif  /* __linux */

        context->pthreads = (pthread_t*)malloc(nb_cores * sizeof(pthread_t));

        /* The first execution unit is for the master thread */
        for( i = 1; i < context->nb_cores; i++ ) {
            pthread_create( &((context)->pthreads[i]),
                            &thread_attr,
                            (void* (*)(void*))__dague_thread_init,
                            (void*)&(startup[i]));
        }
    }

    __dague_thread_init( &startup[0] );

    /* Wait until all threads are done binding themselves */
    dague_barrier_wait( &(context->barrier) );
    context->__dague_internal_finalization_counter++;

    /* Release the temporary array used for starting up the threads */
    free(startup);

    /* Wait until threads are bound before introducing progress threads */
    context->nb_nodes = dague_remote_dep_init(context);
    
#ifdef HAVE_PAPI
    if(PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT)
        printf("PAPI library initialization error! \n");
    else {
        if (PAPI_create_eventset(&eventSet) != PAPI_OK)
            printf("PAPI unable to create event set! \n");
        else {
            for( i = 0; i < num_events; ++i ) {
                int event;
                PAPI_event_name_to_code(event_names[i], &event);

                if (PAPI_add_event(eventSet, event) != PAPI_OK) 
                    printf("PAPI unable to add event: %s \n", event_names[i]);
            }
        }
    }
#endif

    dague_statistics("DAGuE");

    return context;
}

/**
 *
 */
int dague_fini( dague_context_t** pcontext )
{
    dague_context_t* context = *pcontext;
    int i;

#ifdef HAVE_PAPI
    papime_stop();
#endif

    dague_mempool_destruct( &context->context_mempool );
    for(i = 0; i <= MAX_PARAM_COUNT; i++)
        dague_mempool_destruct( &context->datarepo_mempools[i]);

    /* Now wait until every thread is back */
    context->__dague_internal_finalization_in_progress = 1;
    dague_barrier_wait( &(context->barrier) );

    /* The first execution unit is for the master thread */
    for(i = 1; i < context->nb_cores; i++) {
        pthread_join( context->pthreads[i], NULL );
    }

    (void) dague_remote_dep_fini( context );

    dague_set_scheduler( context, NULL );

    for(i = 0; i < context->nb_cores; i++) {
        free(context->execution_units[i]);
        context->execution_units[i] = NULL;
    }
    
#ifdef DAGUE_PROF_TRACE
    dague_profiling_fini( );
#endif  /* DAGUE_PROF_TRACE */

    /* Destroy all resources allocated for the barrier */
    dague_barrier_destroy( &(context->barrier) );

    if( context->nb_cores > 1 ) {
        free(context->pthreads);
    }

#if defined(HAVE_HWLOC)
    dague_hwloc_fini();
#endif  /* defined(HWLOC) */

#if defined(DAGUE_STATS)
    {
        char filename[64];
        char prefix[32];
# if defined(DISTRIBUTED) && defined(HAVE_MPI)
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        snprintf(filename, 64, "dague-%d.stats", rank);
        snprintf(prefix, 32, "%d/%d", rank, size);
# else
        snprintf(filename, 64, "dague.stats");
        prefix[0] = '\0';
# endif
        dague_stats_dump(filename, prefix);
    }
#endif

    dague_object_empty_repository();

    free(context);
    *pcontext = NULL;
    return 0;
}

/**
 * Convert the execution context to a string.
 */
char* dague_service_to_string( const dague_execution_context_t* exec_context,
                               char* tmp,
                               size_t length )
{
    const dague_function_t* function = exec_context->function;
    unsigned int i, index = 0;

    index += snprintf( tmp + index, length - index, "%s", function->name );
    if( index >= length ) return tmp;
    for( i = 0; i < function->nb_parameters; i++ ) {
        index += snprintf( tmp + index, length - index, "%s%d",
                           (i == 0) ? "(" : ", ",
                           exec_context->locals[i].value );
        if( index >= length ) return tmp;
    }
    index += snprintf(tmp + index, length - index, ")");

    return tmp;
}

/**
 * Resolve all IN() dependencies for this particular instance of execution.
 */
static dague_dependency_t
dague_check_IN_dependencies( const dague_object_t *dague_object,
                             const dague_execution_context_t* exec_context )
{
    const dague_function_t* function = exec_context->function;
    int i, j, value, active;
    const dague_flow_t* flow;
    const dep_t* dep;
    dague_dependency_t ret = 0;

    if( !(function->flags & DAGUE_HAS_IN_IN_DEPENDENCIES) ) {
        return 0;
    }

    for( i = 0; (i < MAX_PARAM_COUNT) && (NULL != function->in[i]); i++ ) {
        flow = function->in[i];
        /* this param has no dependency condition satisfied */
#if defined(DAGUE_SCHED_DEPS_MASK)
        active = (1 << flow->flow_index);
#else
        active = 1;
#endif
        for( j = 0; (j < MAX_DEP_IN_COUNT) && (NULL != flow->dep_in[j]); j++ ) {
            dep = flow->dep_in[j];
            if( NULL != dep->cond ) {
                /* Check if the condition apply on the current setting */
                assert( dep->cond->op == EXPR_OP_INLINE );
                value = dep->cond->inline_func(dague_object, exec_context->locals);
                if( 0 == value ) {
                    continue;
                }
            }
            if( dep->dague->nb_parameters == 0 ) {  /* this is only true for memory locations */
                goto dep_resolved;
            }
            if( ACCESS_NONE == flow->access_type ) {
                active = 0;
                goto dep_resolved;
            }
        }
        if( ACCESS_NONE != flow->access_type ) {
            active = 0;
        }
    dep_resolved:
        ret += active;
    }
    return ret;
}

static dague_dependency_t *find_deps(dague_object_t *dague_object,
                                     dague_execution_context_t* restrict exec_context)
{
    dague_dependencies_t *deps;
    int p;

    deps = dague_object->dependencies_array[exec_context->function->deps];
    assert( NULL != deps );

    for(p = 0; p < exec_context->function->nb_parameters - 1; p++) {
        assert( (deps->flags & DAGUE_DEPENDENCIES_FLAG_NEXT) != 0 );
        deps = deps->u.next[exec_context->locals[p].value - deps->min];
        assert( NULL != deps );
    }

    return &(deps->u.dependencies[exec_context->locals[exec_context->function->nb_parameters - 1].value - deps->min]);
}

/**
 * Release the OUT dependencies for a single instance of a task. No ranges are
 * supported and the task is supposed to be valid (no input/output tasks) and
 * local.
 */
int dague_release_local_OUT_dependencies( dague_object_t *dague_object,
                                          dague_execution_unit_t* eu_context,
                                          const dague_execution_context_t* restrict origin,
                                          const dague_flow_t* restrict origin_flow,
                                          dague_execution_context_t* restrict exec_context,
                                          const dague_flow_t* restrict dest_flow,
                                          data_repo_entry_t* dest_repo_entry,
                                          dague_execution_context_t** pready_list )
{
    const dague_function_t* function = exec_context->function;
    dague_dependency_t *deps;
    dague_dependency_t dep_new_value, dep_cur_value;
#if defined(DAGUE_DEBUG)
    char tmp[128];
#endif

    (void)eu_context;

    DEBUG(("Activate dependencies for %s priority %d\n",
           dague_service_to_string(exec_context, tmp, 128), exec_context->priority));
    deps = find_deps(dague_object, exec_context);
    
#if !defined(DAGUE_SCHED_DEPS_MASK)

    if( 0 == *deps ) {
        dep_new_value = 1 + dague_check_IN_dependencies( dague_object, exec_context );
        if( dague_atomic_cas( deps, 0, dep_new_value ) == 1 )
            dep_cur_value = dep_new_value;
        else
            dep_cur_value = dague_atomic_inc_32b( deps );
    } else {
        dep_cur_value = dague_atomic_inc_32b( deps );
    }

#if defined(DAGUE_DEBUG)
    if( dep_cur_value > function->dependencies_goal ) {
        DEBUG(("function %s as reached a dependency count of %d, higher than the goal dependencies count of %d\n",
               dague_service_to_string(exec_context, tmp, 128), dep_cur_value, function->dependencies_goal));
        assert(dep_cur_value <= function->dependencies_goal);
    }
#endif /* DAGUE_DEBUG */

    if( dep_cur_value == function->dependencies_goal ) {

#else  /* defined(DAGUE_SCHED_DEPS_MASK) */

#   if defined(DAGUE_DEBUG)
    if( (*deps) & (1 << dest_flow->flow_index) ) {
        char tmp2[128];
        DEBUG(("Output dependencies 0x%x from %s (flow %s) activate an already existing dependency 0x%x on %s (flow %s)\n",
               dest_flow->flow_index, dague_service_to_string(origin, tmp, 128), origin_flow->name,
               *deps,
               dague_service_to_string(exec_context, tmp2, 128),  dest_flow->name ));
    }
    assert( 0 == (*deps & (1 << dest_flow->flow_index)) );
#   else
    (void) origin; (void) origin_flow;
#   endif 

    dep_new_value = DAGUE_DEPENDENCIES_IN_DONE | (1 << dest_flow->flow_index);
    /* Mark the dependencies and check if this particular instance can be executed */
    if( !(DAGUE_DEPENDENCIES_IN_DONE & (*deps)) ) {
        dep_new_value |= dague_check_IN_dependencies( dague_object, exec_context );
#   ifdef DAGUE_DEBUG
        if( dep_new_value != 0 ) {
            DEBUG(("Activate IN dependencies with mask 0x%x\n", dep_new_value));
        }
#   endif /* DAGUE_DEBUG */
    }

    dep_cur_value = dague_atomic_bor( deps, dep_new_value );

    if( (dep_cur_value & function->dependencies_goal) == function->dependencies_goal ) {

#endif /* defined(DAGUE_SCHED_DEPS_MASK) */

        dague_prof_grapher_dep(origin, exec_context, 1, origin_flow, dest_flow);

#if defined(DAGUE_DEBUG) && defined(DAGUE_SCHED_DEPS_MASK)
        {
            int success;
            dague_dependency_t tmp_mask;
            tmp_mask = *deps;
            success = dague_atomic_cas( deps,
                                        tmp_mask, (tmp_mask | DAGUE_DEPENDENCIES_TASK_DONE) );
            if( !success || (tmp_mask & DAGUE_DEPENDENCIES_TASK_DONE) ) {
                char tmp2[128];
                fprintf(stderr, "I'm not very happy (success %d tmp_mask %4x)!!! Task %s scheduled twice (second time by %s)!!!\n",
                        success, tmp_mask, dague_service_to_string(exec_context, tmp, 128),
                        dague_service_to_string(origin, tmp2, 128));
                assert(0);
            }
        }
#endif  /* defined(DAGUE_DEBUG) && defined(DAGUE_SCHED_DEPS_MASK) */

        /* This service is ready to be executed as all dependencies
         * are solved.  Queue it into the ready_list passed as an
         * argument.
         */
        {
#if defined(DAGUE_DEBUG)
            char tmp2[128];
#endif
            dague_execution_context_t* new_context;
            dague_thread_mempool_t *mpool;
            new_context = (dague_execution_context_t*)dague_thread_mempool_allocate( eu_context->context_mempool );
            mpool = new_context->mempool_owner;  /* this should not be copied over from the old execution context */
            DAGUE_STAT_INCREASE(mem_contexts, sizeof(dague_execution_context_t) + STAT_MALLOC_OVERHEAD);
            memcpy( new_context, exec_context, sizeof(dague_minimal_execution_context_t) );
            new_context->mempool_owner = mpool;

            DEBUG(("%s becomes schedulable from %s with mask 0x%04x on thread %d\n", 
                   dague_service_to_string(exec_context, tmp, 128),
                   dague_service_to_string(origin, tmp2, 128),
                   *deps,
                   eu_context->eu_id));

#if defined(DAGUE_SCHED_CACHE_AWARE)
            new_context->data[0].gc_data = NULL;
#endif
            /* TODO: change this to the real number of input dependencies */
            memset( new_context->data, 0, sizeof(dague_data_pair_t) * MAX_PARAM_COUNT );
            assert( dest_flow->flow_index <= MAX_PARAM_COUNT );
            /**
             * Save the data_repo and the pointer to the data for later use. This will prevent the
             * engine from atomically locking the hash table for at least one of the flow
             * for each execution context.
             */
            new_context->data[(int)dest_flow->flow_index].data_repo = dest_repo_entry;
            new_context->data[(int)dest_flow->flow_index].data      = origin->data[(int)origin_flow->flow_index].data;
            dague_list_add_single_elem_by_priority( pready_list, new_context );
        }

        DAGUE_STAT_INCREASE(counter_nbtasks, 1ULL);

    } else { /* Service not ready */

        dague_prof_grapher_dep(origin, exec_context, 0, origin_flow, dest_flow);

#if defined(DAGUE_SCHED_DEPS_MASK)
        DEBUG(("  => Service %s not yet ready (required mask 0x%02x actual 0x%02x: real 0x%02x)\n",
               dague_service_to_string( exec_context, tmp, 128 ), (int)function->dependencies_goal,
               (int)(dep_cur_value & DAGUE_DEPENDENCIES_BITMASK),
               (int)(dep_cur_value)));
#else
        DEBUG(("  => Service %s not yet ready (requires %d dependencies, %d done)\n",
               dague_service_to_string( exec_context, tmp, 128 ), 
               (int)function->dependencies_goal, dep_cur_value));
#endif
    }

    return 0;
}

#define is_inplace(ctx,flow,dep) NULL
#define is_read_only(ctx,flow,dep) NULL

dague_ontask_iterate_t dague_release_dep_fct(dague_execution_unit_t *eu, 
                                             dague_execution_context_t *newcontext, 
                                             dague_execution_context_t *oldcontext, 
                                             int out_index, int outdep_index, 
                                             int src_rank, int dst_rank,
                                             dague_arena_t* arena,
                                             void *param)
{
    dague_release_dep_fct_arg_t *arg = (dague_release_dep_fct_arg_t *)param;
    const dague_flow_t* target = oldcontext->function->out[out_index];

    if( !(arg->action_mask & (1 << out_index)) ) {
#if defined(DAGUE_DEBUG)
        char tmp[128];
        DEBUG(("On task %s out_index %d not on the action_mask %x\n",
               dague_service_to_string(oldcontext, tmp, 128), out_index, arg->action_mask));
#endif
        return DAGUE_ITERATE_CONTINUE;
    }

#if defined(DISTRIBUTED)
    if( dst_rank != src_rank ) {
        if( arg->action_mask & DAGUE_ACTION_RECV_INIT_REMOTE_DEPS ) {
            void* data;

            data = is_read_only(oldcontext, out_index, outdep_index);
            if(NULL != data) {
                arg->deps->msg.which &= ~(1 << out_index); /* unmark all data that are RO we already hold from previous tasks */
            } else {
                arg->deps->msg.which |= (1 << out_index); /* mark all data that are not RO */
                data = is_inplace(oldcontext, out_index, outdep_index);  /* Can we do it inplace */
            }
            arg->deps->output[out_index].data = data; /* if still NULL allocate it */
            arg->deps->output[out_index].type = arena;
            if(newcontext->priority > arg->deps->max_priority) arg->deps->max_priority = newcontext->priority;
        }
        if( arg->action_mask & DAGUE_ACTION_SEND_INIT_REMOTE_DEPS ) {
            int _array_pos, _array_mask;

            _array_pos = dst_rank / (8 * sizeof(uint32_t));
            _array_mask = 1 << (dst_rank % (8 * sizeof(uint32_t)));
            DAGUE_ALLOCATE_REMOTE_DEPS_IF_NULL(arg->remote_deps, oldcontext, MAX_PARAM_COUNT);
            arg->remote_deps->root = src_rank;
            if( !(arg->remote_deps->output[out_index].rank_bits[_array_pos] & _array_mask) ) {
                arg->remote_deps->output[out_index].type = arena;
                arg->remote_deps->output[out_index].data = oldcontext->data[target->flow_index].data;
                arg->remote_deps->output[out_index].rank_bits[_array_pos] |= _array_mask;
                arg->remote_deps->output[out_index].count++;
                arg->remote_deps_count++;
            }
            if(newcontext->priority > arg->remote_deps->max_priority) arg->remote_deps->max_priority = newcontext->priority;
        }
    }
#else
    (void)src_rank;
    (void)arena;
#endif

    if( (arg->action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS) &&
        (eu->master_context->my_rank == dst_rank) ) {
        arg->output_entry->data[out_index] = oldcontext->data[target->flow_index].data;
        arg->output_usage++;
        AREF( arg->output_entry->data[out_index] );
        arg->nb_released += dague_release_local_OUT_dependencies(oldcontext->dague_object,
                                                                 eu, oldcontext,
                                                                 oldcontext->function->out[out_index],
                                                                 newcontext,
                                                                 oldcontext->function->out[out_index]->dep_out[outdep_index]->flow,
                                                                 arg->output_entry,
                                                                 &arg->ready_list);
    }
    
    return DAGUE_ITERATE_CONTINUE;
}

void dague_dump_object( dague_object_t* object )
{
    (void) object;
}

void dague_dump_execution_context( dague_execution_context_t* exec_context )
{
    char tmp[128];

    printf( "Task %s\n", dague_service_to_string( exec_context, tmp, 128 ) );
}

void dague_destruct_dependencies(dague_dependencies_t* d)
{
    int i;
    if( (d != NULL) && (d->flags & DAGUE_DEPENDENCIES_FLAG_NEXT) ) {
        for(i = d->min; i <= d->max; i++)
            if( NULL != d->u.next[i-d->min] )
                dague_destruct_dependencies(d->u.next[i-d->min]);
    }
    free(d);
}

/**
 *
 */
int dague_set_complete_callback( dague_object_t* dague_object,
                                 dague_completion_cb_t complete_cb, void* complete_cb_data )
{
    if( NULL == dague_object->complete_cb ) {
        dague_object->complete_cb      = complete_cb;
        dague_object->complete_cb_data = complete_cb_data;
        return 0;
    }
    return -1;
}
    dague_completion_cb_t      complete_cb;
    void*                      complete_cb_data;

/**
 *
 */
int dague_get_complete_callback( const dague_object_t* dague_object,
                                 dague_completion_cb_t* complete_cb, void** complete_cb_data )
{
    if( NULL != dague_object->complete_cb ) {
        *complete_cb      = dague_object->complete_cb;
        *complete_cb_data = dague_object->complete_cb_data;
        return 0;
    }
    return -1;
}

/* TODO: Change this code to something better */
static dague_object_t** object_array = NULL;
static uint32_t object_array_size = 1, object_array_pos = 0;

static void dague_object_empty_repository(void)
{
    free(object_array);
    object_array = NULL;
    object_array_size = 1;
    object_array_pos = 0;
}

/**< Retrieve the local object attached to a unique object id */
dague_object_t* dague_object_lookup( uint32_t object_id )
{
    if( object_id > object_array_pos ) {
        return NULL;
    }
    return object_array[object_id];
}

/**< Register the object with the engine. Create the unique identifier for the object */
int dague_object_register( dague_object_t* object )
{
    uint32_t index = ++object_array_pos;

    if( index >= object_array_size ) {
        object_array_size *= 2;
        object_array = (dague_object_t**)realloc(object_array, object_array_size * sizeof(dague_object_t*) );
    }
    object_array[index] = object;
    object->object_id = index;
    return (int)index;
}

