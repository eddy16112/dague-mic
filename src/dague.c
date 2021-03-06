/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
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

#ifdef DAGUE_PROFILING
#include "profiling.h"
#endif

#ifdef HAVE_PAPI
#include <papime.h>
#endif

#ifdef HAVE_HWLOC
#include "hbbuffer.h"
#include "dague_hwloc.h"
#endif

FILE *__dague_graph_file = NULL;

#ifdef DAGUE_PROFILING
int MEMALLOC_start_key, MEMALLOC_end_key;
int schedule_poll_begin, schedule_poll_end;
int schedule_push_begin, schedule_push_end;
int schedule_sleep_begin, schedule_sleep_end;
#endif  /* DAGUE_PROFILING */

#ifdef HAVE_PAPI
int eventSet = PAPI_NULL;
int num_events = 0;
char* event_names[MAX_EVENTS];
#endif

int DAGUE_TILE_SIZE = 0;

const dague_t* dague_find(const dague_object_t *dague_object, const char *fname)
{
    unsigned int i;
    const dague_t* object;

    for( i = 0; i < dague_object->nb_functions; i++ ) {
        object = dague_object->functions_array[i];
        if( 0 == strcmp( object->name, fname ) ) {
            return object;
        }
    }
    return NULL;
}


/**
 *
 */
#if defined(DAGUE_USE_GLOBAL_LIFO)
dague_atomic_lifo_t ready_list;
#endif  /* defined(DAGUE_USE_GLOBAL_LIFO) */

typedef struct __dague_temporary_thread_initialization_t {
    dague_context_t* master_context;
    int th_id;
    int nb_cores;
    int bindto;
} __dague_temporary_thread_initialization_t;

#if !defined(DAGUE_USE_GLOBAL_LIFO) && defined(HAVE_HWLOC)
/** In case of hierarchical bounded buffer, define
 *  the wrappers to functions
 */
#if defined(USE_HIERARCHICAL_QUEUES)
static void push_in_buffer_wrapper(void *store, dague_list_item_t *elt)
{ 
    /* Store is a hbbbuffer */
    dague_hbbuffer_push_all( (dague_hbbuffer_t*)store, elt );
}
#endif

static void push_in_queue_wrapper(void *store, dague_list_item_t *elt)
{
    /* Store is a lifo or a dequeue */
#if defined(DAGUE_USE_LIFO)
    dague_atomic_lifo_push( (dague_atomic_lifo_t*)store, elt );
#else
    dague_dequeue_push_back( (dague_dequeue_t*)store, elt );
#endif
}
#endif

static void* __dague_thread_init( __dague_temporary_thread_initialization_t* startup )
{
    dague_execution_unit_t* eu;

    /* Bind to the specified CORE */
    dague_bindthread(startup->bindto);

    eu = (dague_execution_unit_t*)malloc(sizeof(dague_execution_unit_t));
    if( NULL == eu ) {
        return NULL;
    }
    eu->eu_id          = startup->th_id;
    eu->master_context = startup->master_context;
    (startup->master_context)->execution_units[startup->th_id] = eu;

#ifdef DAGUE_PROFILING
    eu->eu_profile = dague_profiling_thread_init( 65536, "DAGuE Thread %d", eu->eu_id );
#endif
#ifdef DAGUE_USE_LIFO
    eu->eu_task_queue = (dague_atomic_lifo_t*)malloc( sizeof(dague_atomic_lifo_t) );
    if( NULL == eu->eu_task_queue ) {
        free(eu);
        return NULL;
    }
    dague_atomic_lifo_construct( eu->eu_task_queue );
#elif defined(DAGUE_USE_GLOBAL_LIFO)
    /* Everybody share the same global LIFO */
    eu->eu_task_queue = &ready_list;
#elif defined(HAVE_HWLOC)
    /* we set the eu_task_queue later */
#else
    eu->eu_task_queue = (dague_dequeue_t*)malloc( sizeof(dague_dequeue_t) );
    if( NULL == eu->eu_task_queue ) {
        free(eu);
        return NULL;
    }
    dague_dequeue_construct( eu->eu_task_queue );
#if PLACEHOLDER_SIZE
    eu->placeholder_pop  = 0;
    eu->placeholder_push = 0;
#endif  /* PLACEHOLDER_SIZE */
#endif  /* DAGUE_USE_LIFO */

#if defined(DAGUE_CACHE_AWARE)
    eu->closest_cache = NULL;
#endif

#if defined(HAVE_HWLOC)
    {
        int level;
        if( eu->eu_id == 0 ) {
            eu->eu_system_queue = (dague_dequeue_t*)malloc(sizeof(dague_dequeue_t));
            dague_dequeue_construct( eu->eu_system_queue );
            dague_barrier_wait( &startup->master_context->barrier );
        } else {
            dague_barrier_wait( &startup->master_context->barrier );
            eu->eu_system_queue = startup->master_context->execution_units[0]->eu_system_queue;
        }

#if defined(USE_HIERARCHICAL_QUEUES)
        eu->eu_nb_hierarch_queues = dague_hwloc_nb_levels(startup->master_context);
        assert(eu->eu_nb_hierarch_queues > 0 /* Must have at least a system queue and a socket queue to work with hwloc */ );

        eu->eu_hierarch_queues = (dague_hbbuffer_t **)malloc(eu->eu_nb_hierarch_queues * sizeof(dague_hbbuffer_t*) );

        for(level = 0; level < eu->eu_nb_hierarch_queues; level++) {
            int idx = eu->eu_nb_hierarch_queues - 1 - level;
            int master = dague_hwloc_master_id(startup->master_context, level, eu->eu_id);
            if( eu->eu_id == master ) {
                int nbcores = dague_hwloc_nb_cores(startup->master_context, level, master);
                int queue_size = 96 * (level+1) / nbcores;
                if( queue_size < nbcores ) queue_size = nbcores;

                /* The master(s) create the shared queues */               
                eu->eu_hierarch_queues[idx] = dague_hbbuffer_new( queue_size, nbcores,
                                                                  level == 0 ? push_in_queue_wrapper : push_in_buffer_wrapper,
                                                                  level == 0 ? (void*)eu->eu_system_queue : (void*)eu->eu_hierarch_queues[idx+1]);
                DEBUG(("%d creates hbbuffer of size %d (ideal %d) for level %d stored in %d: %p (parent: %p -- %s)\n",
                       eu->eu_id, queue_size, nbcores,
                       level, idx, eu->eu_hierarch_queues[idx],
                       level == 0 ? (void*)eu->eu_system_queue : (void*)eu->eu_hierarch_queues[idx+1],
                       level == 0 ? "System queue" : "upper level hhbuffer"));
                
                /* The master(s) unblock all waiting slaves */
                dague_barrier_wait( &startup->master_context->barrier );
            } else {
                /* Be a slave: wait that the master(s) unblock me */
                dague_barrier_wait( &startup->master_context->barrier );
                
                DEBUG(("%d takes the buffer of %d at level %d stored in %d: %p\n",
                       eu->eu_id, master, level, idx, startup->master_context->execution_units[master]->eu_hierarch_queues[idx]));
                /* The slaves take their queue for this level from their master */
                eu->eu_hierarch_queues[idx] = startup->master_context->execution_units[master]->eu_hierarch_queues[idx];
            }
        }
        eu->eu_task_queue = eu->eu_hierarch_queues[0];
#else /* Don't USE_HIERARCHICAL_QUEUES: USE_FLAT_QUEUES */
        {
            int queue_size = startup->master_context->nb_cores * 4;
            int nq = 0;
            int id;

            eu->eu_nb_hierarch_queues = startup->master_context->nb_cores;
            eu->eu_hierarch_queues = (dague_hbbuffer_t **)malloc(eu->eu_nb_hierarch_queues * sizeof(dague_hbbuffer_t*) );
            /* Each thread creates its own "local" queue, connected to the shared dequeue */
            eu->eu_task_queue = dague_hbbuffer_new( queue_size, 1, push_in_queue_wrapper, 
                                                    (void*)eu->eu_system_queue);
            eu->eu_hierarch_queues[0] =  eu->eu_task_queue;

            dague_barrier_wait( &startup->master_context->barrier );

            /* Then, they know about all other queues, from the closest to the farthest */
            nq = 1;
            for(level = 0; level <= dague_hwloc_nb_levels(); level++) {
                for(id = (eu->eu_id + 1) % startup->master_context->nb_cores; 
                    id != eu->eu_id; 
                    id = (id + 1) %  startup->master_context->nb_cores) {
                    int d;

                    d = dague_hwloc_distance(eu->eu_id, id);
                    if( d == 2*level || d == 2*level + 1 ) {
                        eu->eu_hierarch_queues[nq] = startup->master_context->execution_units[id]->eu_task_queue;
                        DEBUG(("%d: my %d preferred queue is the task queue of %d (%p)\n",
                               eu->eu_id, nq, id, eu->eu_hierarch_queues[nq]));
                        nq++;
                    }
                }
            }
        }
#endif

#if defined(DAGUE_CACHE_AWARE)
#define TILE_SIZE (120*120*sizeof(double))
        for(level = 0; level < dague_hwloc_nb_levels(); level++) {
            master = dague_hwloc_master_id(level, eu->eu_id);
            if( eu->eu_id == master ) {
                int nbtiles = (dague_hwloc_cache_size(level, master) / TILE_SIZE)-1;
                int nbcores = dague_hwloc_nb_cores(level, master);

                /* The master(s) create the cache explorer, using their current closest cache as its father */
                eu->closest_cache = cache_create( nbcores, eu->closest_cache, nbtiles);
                DEBUG(("%d creates cache of size %d for level %d: %p (parent: %p)\n",
                       eu->eu_id, nbtiles,
                       level, eu->closest_cache,
                       eu->closest_cache != NULL ? eu->closest_cache->parent : NULL));
                
                /* The master(s) unblock all waiting slaves */
                dague_barrier_wait( &startup->master_context->barrier );
            } else {
                /* Be a slave: wait that the master(s) unblock me */
                dague_barrier_wait( &startup->master_context->barrier );
                
                /* The closest cache has been created by my master. Thank you, master */
                eu->closest_cache = startup->master_context->execution_units[master]->closest_cache;
                DEBUG(("%d takes the closest cache of %d at level %d: %p\n",
                       eu->eu_id, master, level,  eu->closest_cache));
            }
        }
#endif /* DAGUE_CACHE_AWARE */
    }
#endif  /* defined(HAVE_HWLOC)*/

    /* The main thread will go back to the user level */
    if( 0 == eu->eu_id )
        return NULL;

    return __dague_progress(eu);
}

#ifdef USE_PAPI
extern int num_events;
extern char* event_names[];
#endif

dague_context_t* dague_init( int nb_cores, int* pargc, char** pargv[], int tile_size )
{
    int argc = (*pargc), i;
    char** argv = NULL;

#if defined(HAVE_GETOPT_LONG)
    struct option long_options[] =
        {
            {"dot",         required_argument,  NULL, 'd'},
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

    DAGUE_TILE_SIZE = tile_size;  /* TODO: REMOVE ME */

#if defined(USE_MPI)
    /* Change this to pass the MPI Datatype as parameter to dague_init, or 
     * at least authorize to pass something different that MPI_DOUBLE?
     */
    remote_dep_mpi_create_default_datatype(tile_size, MPI_DOUBLE);
#endif

#if defined(HAVE_HWLOC)
    dague_hwloc_init();
#endif  /* defined(HWLOC) */

    context->__dague_internal_finalization_in_progress = 0;
    context->nb_cores  = (int32_t) nb_cores;
    context->__dague_internal_finalization_counter = 0;
    context->nb_nodes  = 1;
    context->taskstodo = 0;
    context->my_rank   = 0;

#ifdef USE_PAPI
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

    optind = 1;
    do {
        int ret;
#if defined(HAVE_GETOPT_LONG)
        int option_index = 0;
        
        ret = getopt_long (argc, argv, "d:p:b:",
                           long_options, &option_index);
#else
        ret = getopt (argc, argv, "d:p:b:");
#endif  /* defined(HAVE_GETOPT_LONG) */
        if( -1 == ret ) break;  /* we're done */

        switch(ret) {
        case 'd':
            if( NULL == __dague_graph_file ) {
                int len = strlen(optarg) + 32;
                char filename[len];
#if defined(DISTRIBUTED) && defined(USE_MPI)
                int rank;
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                snprintf(filename, len, "%s%d", optarg, rank);
#else
                snprintf(filename, len, "%s", optarg);
#endif
                __dague_graph_file = fopen( filename, "w");
            }
            break;
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

    /* Initialize the barriers */
    dague_barrier_init( &(context->barrier), NULL, nb_cores );

    if( NULL != __dague_graph_file ) {
        fprintf(__dague_graph_file, "digraph G {\n");
        fflush(__dague_graph_file);
    }
#ifdef DAGUE_PROFILING
    dague_profiling_init( "%s", (*pargv)[0] );

    dague_profiling_add_dictionary_keyword( "MEMALLOC", "fill:#FF00FF",
                                            &MEMALLOC_start_key, &MEMALLOC_end_key);
    dague_profiling_add_dictionary_keyword( "Sched POLL", "fill:#8A0886",
                                            &schedule_poll_begin, &schedule_poll_end);
    dague_profiling_add_dictionary_keyword( "Sched PUSH", "fill:#F781F3",
                                            &schedule_push_begin, &schedule_push_end);
    dague_profiling_add_dictionary_keyword( "Sched SLEEP", "fill:#FA58F4",
                                            &schedule_sleep_begin, &schedule_sleep_end);
#endif  /* DAGUE_PROFILING */

#if defined(DAGUE_USE_GLOBAL_LIFO)
    dague_atomic_lifo_construct(&ready_list);
#endif  /* defined(DAGUE_USE_GLOBAL_LIFO) */

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

    /* Now wait until every thread is back */
    context->__dague_internal_finalization_in_progress = 1;
    dague_barrier_wait( &(context->barrier) );

    /* The first execution unit is for the master thread */
    for(i = 1; i < context->nb_cores; i++) {
        pthread_join( context->pthreads[i], NULL );
    }

    (void) dague_remote_dep_fini( context );
    
    for(i = 1; i < context->nb_cores; i++) {
#if defined(DAGUE_USE_LIFO) && !defined(DAGUE_USE_GLOBAL_LIFO)
        free( context->execution_units[i]->eu_task_queue );
        context->execution_units[i]->eu_task_queue = NULL;
#endif  /* defined(DAGUE_USE_LIFO) && !defined(DAGUE_USE_GLOBAL_LIFO) */
#if defined(HAVE_HWLOC)
        /**
         * TODO: use HWLOC to know who is responsible to free this, 
         * and free the inside too 
         */
        free(context->execution_units[i]->eu_hierarch_queues);
        context->execution_units[i]->eu_hierarch_queues = NULL;
        context->execution_units[i]->eu_nb_hierarch_queues = 0;
        //free(context->execution_units[i]->eu_system_queue);
        context->execution_units[i]->eu_system_queue = NULL;
#endif  /* !defined(DAGUE_USE_GLOBAL_LIFO)  && defined(HAVE_HWLOC)*/
    }
    
#ifdef DAGUE_PROFILING
    dague_profiling_fini( );
#endif  /* DAGUE_PROFILING */

    /* Destroy all resources allocated for the barrier */
    dague_barrier_destroy( &(context->barrier) );

    if( context->nb_cores > 1 ) {
        free(context->pthreads);
    }

    if( NULL != __dague_graph_file ) {
        fprintf(__dague_graph_file, "}\n");
        fclose(__dague_graph_file);
        __dague_graph_file = NULL;
    }

#if defined(HAVE_HWLOC)
    dague_hwloc_fini();
#endif  /* defined(HWLOC) */

#if defined(DAGUE_STATS)
    {
        char filename[64];
        char prefix[32];
# if defined(DISTRIBUTED) && defined(USE_MPI)
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
    const dague_t* function = exec_context->function;
    unsigned int i, index = 0;

    index += snprintf( tmp + index, length - index, "%s", function->name );
    if( index >= length ) return tmp;
    for( i = 0; i < function->nb_locals; i++ ) {
        index += snprintf( tmp + index, length - index, "_%d",
                           exec_context->locals[i].value );
        if( index >= length ) return tmp;
    }

    return tmp;
}

/**
 * Convert a dependency to a string under the format X(...) -> Y(...).
 */
static char* dague_dependency_to_string( const dague_execution_context_t* from,
                                         const dague_execution_context_t* to,
                                         char* tmp,
                                         size_t length )
{
    int index = 0;

    dague_service_to_string( from, tmp, length );
    index = strlen(tmp);
    index += snprintf( tmp + index, length - index, " -> " );
    dague_service_to_string( to, tmp + index, length - index );
    return tmp;
}

/**
 * Resolve all IN() dependencies for this particular instance of execution.
 */
static dague_dependency_t dague_check_IN_dependencies( const dague_object_t *dague_object, const dague_execution_context_t* exec_context )
{
    const dague_t* function = exec_context->function;
    int i, j, value;
    const param_t* param;
    const dep_t* dep;
    dague_dependency_t ret = 0;

    if( !(function->flags & DAGUE_HAS_IN_IN_DEPENDENCIES) ) {
        return 0;
    }

    for( i = 0; (i < MAX_PARAM_COUNT) && (NULL != function->in[i]); i++ ) {
        param = function->in[i];

        for( j = 0; (j < MAX_DEP_IN_COUNT) && (NULL != param->dep_in[j]); j++ ) {
            dep = param->dep_in[j];
            if( NULL != dep->cond ) {
                /* Check if the condition apply on the current setting */
                (void)expr_eval( dague_object, dep->cond, exec_context->locals, MAX_LOCAL_COUNT, &value );
                if( 0 == value ) {
                    continue;
                }
            }
            if( dep->dague->nb_locals == 0 ) {
#if defined(DAGUE_USE_COUNTER_FOR_DEPENDENCIES)
                ret += 1;
#else
                ret |= param->param_mask;
#endif
            }
        }
    }
    return ret;
}

#define CURRENT_DEPS_INDEX(K)  (exec_context->locals[(K)].value - deps->min)

static dague_dependencies_t *find_deps(dague_object_t *dague_object,
                                       dague_execution_context_t* restrict exec_context)
{
    dague_dependencies_t *deps;
    int p;

    deps = dague_object->dependencies_array[exec_context->function->deps];
    assert( NULL != deps );

    for(p = 0; p < exec_context->function->nb_locals - 1; p++) {
        assert( (deps->flags & DAGUE_DEPENDENCIES_FLAG_NEXT) != 0 );
        deps = deps->u.next[exec_context->locals[p].value - deps->min];
        assert( NULL != deps );
    }

    return deps;
}

/**
 * Release the OUT dependencies for a single instance of a task. No ranges are
 * supported and the task is supposed to be valid (no input/output tasks) and
 * local.
 */
int dague_release_local_OUT_dependencies( dague_object_t *dague_object,
                                          dague_execution_unit_t* eu_context,
                                          const dague_execution_context_t* restrict origin,
                                          const param_t* restrict origin_param,
                                          dague_execution_context_t* restrict exec_context,
                                          const param_t* restrict dest_param,
                                          dague_execution_context_t** pready_list )
{
    const dague_t* function = exec_context->function;
    dague_dependencies_t *deps;
    int i;
    dague_dependency_t dep_new_value, dep_cur_value;
#if defined(DAGUE_DEBUG)
    char tmp[128];
#endif

    (void)eu_context;

    DEBUG(("Activate dependencies for %s priority %d\n",
           dague_service_to_string(exec_context, tmp, 128), exec_context->priority));
    deps = find_deps(dague_object, exec_context);
    
    i = function->nb_locals - 1;

#if defined(DAGUE_USE_COUNTER_FOR_DEPENDENCIES)

    if( 0 == deps->u.dependencies[CURRENT_DEPS_INDEX(i)] ) {
        dep_new_value = 1 + dague_check_IN_dependencies( dague_object, exec_context );
        if( dague_atomic_cas( &deps->u.dependencies[CURRENT_DEPS_INDEX(i)], 0, dep_new_value ) == 1 )
            dep_cur_value = dep_new_value;
        else
            dep_cur_value = dague_atomic_inc_32b( &deps->u.dependencies[CURRENT_DEPS_INDEX(i)] );
    } else {
        dep_cur_value = dague_atomic_inc_32b( &deps->u.dependencies[CURRENT_DEPS_INDEX(i)] );
    }

#if defined(DAGUE_DEBUG)
    if( dep_cur_value > function->dependencies_goal ) {
        DEBUG(("function %s as reached a dependency count of %d, higher than the goal dependencies count of %d\n",
               dague_service_to_string(exec_context, tmp, 128), dep_cur_value, function->dependencies_goal));
        assert(dep_cur_value <= function->dependencies_goal);
    }
#endif /* DAGUE_DEBUG */

    if( dep_cur_value == function->dependencies_goal ) {

#else  /* !defined(DAGUE_USE_COUNTER_FOR_DEPENDENCIES) */

#   if defined(DAGUE_DEBUG)
    if( deps->u.dependencies[CURRENT_DEPS_INDEX(i)] & dest_param->param_mask ) {
        char tmp2[128];
        DEBUG(("Output dependencies 0x%x from %s (param %s) activate an already existing dependency 0x%x on %s (param %s)\n",
               dest_param->param_mask, dague_service_to_string(origin, tmp, 128), origin_param->name,
               deps->u.dependencies[CURRENT_DEPS_INDEX(i)],
               dague_service_to_string(exec_context, tmp2, 128),  dest_param->name ));
    }
    assert( 0 == (deps->u.dependencies[CURRENT_DEPS_INDEX(i)] & dest_param->param_mask) );
#   endif 

    dep_new_value = DAGUE_DEPENDENCIES_IN_DONE | dest_param->param_mask;
    /* Mark the dependencies and check if this particular instance can be executed */
    if( !(DAGUE_DEPENDENCIES_IN_DONE & deps->u.dependencies[CURRENT_DEPS_INDEX(i)]) ) {
        dep_new_value |= dague_check_IN_dependencies( dague_object, exec_context );
#   ifdef DAGUE_DEBUG
        if( dep_new_value != 0 ) {
            DEBUG(("Activate IN dependencies with mask 0x%x\n", dep_new_value));
        }
#   endif /* DAGUE_DEBUG */
    }

    dep_cur_value = dague_atomic_bor( &deps->u.dependencies[CURRENT_DEPS_INDEX(i)], dep_new_value );

    if( (dep_cur_value & function->dependencies_goal) == function->dependencies_goal ) {

#endif /* defined(DAGUE_USE_COUNTER_FOR_DEPENDENCIES) */


#if defined(DAGUE_GRAPHER) || 1
        if( NULL != __dague_graph_file ) {
            char tmp1[128];
            fprintf(__dague_graph_file, 
                    "%s [label=\"%s=>%s\" color=\"#00FF00\" style=\"solid\"]\n", 
                    dague_dependency_to_string(origin, exec_context, tmp1, 128),
                    origin_param->name, dest_param->name);
            fflush(__dague_graph_file);
        }
#endif  /* defined(DAGUE_GRAPHER) */

#if defined(DAGUE_DEBUG) && !defined(DAGUE_USE_COUNTER_FOR_DEPENDENCIES)
        {
            int success;
            dague_dependency_t tmp_mask;
            tmp_mask = deps->u.dependencies[CURRENT_DEPS_INDEX(i)];
            success = dague_atomic_cas( &deps->u.dependencies[CURRENT_DEPS_INDEX(i)],
                                        tmp_mask, (tmp_mask | DAGUE_DEPENDENCIES_TASK_DONE) );
            if( !success || (tmp_mask & DAGUE_DEPENDENCIES_TASK_DONE) ) {
                char tmp2[128];
                fprintf(stderr, "I'm not very happy (success %d tmp_mask %4x)!!! Task %s scheduled twice (second time by %s)!!!\n",
                        success, tmp_mask, dague_service_to_string(exec_context, tmp, 128),
                        dague_service_to_string(origin, tmp2, 128));
                assert(0);
            }
        }
#endif  /* defined(DAGUE_DEBUG) && !defined(DAGUE_USE_COUNTER_FOR_DEPENDENCIES) */

        /* This service is ready to be executed as all dependencies
         * are solved.  Queue it into the ready_list passed as an
         * argument.
         */
        {
#if defined(DAGUE_DEBUG)
            char tmp2[128];
#endif
            dague_execution_context_t* new_context;
            new_context = (dague_execution_context_t*)malloc(sizeof(dague_execution_context_t));
            DAGUE_STAT_INCREASE(mem_contexts, sizeof(dague_execution_context_t) + STAT_MALLOC_OVERHEAD);
            memcpy( new_context, exec_context, sizeof(dague_execution_context_t) );

            DEBUG(("%s becomes schedulable from %s with mask 0x%04x on thread %d\n", 
                   dague_service_to_string(exec_context, tmp, 128),
                   dague_service_to_string(origin, tmp2, 128),
                   deps->u.dependencies[CURRENT_DEPS_INDEX(i)],
                   eu_context->eu_id));

#if defined(DAGUE_CACHE_AWARE)
            new_context->data[0].gc_data = NULL;
#endif

            dague_list_add_single_elem_by_priority( pready_list, new_context );
        }

        DAGUE_STAT_INCREASE(counter_nbtasks, 1ULL);

    } else { /* Service not ready */

#if defined(DAGUE_GRAPHER) || 1
        if( NULL != __dague_graph_file ) {
            char tmp1[128];
            fprintf(__dague_graph_file, 
                    "%s [label=\"%s=>%s\" color=\"#FF0000\" style=\"dashed\"]\n", 
                    dague_dependency_to_string(origin, exec_context, tmp1, 128),
                    origin_param->name, dest_param->name);
            fflush(__dague_graph_file);
        }
#endif  /* defined(DAGUE_GRAPHER) */

#if defined(DAGUE_USE_COUNTER_FOR_DEPENDENCIES)
        DEBUG(("  => Service %s not yet ready (requires %d dependencies, %d done)\n",
               dague_service_to_string( exec_context, tmp, 128 ), 
               (int)function->dependencies_goal, dep_cur_value));
#else
        DEBUG(("  => Service %s not yet ready (required mask 0x%02x actual 0x%02x: real 0x%02x)\n",
               dague_service_to_string( exec_context, tmp, 128 ), (int)function->dependencies_goal,
               (int)(dep_cur_value & DAGUE_DEPENDENCIES_BITMASK),
               (int)(dep_cur_value)));
#endif
    }

    return 0;
}

#define is_inplace(ctx,param,dep) NULL
#define is_read_only(ctx,param,dep) NULL

dague_ontask_iterate_t dague_release_dep_fct(struct dague_execution_unit_t *eu, 
                                             dague_execution_context_t *newcontext, 
                                             dague_execution_context_t *oldcontext, 
                                             int param_index, int outdep_index, 
                                             int src_rank, int dst_rank,
                                             void *param)
{
    dague_release_dep_fct_arg_t *arg = (dague_release_dep_fct_arg_t *)param;

    if( arg->action_mask & (1 << param_index) ) {
#if defined(DISTRIBUTED)
        if( arg->action_mask & DAGUE_ACTION_RECV_INIT_REMOTE_DEPS ) {
            void* data;
            if(NULL != (data = is_read_only(oldcontext, param_index, outdep_index)))
            {
                arg->deps->msg.which &= ~(1 << param_index); /* unmark all data that are RO we already hold from previous tasks */
                arg->deps->output[param_index].data = data;
            }
            else if(NULL != (data = is_inplace(oldcontext, param_index, outdep_index)))
            {
                arg->deps->output[param_index].data = data; /* inplace there, don't allocate but still receive */
                arg->deps->output[param_index].type = oldcontext->function->out[param_index]->dep_out[outdep_index]->type;
            }
            else
            {
                arg->deps->output[param_index].data = NULL; /* not local or inplace, lets allocate it */ 
                arg->deps->output[param_index].type = oldcontext->function->out[param_index]->dep_out[outdep_index]->type;
            }
        }
        if( arg->action_mask & DAGUE_ACTION_SEND_INIT_REMOTE_DEPS ) {
            int _array_pos, _array_mask;

            _array_pos = dst_rank / (8 * sizeof(uint32_t));
            _array_mask = 1 << (dst_rank % (8 * sizeof(uint32_t)));
            DAGUE_ALLOCATE_REMOTE_DEPS_IF_NULL(arg->remote_deps, oldcontext, MAX_PARAM_COUNT);
            arg->remote_deps->root = src_rank;
            if( !(arg->remote_deps->output[param_index].rank_bits[_array_pos] & _array_mask) ) {
                arg->remote_deps->output[param_index].type = oldcontext->function->out[param_index]->dep_out[outdep_index]->type;
                arg->remote_deps->output[param_index].data = arg->data[param_index];
                arg->remote_deps->output[param_index].rank_bits[_array_pos] |= _array_mask;
                arg->remote_deps->output[param_index].count++;
                arg->remote_deps_count++;
            }
        }
#else
        (void)src_rank;
#endif

        if( (arg->action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS) &&
            (eu->master_context->my_rank == dst_rank) ) {
            arg->output_entry->data[param_index] = arg->data[param_index];
            arg->output_usage++;
            AREF( arg->output_entry->data[param_index] );
            arg->nb_released += dague_release_local_OUT_dependencies(oldcontext->dague_object,
                                                                     eu, oldcontext,
                                                                     oldcontext->function->out[param_index],
                                                                     newcontext,
                                                                     oldcontext->function->out[param_index]->dep_out[outdep_index]->param,
                                                                     &arg->ready_list);
        }
    } else {
#if defined(DAGUE_DEBUG)
        char tmp[128];
        DEBUG(("On task %s param_index %d not on the action_mask %x\n", dague_service_to_string(oldcontext, tmp, 128), param_index, arg->action_mask));
#endif
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

/* TODO: Change this code to something better */
static dague_object_t** object_array = NULL;
static uint32_t object_array_size = 1, object_array_pos = 0;

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

