/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DAGUE_EXECUTION_UNIT_H_HAS_BEEN_INCLUDED
#define DAGUE_EXECUTION_UNIT_H_HAS_BEEN_INCLUDED

#include "dague_config.h"

#ifdef HAVE_HWLOC
#include <hwloc.h>
#endif

#include <pthread.h>
#include <stdint.h>
#include "hbbuffer.h"
#include "mempool.h"
#include "profiling.h"
#include "barrier.h"

struct dague_execution_unit {
    int32_t eu_id;
    pthread_t pthread_id;
#if defined(DAGUE_PROF_TRACE)
    dague_thread_profiling_t* eu_profile;
#endif /* DAGUE_PROF_TRACE */

    void *scheduler_object;

#if defined(DAGUE_SCHED_REPORT_STATISTICS)
    uint32_t sched_nb_tasks_done;
#endif

#if defined(DAGUE_SIM)
    int largest_simulation_date;
#endif

    dague_context_t*        master_context;
    dague_thread_mempool_t* context_mempool;
    dague_thread_mempool_t* datarepo_mempools[MAX_PARAM_COUNT+1];
};

struct dague_context_t {
    volatile int32_t __dague_internal_finalization_in_progress;
    int32_t nb_cores;
    volatile int32_t __dague_internal_finalization_counter;
    int32_t nb_nodes;
    volatile uint32_t active_objects;
    int32_t my_rank;
    dague_barrier_t  barrier;

    size_t remote_dep_fw_mask_sizeof;

    dague_mempool_t context_mempool;
    dague_mempool_t datarepo_mempools[MAX_PARAM_COUNT+1];
    pthread_t* pthreads;

#if defined(DAGUE_SIM)
    int largest_simulation_date;
#endif

#ifdef HAVE_HWLOC
    int comm_th_core;
    hwloc_cpuset_t comm_th_index_mask;
    hwloc_cpuset_t index_core_free_mask;
#endif

    /* This field should always be the last one in the structure. Even if the
     * declared number of execution units is 1 when we allocate the memory
     * we will allocate more (as many as we need), so everything after this
     * field might be overwritten.
     */
    dague_execution_unit_t* execution_units[1];
};

#endif  /* DAGUE_EXECUTION_UNIT_H_HAS_BEEN_INCLUDED */
