/*
 * Copyright (c) 2009-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DAGUE_H_HAS_BEEN_INCLUDED
#define DAGUE_H_HAS_BEEN_INCLUDED

#include "dague_config.h"

typedef struct dague_function_s          dague_function_t;
typedef struct dague_handle_s            dague_handle_t;
typedef struct dague_execution_context_s dague_execution_context_t;
typedef struct dague_dependencies_s      dague_dependencies_t;
typedef struct dague_execution_unit_s    dague_execution_unit_t;    /**< Each virtual process includes multiple execution units (posix threads + local data) */
typedef struct dague_vp_s                dague_vp_t;                /**< Each MPI process includes multiple virtual processes (and a single comm. thread) */
typedef struct dague_context_s           dague_context_t;           /**< The general context that holds all the threads of dague for this MPI process */

typedef void* (*dague_data_allocate_t)(size_t matrix_size);
typedef void (*dague_data_free_t)(void *data);
extern dague_data_allocate_t dague_data_allocate;
extern dague_data_free_t     dague_data_free;

typedef void (*dague_startup_fn_t)(dague_context_t *context,
                                   dague_handle_t *dague_handle,
                                   dague_execution_context_t** startup_list);
typedef int (*dague_completion_cb_t)(dague_handle_t* dague_handle, void*);
typedef void (*dague_destruct_fn_t)(dague_handle_t* dague_handle);

struct dague_handle_s {
    /** All dague_handle_t structures hold these two arrays **/
    uint32_t                   handle_id;
    volatile uint32_t          nb_local_tasks;
    uint32_t                   nb_functions;
    int32_t                    priority;
    uint32_t                   devices_mask;
    dague_context_t           *context;
    dague_startup_fn_t         startup_hook;
    const dague_function_t**   functions_array;
#if defined(DAGUE_PROF_TRACE)
    const int*                 profiling_array;
#endif  /* defined(DAGUE_PROF_TRACE) */
    /* Completion callback. Triggered when the all tasks associated with
     * a particular dague object have been completed.
     */
    dague_completion_cb_t      complete_cb;
    void*                      complete_cb_data;
    dague_destruct_fn_t        destructor;
    dague_dependencies_t**     dependencies_array;
};

const dague_function_t* dague_find(const dague_handle_t *dague_handle, const char *fname);
dague_context_t* dague_init( int nb_cores, int* pargc, char** pargv[]);
int dague_fini( dague_context_t** pcontext );
int dague_enqueue( dague_context_t* context, dague_handle_t* handle);
int dague_progress(dague_context_t* context);

/**
 * Allow to change the default priority of an object. It returns the
 * old priority (the default priorityy of an object is 0). This function
 * can be used during the lifetime of an object, however, only tasks
 * generated after this call will be impacted.
 */
static inline int32_t dague_set_priority( dague_handle_t* object, int32_t new_priority )
{
    int32_t old_priority = object->priority;
    object->priority = new_priority;
    return old_priority;
}

/* Dump functions */
char* dague_snprintf_execution_context( char* str, size_t size,
                                        const dague_execution_context_t* task);

/* Accessors to set and get the completion callback and data */
int dague_set_complete_callback(dague_handle_t* dague_handle,
                                dague_completion_cb_t complete_cb, void* complete_data);
int dague_get_complete_callback(const dague_handle_t* dague_handle,
                                dague_completion_cb_t* complete_cb, void** complete_data);

/**< Retrieve the local object attached to a unique object id */
dague_handle_t* dague_handle_lookup(uint32_t handle_id);
/**< Register the object with the engine. Create the unique identifier for the handle */
int dague_handle_register(dague_handle_t* handle);
/**< Unregister the object with the engine. */
void dague_handle_unregister(dague_handle_t* handle);
/**< Start the dague execution and launch the ready tasks */
int dague_handle_start(dague_handle_t* handle);

/**< Print DAGuE usage message */
void dague_usage(void);

#if defined(DAGUE_SIM)
int dague_getsimulationdate( dague_context_t *dague_context );
#endif
#endif  /* DAGUE_H_HAS_BEEN_INCLUDED */
