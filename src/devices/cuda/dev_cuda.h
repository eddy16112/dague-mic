/*
 * Copyright (c) 2010-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DAGUE_GPU_DATA_H_HAS_BEEN_INCLUDED
#define DAGUE_GPU_DATA_H_HAS_BEEN_INCLUDED

#include <dague_config.h>
#include "dague_internal.h"
#include <dague/class/dague_object.h>
#include <dague/devices/device.h>

#if defined(HAVE_CUDA)
#include "list_item.h"
#include "list.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <dague/devices/device_malloc.h>

#define DAGUE_GPU_USE_PRIORITIES     1

#define DAGUE_MAX_STREAMS            4
#define DAGUE_MAX_EVENTS_PER_STREAM  4

#if defined(DAGUE_PROF_TRACE)
#define DAGUE_PROFILE_CUDA_TRACK_DATA_IN  0x0001
#define DAGUE_PROFILE_CUDA_TRACK_DATA_OUT 0x0002
#define DAGUE_PROFILE_CUDA_TRACK_OWN      0x0004
#define DAGUE_PROFILE_CUDA_TRACK_EXEC     0x0008

extern int dague_cuda_trackable_events;
extern int dague_cuda_movein_key_start;
extern int dague_cuda_movein_key_end;
extern int dague_cuda_moveout_key_start;
extern int dague_cuda_moveout_key_end;
extern int dague_cuda_own_GPU_key_start;
extern int dague_cuda_own_GPU_key_end;
#endif  /* defined(PROFILING) */

extern float *device_load, *device_weight;

typedef struct __dague_gpu_context {
    dague_list_item_t          list_item;
    dague_execution_context_t *ec;
} dague_gpu_context_t;

typedef struct __dague_gpu_exec_stream {
    struct __dague_gpu_context **tasks;
    CUevent *events;
    CUstream cuda_stream;
    int32_t max_events;  /* number of potential events, and tasks */
    int32_t executed;    /* number of executed tasks */
    int32_t start, end;  /* circular buffer management start and end positions */
    dague_list_t *fifo_pending;
#if defined(DAGUE_PROF_TRACE)
    int prof_event_track_enable;
    int prof_event_key_start, prof_event_key_end;
#endif  /* defined(PROFILING) */
} dague_gpu_exec_stream_t;

typedef struct _gpu_device {
    dague_device_t super;
    uint8_t major;
    uint8_t minor;
    uint8_t cuda_index;
    uint8_t max_exec_streams;
    int16_t peer_access_mask;  /**< A bit set to 1 represent the capability of
                                *   the device to access directly the memory of
                                *   the index of the set bit device.
                                */
    CUcontext  ctx;
    CUmodule   hcuModule;
    CUfunction hcuFunction;
    dague_gpu_exec_stream_t* exec_stream;
    dague_list_t* gpu_mem_lru;
    dague_list_t* gpu_mem_owned_lru;
    volatile uint32_t mutex;
    dague_list_t pending;
    gpu_malloc_t *memory;
} gpu_device_t;

#define DAGUE_CUDA_CHECK_ERROR( STR, ERROR, CODE )                      \
    do {                                                                \
        cudaError_t __cuda_error = (cudaError_t) (ERROR);               \
        if( cudaSuccess != __cuda_error ) {                             \
            WARNING(( "%s:%d %s%s\n", __FILE__, __LINE__,               \
                    (STR), cudaGetErrorString(__cuda_error) ));         \
            CODE;                                                       \
        }                                                               \
    } while(0)

int dague_gpu_init(dague_context_t *dague_context);
int dague_gpu_fini(void);

/**
 * Debugging functions.
 */
void dump_exec_stream(dague_gpu_exec_stream_t* exec_stream);
void dump_GPU_state(gpu_device_t* gpu_device);

/****************************************************
 ** GPU-DATA Specific Starts Here **
 ****************************************************/

/**
 * Overload the default data_copy_t with a GPU specialized type
 */
typedef dague_data_copy_t dague_gpu_data_copy_t;

#include "data_distribution.h"

/*
 * Data [un]registering
 */
int dague_gpu_data_register( dague_context_t *dague_context,
                             dague_ddesc_t   *data,
                             int              nbelem,
                             size_t           eltsize );
int dague_gpu_data_unregister( dague_ddesc_t* data );

/*
 * Data movement
 */
int dague_gpu_data_reserve_device_space( gpu_device_t* gpu_device,
                                         dague_execution_context_t *this_task,
                                         int *array_of_eltsize,
                                         int  move_data_count );

int dague_gpu_data_stage_in( gpu_device_t* gpu_device,
                             int32_t type,
                             dague_data_pair_t* task_data,
                             size_t length,
                             CUstream stream );

/**
 * Progress
 */
typedef int (*advance_task_function_t)(gpu_device_t* gpu_device,
                                       dague_gpu_context_t* task,
                                       dague_gpu_exec_stream_t* gpu_stream);

int progress_stream( gpu_device_t* gpu_device,
                     dague_gpu_exec_stream_t* gpu_stream,
                     advance_task_function_t progress_fct,
                     dague_gpu_context_t* task,
                     dague_gpu_context_t** out_task );

#endif /* defined(HAVE_CUDA) */

#endif
