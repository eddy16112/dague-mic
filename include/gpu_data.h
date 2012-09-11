/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DAGUE_GPU_DATA_H_HAS_BEEN_INCLUDED
#define DAGUE_GPU_DATA_H_HAS_BEEN_INCLUDED

#include "dague_config.h"
#include "dague_internal.h"

#if defined(HAVE_CUDA)
#include "list_item.h"
#include "list.h"
#include "profiling.h"
#include "moesi.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "gpu_malloc.h"

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
} dague_gpu_exec_stream_t;

typedef struct _gpu_device {
    dague_list_item_t item;
    CUcontext  ctx;
    CUmodule   hcuModule;
    CUfunction hcuFunction;
    uint8_t index;
    uint8_t device_index;
    uint8_t major;
    uint8_t minor;
    int16_t max_exec_streams;
    int16_t peer_access_mask;  /**< A bit set to 1 represent the capability of
                                *   the device to access directly the memory of
                                *   the index of the set bit device.
                                */
    dague_gpu_exec_stream_t* exec_stream;
    int executed_tasks;
    volatile uint32_t mutex;
    dague_list_t pending;
    uint64_t transferred_data_in;
    uint64_t transferred_data_out;
    uint64_t required_data_in;
    uint64_t required_data_out;
    dague_list_t* gpu_mem_lru;
    dague_list_t* gpu_mem_owned_lru;
#if defined(DAGUE_PROF_TRACE)
    dague_thread_profiling_t *profiling;
#endif  /* defined(PROFILING) */
    gpu_malloc_t *memory;
} gpu_device_t;

#define DAGUE_CUDA_CHECK_ERROR( STR, ERROR, CODE )                      \
    {                                                                   \
        cudaError_t __cuda_error = (cudaError_t) (ERROR);               \
        if( cudaSuccess != __cuda_error ) {                             \
            WARNING(( "%s:%d %s%s\n", __FILE__, __LINE__,               \
                    (STR), cudaGetErrorString(__cuda_error) ));         \
            CODE;                                                       \
        }                                                               \
    }

extern gpu_device_t** gpu_enabled_devices;
int dague_gpu_init(dague_context_t *dague_context,
                   int* puse_gpu,
                   int dague_show_detailed_capabilities);
int dague_gpu_fini( void );

/**
 * Enable and disale GPU-compatible memory if possible
 */
void dague_data_enable_gpu( int nbgpu );
void dague_data_disable_gpu( void );

/**
 * Returns the number of GPUs managed by the DAGuE runtime. This is
 * different than the number of GPUs in the system, as they get
 * enabled based on the GPU mask.
 */
int dague_active_gpu(void);

/****************************************************
 ** GPU-DATA Specific Starts Here **
 ****************************************************/

typedef struct _gpu_elem           gpu_elem_t;


/**
 * Particular overloading of the generic device type
 * for GPUs.
 */
struct _gpu_elem {
    dague_list_item_t   item;
    CUdeviceptr         gpu_mem_ptr;
    moesi_copy_t        moesi;
};

static inline void gpu_elem_construct(gpu_elem_t* gpu_elem, moesi_master_t* master) {
    DAGUE_LIST_ITEM_CONSTRUCT(gpu_elem);
    gpu_elem->moesi.master = master;
    gpu_elem->moesi.device_private = gpu_elem;
}
#define gpu_elem_destruct(gpu_elem)

static inline gpu_elem_t* gpu_elem_obtain_from_master(moesi_master_t* master, int device) {
    moesi_copy_t* copy = master->device_copies[device];
    if( NULL == copy ) return NULL;
    assert( copy->master == master );
    return copy->device_private;
}


typedef enum {
    DAGUE_READ       = ACCESS_READ,
    DAGUE_WRITE      = ACCESS_WRITE,
    DAGUE_READ_DONE  = 0x4,
    DAGUE_WRITE_DONE = 0x8
} dague_data_usage_type_t;

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
 * Init/Finalize kernel
 */
int dague_gpu_kernel_fini(dague_context_t* dague_context,
                          char *kernelname);

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
                                       CUstream cuda_stream);

int progress_stream( gpu_device_t* gpu_device,
                     dague_gpu_exec_stream_t* exec_stream,
                     advance_task_function_t progress_fct,
                     dague_gpu_context_t* task,
                     dague_gpu_context_t** out_task );

/**
 * Compute the adapted unit
 */
void dague_compute_best_unit( uint64_t length, float* updated_value, char** best_unit );

#endif /* defined(HAVE_CUDA) */

#endif
