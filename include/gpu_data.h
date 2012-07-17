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
#include "fifo.h"
#include "profiling.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "gpu_malloc.h"
#include "data_distribution.h"

#define DAGUE_GPU_USE_PRIORITIES     1

#define DAGUE_MAX_STREAMS            4
#define DAGUE_MAX_EVENTS_PER_STREAM  4

#define GPU_MEMORY_PER_TILE 1

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
    void   *function;
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

/**
 * Returns the number of GPUs managed by the DAGuE runtime. This is
 * different than the number of GPUs in the system, as they get
 * enabled based on the GPU mask.
 */
int dague_active_gpu(void);

/**
 * allocate a buffer to hold the data using GPU-compatible memory if needed
 */
void* dague_allocate_data( size_t matrix_size );

/**
 * free a buffer allocated by dague_allocate_data
 */
void dague_free_data(void *address);

/****************************************************
 ** GPU-DATA Specific Starts Here **
 ****************************************************/

/**
 * Data coherency protocol based on MOESI.
 */
#define    DAGUE_DATA_INVALID    ((uint8_t)0x0)
#define    DAGUE_DATA_OWNED      ((uint8_t)0x1)
#define    DAGUE_DATA_EXCLUSIVE  ((uint8_t)0x2)
#define    DAGUE_DATA_SHARED     ((uint8_t)0x4)

typedef uint8_t                    dague_data_coherency_t;
typedef struct _dague_device_elem  dague_device_elem_t;
typedef struct _memory_elem        memory_elem_t;
typedef struct _gpu_elem           gpu_elem_t;

/**
 * Generic type for all the devices.
 */
struct _dague_device_elem {
    dague_list_item_t      item;
    dague_data_coherency_t coherency_state;
    int16_t                readers;
    uint32_t               version;
    memory_elem_t*         memory_elem;
};

/**
 * A memory element targets a specific data. It can be found
 * based on a unique key.
 */
struct _memory_elem {
    uint32_t               key;
    dague_data_coherency_t coherency_state;
    uint16_t               device_owner;
    uint32_t               version;
    void*                  main_memory;
    dague_device_elem_t*   device_elem[1];
};

typedef struct __dague_gpu_data_map {
    dague_ddesc_t*  desc;
    memory_elem_t** data_map;
} dague_gpu_data_map_t;

/**
 * Particular overloading of the generic device type
 * for GPUs.
 */
struct _gpu_elem {
    dague_device_elem_t    generic;
    CUdeviceptr            gpu_mem;
};


typedef enum {
    DAGUE_READ       = ACCESS_READ,
    DAGUE_WRITE      = ACCESS_WRITE,
    DAGUE_READ_DONE  = 0x4,
    DAGUE_WRITE_DONE = 0x8
} dague_data_usage_type_t;

/*
 * Data [un]registering
 */
int dague_gpu_data_register( dague_context_t *dague_context,
                             dague_ddesc_t   *data,
                             int              nbelem,
                             size_t           eltsize );
int dague_gpu_data_unregister();

/*
 * Init/Finalize kernel
 */
int dague_gpu_kernel_fini(dague_context_t* dague_context,
                          char *kernelname);

/*
 * Data coherency and movement
 */
int dague_gpu_data_elt_write_owner( dague_gpu_data_map_t* gpu_map,
                                    uint32_t key );

int dague_gpu_data_get_elt( dague_gpu_data_map_t* gpu_map,
                            uint32_t key,
                            memory_elem_t **pmem_elem );

int dague_gpu_update_data_version( dague_gpu_data_map_t* gpu_map, uint32_t key );

int dague_gpu_find_space_for_elts( gpu_device_t* gpu_device,
                                   dague_execution_context_t *this_task,
                                   int *array_of_eltsize,
                                   int  move_data_count );
/**
 *
 */
int dague_gpu_data_stage_in( gpu_device_t* gpu_device,
                             int32_t type,
                             dague_data_pair_t* task_data,
                             size_t length,
                             CUstream stream );

/**
 *
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
