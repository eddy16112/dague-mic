/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DAGUE_GPU_DATA_H_HAS_BEEN_INCLUDED
#define DAGUE_GPU_DATA_H_HAS_BEEN_INCLUDED

#include "dague_config.h"

#if defined(HAVE_CUDA)
#include "list_item.h"
#include "fifo.h"

#include "profiling.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "gpu_malloc.h"
#include "data_distribution.h"

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

typedef struct _gpu_device {
    dague_list_item_t item;
    CUcontext ctx;
    CUmodule   hcuModule;
    CUfunction hcuFunction;
    CUstream   streams[DAGUE_MAX_STREAMS];
    int max_streams;
#if !defined(DAGUE_GPU_STREAM_PER_TASK)
    int max_in_tasks,
        max_exec_tasks,
        max_out_tasks;
    int max_exec_streams;
    struct dague_execution_context_t **in_array;
    struct dague_execution_context_t **exec_array;
    struct dague_execution_context_t **out_array;
    CUevent *in_array_events;
    CUevent *exec_array_events;
    CUevent *out_array_events;
    int in_submit, in_waiting,
        exec_submit, exec_waiting,
        out_submit, out_waiting;
    dague_list_t *fifo_pending_in;
    dague_list_t *fifo_pending_exec;
    dague_list_t *fifo_pending_out;
#endif  /* DAGUE_GPU_STREAM_PER_TASK */
    uint8_t index;
    uint8_t device_index;
    uint8_t major;
    uint8_t minor;
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
int dague_gpu_init(int* puse_gpu, int dague_show_detailed_capabilities);
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
    uint16_t               readers;
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

int dague_gpu_data_stage_in( gpu_device_t* gpu_device,
                             uint32_t key, int32_t type,
                             memory_elem_t* mem_elem,
                             void* memptr, size_t length,
                             CUstream stream );

static inline gpu_elem_t*
dague_gpu_get_data_on_gpu( gpu_device_t* gpu_device,
                           dague_gpu_data_map_t* gpu_map,
                           uint32_t key,
                           memory_elem_t** mem_elem )
{
    if( 0 > dague_gpu_data_get_elt(gpu_map, key, mem_elem) )
        return NULL;
    return (gpu_elem_t*)((*mem_elem)->device_elem[gpu_device->index]);
}

/**
 * Compute the adapted unit
 */
void dague_compute_best_unit( uint64_t length, float* updated_value, char** best_unit );

#endif /* defined(HAVE_CUDA) */

#endif
