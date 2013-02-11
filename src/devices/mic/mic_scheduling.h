/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"

#if defined(HAVE_CUDA)
#include "dague_internal.h"
#include <dague/devices/mic/dev_mic.h>
#include <dague/devices/device_malloc.h>
#include "profiling.h"
#include "execution_unit.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <errno.h>
#include "lifo.h"

/**
 * Define functions names
 */
#ifndef KERNEL_NAME
#error "KERNEL_NAME must be defined before to include this file"
#endif

#define GENERATE_NAME_v2( _func_, _kernel_ ) _func_##_##_kernel_
#define GENERATE_NAME( _func_, _kernel_ ) GENERATE_NAME_v2( _func_, _kernel_ )

#define mic_kernel_push      GENERATE_NAME( mic_kernel_push     , KERNEL_NAME )
#define mic_kernel_submit    GENERATE_NAME( mic_kernel_submit   , KERNEL_NAME )
#define mic_kernel_pop       GENERATE_NAME( mic_kernel_pop      , KERNEL_NAME )
#define mic_kernel_epilog    GENERATE_NAME( mic_kernel_epilog   , KERNEL_NAME )
#define mic_kernel_profile   GENERATE_NAME( mic_kernel_profile  , KERNEL_NAME )
#define mic_kernel_scheduler GENERATE_NAME( mic_kernel_scheduler, KERNEL_NAME )

/**
 * Try to execute a kernel on a GPU.
 *
 * Returns: one of the dague_hook_return_t values
 */

/**
 * This version is based on 4 streams: one for transfers from the memory to
 * the GPU, 2 for kernel executions and one for tranfers from the GPU into
 * the main memory. The synchronization on each stream is based on CUDA events,
 * such an event indicate that a specific epoch of the lifetime of a task has
 * been completed. Each type of stream (in, exec and out) has a pending FIFO,
 * where tasks ready to jump to the respective step are waiting.
 */
static inline dague_hook_return_t
mic_kernel_scheduler( dague_execution_unit_t *eu_context,
                     dague_mic_context_t    *this_task,
                     int which_mic )
{
    mic_device_t* mic_device;
    CUcontext saved_ctx;
    cudaError_t status;
    int rc, exec_stream = 0;
    dague_mic_context_t *next_task;
#if defined(DAGUE_DEBUG_VERBOSE2)
    char tmp[MAX_TASK_STRLEN];
#endif
    
    mic_device = (mic_device_t*)dague_devices_get(which_mic);
    
    /* Check the GPU status */
    rc = dague_atomic_inc_32b( &(mic_device->mutex) );
    if( 1 != rc ) {  /* I'm not the only one messing with this GPU */
        dague_fifo_push( &(mic_device->pending), (dague_list_item_t*)this_task );
        return DAGUE_HOOK_RETURN_ASYNC;
    }
    
/*    do {
        saved_ctx = gpu_device->ctx;
        dague_atomic_cas( &(gpu_device->ctx), saved_ctx, NULL );
    } while( NULL == saved_ctx );*/
    
#if defined(DAGUE_PROF_TRACE)
    if( dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_OWN )
        dague_profiling_trace( eu_context->eu_profile, dague_cuda_own_GPU_key_start,
                              (unsigned long)eu_context, PROFILE_OBJECT_ID_NULL, NULL );
#endif  /* defined(DAGUE_PROF_TRACE) */
    
/*    status = (cudaError_t)cuCtxPushCurrent(saved_ctx);
    DAGUE_CUDA_CHECK_ERROR( "cuCtxPushCurrent ", status,
                           {return DAGUE_HOOK_RETURN_DISABLE;} );*/
    
check_in_deps:
    if( NULL != this_task ) {
        DEBUG2(( "GPU:\tPush data for %s priority %d\n",
                dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, this_task->ec),
                this_task->ec->priority ));
    }
    rc = progress_stream_mic( mic_device,
                         &(mic_device->exec_stream[0]),
                         mic_kernel_push,
                         this_task, &next_task );
    if( rc < 0 ) {
        if( -1 == rc )
            goto disable_mic;
    }
    this_task = next_task;
    
    /* Stage-in completed for this Task: it is ready to be executed */
 //   exec_stream = (exec_stream + 1) % (gpu_device->max_exec_streams - 2);  /* Choose an exec_stream */
    exec_stream = 2;
    if( NULL != this_task ) {
        DEBUG2(( "GPU:\tExecute %s priority %d\n", 
                dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, this_task->ec),
                this_task->ec->priority ));
    }
    rc = progress_stream_mic( mic_device,
                         &(mic_device->exec_stream[exec_stream]),
                         mic_kernel_submit,
                         this_task, &next_task );
    if( rc < 0 ) {
        if( -1 == rc )
            goto disable_mic;
    }
    this_task = next_task;
    
    /* This task has completed its execution: we have to check if we schedule DtoN */
    if( NULL != this_task ) {
        DEBUG2(( "GPU:\tPop data for %s priority %d\n",
                dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, this_task->ec),
                this_task->ec->priority ));
    }
    /* Task is ready to move the data back to main memory */
    rc = progress_stream_mic( mic_device,
                         &(mic_device->exec_stream[1]),
                         mic_kernel_pop,
                         this_task,
                         &next_task );
    if( rc < 0 ) {
        if( -1 == rc )
            goto disable_mic;
    }
    if( NULL != next_task ) {
        /* We have a succesfully completed task. However, it is not this_task, as
         * it was just submitted into the data retrieval system. Instead, the task
         * ready to move into the next level is the next_task.
         */
        this_task = next_task;
        next_task = NULL;
        goto complete_task;
    }
    this_task = next_task;
    
fetch_task_from_shared_queue:
    assert( NULL == this_task );
    this_task = (dague_mic_context_t*)dague_fifo_try_pop( &(mic_device->pending) );
    if( NULL != this_task ) {
        DEBUG2(( "GPU:\tGet from shared queue %s priority %d\n", 
                dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, this_task->ec),
                this_task->ec->priority ));
    }
    goto check_in_deps;
    
complete_task:
    assert( NULL != this_task );
    DEBUG2(( "GPU:\tComplete %s priority %d\n",
            dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, this_task->ec),
            this_task->ec->priority ));
    /* Everything went fine so far, the result is correct and back in the main memory */
    DAGUE_LIST_ITEM_SINGLETON(this_task);
    mic_kernel_epilog( mic_device, this_task );
    __dague_complete_execution( eu_context, this_task->ec );
    dague_device_load[mic_device->super.device_index] -= dague_device_sweight[mic_device->super.device_index];
    mic_device->super.executed_tasks++;
    free( this_task );
    rc = dague_atomic_dec_32b( &(mic_device->mutex) );
    if( 0 == rc ) {  /* I was the last one */
#if defined(DAGUE_PROF_TRACE)
        if( dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_OWN )
            dague_profiling_trace( eu_context->eu_profile, dague_cuda_own_GPU_key_end,
                                  (unsigned long)eu_context, PROFILE_OBJECT_ID_NULL, NULL );
#endif  /* defined(DAGUE_PROF_TRACE) */
 //       status = (cudaError_t)cuCtxPopCurrent(NULL);
        /* Restore the context so the others can steal it */
 //       dague_atomic_cas( &(gpu_device->ctx), NULL, saved_ctx );
        
  //      DAGUE_CUDA_CHECK_ERROR( "cuCtxPushCurrent ", status,
  //                             {return DAGUE_HOOK_RETURN_ASYNC;} );
        return DAGUE_HOOK_RETURN_ASYNC;
    }
    this_task = next_task;
    goto fetch_task_from_shared_queue;
    
disable_mic:
    /* Something wrong happened. Push all the pending tasks back on the
     * cores, and disable the gpu.
     */
    printf("Critical issue related to the GPU discovered. Giving up\n");
    return DAGUE_HOOK_RETURN_DISABLE;
}

#endif /* HAVE_CUDA */
