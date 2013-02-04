#ifndef _pingpong_gpu_h_
#define _pingpong_gpu_h_

#include <dague.h>
#include <data_distribution.h>
#include <data.h>
#include <dague/devices/cuda/dev_cuda.h>
#include <dague/devices/device_malloc.h>
#include <fifo.h>
#include "scheduling.h"

typedef struct dague_pingpong_args_s {
    dague_gpu_context_t super;
    int pushout;

    dague_ddesc_t *ddescA;
} dague_pingpong_args_t;

int pingpong_cuda(dague_execution_unit_t* eu_context,
               	  dague_execution_context_t* this_task,
				  dague_ddesc_t * descA);

dague_hook_return_t gpu_pingpong_scheduler( dague_execution_unit_t *eu_context,
                     dague_gpu_context_t    *this_task,
                     int which_gpu );

int gpu_pingpong_push( gpu_device_t            *gpu_device,
                       dague_gpu_context_t     *gpu_task,
                       dague_gpu_exec_stream_t *gpu_stream);

int gpu_pingpong_pop( gpu_device_t        *gpu_device,
                      dague_gpu_context_t *gpu_task,
                      dague_gpu_exec_stream_t* gpu_stream);

int gpu_pingpong_epilog( gpu_device_t        *gpu_device,
                         dague_gpu_context_t *gpu_task );


int pingpong_cuda(dague_execution_unit_t* eu_context,
               	  dague_execution_context_t* this_task,
				  dague_ddesc_t * descA)
{
	int i, data_index = 0;
	dague_pingpong_args_t *gpu_task;
    dague_handle_t* handle = this_task->dague_handle;
	
	 /* Step one: which write enabled data we will look at */
    for( i = 0; i < this_task->function->nb_parameters; i++ ) {
        if( this_task->function->in[i]->access_type & ACCESS_WRITE ) {
            data_index = i;
            break;
        }
    }	
	
	gpu_task = (dague_pingpong_args_t*)malloc(sizeof(dague_pingpong_args_t));
	OBJ_CONSTRUCT(gpu_task, dague_list_item_t);
    gpu_task->super.ec = this_task;
    gpu_task->pushout  = 1;
	gpu_task->ddescA   = (dague_ddesc_t*)descA;
	return gpu_pingpong_scheduler( eu_context, (dague_gpu_context_t*)gpu_task, 1 );

	return 1;
}

dague_hook_return_t gpu_pingpong_scheduler( dague_execution_unit_t *eu_context,
                     dague_gpu_context_t    *this_task,
                     int which_gpu )
{
	gpu_device_t* gpu_device;
    CUcontext saved_ctx;
    cudaError_t status;
    int rc, exec_stream = 0;
    dague_gpu_context_t *next_task;
#if defined(DAGUE_DEBUG_VERBOSE2)
    char tmp[MAX_TASK_STRLEN];
#endif
    
    gpu_device = (gpu_device_t*)dague_devices_get(which_gpu);
    
    /* Check the GPU status */
    rc = dague_atomic_inc_32b( &(gpu_device->mutex) );
    if( 1 != rc ) {  /* I'm not the only one messing with this GPU */
        dague_fifo_push( &(gpu_device->pending), (dague_list_item_t*)this_task );
        return DAGUE_HOOK_RETURN_ASYNC;
    }
    
    do {
        saved_ctx = gpu_device->ctx;
        dague_atomic_cas( &(gpu_device->ctx), saved_ctx, NULL );
    } while( NULL == saved_ctx );
    
#if defined(DAGUE_PROF_TRACE)
    if( dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_OWN )
        dague_profiling_trace( eu_context->eu_profile, dague_cuda_own_GPU_key_start,
                              (unsigned long)eu_context, PROFILE_OBJECT_ID_NULL, NULL );
#endif  /* defined(DAGUE_PROF_TRACE) */
    
    status = (cudaError_t)cuCtxPushCurrent(saved_ctx);
    DAGUE_CUDA_CHECK_ERROR( "cuCtxPushCurrent ", status,
                           {return DAGUE_HOOK_RETURN_DISABLE;} );
    
check_in_deps:
    if( NULL != this_task ) {
        DEBUG2(( "GPU[%1d]:\tPush data for %s priority %d\n", gpu_device->cuda_index,
                dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, this_task->ec),
                this_task->ec->priority ));
    }
    rc = progress_stream( gpu_device,
                         &(gpu_device->exec_stream[0]),
                         gpu_pingpong_push,
                         this_task, &next_task );
    if( rc < 0 ) {
        if( -1 == rc )
            goto disable_gpu;
    }
    this_task = next_task;
    
   
    
    /* This task has completed its execution: we have to check if we schedule DtoN */
    if( NULL != this_task ) {
        DEBUG2(( "GPU[%1d]:\tPop data for %s priority %d\n", gpu_device->cuda_index,
                dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, this_task->ec),
                this_task->ec->priority ));
    }
    /* Task is ready to move the data back to main memory */
    rc = progress_stream( gpu_device,
                         &(gpu_device->exec_stream[1]),
                         gpu_pingpong_pop,
                         this_task,
                         &next_task );
    if( rc < 0 ) {
        if( -1 == rc )
            goto disable_gpu;
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
    this_task = (dague_gpu_context_t*)dague_fifo_try_pop( &(gpu_device->pending) );
    if( NULL != this_task ) {
        DEBUG2(( "GPU[%1d]:\tGet from shared queue %s priority %d\n", gpu_device->cuda_index,
                dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, this_task->ec),
                this_task->ec->priority ));
    }
    goto check_in_deps;
    
complete_task:
    assert( NULL != this_task );
    DEBUG2(( "GPU[%1d]:\tComplete %s priority %d\n", gpu_device->cuda_index,
            dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, this_task->ec),
            this_task->ec->priority ));
    /* Everything went fine so far, the result is correct and back in the main memory */
    DAGUE_LIST_ITEM_SINGLETON(this_task);
    gpu_pingpong_epilog( gpu_device, this_task );
    __dague_complete_execution( eu_context, this_task->ec );
  //  dague_device_load[gpu_device->super.device_index] -= dague_device_sweight[gpu_device->super.device_index];
    gpu_device->super.executed_tasks++;
    free( this_task );
    rc = dague_atomic_dec_32b( &(gpu_device->mutex) );
    if( 0 == rc ) {  /* I was the last one */
#if defined(DAGUE_PROF_TRACE)
        if( dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_OWN )
            dague_profiling_trace( eu_context->eu_profile, dague_cuda_own_GPU_key_end,
                                  (unsigned long)eu_context, PROFILE_OBJECT_ID_NULL, NULL );
#endif  /* defined(DAGUE_PROF_TRACE) */
        status = (cudaError_t)cuCtxPopCurrent(NULL);
        /* Restore the context so the others can steal it */
        dague_atomic_cas( &(gpu_device->ctx), NULL, saved_ctx );
        
        DAGUE_CUDA_CHECK_ERROR( "cuCtxPushCurrent ", status,
                               {return DAGUE_HOOK_RETURN_ASYNC;} );
        return DAGUE_HOOK_RETURN_ASYNC;
    }
    this_task = next_task;
    goto fetch_task_from_shared_queue;
    
disable_gpu:
    /* Something wrong happened. Push all the pending tasks back on the
     * cores, and disable the gpu.
     */
    printf("Critical issue related to the GPU discovered. Giving up\n");
    return DAGUE_HOOK_RETURN_DISABLE;
}

int gpu_pingpong_push( gpu_device_t            *gpu_device,
                       dague_gpu_context_t     *gpu_task,
                       dague_gpu_exec_stream_t *gpu_stream)
{
    int i, ret, move_data_count = 0;
    dague_execution_context_t *this_task = gpu_task->ec;
    dague_pingpong_args_t        *args = (dague_pingpong_args_t*)gpu_task;
    dague_data_t              *data;
    dague_data_copy_t         *local;

    for( i = 0; i < this_task->function->nb_parameters; i++ ) {
        if( !(this_task->function->in[0]->access_type & ACCESS_READ) )
            continue;

        data = this_task->data[i].data->original;
        if( NULL == (local = dague_data_get_copy(data, gpu_device->super.device_index)) ) {
            move_data_count++;
        } else {
            /**
             * In case the data copy I got is not on my local device, swap the
             * reference with the most recent version on the local device. Otherwise,
             * use the original copy. This allow copy-on-write to work seamlesly.
             */
            if( this_task->data[i].data->device_index != gpu_device->super.device_index ) {
                /* Attach the GPU copy to the task */
                this_task->data[i].data = local;
            }
        }
    }

    if( 0 != move_data_count ) { /* Try to reserve enough room for all data */
        ret = dague_gpu_data_reserve_device_space( gpu_device,
                                                   this_task,
                                                   move_data_count );
        if( ret < 0 ) {
            goto release_and_return_error;
        }
    }

    assert( NULL != dague_data_copy_get_ptr(this_task->data[0].data) );

    DAGUE_TASK_PROF_TRACE_IF(gpu_stream->prof_event_track_enable,
                             gpu_device->super.profiling,
                             (-1 == gpu_stream->prof_event_key_start ?
                              DAGUE_PROF_FUNC_KEY_START(this_task->dague_handle,
                                                        this_task->function->function_id) :
                              gpu_stream->prof_event_key_start),
                             this_task);

    DEBUG3(("GPU[%1d]:\tIN  Data of %s(%d, %d) on GPU\n",
            gpu_device->cuda_index, this_task->function->in[0]->name, args->Am, args->An));
    ret = dague_gpu_data_stage_in( gpu_device, this_task->function->in[0]->access_type,
                                   &(this_task->data[0]), gpu_stream->cuda_stream );
    if( ret < 0 ) {
        goto release_and_return_error;
    }

  release_and_return_error:
    return ret;
}

int gpu_pingpong_pop( gpu_device_t        *gpu_device,
                      dague_gpu_context_t *gpu_task,
                      dague_gpu_exec_stream_t* gpu_stream)
{
    dague_execution_context_t *this_task = gpu_task->ec;
    dague_pingpong_args_t        *args = (dague_pingpong_args_t*)gpu_task;
    dague_gpu_data_copy_t     *gpu_copy = NULL;
    dague_data_t              *original;
    int return_code = 0, how_many = 0, i;
    cudaError_t status;

    for( i = 0; NULL != this_task->function->in[i]; i++ ) {
        gpu_copy = this_task->data[i].data;
        original = gpu_copy->original;
        if( this_task->function->in[i]->access_type & ACCESS_READ ) {
            gpu_copy->readers--; assert(gpu_copy->readers >= 0);
            if( (0 == gpu_copy->readers) &&
                !(this_task->function->in[i]->access_type & ACCESS_WRITE) ) {
                dague_list_item_ring_chop((dague_list_item_t*)gpu_copy);
                DAGUE_LIST_ITEM_SINGLETON(gpu_copy); /* TODO: singleton instead? */
                dague_ulist_fifo_push(&gpu_device->gpu_mem_lru, (dague_list_item_t*)gpu_copy);
            }
        }
        if( this_task->function->in[i]->access_type & ACCESS_WRITE ) {
            assert( gpu_copy == dague_data_get_copy(gpu_copy->original, gpu_device->super.device_index) );
            /* Stage the transfer of the data back to main memory */
            gpu_device->super.required_data_out += original->nb_elts;
            assert( ((dague_list_item_t*)gpu_copy)->list_next == (dague_list_item_t*)gpu_copy );
            assert( ((dague_list_item_t*)gpu_copy)->list_prev == (dague_list_item_t*)gpu_copy );

            if( args->pushout ) {  /* n == (k + 1) */
                DEBUG3(("GPU[%1d]:\tOUT Data of %s key %d\n", gpu_device->cuda_index,
                        this_task->function->in[i]->name, this_task->data[i].data->original->key));
                DAGUE_TASK_PROF_TRACE_IF(gpu_stream->prof_event_track_enable,
                                         gpu_device->super.profiling,
                                         (-1 == gpu_stream->prof_event_key_start ?
                                          DAGUE_PROF_FUNC_KEY_START(this_task->dague_handle,
                                                                    this_task->function->function_id) :
                                          gpu_stream->prof_event_key_start),
                                         this_task);
                /* TODO: Move the data back into main memory, but not always on the first device (!) */
                original = gpu_copy->original;
                status = (cudaError_t)cuMemcpyDtoHAsync( original->device_copies[0]->device_private,
                                                         (CUdeviceptr)gpu_copy->device_private,
                                                         original->nb_elts, gpu_stream->cuda_stream );
                DAGUE_CUDA_CHECK_ERROR( "cuMemcpyDtoHAsync from device ", status,
                                        { WARNING(("data %s <<%p>> -> <<%p>>\n", this_task->function->in[i]->name,
                                                   gpu_copy->device_private, original->device_copies[0]->device_private));
                                            return_code = -2;
                                            goto release_and_return_error;} );
                gpu_device->super.transferred_data_out += original->nb_elts; /* TODO: not hardcoded, use datatype size */
                how_many++;
            }
        }
    }

  release_and_return_error:
    return (return_code < 0 ? return_code : how_many);
}

int gpu_pingpong_epilog( gpu_device_t        *gpu_device,
                         dague_gpu_context_t *gpu_task )
{
    dague_execution_context_t *this_task = gpu_task->ec;
    dague_pingpong_args_t        *args = (dague_pingpong_args_t*)gpu_task;
    dague_gpu_data_copy_t     *gpu_copy;
    dague_data_t              *original;
    int i;

    for( i = 0; i < this_task->function->nb_parameters; i++ ) {
        if( !(this_task->function->in[i]->access_type & ACCESS_WRITE) ) continue;

        gpu_copy = this_task->data[i].data;
        assert( DATA_COHERENCY_OWNED == gpu_copy->coherency_state );
        gpu_copy->coherency_state = DATA_COHERENCY_SHARED;
        original = gpu_copy->original;
        original->version = gpu_copy->version;
        original->owner_device = -1;

        if( args->pushout ) {  /* n == (k  + 1) */
            dague_ulist_fifo_push(&gpu_device->gpu_mem_lru, (dague_list_item_t*)gpu_copy);
        } else {
            dague_ulist_fifo_push(&gpu_device->gpu_mem_owned_lru, (dague_list_item_t*)gpu_copy);
        }
    }
    return 0;
}

#endif /* _pingpong_gpu_h_ */
