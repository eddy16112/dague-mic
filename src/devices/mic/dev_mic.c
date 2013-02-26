/*
 * Copyright (c) 2010-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "dague_internal.h"
#include <dague/utils/mca_param.h>
#include <dague/constants.h>

#if defined(HAVE_CUDA)
#include "dague.h"
#include "data.h"
#include "dague/devices/mic/dev_mic.h"
#include "dague/devices/device_malloc.h"
#include "profiling.h"
#include "execution_unit.h"
#include "arena.h"
#include "dague/utils/output.h"
#include "dague/devices/mymic.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <errno.h>
#include <dlfcn.h>

#if defined(DAGUE_PROF_TRACE)
/* Accepted values are: DAGUE_PROFILE_CUDA_TRACK_DATA_IN | DAGUE_PROFILE_CUDA_TRACK_DATA_OUT |
 *                      DAGUE_PROFILE_CUDA_TRACK_OWN | DAGUE_PROFILE_CUDA_TRACK_EXEC
 */
int dague_cuda_trackable_events = DAGUE_PROFILE_CUDA_TRACK_EXEC | DAGUE_PROFILE_CUDA_TRACK_DATA_OUT | DAGUE_PROFILE_CUDA_TRACK_DATA_IN | DAGUE_PROFILE_CUDA_TRACK_OWN;
int dague_cuda_movein_key_start;
int dague_cuda_movein_key_end;
int dague_cuda_moveout_key_start;
int dague_cuda_moveout_key_end;
int dague_cuda_own_GPU_key_start;
int dague_cuda_own_GPU_key_end;
#endif  /* defined(PROFILING) */

int dague_mic_output_stream = -1;

/* Dirty selection for now */
float mic_speeds[2][2] ={
    /* C1060, C2050 */
    { 622.08, 1030.4 },
    {  77.76,  515.2 }
};

//TODO:
static int dague_mic_device_fini(dague_device_t* device)
{
	return DAGUE_SUCCESS;
}

mic_mem_t* dague_mic_get_cpu_base(void* ptr, mic_device_t* mic_device)
{
	int i;
	for (i = 0; i < mic_device->num_of_cpu_ptr; i++) {
		if (ptr >= mic_device->cpu_ptr[i].addr && ptr < mic_device->cpu_ptr[i].addr+mic_device->cpu_ptr[i].actual_nbyte) {
			return &mic_device->cpu_ptr[i];
		}
	}
	return NULL;
}

static int dague_mic_host_memory_register(dague_device_t* device, void* ptr, size_t length)
{
    mic_device_t* mic_device = (mic_device_t*)device;
/*    mic_mem_t *mem_host = (mic_mem_t *)malloc(sizeof(mic_mem_t));
	mem_host->addr = ptr;
    if (micHostAlloc(mem_host, length) == MIC_ERROR) {
        return DAGUE_ERROR;
    }
    ptr = mem_host->addr;*/
	mic_device->cpu_ptr[mic_device->num_of_cpu_ptr].addr = ptr;
	off_t offset = micHostRegister(ptr, length);
	if (offset == 0x0) {
		return DAGUE_ERROR;
	}
	mic_device->cpu_ptr[mic_device->num_of_cpu_ptr].offset = offset;
	mic_device->cpu_ptr[mic_device->num_of_cpu_ptr].actual_nbyte = length;
	mic_device->num_of_cpu_ptr ++;
	printf("base %p\n", ptr);
    
    return DAGUE_SUCCESS;
}

//TODO:
static int dague_mic_host_memory_unregister(dague_device_t* device, void* ptr)
{
    return DAGUE_SUCCESS;
}

void* mic_solve_handle_dependencies(mic_device_t* mic_device,
                                     const char* fname)
{
    return NULL;
}

/* TODO: Ugly code to be removed ASAP */
void** mic_gemm_functions = NULL;
/* TODO: Ugly code to be removed ASAP */

static int
dague_mic_handle_register(dague_device_t* device, dague_handle_t* handle)
{
    mic_device_t* mic_device = (mic_device_t*)device;
    uint32_t i, dev_mask = 0x0;

    /**
     * Let's suppose it is not our job to detect if a particular body can
     * run or not. We will need to add some properties that will allow the
     * user to write the code to assess this.
     */
    assert(DAGUE_DEV_CUDA == device->type);
    for( i = 0; i < handle->nb_functions; i++ ) {
        const dague_function_t* function = handle->functions_array[i];
        __dague_chore_t* chores = (__dague_chore_t*)function->incarnations;
        for( uint32_t j = 0; NULL != chores[j].hook; j++ ) {
            dev_mask |= (1 << chores[j].type);
        }
        if(dev_mask & (1 << device->type)) {  /* find the function */
            void* devf = mic_solve_handle_dependencies(mic_device, function->name);
            /* TODO: Ugly code to be removed ASAP */
            if( NULL == mic_gemm_functions ) {
                mic_gemm_functions = (void**)calloc(100, sizeof(void*));
            }
            mic_gemm_functions[mic_device->mic_index] = devf;
            /* TODO: Ugly code to be removed ASAP */
        }
    }
    /* Not a single chore supports this device, there is no reason to check anything further */
    if(!(dev_mask & (1 << device->type))) {
        handle->devices_mask &= ~(device->device_index);
    }

    return DAGUE_SUCCESS;
}

static int
dague_mic_handle_unregister(dague_device_t* device, dague_handle_t* handle)
{
	(void)device; (void)handle;
    return DAGUE_SUCCESS;
}

int dague_mic_init(dague_context_t *dague_context)
{
    int show_caps_index, show_caps = 0;
    int use_cuda_index, use_cuda;
    int cuda_mask, cuda_verbosity;
    int ndevices, i, j, k;
    CUresult status;
    int isdouble = 0;
	int rc;

    use_cuda_index = dague_mca_param_reg_int_name("device_cuda", "enabled",
                                                  "The number of CUDA device to enable for the next PaRSEC context",
                                                  false, false, 0, &use_cuda);
    (void)dague_mca_param_reg_int_name("device_cuda", "mask",
                                       "The bitwise mask of CUDA devices to be enabled (default all)",
                                       false, false, 0xffffffff, &cuda_mask);
    (void)dague_mca_param_reg_int_name("device_cuda", "verbose",
                                       "Set the verbosity level of the CUDA device (negative value turns all output off, higher is less verbose)\n",
                                       false, false, -1, &cuda_verbosity);
    if( 0 == use_cuda ) {
        return -1;  /* Nothing to do around here */
    }

    if( cuda_verbosity >= 0 ) {
        dague_mic_output_stream = dague_output_open(NULL);
        dague_output_set_verbosity(dague_mic_output_stream, cuda_verbosity);
    }

    rc = micInit();
    if (rc != MIC_SUCCESS) {
		return -1;
	}

    micDeviceGetCount(&ndevices);

    if( ndevices < use_cuda ) {
        if( 0 < use_cuda_index )
            dague_mca_param_set_int(use_cuda_index, ndevices);
    }
    /* Update the number of GPU for the upper layer */
    use_cuda = ndevices;
    if( 0 == ndevices ) {
        return -1;
    }
    show_caps_index = dague_mca_param_find("device", NULL, "show_capabilities");
    if(0 < show_caps_index) {
        dague_mca_param_lookup_int(show_caps_index, &show_caps);
    }
#if defined(DAGUE_PROF_TRACE)
    dague_profiling_add_dictionary_keyword( "movein", "fill:#33FF33",
                                            0, NULL,
                                            &dague_cuda_movein_key_start, &dague_cuda_movein_key_end);
    dague_profiling_add_dictionary_keyword( "moveout", "fill:#ffff66",
                                            0, NULL,
                                            &dague_cuda_moveout_key_start, &dague_cuda_moveout_key_end);
    dague_profiling_add_dictionary_keyword( "cuda", "fill:#66ff66",
                                            0, NULL,
                                            &dague_cuda_own_GPU_key_start, &dague_cuda_own_GPU_key_end);
#endif  /* defined(PROFILING) */

    for( i = 0; i < ndevices; i++ ) {
#if CUDA_VERSION >= 3020
        size_t total_mem;
#else
        unsigned int total_mem;
#endif  /* CUDA_VERSION >= 3020 */
        mic_device_t* mic_device;
     //   CUdevprop devProps;
        char szName[256];


        mic_device = (mic_device_t*)calloc(1, sizeof(mic_device_t));
        OBJ_CONSTRUCT(mic_device, dague_list_item_t);
        mic_device->super.name = strdup(szName);	

        mic_device->max_exec_streams = DAGUE_MAX_STREAMS;
        mic_device->exec_stream =
            (dague_mic_exec_stream_t*)malloc(mic_device->max_exec_streams
                                             * sizeof(dague_mic_exec_stream_t));
        for( j = 0; j < mic_device->max_exec_streams; j++ ) {
            cudaError_t cudastatus;
            dague_mic_exec_stream_t* exec_stream = &(mic_device->exec_stream[j]);

            /* Allocate the stream */
            exec_stream->mic_stream = j; // just test, not real stream
            exec_stream->max_events   = DAGUE_MAX_EVENTS_PER_STREAM;
            exec_stream->executed     = 0;
            exec_stream->start        = 0;
            exec_stream->end          = 0;
            exec_stream->fifo_pending = (dague_list_t*)OBJ_NEW(dague_list_t);
            OBJ_CONSTRUCT(exec_stream->fifo_pending, dague_list_t);
            exec_stream->tasks  = (dague_mic_context_t**)malloc(exec_stream->max_events
                                                                * sizeof(dague_mic_context_t*));
           // exec_stream->events = (CUevent*)malloc(exec_stream->max_events * sizeof(CUevent));
			exec_stream->events = NULL;
			exec_stream->events = micInitEventQueue(10);
            /* and the corresponding events */
            for( k = 0; k < exec_stream->max_events; k++ ) {
             //   exec_stream->events[k] = NULL;
                exec_stream->tasks[k]  = NULL;
/*
#if CUDA_VERSION >= 3020
                status = cuEventCreate(&(exec_stream->events[k]), CU_EVENT_DISABLE_TIMING);
#else
                status = cuEventCreate(&(exec_stream->events[k]), CU_EVENT_DEFAULT);
#endif  /* CUDA_VERSION >= 3020 */
  /*              DAGUE_CUDA_CHECK_ERROR( "(INIT) cuEventCreate ", (cudaError_t)status,
                                        {break;} );*/
            }
#if defined(DAGUE_PROF_TRACE)
            exec_stream->prof_event_track_enable = dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_EXEC;
            exec_stream->prof_event_key_start    = -1;
            exec_stream->prof_event_key_end      = -1;
#endif  /* defined(DAGUE_PROF_TRACE) */
        }

		mic_device->cpu_ptr = (mic_mem_t *)malloc(sizeof(mic_mem_t) * MIC_MAX_PTR);
		mic_device->num_of_cpu_ptr = 0;

        mic_device->mic_index                 = (uint8_t)i;
        mic_device->super.type                 = DAGUE_DEV_CUDA;  // TODO: this one should be replaced later. 
        mic_device->super.executed_tasks       = 0;
        mic_device->super.transferred_data_in  = 0;
        mic_device->super.transferred_data_out = 0;
        mic_device->super.required_data_in     = 0;
        mic_device->super.required_data_out    = 0;

        mic_device->super.device_fini              = dague_mic_device_fini;
        mic_device->super.device_memory_register   = dague_mic_host_memory_register;
        mic_device->super.device_memory_unregister = dague_mic_host_memory_unregister;
        mic_device->super.device_handle_register   = dague_mic_handle_register;
        mic_device->super.device_handle_unregister = dague_mic_handle_unregister;

        /**
         * TODO: Find a better ay to evaluate the performance of the current GPU.
         * device_weight[i+1] = ((float)devProps.maxThreadsPerBlock * (float)devProps.clockRate) * 2;
         * device_weight[i+1] *= (concurrency == 1 ? 2 : 1);
         */
        mic_device->super.device_dweight = mic_speeds[1][1];
        mic_device->super.device_sweight = mic_speeds[0][1];

        /* Initialize internal lists */
        OBJ_CONSTRUCT(&mic_device->gpu_mem_lru,       dague_list_t);
        OBJ_CONSTRUCT(&mic_device->gpu_mem_owned_lru, dague_list_t);
        OBJ_CONSTRUCT(&mic_device->pending,           dague_list_t);

#if defined(DAGUE_PROF_TRACE)
        mic_device->super.profiling = dague_profiling_thread_init( 2*1024*1024, "GPU %d.0", i );
        /**
         * Reconfigure the stream 0 and 1 for input and outputs.
         */
        mic_device->exec_stream[0].prof_event_track_enable = dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_DATA_IN;
        mic_device->exec_stream[0].prof_event_key_start    = dague_cuda_movein_key_start;
        mic_device->exec_stream[0].prof_event_key_end      = dague_cuda_movein_key_end;

        mic_device->exec_stream[1].prof_event_track_enable = dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_DATA_OUT;
        mic_device->exec_stream[1].prof_event_key_start    = dague_cuda_moveout_key_start;
        mic_device->exec_stream[1].prof_event_key_end      = dague_cuda_moveout_key_end;
#endif  /* defined(PROFILING) */
        dague_devices_add(dague_context, &(mic_device->super));
    }
	printf("I finish init mic\n");

/* TODO: the following does not work */
#if defined(DAGUE_HAVE_PEER_DEVICE_MEMORY_ACCESS)

#endif

    return 0;
}

int dague_mic_fini(void)
{
    //dague_output_close(dague_cuda_output_stream);
    //dague_cuda_output_stream = -1;

    return DAGUE_SUCCESS;
}

int dague_mic_data_register( dague_context_t *dague_context,
                             dague_ddesc_t   *data,
                             int              nbelem, /* Could be a function of the dague_desc_t */
                             size_t           eltsize )
{
    mic_device_t* mic_device;
    CUresult status;
    uint32_t i;
    (void)eltsize; (void)data;
	printf("I am in mic_data_register\n");

    for(i = 0; i < dague_nb_devices; i++) {
        size_t how_much_we_allocate;

        size_t total_mem, free_mem, initial_free_mem;

        uint32_t mem_elem_per_gpu = 0;

        if( NULL == (mic_device = (mic_device_t*)dague_devices_get(i)) ) continue;
        /* Skip all non CUDA devices */
        if( DAGUE_DEV_CUDA != mic_device->super.type ) continue;

      /*  status = cuCtxPushCurrent( gpu_device->ctx );
        DAGUE_CUDA_CHECK_ERROR( "(dague_gpu_data_register) cuCtxPushCurrent ", status,
                                {continue;} );*/

        /**
         * It appears that CUDA allocate the memory in chunks of 1MB,
         * so we need to adapt to this.
         */
        micMemGetInfo( &initial_free_mem, &total_mem );
		
        free_mem = initial_free_mem;
        /* We allocate 9/10 of the available memory */
        how_much_we_allocate = (9 * initial_free_mem) / 10;

/* TODO:PER TILE IS NOT USED HERE */
#if defined(DAGUE_GPU_CUDA_ALLOC_PER_TILE)
        /*
         * We allocate a bunch of tiles that will be used
         * during the computations
         */
        while( (free_mem > eltsize )
               && (initial_free_mem - how_much_we_allocate)
               && !(mem_elem_per_gpu > (uint32_t)(nbelem/2*3)) ) {
            dague_gpu_data_copy_t* gpu_elem;
            CUdeviceptr device_ptr;
            cudaError_t cuda_status;
#if 0
            /* Enable to stress the GPU memory subsystem and the coherence protocol */
            if( mem_elem_per_gpu > 10 )
                break;
#endif
            gpu_elem = OBJ_NEW(dague_data_copy_t);

            cuda_status = (cudaError_t)cuMemAlloc( &device_ptr, eltsize);
            DAGUE_CUDA_CHECK_ERROR( "cuMemAlloc ", cuda_status,
                                    ({
#if CUDA_VERSION < 3020
                                        unsigned int _free_mem, _total_mem;
#else
                                        size_t _free_mem, _total_mem;
#endif  /* CUDA_VERSION < 3020 */
                                        cuMemGetInfo( &_free_mem, &_total_mem );
                                        WARNING(("Per context: free mem %zu total mem %zu\n",
                                                 _free_mem, _total_mem));
                                        free( gpu_elem );
                                        break;
                                     }) );
            gpu_elem->device_private = (void*)(long)device_ptr;    //TODO: PROBLEM HERE
            gpu_elem->device_index = gpu_device->super.device_index;
            mem_elem_per_gpu++;
            dague_ulist_fifo_push( &gpu_device->gpu_mem_lru, (dague_list_item_t*)gpu_elem );
            cuMemGetInfo( &free_mem, &total_mem );
        }
        if( 0 == mem_elem_per_gpu ) {
            WARNING(("GPU:\tRank %d Cannot allocate memory on GPU %d. Skip it!\n",
                     dague_context->my_rank, i));
            continue;
        }
        DAGUE_OUTPUT_VERBOSE((5, dague_mic_output_stream,
                              "GPU:\tAllocate %u tiles on the GPU memory\n", mem_elem_per_gpu ));
#else
        if( NULL == mic_device->memory ) {
            /*
             * We allocate all the memory on the GPU and we use our memory management
             */
            mem_elem_per_gpu = (how_much_we_allocate + GPU_MALLOC_UNIT_SIZE - 1 ) / GPU_MALLOC_UNIT_SIZE ;
            mic_device->memory = mic_malloc_init( mem_elem_per_gpu, GPU_MALLOC_UNIT_SIZE );

            if( mic_device->memory == NULL ) {
                WARNING(("GPU:\tRank %d Cannot allocate memory on GPU %d. Skip it!\n",
                         dague_context->my_rank, i));
                continue;
            }
            DAGUE_OUTPUT_VERBOSE((5, dague_mic_output_stream,
                                  "GPU:\tAllocate %u segment of size %d on the GPU memory\n",
                                  mem_elem_per_gpu, GPU_MALLOC_UNIT_SIZE ));
        }
#endif

       /* status = cuCtxPopCurrent(NULL);
        DAGUE_CUDA_CHECK_ERROR( "(INIT) cuCtxPopCurrent ", status,
                                {continue;} );*/
    }

    return 0;
}

/**
 * This function release all copies of a data on all devices. It ensure
 * the most recent version is moved back in main memory, and then release
 * the corresponding data from all attached devices.
 *
 * One has to notice that all the data available on the GPU is stored in one of
 * the two used to keep track of the allocated data, either the gpu_mem_lru or
 * the gpu_mem_owner_lru. Thus, going over all the elements in these two lists
 * should be enough to enforce a clean release.
 */
int dague_mic_data_unregister( dague_ddesc_t* ddesc )
{
	return 0;
}

int dague_mic_data_reserve_device_space( mic_device_t* mic_device,
                                         dague_execution_context_t *this_task,
                                         int  move_data_count )
{
    dague_gpu_data_copy_t* temp_loc[MAX_PARAM_COUNT], *gpu_elem, *lru_gpu_elem;
    dague_data_t* master;
    int eltsize = 0, i, j;
    (void)eltsize;

    /**
     * Parse all the input and output flows of data and ensure all have
     * corresponding data on the GPU available.
     */
    for( i = 0; i < this_task->function->nb_parameters; i++ ) {
        if(NULL == this_task->function->in[i]) continue;

		temp_loc[i] = NULL;

        master = this_task->data[i].data_in->original;
        gpu_elem = dague_data_get_copy(master, mic_device->super.device_index);
        /* There is already a copy on the device */
        if( NULL != gpu_elem ) continue;

#if !defined(DAGUE_GPU_CUDA_ALLOC_PER_TILE)
        gpu_elem = OBJ_NEW(dague_data_copy_t);

        eltsize = master->nb_elts;
        eltsize = (eltsize + GPU_MALLOC_UNIT_SIZE - 1) / GPU_MALLOC_UNIT_SIZE;

    malloc_data:
        gpu_elem->device_private = mic_malloc( mic_device->memory, eltsize );
        if( NULL == gpu_elem->device_private ) {
#endif

        find_another_data:
            lru_gpu_elem = (dague_gpu_data_copy_t*)dague_ulist_fifo_pop(&mic_device->gpu_mem_lru);
            if( NULL == lru_gpu_elem ) {
                /* Make sure all remaining temporary locations are set to NULL */
                for( ;  i < this_task->function->nb_parameters; temp_loc[i++] = NULL );
                break;  /* Go and cleanup */
            }
            DAGUE_LIST_ITEM_SINGLETON(lru_gpu_elem);

            /* If there are pending readers, let the gpu_elem loose. This is a weak coordination
             * protocol between here and the dague_gpu_data_stage_in, where the readers don't necessarily
             * always remove the data from the LRU.
             */
            if( 0 != lru_gpu_elem->readers ) {
                goto find_another_data;
            }
            /* Make sure the new GPU element is clean and ready to be used */
            assert( master != lru_gpu_elem->original );
#if !defined(DAGUE_GPU_CUDA_ALLOC_PER_TILE)
            assert(NULL != lru_gpu_elem->original);
#endif
            if( master != lru_gpu_elem->original ) {
                if( NULL != lru_gpu_elem->original ) {
                    dague_data_t* oldmaster = lru_gpu_elem->original;
                    /* Let's check we're not trying to steal one of our own data */
                    for( j = 0; j < this_task->function->nb_parameters; j++ ) {
                        if( NULL == this_task->data[j].data_in ) continue;
                        if( this_task->data[j].data_in->original == oldmaster ) {
                            temp_loc[j] = lru_gpu_elem;
                            goto find_another_data;
                        }
                    }

                    dague_data_copy_detach(oldmaster, lru_gpu_elem, mic_device->super.device_index);
                    DAGUE_OUTPUT_VERBOSE((3, dague_mic_output_stream,
                                          "GPU:\tRepurpose copy %p to mirror block %p (in task %s:i) instead of %p\n",
                                          lru_gpu_elem, master, this_task->function->name, i, oldmaster));

#if !defined(DAGUE_GPU_CUDA_ALLOC_PER_TILE)
                    mic_free( mic_device->memory, (void*)(lru_gpu_elem->device_private) );
                    free(lru_gpu_elem);
                    goto malloc_data;
#endif
                }
            }
            gpu_elem = lru_gpu_elem;

#if !defined(DAGUE_GPU_CUDA_ALLOC_PER_TILE)
        }
#endif
        assert( 0 == gpu_elem->readers );
        gpu_elem->coherency_state = DATA_COHERENCY_INVALID;
        gpu_elem->version = 0;
        dague_data_copy_attach(master, gpu_elem, mic_device->super.device_index);
		this_task->data[i].data_out = gpu_elem;
        move_data_count--;
        temp_loc[i] = gpu_elem;
        dague_ulist_fifo_push(&mic_device->gpu_mem_lru, (dague_list_item_t*)gpu_elem);
    }
    if( 0 != move_data_count ) {
        WARNING(("GPU:\tRequest space on GPU failed for %d out of %d data\n",
                 move_data_count, this_task->function->nb_parameters));
        /* We can't find enough room on the GPU. Insert the tiles in the begining of
         * the LRU (in order to be reused asap) and return without scheduling the task.
         */
        for( i = 0; NULL != this_task->data[i].data_in; i++ ) {
            if( NULL == temp_loc[i] ) continue;
            dague_ulist_lifo_push(&mic_device->gpu_mem_lru, (dague_list_item_t*)temp_loc[i]);
        }
        return -2;
    }
    return 0;
}


/**
 * If the most current version of the data is not yet available on the GPU memory
 * schedule a transfer.
 * Returns:
 *    0: The most recent version of the data is already available on the GPU
 *    1: A copy has been scheduled on the corresponding stream
 *   -1: A copy cannot be issued due to CUDA.
 */
int dague_mic_data_stage_in( mic_device_t* mic_device,
                             int32_t type,
                             dague_data_pair_t* task_data,
                             int stream )
{
    dague_data_copy_t* in_elem = task_data->data_in;
    dague_data_t* original = in_elem->original;
    dague_gpu_data_copy_t* gpu_elem = task_data->data_out;
    int transfer_required = 0;

    /* If the data will be accessed in write mode, remove it from any lists
     * until the task is completed.
     */
    if( ACCESS_WRITE & type ) {
        dague_list_item_ring_chop((dague_list_item_t*)gpu_elem);
        DAGUE_LIST_ITEM_SINGLETON(gpu_elem);
    }

    transfer_required = dague_data_transfer_ownership_to_copy(original, mic_device->super.device_index, (uint8_t)type);
    mic_device->super.required_data_in += original->nb_elts;
    if( transfer_required ) {
        int status;

        DAGUE_OUTPUT_VERBOSE((2, dague_mic_output_stream,
                              "GPU:\tMove data <%x> (%p:%p) to GPU requested\n",
                              original->key, in_elem->device_private, (void*)gpu_elem->device_private));
        /* Push data into the GPU */
		char *mic_base, *mic_now;
		size_t diff;
		mic_base = mic_device->memory->base;
		mic_now = (char *)gpu_elem->device_private;
		diff = mic_now - mic_base;

		mic_mem_t *cpu_base = dague_mic_get_cpu_base(in_elem->device_private, mic_device);	
		if (cpu_base == NULL) {
			printf("cpu ptr can not be found\n");
			return -1;
		}
		size_t diff_cpu = (char*)in_elem->device_private - (char*)cpu_base->addr;

	//	printf("base: %p, cpu_diff: %lu, cpu mem:%p, mic base: %p, mic diff %lu, key%x\n", cpu_base->addr, (unsigned long)diff_cpu, in_elem->device_private, mic_base, diff, original->key);
		
  //      status = (cudaError_t)cuMemcpyHtoDAsync( (CUdeviceptr)gpu_elem->device_private,
    //                                             in_elem->device_private, length, stream );
	//	micVMemcpyAsync((void *)in_elem->device_private, mic_device->memory->offset+diff, original->nb_elts, micMemcpyHostToDevice);
		micMemcpyAsync(cpu_base->offset+diff_cpu, mic_device->memory->offset+diff, original->nb_elts, micMemcpyHostToDevice);
   
        mic_device->super.transferred_data_in += original->nb_elts;
        /* TODO: take ownership of the data */
        return 1;
    }
    /* TODO: data keeps the same coherence flags as before */
    return 0;
}

#if DAGUE_GPU_USE_PRIORITIES
static inline dague_list_item_t* dague_fifo_push_ordered( dague_list_t* fifo,
                                                          dague_list_item_t* elem )
{
    dague_ulist_push_sorted(fifo, elem, dague_execution_context_priority_comparator);
    return elem;
}
#define DAGUE_FIFO_PUSH  dague_fifo_push_ordered
#else
#define DAGUE_FIFO_PUSH  dague_ulist_fifo_push
#endif

int progress_stream_mic( mic_device_t* mic_device,
                    dague_mic_exec_stream_t* exec_stream,
                    mic_advance_task_function_t progress_fct,
                    dague_mic_context_t* task,
                    dague_mic_context_t** out_task )
{
    int saved_rc = 0, rc;
    *out_task = NULL;

    if( NULL != task ) {
        DAGUE_FIFO_PUSH(exec_stream->fifo_pending, (dague_list_item_t*)task);
        task = NULL;
    }
 grab_a_task:
    if( NULL == exec_stream->tasks[exec_stream->start] ) {
        /* get the best task */
        task = (dague_mic_context_t*)dague_ulist_fifo_pop(exec_stream->fifo_pending);
    }
    if( NULL == task ) {
        /* No more room on the event list or no tasks. Keep moving */
        goto check_completion;
    }
    DAGUE_LIST_ITEM_SINGLETON((dague_list_item_t*)task);

    assert( NULL == exec_stream->tasks[exec_stream->start] );
    /**
     * In case the task is succesfully progressed, the corresponding profiling
     * event is triggered.
     */
    rc = progress_fct( mic_device, task, exec_stream );
    if( 0 > rc ) {
        if( -1 == rc ) return -1;  /* Critical issue */
        assert(0); // want to debug this. It happens too often
        /* No more room on the GPU. Push the task back on the queue and check the completion queue. */
        DAGUE_FIFO_PUSH(exec_stream->fifo_pending, (dague_list_item_t*)task);
        DAGUE_OUTPUT_VERBOSE((3, dague_mic_output_stream,
                              "GPU: Reschedule %s(task %p) priority %d: no room available on the GPU for data\n",
                              task->ec->function->name, (void*)task->ec, task->ec->priority ));
        saved_rc = rc;  /* keep the info for the upper layer */
    } else {
        /**
         * Do not skip the cuda event generation. The problem is that some of the inputs
         * might be in the pipe of being transferred to the GPU. If we activate this task
         * too early, it might get executed before the data is available on the GPU.
         * Obviously, this lead to incorrect results.
         */
    //    rc = cuEventRecord( exec_stream->events[exec_stream->start], exec_stream->cuda_stream );
	//	printf("record event %d on stream %d\n", exec_stream->start, exec_stream->mic_stream);
		rc = micEventRecord(exec_stream->events, exec_stream->start, exec_stream->mic_stream);
        exec_stream->tasks[exec_stream->start] = task;
        exec_stream->start = (exec_stream->start + 1) % exec_stream->max_events;
        DAGUE_OUTPUT_VERBOSE((3, dague_mic_output_stream,
                              "GPU: Submitted %s(task %p) priority %d\n",
                              task->ec->function->name, (void*)task->ec, task->ec->priority ));
    }
    task = NULL;

 check_completion:
    if( (NULL == *out_task) && (NULL != exec_stream->tasks[exec_stream->end]) ) {
  //      rc = cuEventQuery(exec_stream->events[exec_stream->end]);
		rc = micEventQuery(exec_stream->events, exec_stream->end);
     //   if( CUDA_SUCCESS == rc ) {
		//rc = MIC_SUCCESS;
		if( MIC_SUCCESS == rc ) {
            /* Save the task for the next step */
	//		printf("release event %d on stream %d\n", exec_stream->end, exec_stream->mic_stream);
            task = *out_task = exec_stream->tasks[exec_stream->end];
            DAGUE_OUTPUT_VERBOSE((3, dague_mic_output_stream,
                                  "GPU: Event for task %s(task %p) encountered\n",
                                  task->ec->function->name, (void*)task->ec ));
            exec_stream->tasks[exec_stream->end] = NULL;
            exec_stream->end = (exec_stream->end + 1) % exec_stream->max_events;
            DAGUE_TASK_PROF_TRACE_IF(exec_stream->prof_event_track_enable,
                                     mic_device->super.profiling,
                                     (-1 == exec_stream->prof_event_key_end ?
                                      DAGUE_PROF_FUNC_KEY_END(task->ec->dague_handle,
                                                              task->ec->function->function_id) :
                                      exec_stream->prof_event_key_end),
                                     task->ec);
            task = NULL;  /* Try to schedule another task */
            goto grab_a_task;
        }
        if( CUDA_ERROR_NOT_READY != rc ) {
            DAGUE_CUDA_CHECK_ERROR( "cuEventQuery ", rc,
                                    {return -1;} );
        }
    }
    return saved_rc;
}


#endif /* HAVE_CUDA */
