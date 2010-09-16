/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dplasma_config.h"
#include "gpu_data.h"
#include "dplasma.h"
#include "scheduling.h"

#include <stdio.h>
#include <cublas.h>
#include <plasma.h>

#include "data_management.h"
extern DPLASMA_desc ddescA;

#define DPLASMA_CONTEXT_PER_GPU 1

#if DPLASMA_SMART_SCHEDULING
float	cpu_usage = 4.0;                 /* CPU core is slower than the fastest GPU 7.0 times */
static gpu_device_t* get_best_gpu(int);  /* Function to get best choice of avaiblable Contexts !! */
int *waiting;
int max_wait = 1;
#else
/* We don't use gpu_devices, instead we use a subset of gpu-array
 * gpu_array - list of GPU by order of their performance
 */
gpu_device_t** gpu_devices = NULL;
#endif

int dplasma_show_detailed_capabilities = 0;
volatile int32_t cpu_counter = 0;
int ndevices = 0;
#if defined(DPLASMA_PROFILING)
static int movein_key_start;
static int movein_key_end;
static int compute_key_start;
static int compute_key_end;
static int moveout_key_start;
static int moveout_key_end;
#endif  /* defined(PROFILING) */

int spotrf_cuda_init( int* puse_gpu )
{
    cublasStatus cublas_status;
    CUdevice hcuDevice;
    int i, j;

    if( (*puse_gpu) == -1 ) {
        return -1;  /* Nothing to do around here */
    }
    cuInit(0);

    cuDeviceGetCount( &ndevices );

    if( ndevices > (*puse_gpu) )
        ndevices = (*puse_gpu);
    /* Update the number of GPU for the upper layer */
    *puse_gpu = ndevices;
    if( 0 == ndevices ) {
        return -1;
    }
    gpu_devices = (gpu_device_t**)calloc(ndevices, sizeof(gpu_device_t));

#if DPLASMA_SMART_SCHEDULING
	CUresult status;
	CUdevice hcuDevice_; /* use to compare */
	/* Choose GPU by requirement [Capability - sometimes > 1.3 because of needs of double precision ]*/
    int pi, pi_, tmp ,major, minor;
	int rmajor = 1, rminor = 1;
	/* gpu_array - list of GPU which we're gonna use ! */
	gpu_array = (gpu_item_t*)calloc(ndevices, sizeof(gpu_item_t));
	for(i = j = 0; i < ndevices; i++){
		status = cuDeviceGet( &hcuDevice, i );
		/* PASS requirement ?*/
		cuDeviceComputeCapability(&major, &minor, hcuDevice);
		if(major > rmajor || (major == rmajor && minor >= rminor)){
			gpu_array[j].gpu_id = i;
			
			/* Assign usage 
			 * the FASTEST will be 1.0 
			 * then the next would be 5.0 due to slower than the FASTEST 5.0 times */
			if( major == 1 & minor == 3 ) {
				gpu_array[j].func1_usage = 1.0;
			}else{
				gpu_array[j].func1_usage = 2.0;
			}
			/* working status - would it divided by function ? , probably */
			gpu_array[j].working = 0;
			gpu_array[j].func1_current = 0;	/* function 1 designed for top priority*/
			gpu_array[j].func2_current = 0;	/* function 2 designed for low priority trying to catch left GPU from func1*/

			dplasma_atomic_lifo_construct(&(gpu_array[j].gpu_devices)); /* gpu_devies Context*/
			/* EACH GPU have their own stack , we don't mix up due to need of selecting smart choice while we have different cards */
			/* Also, if there's a Card that can have many Context, we will be able to control it properly */
			j++;
		}
	}
	/* number of device changed due to requirement */
	ndevices = j;
	
	/* Sort GPU by Multiprocessor count */
	/* gpu_array[ best -> worst ] */
	/* We will be able to choose best available , near 0 */
	if( ndevices > 1 ) {
		for( i = 0; i < (ndevices - 1); i++ ) {
            for( j = 0; j < (ndevices - 1); j++ ) {
                status = cuDeviceGet( &hcuDevice, gpu_array[j].gpu_id );
				DPLASMA_CUDA_CHECK_ERROR( "cuDeviceGet ", status, {*puse_gpu = 0; return -1;} );
				cuDeviceGetAttribute(&pi, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, hcuDevice);
				
				status = cuDeviceGet( &hcuDevice_, gpu_array[j+1].gpu_id );
				DPLASMA_CUDA_CHECK_ERROR( "cuDeviceGet ", status, {*puse_gpu = 0; return -1;} );
				cuDeviceGetAttribute(&pi_, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, hcuDevice_);
				if(pi < pi_){
					tmp = gpu_array[j].gpu_id;
					gpu_array[j].gpu_id = gpu_array[j+1].gpu_id;
					gpu_array[j+1].gpu_id = tmp;
				}else if(pi == pi_){
					cuDeviceGetAttribute(&pi, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, hcuDevice);
					cuDeviceGetAttribute(&pi_, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, hcuDevice_);
					if(pi < pi_){
						tmp = gpu_array[j].gpu_id;
						gpu_array[j].gpu_id = gpu_array[j+1].gpu_id;
						gpu_array[j+1].gpu_id = tmp;
					}
				}
			}
		}
	}
	/* Setup default vaule */
	waiting = (int*)calloc(max_wait, sizeof(int));
	for( i = 0; i < max_wait ; i++){
            waiting[i] = 0;
	}
#endif

	for( i = 0; i < ndevices; i++ ) {
        unsigned int total_mem, tile_size, thread_gpu_mem, free_mem;
        unsigned int nb_allocations = 0;
        dplasma_atomic_lifo_t* gpu_mem_lifo;
        gpu_device_t* gpu_device;
        CUdevprop devProps;
        char szName[256];
        CUresult status;
        int major, minor;

#if DPLASMA_SMART_SCHEDULING
	    /* instead of go from ID 0 -> n - 1 
	     * we have to use ID which's in gpu_array 
	     * because they might not be in order due to sorting/reordering */
	    status = cuDeviceGet( &hcuDevice, gpu_array[i].gpu_id);
	    DPLASMA_CUDA_CHECK_ERROR( "cuDeviceGet ", status, {*puse_gpu = 0; return -1;} );
#else
	    status = cuDeviceGet( &hcuDevice, i );
	    DPLASMA_CUDA_CHECK_ERROR( "cuDeviceGet ", status, {*puse_gpu = 0; return -1;} );
#endif
        status = cuDeviceGetName( szName, 256, hcuDevice );
        DPLASMA_CUDA_CHECK_ERROR( "cuDeviceGetName ", status, {*puse_gpu = 0; return -1;} );

        status = cuDeviceComputeCapability( &major, &minor, hcuDevice);
        DPLASMA_CUDA_CHECK_ERROR( "cuDeviceComputeCapability ", status, {*puse_gpu = 0; return -1;} );

        status = cuDeviceGetProperties( &devProps, hcuDevice );
        DPLASMA_CUDA_CHECK_ERROR( "cuDeviceGetProperties ", status, {*puse_gpu = 0; return -1;} );

#if DPLASMA_SMART_SCHEDULING
	    printf("Device %d (capability %d.%d): %s\n", gpu_array[i].gpu_id, major, minor, szName );
#else
        printf("Device %d (capability %d.%d): %s\n", i, major, minor, szName );
#endif
        if( dplasma_show_detailed_capabilities ) {
            printf("\tmaxThreadsPerBlock : %d\n", devProps.maxThreadsPerBlock );
            printf("\tmaxThreadsDim      : [%d %d %d]\n", devProps.maxThreadsDim[0],
                   devProps.maxThreadsDim[1], devProps.maxThreadsDim[2] );
            printf("\tmaxGridSize        : [%d %d %d]\n", devProps.maxGridSize[0],
                   devProps.maxGridSize[1], devProps.maxGridSize[2] );
            printf("\tsharedMemPerBlock  : %d\n", devProps.sharedMemPerBlock );
            printf("\tconstantMemory     : %d\n", devProps.totalConstantMemory );
            printf("\tSIMDWidth          : %d\n", devProps.SIMDWidth );
            printf("\tmemPitch           : %d\n", devProps.memPitch );
            printf("\tregsPerBlock       : %d\n", devProps.regsPerBlock );
            printf("\tclockRate          : %d\n", devProps.clockRate );
#if 0
            > 1.2 printf("\tdeviceOverlap    : %ssupported\n", (devProps.deviceOverlap ? "" : "not ") );
            > 2.0 printf("\tconcurrentKernels: %ssupported\n", (devProps.concurrentKernels ? "" : "not ") );
#endif
        }
        status = cuDeviceTotalMem( &total_mem, hcuDevice );
        DPLASMA_CUDA_CHECK_ERROR( "cuDeviceTotalMem ", status, {*puse_gpu = 0; return -1;} );

        for( j = 0; j < DPLASMA_CONTEXT_PER_GPU; j++ ) {
            cudaError_t cuda_status;

            gpu_device = (gpu_device_t*)calloc(1, sizeof(gpu_device_t));
            gpu_devices[i] = gpu_device;
            dplasma_dequeue_construct(&gpu_device->pending);
            gpu_device->major = major;
            gpu_device->minor = minor;

            /* cuCtxCreate: Function works on floating contexts and current context */
            status = cuCtxCreate( &(gpu_device->ctx), 0 /*CU_CTX_BLOCKING_SYNC*/, hcuDevice );
            DPLASMA_CUDA_CHECK_ERROR( "(INIT) cuCtxCreate ", status,
                                      {free(gpu_device); return -1;} );

            {
                char module_path[20];
                assert(major < 10 && minor < 10);
                snprintf(module_path, 20, "sgemm-sm_%1d%1d.cubin", major, minor);
                status = cuModuleLoad(&(gpu_device->hcuModule), module_path);
                DPLASMA_CUDA_CHECK_ERROR( "(INIT) cuModuleLoad ", status,
                                          {
                                              cuCtxDestroy( gpu_device->ctx );
                                              free(gpu_device);
                                              break;
                                          } );
                    
                status = cuModuleGetFunction( &(gpu_device->hcuFunction), gpu_device->hcuModule, "sgemmNT" );
                DPLASMA_CUDA_CHECK_ERROR( "(INIT) cuModuleGetFunction ", status,
                                          {
                                              cuCtxDestroy( gpu_device->ctx );
                                              free(gpu_device);
                                              break;
                                          } );
                if( 1 == major ) {
                    cuFuncSetBlockShape( gpu_device->hcuFunction, 16, 4, 1 );
                } else {
                    cuFuncSetBlockShape( gpu_device->hcuFunction, 64, 4, 1 );
                }
            }

            /**
             * Prepare the reusable memory on the GPU.
             */
            dplasma_data_map_init( gpu_device, &ddescA );
            /**
             * It appears that CUDA allocate the memory in chunks of 1MB,
             * so we need to adapt to this.
             */
            tile_size = ddescA.bsiz * sizeof(float);
            cuMemGetInfo( &free_mem, &total_mem );
            /* We allocate 9/10 of the total memory */
            thread_gpu_mem = (total_mem - total_mem / 10) / DPLASMA_CONTEXT_PER_GPU;

            while( free_mem > (total_mem - thread_gpu_mem) ) {
                gpu_elem_t* gpu_elem;
                if( nb_allocations > ((ddescA.mt * ddescA.nt) >> 1) )
                    break;
                gpu_elem = (gpu_elem_t*)malloc(sizeof(gpu_elem_t));
                dplamsa_linked_list_item_construct( (dplasma_list_item_t*)gpu_elem );
                
                cuda_status = cuMemAlloc( &(gpu_elem->gpu_mem), tile_size);
                DPLASMA_CUDA_CHECK_ERROR( "cuMemAlloc ", cuda_status,
                                          ({
                                              unsigned int free_mem, total_mem;
                                              cuMemGetInfo( &free_mem, &total_mem );
                                              printf("Per context: free mem %u total mem %u\n", free_mem, total_mem);
                                              free( gpu_elem );
                                              break;
                                          }) );
                nb_allocations++;
                gpu_elem->memory_elem = NULL;
                dplasma_linked_list_add_tail( gpu_device->gpu_mem_lru, (dplasma_list_item_t*)gpu_elem );
                cuMemGetInfo( &free_mem, &total_mem );
            }
            printf( "Allocate %d tiles on the GPU memory\n", nb_allocations );

            /**
             * Allocate the streams
             */
            {
                int stream_id;
                gpu_device->max_streams = DPLASMA_MAX_STREAMS;
                for( stream_id = 0; stream_id < DPLASMA_MAX_STREAMS; stream_id++ ) {
                    cuda_status = cuStreamCreate( &(gpu_device->streams[stream_id]), 0 );
                    DPLASMA_CUDA_CHECK_ERROR( "cuStreamCreate ", cuda_status,
                                          ({
                                              gpu_device->max_streams = stream_id - 1;
                                              break;
                                          }) );
                }
            }
#if DPLASMA_SMART_SCHEDULING
            /* because GPU might not be in sequence
             * but we can get the GPU ID from gpu_array[i].gpu_id
             * Don't forget - we're running around gpu_array[i] -> n-1
             *  */
            gpu_device->id = gpu_array[i].gpu_id;
#else
            gpu_device->id  = i;
#endif
            gpu_device->executed_tasks = 0;
            gpu_device->transferred_data_in = 0;
            gpu_device->transferred_data_out = 0;

            status = cuCtxPopCurrent(NULL);
            DPLASMA_CUDA_CHECK_ERROR( "(INIT) cuCtxPopCurrent ", status,
                                      {free(gpu_device); return -1;} );
            
#if defined(DPLASMA_PROFILING)
#if DPLASMA_SMART_SCHEDULING
            gpu_device->profiling = dplasma_profiling_thread_init( 6*4096, "GPU %d.%d", gpu_array[i].gpu_id, j);
#else
            gpu_device->profiling = dplasma_profiling_thread_init( 6*4096, "GPU %d.%d", i, j );
#endif
#endif  /* defined(PROFILING) */

#if DPLASMA_SMART_SCHEDULING
            /* After use context, GPUs will not know which gpu_array id they are up to */
            /* I will put lifoid for them !! */
            gpu_device->lifoid = i;
            dplasma_atomic_lifo_push( &(gpu_array[i].gpu_devices), (dplasma_list_item_t*)gpu_device);
            printf("\t\tPush context into list[%d] for GPU[%d]\n",i,gpu_array[i].gpu_id);
#endif

        }
    }

#if defined(DPLASMA_PROFILING)
    dplasma_profiling_add_dictionary_keyword( "movein", "fill:#33FF33",
                                              &movein_key_start, &movein_key_end);
    dplasma_profiling_add_dictionary_keyword( "compute", "fill:#ff33cc",
                                              &compute_key_start, &compute_key_end);
    dplasma_profiling_add_dictionary_keyword( "moveout", "fill:#ffff66",
                                              &moveout_key_start, &moveout_key_end);
#endif  /* defined(PROFILING) */

    return 0;
}

static void compute_best_unit( uint64_t length, float* updated_value, char** best_unit )
{
    float measure = (float)length;

    *best_unit = "B";
    if( measure > 1024.0 ) { /* 1KB */
        *best_unit = "KB";
        measure = measure / 1024.0;
        if( measure > 1024.0 ) { /* 1MB */
            *best_unit = "MB";
            measure = measure / 1024.0;
            if( measure > 1024.0 ) {
                *best_unit = "GB";
                measure = measure / 1024.0;
            }
        }
    }
    *updated_value = measure;
    return;
}

int spotrf_cuda_fini( int use_gpu )
{
    cudaError_t status;
    gpu_elem_t* gpu_elem;
    gpu_device_t* gpu_device;
    int total = 0, *gpu_counter, i, j, overlap_counter, active_devices = 0;
    uint64_t *transferred_in, *transferred_out, total_data_in = 0, total_data_out = 0;
    uint64_t *required_in, *required_out;
    float gtotal = 0.0, best_data_in, best_data_out;
    char *data_in_unit, *data_out_unit;

    if (ndevices <= 0)
        return 0;

    /* GPU counter for GEMM / each */
    gpu_counter = (int*)calloc(ndevices, sizeof(int));
    transferred_in  = (uint64_t*)calloc(ndevices, sizeof(uint64_t));
    transferred_out = (uint64_t*)calloc(ndevices, sizeof(uint64_t));
    required_in     = (uint64_t*)calloc(ndevices, sizeof(uint64_t));
    required_out    = (uint64_t*)calloc(ndevices, sizeof(uint64_t));

    for(i = 0; i < ndevices ; i++)
#if DPLASMA_SMART_SCHEDULING	
        /* RUN INTO gpu_array[ 0 -> ndevices - 1] */
	    while( NULL != (gpu_device = (gpu_device_t*)dplasma_atomic_lifo_pop(&(gpu_array[i].gpu_devices))) )
#endif
            {
#if !DPLASMA_SMART_SCHEDULING	
                gpu_device = gpu_devices[i];
#endif
                status = cuCtxPushCurrent( gpu_device->ctx );
                DPLASMA_CUDA_CHECK_ERROR( "(FINI) cuCtxPushCurrent ", status,
                                          {continue;} );
                status = cuCtxSynchronize();
                DPLASMA_CUDA_CHECK_ERROR( "cuCtxSynchronize", status,
                                          {continue;} );
                /* Save the statistics */
                gpu_counter[gpu_device->id]     += gpu_device->executed_tasks;
                transferred_in[gpu_device->id]  += gpu_device->transferred_data_in;
                transferred_out[gpu_device->id] += gpu_device->transferred_data_out;
                required_in[gpu_device->id]     += gpu_device->required_data_in;
                required_out[gpu_device->id]    += gpu_device->required_data_out;

                /**
                 * Release the GPU memory.
                 */
                while( NULL != (gpu_elem = (gpu_elem_t*)dplasma_linked_list_remove_head( gpu_device->gpu_mem_lru )) ) {
                    cuMemFree( gpu_elem->gpu_mem );
                    free( gpu_elem );
                }
                /**
                 * Release all streams
                 */
                for( j = 0; j < gpu_device->max_streams; j++ ) {
                    cuStreamDestroy( gpu_device->streams[j] );
                }

                status = cuCtxDestroy( gpu_device->ctx );
                DPLASMA_CUDA_CHECK_ERROR( "(FINI) cuCtxDestroy ", status,
                                          {continue;} );
                free(gpu_device->gpu_mem_lru);
                free(gpu_device);
                active_devices++;
            }

    /* No active devices */
    if( 0 == active_devices )
        return 0;

    /* Print statisitics */
    for( i = 0; i < ndevices; i++ ) {
        total += gpu_counter[i];
        total_data_in  += transferred_in[i];
        total_data_out += transferred_out[i];
    }
    if( 0 == total_data_in ) total_data_in = 1;
    if( 0 == total_data_out ) total_data_out = 1;

    gtotal = total + cpu_counter;
    printf("------------------------------------------------------------------------------\n");
    printf("|PU       |  # GEMM   |    %%   |   Data In   |    %%   |   Data Out  |    %%   |\n");
    printf("|---------|-----------|--------|-------------|--------|-------------|--------|\n");
    for( i = 0; i < ndevices; i++ ) {
        compute_best_unit( transferred_in[i],  &best_data_in, &data_in_unit );
        compute_best_unit( transferred_out[i], &best_data_out, &data_out_unit );
        printf("|GPU:  %2d |%10d | %6.2f |%10.2f%2s | %6.2f |%10.2f%2s | %6.2f |\n",
               i, gpu_counter[i], (gpu_counter[i]/gtotal)*100.00,
               best_data_in, data_in_unit, (((float)transferred_in[i]) / required_in[i]) * 100.0,
               best_data_out, data_out_unit, (((float)transferred_out[i]) / required_out[i]) * 100.0 );
    }
    printf("|---------|-----------|--------|-------------|--------|-------------|--------|\n");
    compute_best_unit( total_data_in,  &best_data_in, &data_in_unit );
    compute_best_unit( total_data_out, &best_data_out, &data_out_unit );
    printf("|All GPUs |%10d | %6.2f |%10.2f%2s | %6.2f |%10.2f%2s | %6.2f |\n",
           total, (total/gtotal)*100.00,
           best_data_in, data_in_unit, 100.0,
           best_data_out, data_out_unit, 100.0);
    printf("|All CPUs |%10d | %6.2f |%10.2f%2s | %6.2f |%10.2f%2s | %6.2f |\n",
           cpu_counter, (cpu_counter / gtotal)*100.00,
           0.0, " ", 0.0, 0.0, " ", 0.0);
    /*printf("|---------|-----------|--------|-------------|--------|-------------|--------|\n");
      printf("|Overlap  |%10d  %10.5s |\n", overlap_counter, "times");*/
    printf("------------------------------------------------------------------------------\n");
    free(gpu_counter);
    free(transferred_in);
    free(transferred_out);
    free(required_in);
    free(required_out);
}

#define ALIGN_UP(OFFSET, ALIGN) \
    (OFFSET) = ((OFFSET) + (ALIGN) - 1) & ~((ALIGN) - 1)
#define CU_PUSH_POINTER( FUNCTION, OFFSET, PTR )                        \
        do {                                                            \
            void* __ptr = (void*)(size_t)(PTR);                         \
            ALIGN_UP((OFFSET), __alignof(void*));                       \
            cuParamSetv( (FUNCTION), (OFFSET), &__ptr, sizeof(void*));  \
            (OFFSET) += sizeof(void*);                                  \
        } while (0)
#define CU_PUSH_INT( FUNCTION, OFFSET, VALUE )                          \
        do {                                                            \
            ALIGN_UP((OFFSET), __alignof(int));                         \
            cuParamSeti( (FUNCTION), (OFFSET), (VALUE) );               \
            (OFFSET) += sizeof(int);                                    \
        } while (0)
#define CU_PUSH_FLOAT( FUNCTION, OFFSET, VALUE )                        \
        do {                                                            \
            ALIGN_UP((OFFSET), __alignof(float));                       \
            cuParamSetf( (FUNCTION), (OFFSET), (VALUE) );               \
            (OFFSET) += sizeof(float);                                  \
        } while (0)

static int
gpu_sgemm_internal( gpu_device_t* gpu_device,
                    dplasma_execution_unit_t* eu_context,
                    dplasma_execution_context_t* exec_context,
                    CUstream stream,
                    int uplo, void* A, void* B, void* C, int k, int n, int m )
 {
    gpu_elem_t *gpu_elem_A = NULL, *gpu_elem_B = NULL, *gpu_elem_C = NULL;
    int offset, on_gpu, return_code = 0, tile_size;  /* by default suppose an error */
    dplasma_execution_context_t* new_context;
    float alpha = -1.0, beta = 1.0;
    int grid_width, grid_height;
    CUdeviceptr d_A, d_B, d_C;
    cudaError_t status;
    void* ptr;

    tile_size = ddescA.mb*ddescA.nb*sizeof(float);

 loop_around_gpu_submission:
#if defined(DPLASMA_PROFILING)
    dplasma_profiling_trace( gpu_device->profiling, movein_key_start, 0 );
#endif  /* defined(PROFILING) */
    /*cuStreamCreate(&stream, 0);*/
    on_gpu = dplasma_data_is_on_gpu(gpu_device, &ddescA, DPLASMA_READ, n, k, &gpu_elem_A);
    gpu_elem_A->memory_elem->memory = A;
    d_A = gpu_elem_A->gpu_mem;
    gpu_device->required_data_in += tile_size;
    if( !on_gpu ) {
        /* Push A into the GPU */
        status = cuMemcpyHtoDAsync( d_A, A, tile_size, stream );
        DPLASMA_CUDA_CHECK_ERROR( "cuMemcpyHtoDAsync to device (d_A) ", status, 
                                  {printf("<<%p>> -> <<%p>> [%d]\n", (void*)A, (void*)(long)d_A, tile_size); return_code = -2; goto release_and_return_error;} );
        gpu_device->transferred_data_in += tile_size;
    }

    on_gpu = dplasma_data_is_on_gpu(gpu_device, &ddescA, DPLASMA_READ, m, k, &gpu_elem_B);
    d_B = gpu_elem_B->gpu_mem;
    gpu_elem_B->memory_elem->memory = B;
    gpu_device->required_data_in += tile_size;
    if( !on_gpu ) {
        /* Push B into the GPU */
        status = cuMemcpyHtoDAsync( d_B, B, tile_size, stream );
        DPLASMA_CUDA_CHECK_ERROR( "cuMemcpyHtoDAsync to device (d_B) ", status,
                                  {printf("<<%p>> -> <<%p>>\n", (void*)B, (void*)(long)d_B); return_code = -2; goto release_and_return_error;} );
        gpu_device->transferred_data_in += tile_size;
    }

    on_gpu = dplasma_data_is_on_gpu(gpu_device, &ddescA, DPLASMA_READ | DPLASMA_WRITE, m, n, &gpu_elem_C);
    d_C = gpu_elem_C->gpu_mem;
    gpu_elem_C->memory_elem->memory = C;
    gpu_device->required_data_in += tile_size;
    if( !on_gpu ) {
        /* Push C into the GPU */
        status = cuMemcpyHtoDAsync( d_C, C, tile_size, stream );
        DPLASMA_CUDA_CHECK_ERROR( "cuMemcpyHtoDAsync to device (d_C) ", status,
                                  {printf("<<%p>> -> <<%p>>\n", (void*)C, (void*)(long)d_C); return_code = -2; goto release_and_return_error;} );
        gpu_device->transferred_data_in += tile_size;
    }
#if defined(DPLASMA_PROFILING)
    dplasma_profiling_trace( gpu_device->profiling, movein_key_end, 0 );
#endif  /* defined(PROFILING) */

#if defined(DPLASMA_PROFILING)
    dplasma_profiling_trace( gpu_device->profiling, compute_key_start, 1 );
#endif  /* defined(PROFILING) */
    offset = 0;
    CU_PUSH_POINTER( gpu_device->hcuFunction, offset, d_B );
    CU_PUSH_INT(     gpu_device->hcuFunction, offset, ddescA.nb );
    CU_PUSH_POINTER( gpu_device->hcuFunction, offset, d_A );
    CU_PUSH_INT(     gpu_device->hcuFunction, offset, ddescA.nb );
    CU_PUSH_POINTER( gpu_device->hcuFunction, offset, d_C );
    CU_PUSH_INT(     gpu_device->hcuFunction, offset, ddescA.nb );
    CU_PUSH_INT(     gpu_device->hcuFunction, offset, ddescA.nb );
    CU_PUSH_FLOAT(   gpu_device->hcuFunction, offset, alpha );
    CU_PUSH_FLOAT(   gpu_device->hcuFunction, offset, beta );
    cuParamSetSize( gpu_device->hcuFunction, offset );

    /* cuLaunch: we kick off the CUDA */
    if( 1 == gpu_device->major ) {
        grid_width  = ddescA.nb / 64 + (ddescA.nb % 64 != 0);
        grid_height = ddescA.nb / 16 + (ddescA.nb % 16 != 0);
    } else {
        grid_width  = ddescA.nb / 64 + (ddescA.nb % 64 != 0);
        grid_height = ddescA.nb / 64 + (ddescA.nb % 64 != 0);
    }
    status = cuLaunchGridAsync( gpu_device->hcuFunction,
                                grid_width, grid_height, stream);

    DPLASMA_CUDA_CHECK_ERROR( "cuLaunchGridAsync ", status,
                              {return -1;} );

#if defined(DPLASMA_PROFILING)
    dplasma_profiling_trace( gpu_device->profiling, compute_key_end, 1 );
#endif  /* defined(PROFILING) */

    /* Pop C from the GPU */
    gpu_device->required_data_out += tile_size;
    if( (n == k+1) ) {
#if defined(DPLASMA_PROFILING)
        dplasma_profiling_trace( gpu_device->profiling, moveout_key_start, 2 );
#endif  /* defined(PROFILING) */
        /* Pop C from the GPU */
        status = cuMemcpyDtoHAsync( C, d_C, tile_size, stream );
        DPLASMA_CUDA_CHECK_ERROR( "cuMemcpyDtoHAsync from device (d_C) ", status,
                                  {printf("<<%p>> -> <<%p>>\n", (void*)(long)d_C, (void*)C); return_code = -2; goto release_and_return_error;} );
        gpu_device->transferred_data_out += tile_size;
#if defined(DPLASMA_PROFILING)
        dplasma_profiling_trace( gpu_device->profiling, moveout_key_end, 2 );
#endif  /* defined(PROFILING) */
    }

 release_and_return_error:
    return return_code;
}

/* Try to execute a GEMM on a GPU.
 *
 * Returns:
 *  0 - if the GEMM should be executed by some other meaning (in this case the
 *         execution context is not released).
 * -1 - if the GEMM is scheduled to be executed on a GPU.
 */
int gpu_sgemm( dplasma_execution_unit_t* eu_context,
               dplasma_execution_context_t* exec_context,
               int uplo, void* A, void* B, void* C, int k, int n, int m )
{
    int which_gpu, rc, stream_rc, waiting = 0, submit = 0;
    gpu_device_t* gpu_device;
    gc_data_t *gA, *gB, *gC;
    gpu_elem_t *gpu_elem_C;
    cudaError_t status;
    dplasma_execution_context_t* progress_array[DPLASMA_MAX_STREAMS];

    /* We always schedule the task on the GPU owning the C tile. */
    which_gpu = dplasma_data_tile_write_owner( &ddescA, m, n );
    if( which_gpu < 0 ) {  /* this is the first time we see this tile. Let's decide which GPU will work on it. */
        which_gpu = 0; /* TODO */
    }
    gpu_device = gpu_devices[which_gpu];
    
    /* Check the GPU status */
    rc = dplasma_atomic_inc_32b( &(gpu_device->mutex) );
    if( 1 != rc ) {  /* I'm not the only one messing with this GPU */
        DPLASMA_LIST_ITEM_SINGLETON( (dplasma_list_item_t*)exec_context );
        dplasma_dequeue_push_back( &(gpu_device->pending), (dplasma_list_item_t*)exec_context );
        return -1;
    }

    status = cuCtxPushCurrent(gpu_device->ctx);
    DPLASMA_CUDA_CHECK_ERROR( "cuCtxPushCurrent ", status,
                              {return -2;} );
    for( rc = 0; rc < DPLASMA_MAX_STREAMS; rc++ )
        progress_array[rc] = NULL;

 more_work_to_do:
    if( (NULL != exec_context) && (NULL == progress_array[submit]) ) {
        progress_array[submit] = exec_context;

        /* Push this task into the GPU */
        rc = gpu_sgemm_internal( gpu_device, eu_context, exec_context, gpu_device->streams[submit],
                                 uplo, A, B, C, k, n, m );
        if( 0 != rc ) {  /* something fishy happened. Reschedule the pending tasks on the cores */
            goto disable_gpu;
        }
        /*printf( "GPU submit %p (k = %d, m = %d, n = %d) [%d]\n", (void*)progress_array[submit], k, m, n, submit );*/
        submit = (submit + 1) % gpu_device->max_streams;
        exec_context = NULL;
    }

    if( NULL != progress_array[waiting] ) {
    wait_for_completion:
        stream_rc = cuStreamQuery(gpu_device->streams[waiting]);
        if( CUDA_ERROR_NOT_READY == stream_rc ) {
            goto fetch_more_work;
            /* Task not yet completed */
        } else if( CUDA_SUCCESS == stream_rc ) {  /* Done with this task */
            goto complete_previous_work;
        } else {
            DPLASMA_CUDA_CHECK_ERROR( "cuStreamQuery ", status,
                                      {return -2;} );
        }
    }

    goto more_work_to_do;

 complete_previous_work:
    /* Everything went fine so far, the result is correct and back in the main memory */
    /*printf( "GPU complete %p (k = %d, m = %d, n = %d) [%d]\n", (void*)progress_array[waiting], k, m, n, waiting );*/
    dplasma_complete_execution( eu_context, progress_array[waiting] );
    progress_array[waiting] = NULL;
    waiting = (waiting + 1) % gpu_device->max_streams;

    gpu_device->executed_tasks++;
    rc = dplasma_atomic_dec_32b( &(gpu_device->mutex) );
    if( 0 == rc ) {  /* I was the last one */
        status = cuCtxPopCurrent(NULL);
        DPLASMA_CUDA_CHECK_ERROR( "cuCtxPushCurrent ", status,
                                  {return -1;} );
        return -1;
    }

 fetch_more_work:
    /* Do we still have room in the progress_array? */
    if( NULL != progress_array[submit] )
        goto wait_for_completion;

    exec_context = (dplasma_execution_context_t*)dplasma_dequeue_pop_front( &(gpu_device->pending) );
    if( NULL == exec_context ) {  /* Collisions, save time and come back here later */
        goto more_work_to_do;
    }

    k = exec_context->locals[0].value;
    m = exec_context->locals[1].value;
    n = exec_context->locals[2].value;

    gC = exec_context->pointers[1];
    gA = exec_context->pointers[3];
    gB = exec_context->pointers[5];
    A = GC_DATA(gA);
    B = GC_DATA(gB);
    C = GC_DATA(gC);
    goto more_work_to_do;

    /* a device ... */
    do {
        exec_context = (dplasma_execution_context_t*)dplasma_dequeue_pop_front( &(gpu_device->pending) );
        if( NULL != exec_context ) {
 disable_gpu:
            __dplasma_schedule( eu_context, exec_context, 0 );
            rc = dplasma_atomic_dec_32b( &(gpu_device->mutex) );
        }
    } while( rc != 0 );
    status = cuCtxPopCurrent(NULL);
    DPLASMA_CUDA_CHECK_ERROR( "cuCtxPushCurrent ", status,
                              {} );
    return -2;
}

#if DPLASMA_SMART_SCHEDULING
gpu_device_t* get_best_gpu(int priority)
{
	/* catch gpu first */ 
	int gpu_id;
	gpu_device_t* gpu_device;
	for(gpu_id = 0; gpu_id < ndevices; gpu_id++){
		if( NULL != (gpu_device = (gpu_device_t*)dplasma_atomic_lifo_pop(&(gpu_array[gpu_id].gpu_devices)))){
			/* unleash seat -- concurreny !! to sgemm function now !*/
			gpu_array[gpu_id].working = 1;
			return gpu_device;
		}
	}
	/* there is no GPU available */
	int w;
	w = max_wait-1;
	for(;;){
		if(dplasma_atomic_cas(&(waiting[w]),0,1) == 1){
			w--;
			if(w < max_wait-1 && w >= 0){
				waiting[w+1] = 0;	
			}else if(w < 0){
				for(;;){
					for(gpu_id = 0; gpu_id < ndevices; gpu_id++){
						if( NULL != (gpu_device = (gpu_device_t*)dplasma_atomic_lifo_pop(&(gpu_array[gpu_id].gpu_devices)))){
							/* unleash seat -- concurreny !! to sgemm function now !*/
							gpu_array[gpu_id].working = 1;
							waiting[0] = 0;
							return gpu_device;
						}
					}       
				}
			}
		}else{
			return NULL;
		}
	}
	return NULL;
}
#endif

