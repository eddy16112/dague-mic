/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "data_dist/sparse-matrix/pastix_internal/pastix_internal.h"
#include "data_dist/sparse-matrix/sparse-matrix.h"
#include "cuda_sparse_sgemm.h"
#include "gpu_data.h"
#include "dague.h"
#include "execution_unit.h"
#include "scheduling.h"
#include "fifo.h"

#include <cblas.h>
#include <plasma.h>
#include <core_blas.h>

#include <stdio.h>
#include <cublas.h>
#include <dlfcn.h>

#include "data_distribution.h"


void core_spotrfsp1d_gemm(dague_int_t cblknum,
                          dague_int_t bloknum,
                          dague_int_t fcblknum,
                          float *L,
                          float *C,
                          float *work,
                          SolverMatrix *datacode);


#define DPLASMA_SCHEDULING 1
#define DPLASMA_ONLY_GPU 0
#define DAGUE_GPU_USE_PRIORITIES 1


#define DSPARSE_INDIVUAL_BLOCKTAB
static inline int dague_imax(int a, int b) { return (a >= b) ? a : b; };

/* #undef printf */
/* #define DEBUG(__params__) printf __params__ */

static volatile uint32_t cpu_counter = 0;
static int ndevices = 0;
#if DPLASMA_SCHEDULING
uint32_t *gpu_set;
int *gpu_load;
const uint32_t MAX_QUEUE = 55;
#endif

#warning "Don't forget to change int in dague_int_t"
typedef int my_tmp_int_t;

static int OHM_N = 5;
static int OHM_M = 3;

#define TRACE_WITH_REF(prof, key, eid, refdesc, refdescid) do {         \
        dague_profile_ddesc_info_t info;                                \
        info.desc = refdesc;                                            \
        info.id = refdescid;                                            \
        dague_profiling_trace(prof, key, eid, (void*)&info);            \
    } while(0)

static void compute_best_unit( uint64_t length, float* updated_value, char** best_unit );

static sparse_matrix_desc_t* UGLY_A;
static SolverMatrix*       datacode;

int sparse_sgemm_cuda_init( dague_context_t* dague_context, sparse_matrix_desc_t *sparseA )
{
    CUdevice hcuDevice;
    int i, j;
    char *env;
    size_t sparse_size;
    (void)dague_context;
#if !defined(DSPARSE_INDIVUAL_BLOCKTAB)
    my_tmp_int_t *blocktab;
    size_t       blocktab_size;
#endif

    UGLY_A = sparseA;
    datacode = &(UGLY_A->pastix_data->solvmatr);
    /* TODO : some test needed here ??? */

    /* /\** */
    /*  * Right now the sgemm function available with DPLASMA can only handle */
    /*  * square tiles with a size multiple of 64. */
    /*  *\/ */
    /* if( (sparseA->mb != sparseA->nb) || ((tileA->nb % 64) != 0) ) { */
    /*     printf("#\n# The CUDA GEMM version provided by DPLASMA is limitted to square sparseA\n" */
    /*            "# with a size multiple of 64.\n"); */
    /*     return -1; */
    /* } */

    /* TODO : what is that ??? */
    env = getenv("OHM_N");
    if( NULL != env )
        OHM_N = atoi(env);

    env = getenv("OHM_M");
    if( NULL != env )
        OHM_M = atoi(env);

    ndevices = dague_using_gpu();
#if DPLASMA_SCHEDULING
    gpu_set = (uint32_t*)calloc(400, sizeof(uint32_t));
    for( i = 0; i < 400 ; i++){
        gpu_set[i] = 0;
    }
    gpu_load = (int*)calloc(ndevices, sizeof(int));
    for( i = 0; i < ndevices;i++){
        gpu_load[i] = 0;
    }
#endif

    /*
     * Compute the size of each chunk on the GPU
     */
    sparse_size = SOLV_COEFMAX * sparse_matrix_size_of(sparseA->mtype);
#if defined(DSPARSE_INDIVUAL_BLOCKTAB)
    {
        int maxblockpercol, cblknum;
        int maxblocktabsize;

        maxblockpercol = 0;
        for ( cblknum = 0; cblknum < SYMB_CBLKNBR; cblknum++) {
            maxblockpercol = dague_imax( maxblockpercol,
                                        (SYMB_BLOKNUM(cblknum+1) -
                                         SYMB_BLOKNUM(cblknum)));
        }
        
        sparse_size = ((sparse_size +31)/32)*32;

        maxblocktabsize = maxblockpercol*sizeof(int);
        maxblocktabsize = ((maxblocktabsize+31)/32)*32;
        
        sparse_size += maxblocktabsize;
    }
#endif

    /*
     * Create the blocktab that will be transfered to the GPUs
     */
#if !defined(DSPARSE_INDIVUAL_BLOCKTAB)
    {
        my_tmp_int_t iterblock;

        blocktab_size = 2 * (SYMB_BLOKNBR) * sizeof(my_tmp_int_t);
        blocktab = (my_tmp_int_t*)malloc( blocktab_size );

        for (iterblock = 0; iterblock < SYMB_BLOKNBR; iterblock++)
        {
            blocktab[2*iterblock]   = SYMB_FROWNUM(iterblock);
            blocktab[2*iterblock+1] = SYMB_LROWNUM(iterblock);
        }

        sparseA->d_blocktab = (CUdeviceptr *)malloc(ndevices * sizeof(CUdeviceptr));
    }
#endif

    fprintf(stdout, "ndevices %d\n", ndevices);
    for( i = 0; i < ndevices; i++ ) {
        size_t thread_gpu_mem;
#if CUDA_VERSION < 3020
        unsigned int total_mem, free_mem;
#else
        size_t total_mem, free_mem;
#endif  /* CUDA_VERSION < 3020 */
        uint32_t nb_allocations = 0;
        gpu_device_t* gpu_device;
        CUresult status;
        int major, minor;
        char module_path[FILENAME_MAX];

        status = cuDeviceGet( &hcuDevice, i );
        DAGUE_CUDA_CHECK_ERROR( "cuDeviceGet ", status,
                                { ndevices = 0; return -1; } );

        status = cuDeviceComputeCapability( &major, &minor, hcuDevice);
        DAGUE_CUDA_CHECK_ERROR( "cuDeviceComputeCapability ", status,
                                { ndevices = 0; return -1; } );

        gpu_device = gpu_devices[i];
        status = cuCtxPushCurrent( gpu_device->ctx );
        DAGUE_CUDA_CHECK_ERROR( "(INIT) cuCtxPushCurrent ", status,
                                {   free(gpu_device);
                                    gpu_devices[i] = NULL;
                                    continue; } );

        /* If not disallowed by env, load from static linked kernels */
        /* This is non functional, as the ptr is not a CuFunction. */
        gpu_device->hcuFunction = NULL;
        env = getenv("DAGUE_CUBIN_NOSTATIC");
        fprintf(stdout, "env %s\n", env);
        if(!env || (('1' != env[0]) && ('y' != env[0])))
        {
            void* dlh;
            snprintf(module_path, FILENAME_MAX, "sparse_sgemm_kernel_N_T_64_16_4_16_4_SM%d%d",
                     gpu_device->major, gpu_device->minor);
            dlh = dlopen(NULL, RTLD_NOW);
            if(NULL == dlh)
                printf("Error parsing static libs: %s\n", dlerror());
            gpu_device->hcuFunction = dlsym(dlh, module_path);
            dlclose(dlh);
        }

        /* If not found statically, cuload it */
        if(NULL == gpu_device->hcuFunction)
        {
            fprintf(stdout, "function is NULL\n");
            env = getenv("DAGUE_CUBIN_PATH");
            snprintf(module_path, FILENAME_MAX, "%s/sparse_sgemm-sm_%1d%1d.cubin",
                     env?env:"../cores", gpu_device->major, gpu_device->minor);
            status = cuModuleLoad(&(gpu_device->hcuModule), module_path);
            fprintf(stdout, "status %d %s\n", status, module_path);
            DAGUE_CUDA_CHECK_ERROR( "(INIT) cuModuleLoad ", status,
                                    {
                                        fprintf(stderr,
                                                "*** unable to load `%s'\n",
                                                module_path);
                                        cuCtxDestroy( gpu_device->ctx );
                                        free(gpu_device);
                                        gpu_devices[i] = NULL;
                                        continue;
                                    } );
            snprintf(module_path, FILENAME_MAX,
                     "sparse_sgemm_kernel_N_T_64_16_4_16_4_SM%d%d", gpu_device->major,
                     gpu_device->minor);
            printf("CUDA MODULE %s\n", module_path);
            status = cuModuleGetFunction( &(gpu_device->hcuFunction),
                                          gpu_device->hcuModule, module_path );
            fprintf(stdout, "status %d %s\n", status, module_path);
            DAGUE_CUDA_CHECK_ERROR( "(INIT) cuModuleGetFunction ", status,
                                    {
                                        cuCtxDestroy( gpu_device->ctx );
                                        free(gpu_device);
                                        gpu_devices[i] = NULL;
                                        continue;
                                    } );
        }

        if(NULL == gpu_device->hcuFunction) return -1;

        if( 1 == gpu_device->major ) {
            cuFuncSetBlockShape( gpu_device->hcuFunction, 16, 4, 1 );
        } else {
            cuFuncSetBlockShape( gpu_device->hcuFunction, 16, 4, 1 );
        }

        /*
         * Transfer the blocktab before to allocate the chunks of memory
         */
#if !defined(DSPARSE_INDIVUAL_BLOCKTAB)
        {
            status = (cudaError_t)cuMemAlloc( &(sparseA->d_blocktab[i]),
                                              blocktab_size );
            
            DAGUE_CUDA_CHECK_ERROR( "cuMemAlloc ", status,
                                    ({
                                        fprintf(stderr, "Cannot Allocat blocktab on GPU\n");
                                        assert(-1);
                                        break;
                                    }) );
            
            status = (cudaError_t)cuMemcpyHtoD( sparseA->d_blocktab[i],
                                                blocktab,
                                                blocktab_size );

            DAGUE_CUDA_CHECK_ERROR( "cuMemcpyHtoD ", status,
                                    ({
                                        fprintf(stderr, "Cannot transfer the blocktab to GPU\n");
                                        assert(-1);
                                        break;
                                    }) );
        }
#endif        

        /**
         * Prepare the reusable memory on the GPU.
         */
        sparse_gpu_data_map_init( gpu_device, sparseA );

        /**
         * It appears that CUDA allocate the memory in chunks of 1MB,
         * so we need to adapt to this.
         */
        cuMemGetInfo( &free_mem, &total_mem );

        /* We allocate 9/10 of the total memory */
        thread_gpu_mem = (free_mem - free_mem / 10);

        fprintf(stdout, "mem %ld %ld %ld", free_mem, total_mem, thread_gpu_mem);
        
        while( free_mem > (total_mem - thread_gpu_mem) ) {
            gpu_elem_t* gpu_elem;
            cudaError_t cuda_status;

            if( nb_allocations > (uint32_t)((SYMB_CBLKNBR) >> 1) )
                break;
            gpu_elem = (gpu_elem_t*)malloc(sizeof(gpu_elem_t));
            dague_linked_list_item_construct( (dague_list_item_t*)gpu_elem );

            cuda_status = (cudaError_t)cuMemAlloc( &(gpu_elem->gpu_mem),
                                                   sparse_size);
            DAGUE_CUDA_CHECK_ERROR( "cuMemAlloc ", cuda_status,
                                    ({
#if CUDA_VERSION < 3020
                                        unsigned int _free_mem, _total_mem;
#else
                                        size_t _free_mem, _total_mem;
#endif  /* CUDA_VERSION < 3020 */
                                        cuMemGetInfo( &_free_mem, &_total_mem );
                                        printf("Per context: "
                                               "free mem %zu total mem %zu\n",
                                               _free_mem, _total_mem);
                                        free( gpu_elem );
                                        break;
                                    }) );
            nb_allocations++;
            gpu_elem->memory_elem = NULL;
            dague_linked_list_add_tail( gpu_device->gpu_mem_lru,
                                        (dague_list_item_t*)gpu_elem );
            cuMemGetInfo( &free_mem, &total_mem );
        }
        if( 0 == nb_allocations ) {
            printf("Rank %d Cannot allocate memory on GPU %d. Skip it!\n",
                   dague_context->my_rank, i);
            cuCtxDestroy( gpu_device->ctx );
            free(gpu_device);
            gpu_devices[i] = NULL;
            continue;
        }
        printf( "Allocate %u tiles on the GPU memory\n", nb_allocations );
#if !defined(DAGUE_GPU_STREAM_PER_TASK)
        /* Prepare the management arrays */
        gpu_device->max_in_tasks   = DAGUE_MAX_EVENTS_PER_STREAM;
        gpu_device->max_exec_tasks = DAGUE_MAX_EVENTS_PER_STREAM;
        gpu_device->max_out_tasks  = DAGUE_MAX_EVENTS_PER_STREAM;
        gpu_device->in_submit   = gpu_device->in_waiting   = 0;
        gpu_device->exec_submit = gpu_device->exec_waiting = 0;
        gpu_device->out_submit  = gpu_device->out_waiting  = 0;

        gpu_device->max_exec_streams = gpu_device->max_streams - 2;

        gpu_device->fifo_pending_in = (struct dague_fifo_t*)malloc( sizeof(struct dague_fifo_t) );
        dague_fifo_construct( gpu_device->fifo_pending_in );
        gpu_device->fifo_pending_exec = (struct dague_fifo_t*)malloc( sizeof(struct dague_fifo_t) );
        dague_fifo_construct( gpu_device->fifo_pending_exec );
        gpu_device->fifo_pending_out = (struct dague_fifo_t*)malloc( sizeof(struct dague_fifo_t) );
        dague_fifo_construct( gpu_device->fifo_pending_out );

        gpu_device->in_array = (struct dague_execution_context_t**)malloc(gpu_device->max_in_tasks * sizeof(struct dague_execution_context_t*));
        gpu_device->in_array_events = (CUevent*)malloc(gpu_device->max_in_tasks * sizeof(CUevent));
        for( j= 0; j < gpu_device->max_in_tasks; j++ ) {
            gpu_device->in_array[j] = NULL;
#if CUDA_VERSION >= 3020
            status = cuEventCreate(&(gpu_device->in_array_events[j]),
                                   CU_EVENT_DISABLE_TIMING);
#else
            status = cuEventCreate(&(gpu_device->in_array_events[j]),
                                   CU_EVENT_DEFAULT);
#endif  /* CUDA_VERSION >= 3020 */
            DAGUE_CUDA_CHECK_ERROR( "(INIT) cuEventCreate ",
                                    (cudaError_t)status,
                                    {continue;} );
        }
        gpu_device->exec_array = (struct dague_execution_context_t**)malloc(gpu_device->max_exec_tasks * sizeof(struct dague_execution_context_t*));
        gpu_device->exec_array_events = (CUevent*)malloc(gpu_device->max_exec_tasks * sizeof(CUevent));
        for( j= 0; j < gpu_device->max_exec_tasks; j++ ) {
            gpu_device->exec_array[j] = NULL;
#if CUDA_VERSION >= 3020
            status = cuEventCreate(&(gpu_device->exec_array_events[j]),
                                   CU_EVENT_DISABLE_TIMING);
#else
            status = cuEventCreate(&(gpu_device->exec_array_events[j]),
                                   CU_EVENT_DEFAULT);
#endif  /* CUDA_VERSION >= 3020 */
            DAGUE_CUDA_CHECK_ERROR( "(INIT) cuEventCreate ",
                                    (cudaError_t)status,
                                    {continue;} );
        }
        gpu_device->out_array = (struct dague_execution_context_t**)malloc(gpu_device->max_out_tasks * sizeof(struct dague_execution_context_t*));
        gpu_device->out_array_events = (CUevent*)malloc(gpu_device->max_out_tasks * sizeof(CUevent));
        for( j= 0; j < gpu_device->max_out_tasks; j++ ) {
            gpu_device->out_array[j] = NULL;
#if CUDA_VERSION >= 3020
            status = cuEventCreate(&(gpu_device->out_array_events[j]),
                                   CU_EVENT_DISABLE_TIMING);
#else
            status = cuEventCreate(&(gpu_device->out_array_events[j]),
                                   CU_EVENT_DEFAULT);
#endif  /* CUDA_VERSION >= 3020 */
            DAGUE_CUDA_CHECK_ERROR( "(INIT) cuEventCreate ",
                                    (cudaError_t)status,
                                    {continue;} );
        }
#endif  /* !defined(DAGUE_GPU_STREAM_PER_TASK) */
        status = cuCtxPopCurrent(NULL);
        DAGUE_CUDA_CHECK_ERROR( "(INIT) cuCtxPopCurrent ", status,
                                {free(gpu_device); return -1;} );
    }


#if !defined(DSPARSE_INDIVUAL_BLOCKTAB)
    free(blocktab);
#endif

    return 0;
}

int sparse_sgemm_cuda_fini(dague_context_t* dague_context)
{
    cudaError_t status;
    gpu_elem_t* gpu_elem;
    gpu_device_t* gpu_device;
    int total = 0, *gpu_counter, i, j, active_devices = 0;
    uint64_t *transferred_in, *transferred_out;
    uint64_t total_data_in = 0, total_data_out = 0;
    uint64_t *required_in, *required_out;
    float gtotal = 0.0, best_data_in, best_data_out;
    char *data_in_unit, *data_out_unit;
    (void)dague_context;

    if (ndevices <= 0)
        return 0;

    /* GPU counter for GEMM / each */
    gpu_counter     = (int*)calloc(ndevices, sizeof(int));
    transferred_in  = (uint64_t*)calloc(ndevices, sizeof(uint64_t));
    transferred_out = (uint64_t*)calloc(ndevices, sizeof(uint64_t));
    required_in     = (uint64_t*)calloc(ndevices, sizeof(uint64_t));
    required_out    = (uint64_t*)calloc(ndevices, sizeof(uint64_t));

    for(i = 0; i < ndevices; i++) {
        gpu_device = gpu_devices[i];

        if( NULL == gpu_device )
            continue;

        status = (cudaError_t)cuCtxPushCurrent( gpu_device->ctx );
        DAGUE_CUDA_CHECK_ERROR( "(FINI) cuCtxPushCurrent ", status,
                                {continue;} );
        status = (cudaError_t)cuCtxSynchronize();
        DAGUE_CUDA_CHECK_ERROR( "cuCtxSynchronize", status,
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
        while( NULL != (gpu_elem = (gpu_elem_t*)dague_linked_list_remove_head( gpu_device->gpu_mem_lru )) ) {
            cuMemFree( gpu_elem->gpu_mem );
            free( gpu_elem );
        }
        /**
         * Release all streams
         */
        for( j = 0; j < gpu_device->max_streams; j++ ) {
            cuStreamDestroy( gpu_device->streams[j] );
        }
#if !defined(DAGUE_GPU_STREAM_PER_TASK)
        /* Release all registered events */
        for( j= 0; j < gpu_device->max_in_tasks; j++ ) {
            assert( NULL == gpu_device->in_array[j] );
            status = (cudaError_t)cuEventDestroy(gpu_device->in_array_events[j]);
            DAGUE_CUDA_CHECK_ERROR( "(FINI) cuEventDestroy ", status,
                                    {continue;} );
        }
        free(gpu_device->in_array); gpu_device->in_array = NULL;
        free(gpu_device->in_array_events); gpu_device->in_array_events = NULL;
        for( j= 0; j < gpu_device->max_exec_tasks; j++ ) {
            assert( NULL == gpu_device->exec_array[j] );
            status = (cudaError_t)cuEventDestroy(gpu_device->exec_array_events[j]);
            DAGUE_CUDA_CHECK_ERROR( "(FINI) cuEventDestroy ", status,
                                    {continue;} );
        }
        free(gpu_device->exec_array); gpu_device->exec_array = NULL;
        free(gpu_device->exec_array_events); gpu_device->exec_array_events = NULL;
        for( j= 0; j < gpu_device->max_out_tasks; j++ ) {
            assert( NULL == gpu_device->out_array[j] );
            status = (cudaError_t)cuEventDestroy(gpu_device->out_array_events[j]);
            DAGUE_CUDA_CHECK_ERROR( "(FINI) cuEventDestroy ", status,
                                    {continue;} );
        }
        free(gpu_device->out_array); gpu_device->out_array = NULL;
        free(gpu_device->out_array_events); gpu_device->out_array_events = NULL;
        free( gpu_device->fifo_pending_in ); gpu_device->fifo_pending_in = NULL;
        free( gpu_device->fifo_pending_exec ); gpu_device->fifo_pending_exec = NULL;
        free( gpu_device->fifo_pending_out ); gpu_device->fifo_pending_out = NULL;
#endif  /* !defined(DAGUE_GPU_STREAM_PER_TASK) */

#if !defined(DSPARSE_INDIVUAL_BLOCKTAB)
        cuMemFree(UGLY_A->d_blocktab[i]);
#endif

        status = (cudaError_t)cuCtxDestroy( gpu_device->ctx );
        DAGUE_CUDA_CHECK_ERROR( "(FINI) cuCtxDestroy ", status,
                                {continue;} );
        free(gpu_device->gpu_mem_lru);
        free(gpu_device);
        active_devices++;
    }

#if !defined(DSPARSE_INDIVUAL_BLOCKTAB)
    free(UGLY_A->d_blocktab);
#endif

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
    gtotal = (float)total + (float)cpu_counter;
    printf("------------------------------------------------------------------------------\n");
    printf("|PU % 5d |  # GEMM   |    %%   |   Data In   |    %%   |   Data Out  |    %%   |\n", dague_context->my_rank);
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
    printf("|All CPUs |%10u | %6.2f |%10.2f%2s | %6.2f |%10.2f%2s | %6.2f |\n",
           cpu_counter, (cpu_counter / gtotal)*100.00,
           0.0, " ", 0.0, 0.0, " ", 0.0);
    printf("------------------------------------------------------------------------------\n");
    free(gpu_counter);
    free(transferred_in);
    free(transferred_out);
    free(required_in);
    free(required_out);

    return 0;
}

#define ALIGN_UP(OFFSET, ALIGN) \
    (OFFSET) = ((OFFSET) + (ALIGN) - 1) & ~((ALIGN) - 1)
#define CU_PUSH_POINTER( FUNCTION, OFFSET, PTR )                        \
    do {                                                                \
        void* __ptr = (void*)(size_t)(PTR);                             \
        ALIGN_UP((OFFSET), __alignof(void*));                           \
        cuParamSetv( (FUNCTION), (OFFSET), &__ptr, sizeof(void*));      \
        (OFFSET) += sizeof(void*);                                      \
    } while (0)
#define CU_PUSH_INT( FUNCTION, OFFSET, VALUE )                          \
    do {                                                                \
        ALIGN_UP((OFFSET), __alignof(int));                             \
        cuParamSeti( (FUNCTION), (OFFSET), (VALUE) );                   \
        (OFFSET) += sizeof(int);                                        \
    } while (0)
#define CU_PUSH_FLOAT( FUNCTION, OFFSET, VALUE )                        \
    do {                                                                \
        ALIGN_UP((OFFSET), __alignof(float));                           \
        cuParamSetf( (FUNCTION), (OFFSET), (VALUE) );                   \
        (OFFSET) += sizeof(float);                                      \
    } while (0)

#define ddescA(ec) (UGLY_A)
#define ddescB(ec) ddescA(ec)
#define ddescC(ec) ddescA(ec)
#define CBLK_SIZE(cblk)                                                 \
    (SOLV_STRIDE(cblk)*(SYMB_LCOLNUM(cblk)-SYMB_FCOLNUM(cblk)+1))

#define TYPE_SIZE(this_task)                                    \
    (sparse_matrix_size_of(ddescA(this_task)->mtype))
/**
 *  This function schedule the move of all the data required for a
 *  specific task from the main memory into the GPU memory.
 *
 *  Returns: negative number if any error occured.
 *           positive: the number of data to be moved.
 */
static inline int
gpu_sgemm_internal_push( gpu_device_t* gpu_device,
                         dague_execution_context_t* this_task,
                         CUstream stream )
{
    gpu_elem_t *gpu_elem_A = NULL, *gpu_elem_C = NULL;
    dague_arena_chunk_t *aA, *aC;
    int cblk_size, return_code = 0, on_gpu, how_many = 0;
    CUdeviceptr d_A, d_C;
    cudaError_t status;
    void *A, *C;
    int bloknum, fcblknum, cblknum, phony, prev, next;

    bloknum  = this_task->locals[0].value;
    fcblknum = this_task->locals[1].value;
    cblknum  = this_task->locals[2].value;
    phony    = this_task->locals[3].value;
    prev     = this_task->locals[4].value;
    next     = this_task->locals[5].value;

    aA = this_task->data[0].data;
    aC = this_task->data[1].data;
    A = ADATA(aA);
    C = ADATA(aC);

#if defined(DAGUE_PROF_TRACE)
    if( dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_DATA_IN )
        dague_profiling_trace( gpu_device->profiling,
                               dague_cuda_movein_key_start,
                               (unsigned long)this_task, NULL );
#endif  /* defined(DAGUE_PROF_TRACE) */

    DEBUG(("Request Data of cblk %d on GPU\n", cblknum));
    cblk_size = CBLK_SIZE(cblknum) * TYPE_SIZE(this_task);
    on_gpu = sparse_gpu_data_is_on_gpu(gpu_device, ddescA(this_task), DAGUE_READ,
                                cblknum, &gpu_elem_A);
    gpu_elem_A->memory_elem->memory = A;
    d_A = gpu_elem_A->gpu_mem;
    gpu_device->required_data_in += cblk_size;
    if( !on_gpu ) {
        /* Push A into the GPU */
        status = (cudaError_t)cuMemcpyHtoDAsync( d_A, A, cblk_size, stream );
        DAGUE_CUDA_CHECK_ERROR( "cuMemcpyHtoDAsync to device (d_A) ", status,
                                {   printf("<<%p>> -> <<%p>> [%d]\n",
                                        (void*)A, (void*)(long)d_A, cblk_size);
                                    return_code = -2;
                                    assert(0);
                                    goto release_and_return_error;} );

        status = cuStreamSynchronize( stream );
        DAGUE_CUDA_CHECK_ERROR( "cuStreamSynchronize", status,
                                {   fprintf(stderr, "Tout a peter envoi A\n");
                                    assert(0); } );

        gpu_device->transferred_data_in += cblk_size;
        how_many++;
    }
    this_task->data[0].gpu_data = (struct gpu_elem_t *)gpu_elem_A;

    DEBUG(("Request Data of fcblk %d on GPU\n", fcblknum));
    cblk_size = CBLK_SIZE(fcblknum) * TYPE_SIZE(this_task);
    on_gpu = sparse_gpu_data_is_on_gpu(gpu_device, ddescC(this_task),
                                DAGUE_READ | DAGUE_WRITE, fcblknum, &gpu_elem_C);
    d_C = gpu_elem_C->gpu_mem;
    gpu_elem_C->memory_elem->memory = C;
    gpu_device->required_data_in += cblk_size;
    if( !on_gpu ) {
        /* Push C into the GPU */
        status = (cudaError_t)cuMemcpyHtoDAsync( d_C, C, cblk_size, stream );
        DAGUE_CUDA_CHECK_ERROR( "cuMemcpyHtoDAsync to device (d_C) ", status,
                                {   printf("<<%p>> -> <<%p>>\n", (void*)C,
                                           (void*)(long)d_C);
                                    return_code = -3;
                                    assert(0);
                                    goto release_and_return_error; } );

        status = cuStreamSynchronize( stream );
        DAGUE_CUDA_CHECK_ERROR( "cuStreamSynchronize", status,
                                {   fprintf(stderr, "Tout a peter envoi C\n");
                                    assert(0); } );

        gpu_device->transferred_data_in += cblk_size;
        how_many++;
    }
    this_task->data[1].gpu_data = (struct gpu_elem_t *)gpu_elem_C;
 release_and_return_error:
    return (return_code < 0 ? return_code : how_many);
}

static inline int
gpu_sgemm_internal_submit( gpu_device_t* gpu_device,
                           dague_execution_context_t* this_task,
                           CUstream stream )
{
    gpu_elem_t *gpu_elem_A = NULL, *gpu_elem_C = NULL;
    CUdeviceptr d_A, d_C, d_blok, d_blok_idx, d_facing_blok_idx;
    cudaError_t status;
    int grid_width, grid_height;
    int m, n, k;
    float alpha = -1.0, beta = 1.0;
    int offset;
    int bloknum, fcblknum, cblknum, phony, prev, next, bloknbr;

    bloknum  = this_task->locals[0].value;
    fcblknum = this_task->locals[1].value;
    cblknum  = this_task->locals[2].value;
    phony    = this_task->locals[3].value;
    prev     = this_task->locals[4].value;
    next     = this_task->locals[5].value;

    gpu_elem_A = (gpu_elem_t *)this_task->data[0].gpu_data;
    gpu_elem_C = (gpu_elem_t *)this_task->data[1].gpu_data;
    d_A = gpu_elem_A->gpu_mem;
    d_C = gpu_elem_C->gpu_mem;


#if defined(DSPARSE_INDIVUAL_BLOCKTAB)
    {
        int bloknbr, j, b, return_code, how_many = 0;
        int * blok_idx;
        int * facing_blok_idx;
        size_t stride_size;

        bloknbr = SYMB_BLOKNUM(cblknum+1) - bloknum;
        stride_size = 2*bloknbr*sizeof(int);
        b = SYMB_BLOKNUM( fcblknum );
        blok_idx = (int*)malloc(stride_size);
        facing_blok_idx = &(blok_idx[bloknbr]);

        for (j=bloknum; j<SYMB_BLOKNUM(cblknum + 1); j++) {
            /* Find facing bloknum */
            while (
#ifdef NAPA_SOPALIN /* ILU(k) */
                   !(((SYMB_FROWNUM(j)>=SYMB_FROWNUM(b)) &&
                      (SYMB_LROWNUM(j)<=SYMB_LROWNUM(b))) ||
                     ((SYMB_FROWNUM(j)<=SYMB_FROWNUM(b)) &&
                      (SYMB_LROWNUM(j)>=SYMB_LROWNUM(b))) ||
                     ((SYMB_FROWNUM(j)<=SYMB_FROWNUM(b)) &&
                      (SYMB_LROWNUM(j)>=SYMB_FROWNUM(b))) ||
                     ((SYMB_FROWNUM(j)<=SYMB_LROWNUM(b)) &&
                      (SYMB_LROWNUM(j)>=SYMB_LROWNUM(b))))
#else
                   !((SYMB_FROWNUM(j)>=SYMB_FROWNUM(b)) &&
                     (SYMB_LROWNUM(j)<=SYMB_LROWNUM(b)))
#endif
                   )
                {
                    b++;
                    assert( b < SYMB_BLOKNUM( fcblknum+1 ) );
                }
            blok_idx[j-bloknum]        = SOLV_COEFIND(j) - SOLV_COEFIND(bloknum);
            facing_blok_idx[j-bloknum] = SOLV_COEFIND(b) + SYMB_FROWNUM(j) - SYMB_FROWNUM(b) +
                SOLV_STRIDE(fcblknum)*(SYMB_FROWNUM(bloknum) - SYMB_FCOLNUM(fcblknum));
        }


        int displacement = SOLV_COEFMAX * TYPE_SIZE(this_task);
        displacement = ((displacement+31)/32)*32;

        d_blok_idx = d_C  + displacement;
       
        status = (cudaError_t)cuMemcpyHtoD( d_blok_idx,
                                            blok_idx,
                                            stride_size);

        DAGUE_CUDA_CHECK_ERROR( "cuMemcpyHtoD to device (d_stride) ",
                                status,
                                {   fprintf(stderr, "<<%p>> -> <<%p>> [%d]\n",
                                            (void*)blok_idx,
                                            (void*)(long)d_blok_idx, (int)stride_size);
                                    return_code = -2;
                                    /*exit(666);*/
                                    assert(666==0);
                                    return (return_code < 0 ? return_code : how_many);} );
        gpu_device->transferred_data_in += stride_size;
        how_many++;
        
        free(blok_idx);

    }
#endif

    DEBUG(("Request GPU runs SPARSE_GEMM(%d, %d, %d)\n",
           bloknum, cblknum, fcblknum));

#if defined(DAGUE_PROF_TRACE)
    if( dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_EXEC ) {
        dague_ddesc_t *ddesca = (dague_ddesc_t *)ddescA(this_task);
        int data_id =
            ddesca->data_key(ddesca, this_task->locals[2].value,
                             this_task->locals[0].value);
        uint64_t task_id =
            this_task->function->key( this_task->dague_object,
                                      this_task->locals );
        TRACE_WITH_REF(gpu_device->profiling,
                       DAGUE_PROF_FUNC_KEY_START(this_task->dague_object,
                                                 this_task->function->function_id),
                       task_id, ddesca, data_id);
    }
#endif  /* defined(DAGUE_PROF_TRACE) */


    float *myA    = (float*)malloc(SOLV_COEFMAX * sizeof(float) );
    float *myCcpu = (float*)malloc(SOLV_COEFMAX * sizeof(float) );
    float *myCgpu = (float*)malloc(SOLV_COEFMAX * sizeof(float) );

    cudaThreadSynchronize();
    cudaMemcpy( myA,    (void*)d_A, SOLV_COEFMAX * sizeof(float), cudaMemcpyDeviceToHost );
    cudaMemcpy( myCcpu, (void*)d_C, SOLV_COEFMAX * sizeof(float), cudaMemcpyDeviceToHost );

    core_spotrfsp1d_gemm( cblknum, bloknum, fcblknum, myA, myCcpu, myCgpu, datacode );

    
    offset = 0;
    bloknbr = SYMB_BLOKNUM(cblknum+1) - bloknum;
    d_blok  = d_A + SOLV_COEFIND(bloknum)*TYPE_SIZE(this_task);
    m = SOLV_STRIDE(cblknum)  - SOLV_COEFIND(bloknum);
    n = SYMB_LROWNUM(bloknum) - SYMB_FROWNUM(bloknum) + 1;
    k = SYMB_LCOLNUM(cblknum) - SYMB_FCOLNUM(cblknum) + 1;

#if !defined(DSPARSE_INDIVUAL_BLOCKTAB)
    d_C = d_C + ((SYMB_FROWNUM(bloknum) - SYMB_FCOLNUM(fcblknum))*SOLV_STRIDE(fcblknum)) *TYPE_SIZE(this_task);
#endif

    CU_PUSH_POINTER( gpu_device->hcuFunction, offset, d_C );
    CU_PUSH_POINTER( gpu_device->hcuFunction, offset, d_blok);
    CU_PUSH_POINTER( gpu_device->hcuFunction, offset, d_blok);
    CU_PUSH_INT(     gpu_device->hcuFunction, offset, m);
    CU_PUSH_INT(     gpu_device->hcuFunction, offset, n);
    CU_PUSH_INT(     gpu_device->hcuFunction, offset, k);
    CU_PUSH_INT(     gpu_device->hcuFunction, offset, SOLV_STRIDE(cblknum) );
    CU_PUSH_INT(     gpu_device->hcuFunction, offset, SOLV_STRIDE(cblknum) );
    CU_PUSH_INT(     gpu_device->hcuFunction, offset, SOLV_STRIDE(fcblknum));
    CU_PUSH_FLOAT(   gpu_device->hcuFunction, offset, alpha );
    CU_PUSH_FLOAT(   gpu_device->hcuFunction, offset, beta );
    CU_PUSH_INT(     gpu_device->hcuFunction, offset, bloknbr);
#if defined(DSPARSE_INDIVUAL_BLOCKTAB)
    d_facing_blok_idx = d_blok_idx + bloknbr*sizeof(int);
    CU_PUSH_POINTER( gpu_device->hcuFunction, offset, d_blok_idx);
    CU_PUSH_POINTER( gpu_device->hcuFunction, offset, d_facing_blok_idx);
#else
    {
        my_tmp_int_t fblcknbr = SYMB_BLOKNUM(fcblknum+1) - SYMB_BLOKNUM(fcblknum);
        CUdeviceptr d_blocktab, d_fbloktab;
        
        d_blocktab = UGLY_A->d_blocktab[gpu_device->id] + 2 * bloknum                * sizeof(my_tmp_int_t);
        d_fbloktab = UGLY_A->d_blocktab[gpu_device->id] + 2 * SYMB_BLOKNUM(fcblknum) * sizeof(my_tmp_int_t);

        CU_PUSH_POINTER( gpu_device->hcuFunction, offset, d_blocktab );
        CU_PUSH_INT(     gpu_device->hcuFunction, offset, fblcknbr   );
        CU_PUSH_POINTER( gpu_device->hcuFunction, offset, d_fbloktab );
    }
#endif

    cuParamSetSize(  gpu_device->hcuFunction, offset );

    /* cuLaunch: we kick off the CUDA */
    /* if( 1 == gpu_device->major ) { */
    grid_width  = m / 64 + (m % 64 != 0);
    grid_height = n / 16 + (n % 16 != 0);
/*     } else { */
/*         /\* Change bx and by to match the values in the fermi gemm code *\/ */
/* #define bx 4 */
/* #define by 4 */
/*         /\* grid_width  = ddescA(this_task)->nb / (16*bx) + (ddescA(this_task)->nb % (16*bx) != 0); *\/ */
/*         /\* grid_height = ddescA(this_task)->nb / (16*by) + (ddescA(this_task)->nb % (16*by) != 0); *\/ */
/*         fprintf(stdout, "TODO\n"); */
/*         assert(0); */
/*     } */

    status = (cudaError_t)cuLaunchGridAsync( gpu_device->hcuFunction,
                                             grid_width, grid_height, stream);

    status = cuStreamSynchronize( stream );
    DAGUE_CUDA_CHECK_ERROR( "cuStreamSynchronize", status,
                            {   fprintf(stderr, "Tout a peter kernel\n");
                                assert(0); } );


    cudaMemcpy( myCgpu, (void*)d_C, SOLV_COEFMAX * sizeof(float), cudaMemcpyDeviceToHost );
    float one = -1.0;
    cblas_saxpy( SOLV_STRIDE(fcblknum) * (SYMB_LCOLNUM(fcblknum) - SYMB_FCOLNUM( fcblknum ) + 1 ),
                 one, myCcpu, 1, myCgpu, 1);
    
    CORE_slange( PlasmaMaxNorm, SOLV_STRIDE(fcblknum), (SYMB_LCOLNUM(fcblknum) - SYMB_FCOLNUM( fcblknum ) + 1 ), 
                 myCgpu, SOLV_STRIDE(fcblknum), NULL, &one );

    
    if ( one != 0.0 ) {
        fprintf(stderr, "WARNING: Norm for bloknum=%d is %e\n", bloknum, one);
    }

    DAGUE_CUDA_CHECK_ERROR( "cuLaunchGridAsync ", status,
                              {return -1;} );

    return 0;
}

/**
 *  This function schedule the move of all the modified data for a
 *  specific task from the GPU memory into the main memory.
 *
 *  Returns: negative number if any error occured.
 *           positive: the number of data to be moved.
 */
static inline int
gpu_sgemm_internal_pop( gpu_device_t* gpu_device,
                        dague_execution_context_t* this_task,
                        CUstream stream )
{
    dague_arena_chunk_t *aC;
    gpu_elem_t *gpu_elem_C = NULL;
    int return_code = 0, cblk_size, how_many = 0;
    cudaError_t status;
    CUdeviceptr d_C;
    void* C;
    int bloknum, fcblknum, cblknum, phony, prev, next;

    bloknum  = this_task->locals[0].value;
    fcblknum = this_task->locals[1].value;
    cblknum  = this_task->locals[2].value;
    phony    = this_task->locals[3].value;
    prev     = this_task->locals[4].value;
    next     = this_task->locals[5].value;

    gpu_elem_C = (gpu_elem_t *)this_task->data[1].gpu_data;
    aC  = this_task->data[1].data;
    d_C = gpu_elem_C->gpu_mem;
    C = ADATA(aC);

    cblk_size = CBLK_SIZE(fcblknum) * TYPE_SIZE(this_task);

    /* Pop C from the GPU */
    gpu_device->required_data_out += cblk_size;
    if( (next == 0) ) {
        DEBUG(("Request out of GPU for C(%d)\n", fcblknum));
#if defined(DAGUE_PROF_TRACE)
        if( dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_DATA_OUT )
            dague_profiling_trace( gpu_device->profiling,
                                   dague_cuda_moveout_key_start,
                                   (unsigned long)this_task, NULL );
#endif  /* defined(DAGUE_PROF_TRACE) */
        /* Pop C from the GPU */
        status = (cudaError_t)cuMemcpyDtoHAsync( C, d_C, cblk_size, stream );
        DAGUE_CUDA_CHECK_ERROR( "cuMemcpyDtoHAsync from device (d_C) ", status,
                                {   printf("<<%p>> -> <<%p>>\n",
                                           (void*)(long)d_C, (void*)C);
                                    return_code = -2;
                                    goto release_and_return_error;} );

    status = cuStreamSynchronize( stream );
    DAGUE_CUDA_CHECK_ERROR( "cuStreamSynchronize", status,
                            {   fprintf(stderr, "Tout a peter retour C\n");
                                assert(0); } );

        gpu_device->transferred_data_out += cblk_size;
        how_many++;
    }
 release_and_return_error:
    return (return_code < 0 ? return_code : how_many);
}

/* Try to execute a GEMM on a GPU.
 *
 * Returns:
 *  0 - if the GEMM should be executed by some other meaning (in this case the
 *         execution context is not released).
 * -1 - if the GEMM is scheduled to be executed on a GPU.
 */

#if !defined(DAGUE_GPU_STREAM_PER_TASK)

#if DAGUE_GPU_USE_PRIORITIES
static inline dague_list_item_t*
dague_fifo_push_ordered( dague_fifo_t* fifo,
                         dague_list_item_t* elem )
{
    dague_execution_context_t* ec;
    dague_execution_context_t* input = (dague_execution_context_t*)elem;
    dague_list_item_t* current = (dague_list_item_t*)fifo->fifo_ghost.list_next;

    if( 0 == input->priority ) {
        while( current != &(fifo->fifo_ghost) ) {
            ec = (dague_execution_context_t*)current;
            if( ec->priority < input->priority )
                break;
            current = (dague_list_item_t *)current->list_next;
        }
    } else {
        current = &(fifo->fifo_ghost);
    }
    /* Add the input element before the current one */
    elem->list_prev = current->list_prev;
    elem->list_next = current;
    elem->list_prev->list_next = elem;
    elem->list_next->list_prev = elem;
    return elem;
}
#define DAGUE_FIFO_PUSH  dague_fifo_push_ordered
#else
#define DAGUE_FIFO_PUSH  dague_fifo_push
#endif

/**
 * This version is based on 4 streams: one for transfers from the memory to
 * the GPU, 2 for kernel executions and one for tranfers from the GPU into
 * the main memory. The synchronization on each stream is based on CUDA events,
 * such an event indicate that a specific epoch of the lifetime of a task has
 * been completed. Each type of stream (in, exec and out) has a pending FIFO,
 * where tasks ready to jump to the respective step are waiting.
 */
int sparse_gpu_sgemm( dague_execution_unit_t* eu_context,
               dague_execution_context_t* this_task,
               int uplo )
{
    int which_gpu, rc, exec_stream = 0;
    gpu_device_t* gpu_device;
    CUcontext saved_ctx;
    cudaError_t status;
    int fcblk;

    fcblk = this_task->locals[1].value;

    (void)uplo;
    //DEBUG(("GEMM( k = %d, m = %d, n = %d )\n", k, m, n));
    /* We always schedule the task on the GPU owning the C cblk. */
    which_gpu = sparse_gpu_data_cblk_write_owner( ddescA(this_task), fcblk);
    /*    printf("k=%d, m=%d, n=%d\n",k,m,n);*/
    if( which_gpu < 0 ) {
        /* this is the first time we see this tile.
           Let's decide which GPU will work on it. */
        which_gpu = 0; /* TODO */
#if DPLASMA_SCHEDULING
        if(ndevices > 1){
            /* reverse odd-even */
            /* homogeneous GPU */
            {
                if(fcblk % 2 == 0){
                    which_gpu = gpu_set[fcblk%400] % ndevices;
                }
                else{
                    which_gpu = ndevices - (gpu_set[fcblk%400] % ndevices + 1);
                }
            }

            /* heterogenous GPU */
            /* weight by percentage of getting n of (n)
               with performance factor */
            {


            }

            dague_atomic_inc_32b( &(gpu_set[fcblk%400]) );
        }
#if DPLASMA_ONLY_GPU
#else
        /*
        **Rectangular Mesh **
        * 1. Fact, a number of tile ahd GEMMs comes from Matrix size
        *  and tile size.
        *  We may have to change m,n in every tile size/ matrix size
        * 2. m and n is assign the size of squares which're going to mark over
        *  the triangular bunch of GEMMs
        * 3. m % (?) == (?) and n % (?) == (?) marks which tile is gonna be
        *  executed on CPU
        * 4. all (?) values affect "square size" and "position" -- which affects
        *  how many GEMMs will be executed on CPU
        * 5. Once we superpose/pile up "many square(m,n) -- like a mesh" on to
        *  triangular GEMMs, we will be able to caluculate
        *  how many GEMMs will be on CPU, also know which tiles
        * 6. The number GEMMs on GPU and CPU would meet
        *  "how many times GPU faster than CPU "
        *   * I usually use m % 3 == 0 && n % 2 == 0 on C1060 (3x2 square)
        *   * I usaully use m % 4 == 0 && n % 2 == 0 on C2050 (4x2 square)
        *   chance is lower that 1:6 or 1:8 becasue we pile up this square on to
        *   triangular
        *   * Why this method ?
        *    - try to finish "each bunch of GEMMs" as soon as poosible
        *      with GPU+CPU
        *    - plus "balancing" between CPU/GPU
        **/

        /* if( ((m % OHM_M) == 0) && ( (n % OHM_N) == 0) ){ */
        /*     dague_atomic_inc_32b( &(cpu_counter) ); */
        /*     return -99; */
        /* } */
#endif

#endif
    }
    gpu_device = gpu_devices[which_gpu];

#if DPLASMA_SCHEDULING
    //#warning "I don't know what is that - XL"
    /* keep n -- not being used yet*/
    gpu_load[gpu_device->id] += fcblk;
#endif

    /* Check the GPU status */
    rc = dague_atomic_inc_32b( &(gpu_device->mutex) );
    if( 1 != rc ) {  /* I'm not the only one messing with this GPU */
        DAGUE_LIST_ITEM_SINGLETON( (dague_list_item_t*)this_task );
        dague_dequeue_push_back( &(gpu_device->pending),
                                 (dague_list_item_t*)this_task );
        return -1;
    }

    /**
     * There might be a small race condition here,
     * between the moment when the previous
     * owner of the GPU context release it, and the moment where I can get it.
     */
    do {
        saved_ctx = gpu_device->ctx;
        dague_atomic_cas( &(gpu_device->ctx), saved_ctx, NULL );
    } while( NULL == saved_ctx );

#if defined(DAGUE_PROF_TRACE)
    if( dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_OWN )
        dague_profiling_trace( eu_context->eu_profile,
                               dague_cuda_own_GPU_key_start,
                               (unsigned long)eu_context, NULL );
#endif  /* defined(DAGUE_PROF_TRACE) */

    status = (cudaError_t)cuCtxPushCurrent(saved_ctx);
    DAGUE_CUDA_CHECK_ERROR( "cuCtxPushCurrent ", status,
                            {return -2;} );

    DEBUG(( "Add sparse_gemm(blok = %d, cblk = %d, fcblk = %d) priority %d\n",
            this_task->locals[0].value,
            this_task->locals[2].value,
            this_task->locals[1].value,
            this_task->priority ));
 check_in_deps:
    if( NULL != this_task ) {
        if( NULL != gpu_device->in_array[gpu_device->in_submit] ) {
            /* No more room on the event list. Store the execution context */
            DAGUE_FIFO_PUSH(gpu_device->fifo_pending_in,
                            (dague_list_item_t*)this_task);
            this_task = NULL;
        } else {
            /* Get the oldest task */
            if( !dague_fifo_is_empty(gpu_device->fifo_pending_in) ) {
                DAGUE_FIFO_PUSH(gpu_device->fifo_pending_in,
                                (dague_list_item_t*)this_task);
                this_task = (dague_execution_context_t*)dague_fifo_pop(gpu_device->fifo_pending_in);
            }
        }
    } else {
        if( NULL == gpu_device->in_array[gpu_device->in_submit] ) {
            this_task = (dague_execution_context_t*)dague_fifo_pop(gpu_device->fifo_pending_in);
        }
    }
    if( NULL != this_task ) {
        assert( NULL == gpu_device->in_array[gpu_device->in_submit] );
        rc = gpu_sgemm_internal_push( gpu_device, this_task,
                                      gpu_device->streams[0] );
        /**
         * Do not skip the cuda event generation.
         * The problem is that some of the inputs
         * might be in the pipe of being transferred to the GPU.
         * If we activate this task too early,
         * it might get executed before the data is available on the GPU.
         * Obviously, this lead to bad results.
         */
        /*if( 0 == rc ) goto exec_task;*/  /* No data to be moved for this task */
        gpu_device->in_array[gpu_device->in_submit] = this_task;
        DEBUG(("GPU Request number %d/%d\n",
               (int)(gpu_device->in_array_events[gpu_device->in_submit]),
                     (int)(gpu_device->streams[0])) );
        this_task = NULL;
        if( 0 > rc ) { assert(0); goto disable_gpu; }
        rc = cuEventRecord( gpu_device->in_array_events[gpu_device->in_submit],
                            gpu_device->streams[0] );
        gpu_device->in_submit = (gpu_device->in_submit + 1) % gpu_device->max_in_tasks;
    }
    assert( NULL == this_task );
    if( NULL != gpu_device->in_array[gpu_device->in_waiting] ) {
        rc = cuEventQuery(gpu_device->in_array_events[gpu_device->in_waiting]);
        if( CUDA_ERROR_NOT_READY == rc ) {
            goto check_exec_completion;
        } else if( CUDA_SUCCESS == rc ) {
            /* Save the task for the next step */
            DEBUG(("Completion of GPU Request number %d\n",
                   gpu_device->in_array_events[gpu_device->in_waiting]));
            this_task = gpu_device->in_array[gpu_device->in_waiting];
#if defined(DAGUE_PROF_TRACE)
            if( dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_DATA_IN )
                dague_profiling_trace( gpu_device->profiling,
                                       dague_cuda_movein_key_end,
                                       (unsigned long)this_task, NULL );
#endif  /* defined(DAGUE_PROF_TRACE) */
            gpu_device->in_array[gpu_device->in_waiting] = NULL;
            gpu_device->in_waiting = (gpu_device->in_waiting + 1) % gpu_device->max_in_tasks;
            goto exec_task;
        } else {
            DAGUE_CUDA_CHECK_ERROR( "cuEventQuery ", rc,
                                    {assert(0); goto disable_gpu;} );
        }
    }
 exec_task:
    if( NULL != this_task ) {
        if( NULL != gpu_device->exec_array[gpu_device->exec_submit] ) {
            /* No more room on the event list. Store the execution context */
            DAGUE_FIFO_PUSH(gpu_device->fifo_pending_exec,
                            (dague_list_item_t*)this_task);
            this_task = NULL;
        } else {
            /* Get the oldest task */
            if( !dague_fifo_is_empty(gpu_device->fifo_pending_exec) ) {
                DAGUE_FIFO_PUSH(gpu_device->fifo_pending_exec,
                                (dague_list_item_t*)this_task);
                this_task = (dague_execution_context_t*)dague_fifo_pop(gpu_device->fifo_pending_exec);
            }
        }
    } else {
        if( NULL == gpu_device->exec_array[gpu_device->exec_submit] ) {
            this_task = (dague_execution_context_t*)dague_fifo_pop(gpu_device->fifo_pending_exec);
        }
    }
    if( NULL != this_task ) {
        assert( NULL == gpu_device->exec_array[gpu_device->exec_submit] );
        /* Choose an exec_stream */
        exec_stream = (exec_stream + 1) % (gpu_device->max_exec_streams);
        DEBUG(( "Execute sparse gemm"
                "(blok = %d, cblk = %d, fcblk = %d) priority %d\n",
                this_task->locals[0].value, 
                this_task->locals[2].value,
                this_task->locals[1].value,
                this_task->priority ));
        rc = gpu_sgemm_internal_submit( gpu_device, this_task,
                                        gpu_device->streams[2 + exec_stream] );
        DEBUG(("GPU Request number %d/%d\n",
               gpu_device->exec_array_events[gpu_device->exec_submit],
               gpu_device->streams[2 + exec_stream]));
        gpu_device->exec_array[gpu_device->exec_submit] = this_task;
        this_task = NULL;
        if( 0 != rc ) { assert(0); goto disable_gpu; }
        rc = cuEventRecord( gpu_device->exec_array_events[gpu_device->exec_submit],
                            gpu_device->streams[2 + exec_stream] );
        gpu_device->exec_submit = (gpu_device->exec_submit + 1) % gpu_device->max_exec_tasks;
    }
 check_exec_completion:
    assert( NULL == this_task );
    if( NULL != gpu_device->exec_array[gpu_device->exec_waiting] ) {
        rc = cuEventQuery(gpu_device->exec_array_events[gpu_device->exec_waiting]);
        if( CUDA_ERROR_NOT_READY == rc ) {
            goto check_out_deps;
        } else if( CUDA_SUCCESS == rc ) {
            /* Save the task for the next step */
            DEBUG(("Completion of GPU Request number %d\n",
                   gpu_device->exec_array_events[gpu_device->exec_waiting]));
            this_task = gpu_device->exec_array[gpu_device->exec_waiting];
#if defined(DAGUE_PROF_TRACE)
            if( dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_EXEC ) {
                dague_ddesc_t *ddesca = (dague_ddesc_t *)ddescA(this_task);
                int data_id =
                    ddesca->data_key(ddesca,
                                     this_task->locals[2].value,
                                     this_task->locals[0].value);
                uint64_t task_id =
                    this_task->function->key( this_task->dague_object,
                                              this_task->locals );
                TRACE_WITH_REF(gpu_device->profiling,
                               DAGUE_PROF_FUNC_KEY_END(this_task->dague_object,
                                                       this_task->function->function_id),
                               task_id, ddesca, data_id);
            }
#endif  /* defined(DAGUE_PROF_TRACE) */
            gpu_device->exec_array[gpu_device->exec_waiting] = NULL;
            gpu_device->exec_waiting = (gpu_device->exec_waiting + 1) % gpu_device->max_exec_tasks;
            goto out_task;
        } else {
            DAGUE_CUDA_CHECK_ERROR( "cuEventQuery ", rc,
                                    {assert(0); goto disable_gpu;} );
        }
    }
 out_task:
    if( NULL != this_task ) {
        if( NULL != gpu_device->out_array[gpu_device->out_submit] ) {
            /* No more room on the event list. Store the execution context */
            DAGUE_FIFO_PUSH(gpu_device->fifo_pending_out,
                            (dague_list_item_t*)this_task);
            this_task = NULL;
        } else {
            /* Get the oldest task */
            if( !dague_fifo_is_empty(gpu_device->fifo_pending_out) ) {
                DAGUE_FIFO_PUSH(gpu_device->fifo_pending_out,
                                (dague_list_item_t*)this_task);
                this_task = (dague_execution_context_t*)dague_fifo_pop(gpu_device->fifo_pending_out);
            }
        }
    } else {
        if( NULL == gpu_device->out_array[gpu_device->out_submit] ) {
            this_task = (dague_execution_context_t*)dague_fifo_pop(gpu_device->fifo_pending_out);
        }
    }
    if( NULL != this_task ) {
        assert( NULL == gpu_device->out_array[gpu_device->out_submit] );
        rc = gpu_sgemm_internal_pop( gpu_device, this_task,
                                     gpu_device->streams[1] );
        DEBUG(("GPU Request number %d/%d\n",
               gpu_device->out_array_events[gpu_device->out_submit],
               gpu_device->streams[1]));
        if( 0 == rc ) goto complete_task;  /* no data to be moved */
        gpu_device->out_array[gpu_device->out_submit] = this_task;
        this_task = NULL;
        if( 0 > rc ) { assert(0); goto disable_gpu; }
        rc = cuEventRecord( gpu_device->out_array_events[gpu_device->out_submit],
                            gpu_device->streams[1] );
        gpu_device->out_submit = (gpu_device->out_submit + 1) % gpu_device->max_out_tasks;
    }
 check_out_deps:
    assert( NULL == this_task );
    if( NULL != gpu_device->out_array[gpu_device->out_waiting] ) {
        rc = cuEventQuery(gpu_device->out_array_events[gpu_device->out_waiting]);
        if( CUDA_ERROR_NOT_READY == rc ) {
            goto check_in_deps;
        } else if( CUDA_SUCCESS == rc ) {
            /* Save the task for the next step */
            DEBUG(("Completion of GPU Request number %d\n",
                   gpu_device->out_array_events[gpu_device->out_waiting]));
            this_task = gpu_device->out_array[gpu_device->out_waiting];
#if defined(DAGUE_PROF_TRACE)
            if( dague_cuda_trackable_events &
                DAGUE_PROFILE_CUDA_TRACK_DATA_OUT )
                dague_profiling_trace( gpu_device->profiling,
                                       dague_cuda_moveout_key_end,
                                       (unsigned long)this_task, NULL );
#endif  /* defined(DAGUE_PROF_TRACE) */
            gpu_device->out_array[gpu_device->out_waiting] = NULL;
            gpu_device->out_waiting = (gpu_device->out_waiting + 1) % gpu_device->max_out_tasks;
            goto complete_task;
        } else {
            DAGUE_CUDA_CHECK_ERROR( "cuEventQuery ", rc,
                                    {assert(0); goto disable_gpu;} );
        }
    }

 fetch_task_from_shared_dequeue:
    assert( NULL == this_task );
    this_task = (dague_execution_context_t*)dague_dequeue_pop_front( &(gpu_device->pending) );
    if( NULL != this_task ) {
        DEBUG(( "Add gemm(blok = %d, cblk = %d, fcblk = %d) priority %d\n",
                this_task->locals[0].value, 
                this_task->locals[2].value, 
                this_task->locals[1].value,
                this_task->priority ));
    }
    goto check_in_deps;

 complete_task:
    /* Everything went fine so far, the result is correct and back in the main
       memory */
    DAGUE_LIST_ITEM_SINGLETON(this_task);
    dague_complete_execution( eu_context, this_task );
    gpu_device->executed_tasks++;
    rc = dague_atomic_dec_32b( &(gpu_device->mutex) );
    if( 0 == rc ) {  /* I was the last one */
        assert( (NULL == gpu_device->in_array[gpu_device->in_waiting]) &&
                (NULL == gpu_device->exec_array[gpu_device->exec_waiting]) &&
                (NULL == gpu_device->out_array[gpu_device->out_waiting]) );
#if defined(DAGUE_PROF_TRACE)
        if( dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_OWN )
            dague_profiling_trace( eu_context->eu_profile,
                                   dague_cuda_own_GPU_key_end,
                                   (unsigned long)eu_context, NULL );
#endif  /* defined(DAGUE_PROF_TRACE) */
        status = (cudaError_t)cuCtxPopCurrent(NULL);
        /* Restore the context so the others can steal it */
        dague_atomic_cas( &(gpu_device->ctx), NULL, saved_ctx );

        DAGUE_CUDA_CHECK_ERROR( "cuCtxPushCurrent ", status,
                                {return -1;} );
        return -1;
    }
    this_task = NULL;
    goto fetch_task_from_shared_dequeue;

 disable_gpu:
    /* Something wrong happened. Push all the pending tasks back on the
     * cores, and disable the gpu.
     */
    fprintf(stderr, "Big Badaboumm\n");
    /*exit(-20);*/
    assert(-20 == 0);
    return -2;
}
#else
static int
gpu_sgemm_internal( gpu_device_t* gpu_device,
                    dague_execution_unit_t* eu_context,
                    dague_execution_context_t* this_task,
                    CUstream stream, int uplo )
{
    int return_code = 0;  /* by default suppose an error */

    (void)eu_context;
    (void)uplo;

   // DEBUG(("Execute GEMM( k = %d, m = %d, n = %d ) [%d] on device %d stream %p\n",
     //      k, m, n, this_task->priority, gpu_device->id, (void*)stream));

    return_code = gpu_sgemm_internal_push( gpu_device,
                                           this_task,
                                           stream );
    if( 0 > return_code ) goto release_and_return_error;

    return_code = gpu_sgemm_internal_submit( gpu_device,
                                             this_task,
                                             stream );
    if( 0 != return_code ) goto release_and_return_error;

    return_code = gpu_sgemm_internal_pop( gpu_device,
                                          this_task,
                                          stream );

 release_and_return_error:
    return (return_code < 0 ? return_code : 0);
}

/**
 * This version is based on 4 streams, each of them potentially containing
 * all transfers from memory to the GPU, the kernel execution on the GPU and
 * the transfers from the GPU to the main memory. The synchronizations are
 * based on the fact that each stream contains only tasks related to a single
 * kernel, so waiting for the stream to be empty means everything related to
 * a task has been completed. There might be overlap between the operations on
 * different streams, however it is difficult to schedule in advance transfers
 * related to kernel that will be executed later.
 */
int sparse_gpu_sgemm( dague_execution_unit_t* eu_context,
                      dague_execution_context_t* this_task,
                      int uplo )
{
    int which_gpu, rc, stream_rc, waiting = 0, submit = 0;
    gpu_device_t* gpu_device;
    cudaError_t status;
    dague_execution_context_t* progress_array[DAGUE_MAX_STREAMS];
    int blok, cblk, fcblk;

    blok  = this_task->locals[0].value;
    fcblk = this_task->locals[1].value;
    cblk  = this_task->locals[2].value;

    //DEBUG(("GEMM( k = %d, m = %d, n = %d )\n", k, m, n));
    /* We always schedule the task on the GPU owning the C cblk. */
    which_gpu = gpu_data_cblk_write_owner( ddescA(this_task), fcblk);
/*    printf("k=%d, m=%d, n=%d\n",k,m,n);*/
    if( which_gpu < 0 ) {
        /* this is the first time we see this tile.
           Let's decide which GPU will work on it. */
        which_gpu = 0; /* TODO */
#if DPLASMA_SCHEDULING
        if(ndevices > 1) {
        /* reverse odd-even */
        /* homogeneous GPU */
        if(fcblk % 2 == 0) {
            which_gpu = gpu_set[fcblk%400] % ndevices;
        }
        else {
            which_gpu = ndevices - (gpu_set[fcblk%400] % ndevices + 1);
        }

        /* heterogenous GPU */
        /* weight by percentage of getting n of (n) with performance factor */
        {
        }
        dague_atomic_inc_32b( &(gpu_set[fcblk%400]) );
    }
    /*c1060 4 - 2  384-448  3-0-2-0 960 */
    /*c2050 5 - 2 448       4-2 960 */

#if DPLASMA_ONLY_GPU

#else

     /*
      **Rectangular Mesh **
      *
      * 1. Fact, number of tile,GEMMs is come from Matrix size and tile size
      *    we may have to change m,n in every tile size/ matrix size
      * 2. m and n is assign the size of squares which're going to mark over
      *    the triangular bunch of GEMMs
      * 3. m % (?) == (?) and n % (?) == (?) marks which tile is gonna be
      *    executed on CPU
      * 4. all (?) values affect "square size" and "position"-- which affects
      *    how many GEMMs will be executed on CPU
      * 5. Once we superpose/pile up "many square(m,n) -- like a mesh" on to
      *    triangular GEMMs, we will be able to caluculate how many GEMMs
      *     will be on CPU, also know which tiles
      * 6. The number GEMMs on GPU and CPU would meet "how many times GPU faster
      *    than CPU "
      *    I usually use m % 3 == 0 && n % 2 == 0 on C1060 (3x2 square)
      *    I usaully use m % 4 == 0 && n % 2 == 0 on C2050 (4x2 square)
      *    chance is lower that 1:6 or 1:8 becasue we pile up this square on to
      *    triangular
      *
      * Why this method ?
      * - try to finish "each bunch of GEMMs" as soon as poosible with GPU+CPU
      * - plus "balancing" between CPU/GPU
      */
        /* if( ((m % OHM_M) == 0) && ( (n % OHM_N) == 0) ){ */
        /*     dague_atomic_inc_32b( &(cpu_counter) ); */
        /*     return -99; */
        /* } */
#endif
#endif
    }
    gpu_device = gpu_devices[which_gpu];

#if DPLASMA_SCHEDULING

    /* keep n -- not being used yet*/
    gpu_load[gpu_device->id]+=n;
#endif

    /* Check the GPU status */
    rc = dague_atomic_inc_32b( &(gpu_device->mutex) );
    if( 1 != rc ) {  /* I'm not the only one messing with this GPU */
        DAGUE_LIST_ITEM_SINGLETON( (dague_list_item_t*)this_task );
        dague_dequeue_push_back( &(gpu_device->pending),
                                 (dague_list_item_t*)this_task );
        return -1;
    }

    status = (cudaError_t)cuCtxPushCurrent(gpu_device->ctx);
    DAGUE_CUDA_CHECK_ERROR( "cuCtxPushCurrent ", status,
                              {return -2;} );
    for( rc = 0; rc < DAGUE_MAX_STREAMS; rc++ )
        progress_array[rc] = NULL;

 more_work_to_do:
    if( (NULL != this_task) && (NULL == progress_array[submit]) ) {
        progress_array[submit] = this_task;

        /* Push this task into the GPU */
        rc = gpu_sgemm_internal( gpu_device, eu_context,
                                 this_task, gpu_device->streams[submit], uplo );
        if( 0 != rc ) {
            /* something fishy happened.
               Reschedule the pending tasks on the cores */
            goto disable_gpu;
        }
        /*printf( "GPU submit %p (k = %d, m = %d, n = %d) [%d]\n",
          (void*)progress_array[submit], k, m, n, submit );*/
        submit = (submit + 1) % gpu_device->max_streams;
        this_task = NULL;
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
            DAGUE_CUDA_CHECK_ERROR( "cuStreamQuery ", stream_rc,
                                      {return -2;} );
        }
    }

    if( NULL == this_task ) {
        goto fetch_more_work;
    }
    goto more_work_to_do;

 complete_previous_work:
    /* Everything went fine so far,
       the result is correct and back in the main memory */
    /*printf( "GPU complete %p (k = %d, m = %d, n = %d) [%d]\n",
      (void*)progress_array[waiting], k, m, n, waiting );*/
    dague_complete_execution( eu_context, progress_array[waiting] );
    progress_array[waiting] = NULL;
    waiting = (waiting + 1) % gpu_device->max_streams;

    gpu_device->executed_tasks++;
/*	dague_atomic_dec_32b( &(gpu_device->workload) );*/
    rc = dague_atomic_dec_32b( &(gpu_device->mutex) );
    if( 0 == rc ) {  /* I was the last one */
        status = (cudaError_t)cuCtxPopCurrent(NULL);
        DAGUE_CUDA_CHECK_ERROR( "cuCtxPushCurrent ", status,
                                  {return -1;} );
        return -1;
    }

 fetch_more_work:
    /* Do we still have room in the progress_array? */
    if( NULL != progress_array[submit] )
        goto wait_for_completion;

    this_task = (dague_execution_context_t*)dague_dequeue_pop_front( &(gpu_device->pending) );
    if( NULL == this_task ) {
        /* Collisions, save time and come back here later */
        goto more_work_to_do;
    }

    m = this_task->locals[1].value;
    n = this_task->locals[2].value;

    goto more_work_to_do;

    /* a device ... */
 disable_gpu:
    __dague_schedule( eu_context, this_task);
    rc = dague_atomic_dec_32b( &(gpu_device->mutex) );
    while( rc != 0 ) {
        this_task = (dague_execution_context_t*)dague_dequeue_pop_front( &(gpu_device->pending) );
        if( NULL != this_task ) {
            __dague_schedule( eu_context, this_task);
            rc = dague_atomic_dec_32b( &(gpu_device->mutex) );
        }
    }
    status = (cudaError_t)cuCtxPopCurrent(NULL);
    DAGUE_CUDA_CHECK_ERROR( "cuCtxPushCurrent ", status,
                              {} );
    return -2;
}
#endif  /* !defined(DAGUE_GPU_STREAM_PER_TASK) */

/****************************************************
 ** GPU-DATA Specific Starts Here **
 ****************************************************/

#include "gpu_data.h"
#include "data_distribution.h"
#include "linked_list.h"

static memory_elem_t** data_map = NULL;
extern int ndevices;

int sparse_gpu_mark_data_usage( sparse_matrix_desc_t* data, int type, int cblk )
{
    memory_elem_t* this_data;
    (void)data;
    
    if( (NULL == data_map) ||
        (NULL == (this_data = data_map[cblk])) ) {
        /* Data not on the GPU. Nothing to do */
        return 0;
    }
    if( type & DAGUE_WRITE ) {
        this_data->memory_version++;
        this_data->writer++;
    }
    if( type & DAGUE_READ ) {
        this_data->readers++;
    }
    return 0;
}

int sparse_gpu_data_map_init( gpu_device_t* gpu_device,
                       sparse_matrix_desc_t* data )
{
    (void)data;
    if( NULL == data_map ) {
        data_map = (memory_elem_t**)calloc(SYMB_CBLKNBR,
                                           sizeof(memory_elem_t*));
    }
    gpu_device->gpu_mem_lru = (dague_linked_list_t*)malloc(sizeof(dague_linked_list_t));
    dague_linked_list_construct(gpu_device->gpu_mem_lru);
    return 0;
}

int sparse_gpu_data_cblk_write_owner( sparse_matrix_desc_t* data,
                                      int cblk )
{
    memory_elem_t* memory_elem;
    gpu_elem_t* gpu_elem;
    int i;
    (void)data;

    DEBUG( ("data_map %p, cblk %d\n", data_map, cblk) );
    if( NULL == (memory_elem = data_map[cblk]) ) {
        return -1;
    }
    for( i = 0; i < ndevices; i++ ) {
        gpu_elem = memory_elem->gpu_elems[i];
        if( NULL == gpu_elem )
            continue;
        if( gpu_elem->type & DAGUE_WRITE )
            return i;
    }
    return -2;
}

int sparse_gpu_data_get_cblk( sparse_matrix_desc_t* data,
                       int cblk,
                       memory_elem_t **pmem_elem )
{
    memory_elem_t* memory_elem;
    int rc = 0;  /* the cblk already existed */
    (void)data;

    if( NULL == (memory_elem = data_map[cblk]) ) {
        memory_elem = (memory_elem_t*)calloc(1, sizeof(memory_elem_t) +
                                             (ndevices-1) *
                                             sizeof(gpu_elem_t*));
        memory_elem->memory_version = 0;
        memory_elem->readers = 0;
        memory_elem->writer = 0;
        memory_elem->memory = NULL;
        rc = 1;  /* the cblk has just been created */
        if( 0 == dague_atomic_cas( &(data_map[cblk]),
                                   NULL, memory_elem ) ) {
            free(memory_elem);
            rc = 0;  /* the cblk already existed */
            memory_elem = data_map[cblk];
        }
    }
    *pmem_elem = memory_elem;
    return rc;
}

/**
 * This function check if the target tile is already on the GPU memory.
 * If it is the case, it check if the version on the GPU match with
 * the one in memory. In all cases, it propose a section in the GPU
 * memory where the data should be transferred.
 *
 * It return 1 if no transfer should be initiated, a 0 if a transfer is
 * necessary, and a negative value if no memory is currently available
 * on the GPU.
 */
int sparse_gpu_data_is_on_gpu( gpu_device_t* gpu_device,
                               sparse_matrix_desc_t* data,
                               int type, int cblk,
                               gpu_elem_t **pgpu_elem)
{
    memory_elem_t* memory_elem;
    gpu_elem_t* gpu_elem;

    sparse_gpu_data_get_cblk( data, cblk, &memory_elem );

    if( NULL == (gpu_elem = memory_elem->gpu_elems[gpu_device->id]) ) {
        /* Get the LRU element on the GPU and transfer it to this new data */
        gpu_elem = (gpu_elem_t*)dague_linked_list_remove_head(gpu_device->gpu_mem_lru);
        if( memory_elem != gpu_elem->memory_elem ) {
            if( NULL != gpu_elem->memory_elem ) {
                memory_elem_t* old_mem = gpu_elem->memory_elem;
                old_mem->gpu_elems[gpu_device->id] = NULL;
            }
            gpu_elem->type = 0;
        }
        gpu_elem->type |= type;
        gpu_elem->memory_elem = memory_elem;
        memory_elem->gpu_elems[gpu_device->id] = gpu_elem;
        *pgpu_elem = gpu_elem;
        dague_linked_list_add_tail(gpu_device->gpu_mem_lru, (dague_list_item_t*)gpu_elem);
    } else {
        dague_linked_list_remove_item(gpu_device->gpu_mem_lru, (dague_list_item_t*)gpu_elem);
        dague_linked_list_add_tail(gpu_device->gpu_mem_lru, (dague_list_item_t*)gpu_elem);
        gpu_elem->type |= type;
        *pgpu_elem = gpu_elem;
        if( memory_elem->memory_version == gpu_elem->gpu_version ) {
            /* The GPU version of the data matches the one in memory. We're done */
            return 1;
        }
        /* The version on the GPU doesn't match the one in memory. Let the
         * upper level know a transfer is required.
         */
    }
    gpu_elem->gpu_version = memory_elem->memory_version;
    /* Transfer is required */
    return 0;
}


static void
compute_best_unit( uint64_t length, float* updated_value, char** best_unit )
{
    float measure = (float)length;

    *best_unit = "B";
    if( measure > 1024.0f ) { /* 1KB */
        *best_unit = "KB";
        measure = measure / 1024.0f;
        if( measure > 1024.0f ) { /* 1MB */
            *best_unit = "MB";
            measure = measure / 1024.0f;
            if( measure > 1024.0f ) {
                *best_unit = "GB";
                measure = measure / 1024.0f;
            }
        }
    }
    *updated_value = measure;
    return;
}
