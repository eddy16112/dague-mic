/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> z c d s
 *
 */

#include "dague_config.h"
#include <stdlib.h>
#include <dlfcn.h>
#include <plasma.h>
#include <cublas.h>
#include "dague.h"
#include "gpu_data.h"
#include "execution_unit.h"
#include "scheduling.h"
#include "fifo.h"
#include "datarepo.h"
#include "data_distribution.h"
#include "data_dist/sparse-matrix/pastix_internal/pastix_internal.h"
#include "data_dist/sparse-matrix/sparse-matrix.h"

#include "cuda_zpotrf1d_gemm.h"

#define KERNEL_NAME zpotrfsp1d

int gpu_kernel_init_zpotrfsp1d( dague_context_t* dague_context, 
                                sparse_matrix_desc_t *tileA );

static inline
int gpu_kernel_push_zpotrfsp1d( gpu_device_t* gpu_device,
                                dague_execution_context_t* this_task,
                                CUstream stream );

static inline
int gpu_kernel_submit_zpotrfsp1d( gpu_device_t* gpu_device,
                                  dague_execution_context_t* this_task,
                                  CUstream stream );

static inline
int gpu_kernel_pop_zpotrfsp1d( gpu_device_t* gpu_device,
                               dague_execution_context_t* this_task,
                               CUstream stream );

static inline
int  gpu_kernel_epilog_zpotrfsp1d( gpu_device_t* gpu_device,
                                   dague_execution_context_t* this_task );

static inline 
void gpu_kernel_profile_zpotrfsp1d( gpu_device_t              *gpu_device,
                                    dague_execution_context_t *this_task,
                                    dague_ddesc_t             *ddesca );

#include "gpu_scheduling.h"

#if DPLASMA_SCHEDULING
uint32_t *gpu_set;
#endif

static sparse_matrix_desc_t *UGLY_A;
static SolverMatrix         *datacode;
static int ndevices = 0;

/* FIXME */
#define dague_gpu_1gpu_fini( ... )

#if defined(DAGUE_PROF_TRACE)
static inline
void gpu_kernel_profile_zpotrfsp1d( gpu_device_t              *gpu_device,
                                    dague_execution_context_t *this_task,
                                    dague_ddesc_t             *ddesca )
{
    if( dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_EXEC ) {
        int data_id =
            ddesca->data_key(ddesca, this_task->locals[2].value, this_task->locals[0].value);
        uint64_t task_id =
            this_task->function->key( this_task->dague_object, this_task->locals );
        
        dague_profile_ddesc_info_t info;
        info.desc = ddesca;
        info.id = data_id;
        dague_profiling_trace( gpu_device->profiling,
                               DAGUE_PROF_FUNC_KEY_START(this_task->dague_object,
                                                         this_task->function->function_id),
                               task_id, this_task->dague_object->object_id,
                               (void*)&info);
    }
}
#endif  /* defined(DAGUE_PROF_TRACE) */

int gpu_kernel_init_zpotrfsp1d( dague_context_t* dague_context, 
                                sparse_matrix_desc_t *sparseA )
{
    char *env;
    int i, dindex;
    int nbgpus;
    (void)dague_context;

    UGLY_A = sparseA;
    datacode = &(UGLY_A->pastix_data->solvmatr);

    nbgpus = dague_active_gpu();
#if DPLASMA_SCHEDULING
    gpu_set = (uint32_t*)calloc( SYMB_CBLKNBR, sizeof(uint32_t));
#endif

    for( i = dindex = 0; i < nbgpus; i++ ) {
        gpu_device_t* gpu_device;
        CUresult status;
        char module_path[FILENAME_MAX];

        gpu_device = gpu_enabled_devices[i];

        status = cuCtxPushCurrent( gpu_device->ctx );
        DAGUE_CUDA_CHECK_ERROR( "(INIT) cuCtxPushCurrent ", status, { dague_gpu_1gpu_fini(gpu_device); continue;} );

#if 0
        /* If not disallowed by env, load from static linked kernels */
        /* This is non functional, as the ptr is not a CuFunction. */
        gpu_device->hcuFunction = NULL;
        env = getenv("DAGUE_CUBIN_NOSTATIC");
        if( !env || (('1' != env[0]) && ('y' != env[0])) ) {
            void* dlh;
            snprintf(module_path, FILENAME_MAX, "magmablas_zgemm_SM%d%d",
                     gpu_device->major, gpu_device->minor);
            dlh = dlopen(NULL, RTLD_NOW);
            if(NULL == dlh) ERROR(("Error parsing static libs: %s\n", dlerror()));
            gpu_device->hcuFunction = dlsym(dlh, module_path);
            dlclose(dlh);
        }

        /* If not found statically, cuload it */
        if(NULL == gpu_device->hcuFunction) {
            env = getenv("DAGUE_CUBIN_PATH");
            snprintf(module_path, FILENAME_MAX, "%s/zpotrfsp1d-sm_%1d%1d.cubin",
                     env?env:"../cores", gpu_device->major, gpu_device->minor);
            status = cuModuleLoad(&(gpu_device->hcuModule), module_path);
            DAGUE_CUDA_CHECK_ERROR( "(INIT) cuModuleLoad ", status,
                                    {
                                        WARNING(("GPU:\tUnable to load `%s'\n", module_path));
                                        dague_gpu_1gpu_fini(gpu_device); 
                                        continue;
                                    } );
            snprintf(module_path, FILENAME_MAX, "zpotrfsp1dNT_SM%d%d", gpu_device->major, gpu_device->minor);
            DEBUG3(("CUDA MODULE %s\n", module_path));
            status = cuModuleGetFunction( &(gpu_device->hcuFunction), gpu_device->hcuModule, module_path );
            DAGUE_CUDA_CHECK_ERROR( "(INIT) cuModuleGetFunction ", status,
                                    {
                                        WARNING(("GPU:\tUnable to find the function `%s'\n", module_path));
                                        dague_gpu_1gpu_fini(gpu_device); 
                                        continue;
                                    } );
        }
        if(NULL == gpu_device->hcuFunction) return -1;
#else
        
        if (  gpu_device->major < 2 ) {
            dague_gpu_1gpu_fini( gpu_device );
            continue;
        }

#endif

        status = cuCtxPopCurrent(NULL);
        DAGUE_CUDA_CHECK_ERROR( "(INIT) cuCtxPopCurrent ", status,
                                {dague_gpu_1gpu_fini(gpu_device); continue;} );
        gpu_device->index = (uint8_t)dindex;
        gpu_enabled_devices[dindex++] = gpu_device;
    }

    /* Update the number of GPUs available */
    dague_data_enable_gpu( dindex );
    ndevices = dindex;

    return 0;
}

#define CBLK_SIZE(cblk)                                                 \
    (SOLV_STRIDE(cblk)*(SYMB_LCOLNUM(cblk)-SYMB_FCOLNUM(cblk)+1))

#define TYPE_SIZE                               \
    (sparse_matrix_size_of( UGLY_A->mtype) )

/**
 *  This function schedule the move of all the data required for a
 *  specific task from the main memory into the GPU memory.
 *
 *  Returns:
 *     a positive number: the number of data to be moved.
 *     -1: data cannot be moved into the GPU.
 *     -2: No more room on the GPU to move this data.
 */
static inline int
gpu_kernel_push_zpotrfsp1d( gpu_device_t* gpu_device,
                       dague_execution_context_t* this_task,
                       CUstream stream )
{
    int sizeloc[MAX_PARAM_COUNT];
    int ret;
    int move_data_count = 0;
    int eltsize = 0;
    gpu_elem_t* gpu_elem;
    (void)eltsize;

    int bloknum, fcblknum, cblknum, phony, prev, next;
    bloknum  = this_task->locals[0].value;
    fcblknum = this_task->locals[1].value;
    cblknum  = this_task->locals[2].value;
    phony    = this_task->locals[3].value;
    prev     = this_task->locals[4].value;
    next     = this_task->locals[5].value;

    fprintf(stderr, "Push\n");

    gpu_elem = dague_gpu_get_data_on_gpu(gpu_device, &dague_gpu_map, KERNEL_KEY( cblknum ),
                                         &(this_task->data[0].mem2dev_data) );
    if( NULL == gpu_elem ) move_data_count++;

    gpu_elem = dague_gpu_get_data_on_gpu(gpu_device, &dague_gpu_map, KERNEL_KEY( fcblknum ),
                                         &(this_task->data[1].mem2dev_data) );
    if( NULL == gpu_elem ) move_data_count++;

    if( 0 != move_data_count ) { /* Try to reserve enough room for all data */
        sizeloc[0] = CBLK_SIZE(  cblknum ) * TYPE_SIZE;
        sizeloc[1] = CBLK_SIZE( fcblknum ) * TYPE_SIZE;

        ret = dague_gpu_find_space_for_elts( gpu_device,
                                             this_task,
                                             sizeloc,
                                             move_data_count );
        if( ret < 0 ) {
            goto release_and_return_error;
        }
    }

#if defined(DAGUE_PROF_TRACE)
    if( dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_DATA_IN )
        dague_profiling_trace( gpu_device->profiling, dague_cuda_movein_key_start,
                               (unsigned long)this_task, this_task->dague_object->object_id,
                               NULL );
#endif  /* defined(DAGUE_PROF_TRACE) */

    DEBUG3(("GPU:\tRequest Data of %s(%d, %d, %d, %d, %d) on GPU\n", 
            this_task->function->in[0]->name, 
            bloknum, fcblknum, cblknum, prev, next));
    ret = dague_gpu_data_stage_in( gpu_device, KERNEL_KEY(cblknum), this_task->function->in[0]->access_type,
                                   this_task->data[0].mem2dev_data,
                                   ADATA(this_task->data[0].data), sizeloc[0], stream );
    if( ret < 0 ) {
        goto release_and_return_error;
    }

    DEBUG3(("GPU:\tRequest Data of %s(%d, %d, %d, %d, %d) on GPU\n", 
            this_task->function->in[1]->name, 
            bloknum, fcblknum, cblknum, prev, next));
    ret = dague_gpu_data_stage_in( gpu_device, KERNEL_KEY(fcblknum), this_task->function->in[1]->access_type,
                                   this_task->data[1].mem2dev_data,
                                   ADATA(this_task->data[1].data), sizeloc[1], stream );
    if( ret < 0 ) {
        goto release_and_return_error;
    }

    assert( NULL != this_task->data[0].mem2dev_data->device_elem[gpu_device->index] );
    assert( NULL != this_task->data[1].mem2dev_data->device_elem[gpu_device->index] );
  release_and_return_error:
    return ret;
}

static inline int
gpu_kernel_submit_zpotrfsp1d( gpu_device_t* gpu_device,
                         dague_execution_context_t* this_task,
                         CUstream stream )
{
    gpu_elem_t *gpu_elem_A = NULL, *gpu_elem_C = NULL;
    CUdeviceptr d_A, d_C;
    cudaError_t status;
#if defined(PRECISION_z) || defined(PRECISION_c)
    cuDoubleComplex alpha = make_cuDoubleComplex(-1.0, 0.0);
    cuDoubleComplex beta  = make_cuDoubleComplex( 1.0, 0.0);
#else
    double alpha = -1.0;
    double beta  = 1.0;
#endif

    int bloknum, fcblknum, cblknum, phony, prev, next;
    bloknum  = this_task->locals[0].value;
    fcblknum = this_task->locals[1].value;
    cblknum  = this_task->locals[2].value;
    phony    = this_task->locals[3].value;
    prev     = this_task->locals[4].value;
    next     = this_task->locals[5].value;

    gpu_elem_A = (gpu_elem_t *)this_task->data[0].mem2dev_data->device_elem[gpu_device->index];
    gpu_elem_C = (gpu_elem_t *)this_task->data[1].mem2dev_data->device_elem[gpu_device->index];
    d_A = gpu_elem_A->gpu_mem;
    d_C = gpu_elem_C->gpu_mem;

    fprintf(stderr, "Submit\n");

    DEBUG2(("GPU:\tRequest GPU runs %s(%d, %d, %d, %d, %d, %d)\n", 
            this_task->function->name,
            this_task->locals[0],
            this_task->locals[1],
            this_task->locals[2],
            this_task->locals[3],
            this_task->locals[4],
            this_task->locals[5]));

#if defined(DAGUE_PROF_TRACE)
    gpu_kernel_profile( gpu_device, this_task, dague_gpu_map.desc);
#endif  /* defined(DAGUE_PROF_TRACE) */

    {
        int bloknbr = SYMB_BLOKNUM(cblknum+1) - bloknum;
        int m = SOLV_STRIDE(cblknum)  - SOLV_COEFIND(bloknum);
        int n = SYMB_LROWNUM(bloknum) - SYMB_FROWNUM(bloknum) + 1;
        int k = SYMB_LCOLNUM(cblknum) - SYMB_FCOLNUM(cblknum) + 1;
        CUdeviceptr d_blok = d_A + SOLV_COEFIND(bloknum)*TYPE_SIZE;

        int fblcknbr = SYMB_BLOKNUM(fcblknum+1) - SYMB_BLOKNUM(fcblknum);
        CUdeviceptr d_blocktab, d_fbloktab;

        d_blocktab = UGLY_A->d_blocktab[gpu_device->index] + 2 * bloknum                * sizeof(my_tmp_int_t);
        d_fbloktab = UGLY_A->d_blocktab[gpu_device->index] + 2 * SYMB_BLOKNUM(fcblknum) * sizeof(my_tmp_int_t);
        
#if 0
        int (*func)(char, char, int, int, int,
                    cuDoubleComplex, cuDoubleComplex *, int,
                    cuDoubleComplex *, int,
                    cuDoubleComplex, cuDoubleComplex *, int,
                    int, const int *, int, const int *,
                    CUstream) = NULL;
        
        func = (__typeof__(func)) (uintptr_t) (gpu_device->hcuFunction);

        /* WARNING: check on multi-gpus how the texture works */
        status = func('N', 'T', m, n, k, 
                      alpha, (cuDoubleComplex*)d_blok, SOLV_STRIDE(cblknum),
                             (cuDoubleComplex*)d_blok, SOLV_STRIDE(cblknum),
                      beta,  (cuDoubleComplex*)d_C,    SOLV_STRIDE(fcblknum),
                      bloknbr, (const int *)d_blocktab, fblcknbr, (const int *)d_fbloktab,
                      stream );
#else

        fprintf(stderr, "Submit magmablas_zgemm_SM20\n");
        magmablas_zgemm_SM20('N', 'T', m, n, k, 
                             alpha, (cuDoubleComplex*)d_blok, SOLV_STRIDE(cblknum),
                                    (cuDoubleComplex*)d_blok, SOLV_STRIDE(cblknum),
                             beta,  (cuDoubleComplex*)d_C,    SOLV_STRIDE(fcblknum),
                             bloknbr, (const int *)d_blocktab, fblcknbr, (const int *)d_fbloktab,
                             stream );

#endif

    }

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
gpu_kernel_pop_zpotrfsp1d( gpu_device_t* gpu_device,
                           dague_execution_context_t* this_task,
                           CUstream stream )
{
    gpu_elem_t *gpu_elem = NULL;
    int return_code = 0, tile_size, how_many = 0, i;
    cudaError_t status;

    int bloknum, fcblknum, cblknum, phony, prev, next;
    bloknum  = this_task->locals[0].value;
    fcblknum = this_task->locals[1].value;
    cblknum  = this_task->locals[2].value;
    phony    = this_task->locals[3].value;
    prev     = this_task->locals[4].value;
    next     = this_task->locals[5].value;

    fprintf(stderr, "pop\n");

    /* Generic */
    for( i = 0; i < 2/*this_task->function->nb_parameters*/; i++ ) {
        gpu_elem = (gpu_elem_t*)this_task->data[i].mem2dev_data->device_elem[gpu_device->index];
        assert( gpu_elem->generic.memory_elem == this_task->data[i].mem2dev_data );
        if( this_task->function->in[i]->access_type & ACCESS_READ ) {
            gpu_elem->generic.readers--;
            if( (0 == gpu_elem->generic.readers) &&
                !(this_task->function->in[i]->access_type & ACCESS_WRITE) ) {
                dague_ulist_remove( gpu_device->gpu_mem_owned_lru, (dague_list_item_t*)gpu_elem);
                DAGUE_LIST_ITEM_SINGLETON(gpu_elem);
                dague_ulist_fifo_push(gpu_device->gpu_mem_lru, (dague_list_item_t*)gpu_elem);
            }
        }
        if( this_task->function->in[i]->access_type & ACCESS_WRITE ) {
            /* If we're not using this anymore on the GPU it should be moved back to the CPU */
        }
    }

    /* TODO: Put output data */
    gpu_elem = (gpu_elem_t*)this_task->data[1].mem2dev_data->device_elem[gpu_device->index];
    tile_size = CBLK_SIZE( fcblknum ) * TYPE_SIZE;

    /* Stage the transfer of the data back to main memory */
    gpu_device->required_data_out += tile_size;
    assert( ((dague_list_item_t*)gpu_elem)->list_next == (dague_list_item_t*)gpu_elem );
    assert( ((dague_list_item_t*)gpu_elem)->list_prev == (dague_list_item_t*)gpu_elem );

    /* TODO: Adapt testing to code */
    if( next == 0 ) { 
        DEBUG3(("GPU Request out of GPU for %s key %d\n",
                this_task->function->in[1]->name,
                this_task->data[1].mem2dev_data->key));

#if defined(DAGUE_PROF_TRACE)
        if( dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_DATA_OUT )
            dague_profiling_trace( gpu_device->profiling, dague_cuda_moveout_key_start,
                                   (unsigned long)this_task, this_task->dague_object->object_id,
                                   NULL );
#endif  /* defined(DAGUE_PROF_TRACE) */

        /* Pop C from the GPU */
        status = (cudaError_t)cuMemcpyDtoHAsync( ADATA(this_task->data[1].data), gpu_elem->gpu_mem, tile_size, stream );
        DAGUE_CUDA_CHECK_ERROR( "cuMemcpyDtoHAsync from device ", status,
                                { WARNING(("data %s <<%p>> -> <<%p>>\n", this_task->function->in[1]->name,
                                           (void*)(long)gpu_elem->gpu_mem, (void*)ADATA(this_task->data[1].data)));
                                  return_code = -2;
                                  goto release_and_return_error;} );
        gpu_device->transferred_data_out += tile_size;
        how_many++;
    }
 release_and_return_error:
    return (return_code < 0 ? return_code : how_many);
}

static inline int
gpu_kernel_epilog_zpotrfsp1d( gpu_device_t* gpu_device,
                         dague_execution_context_t* this_task )
{
    gpu_elem_t* gpu_elem;
    int i;
    
    fprintf(stderr, "Epilog\n");
    for( i = 0; i < 2/*this_task->function->nb_parameters*/; i++ ) {
        if( !(this_task->function->in[i]->access_type & ACCESS_WRITE) ) continue;

        gpu_elem = (gpu_elem_t*)this_task->data[i].mem2dev_data->device_elem[gpu_device->index];
        assert( DAGUE_DATA_OWNED == gpu_elem->generic.coherency_state );
        gpu_elem->generic.coherency_state = DAGUE_DATA_SHARED;
        gpu_elem->generic.memory_elem->version = gpu_elem->generic.version;
        this_task->data[1].mem2dev_data->device_owner = -1;

        if( this_task->locals[5].value == 0 ) {  /* next == 0 */
            dague_ulist_fifo_push(gpu_device->gpu_mem_lru, (dague_list_item_t*)gpu_elem);
        } else {
            dague_ulist_fifo_push(gpu_device->gpu_mem_owned_lru, (dague_list_item_t*)gpu_elem);
        }
    }
    return 0;
}


/**
 * Try to execute a GEMM on a GPU.
 *
 * Returns:
 *  0 - if the GEMM should be executed by some other meaning (in this case the
 *         execution context is not released).
 * -1 - if the GEMM is scheduled to be executed on a GPU.
 */

#if !defined(DAGUE_GPU_STREAM_PER_TASK)

/**
 * This version is based on 4 streams: one for transfers from the memory to
 * the GPU, 2 for kernel executions and one for tranfers from the GPU into
 * the main memory. The synchronization on each stream is based on CUDA events,
 * such an event indicate that a specific epoch of the lifetime of a task has
 * been completed. Each type of stream (in, exec and out) has a pending FIFO,
 * where tasks ready to jump to the respective step are waiting.
 */
int gpu_zpotrfsp1d( dague_execution_unit_t* eu_context,
                    dague_execution_context_t* this_task )
{
    int which_gpu;
    int fcblk;

    fprintf(stderr, "Call GPU kernel");

    fcblk = this_task->locals[1].value;

    /* We always schedule the task on the GPU owning the C tile. */
    which_gpu = dague_gpu_data_elt_write_owner( &dague_gpu_map, KERNEL_KEY( fcblk ) );
    if( which_gpu < 0 ) {  /* this is the first time we see this tile. Let's decide which GPU will work on it. */
        which_gpu = 0; /* TODO */
#if DPLASMA_SCHEDULING
        if( ndevices > 1){
            /* reverse odd-even */
            /* homogeneous GPU */
            {
                if(fcblk % 2 == 0){
                    which_gpu = gpu_set[fcblk] % ndevices;
                }
                else{
                    which_gpu = ndevices - (gpu_set[fcblk] % ndevices + 1);
                }
            }

            /* heterogenous GPU */
            /* weight by percentage of getting n of (n) with performance factor */
            {


            }

            dague_atomic_inc_32b( &(gpu_set[fcblk]) );
        }
#if DPLASMA_ONLY_GPU
#else
        /*
        **Rectangular Mesh **
        1. Fact, a number of tile ahd GEMMs comes from Matrix size and tile size
        - we may have to change m,n in every tile size/ matrix size
        2. m and n is assign the size of squares which're going to mark over the
        * triangular bunch of GEMMs
        * 3. m % (?) == (?) and n % (?) == (?) marks which tile is gonna be executed on CPU
        * 4. all (?) values affect "square size" and "position"-- which affects how many GEMMs will be executed on CPU
        * 5. Once we superpose/pile up "many square(m,n) -- like a mesh" on to triangular GEMMs, we will be able to caluculate how many GEMMs will be on CPU, also know which tiles 
        * 6. The number GEMMs on GPU and CPU would meet "how many times GPU faster than CPU "
        * I usually use m % 3 == 0 && n % 2 == 0 on C1060 (3x2 square)
        * I usaully use m % 4 == 0 && n % 2 == 0 on C2050 (4x2 square)
        * chance is lower that 1:6 or 1:8 becasue we pile up this square on to triangular
        * Why this method ?
        *  - try to finish "each bunch of GEMMs" as soon as poosible with GPU+CPU
        *  - plus "balancing" between CPU/GPU
        **/

/*         if( ((m % OHM_M) == 0) && ( (n % OHM_N) == 0) ){ */
/*             dague_atomic_inc_32b( &(dague_cpu_counter) ); */
/*             return -99; */
/*         } */
#endif  /* DPLASMA_ONLY_GPU */
#endif  /* DPLASMA_SCHEDULING */
    }

    int rc = gpu_kernel_scheduler_zpotrfsp1d( eu_context, this_task, which_gpu );
    fprintf(stderr, "rc = %d\n", rc);
    return rc;
}

#else
#error "This case is not correct right now"
#endif  /* !defined(DAGUE_GPU_STREAM_PER_TASK) */

