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
//#include <plasma.h>
//#include <core_blas.h>
#if defined(PRECISION_z) || defined(PRECISION_c)
#include <cuComplex.h>
#endif
//#include <cublas.h>
#include "dague.h"
#include "gpu_data.h"
#include "execution_unit.h"
#include "scheduling.h"
#include "fifo.h"
#include "datarepo.h"
#include "data_distribution.h"
#include "data_dist/sparse-matrix/pastix_internal/pastix_internal.h"
#include "data_dist/sparse-matrix/sparse-matrix.h"

#include "cuda_zgetrfsp_gemm.h"

#define KERNEL_NAME zgetrfsp_gemm

typedef void (*cuda_zgetrfsp_gemm_t) ( char TRANSA, char TRANSB, int m, int n, int k,
                                       cuDoubleComplex alpha, cuDoubleComplex *d_A, int lda,
                                                              cuDoubleComplex *d_B, int ldb,
                                       cuDoubleComplex beta,  cuDoubleComplex *d_C, int ldc,
                                       int blocknbr, const int *blocktab, int fblocknbr, const int *fblocktab,
                                       CUstream stream );

cuda_zgetrfsp_gemm_t* zgetrfsp_gemm_functions;

#define FORCE_UNDEFINED_SYMBOL(x) void* __ ## x ## _getrf =(void*)&x;
/* extern cuda_zgetrfsp_gemm_t zgemm_sparse_SM11; */
/* FORCE_UNDEFINED_SYMBOL(zgemm_sparse_SM11) */
/* extern cuda_zgetrfsp_gemm_t zgemm_sparse_SM13; */
/* FORCE_UNDEFINED_SYMBOL(zgemm_sparse_SM13) */
extern cuda_zgetrfsp_gemm_t zgemm_sparse_SM20;
FORCE_UNDEFINED_SYMBOL(zgemm_sparse_SM20)

static inline
int gpu_kernel_push_zgetrfsp_gemm( gpu_device_t* gpu_device,
                                   dague_gpu_context_t* this_task,
                                   CUstream stream );

static inline
int gpu_kernel_submit_zgetrfsp_gemm( gpu_device_t* gpu_device,
                                     dague_gpu_context_t* this_task,
                                     CUstream stream );

static inline
int gpu_kernel_pop_zgetrfsp_gemm( gpu_device_t* gpu_device,
                                  dague_gpu_context_t* this_task,
                                  CUstream stream );

static inline
int  gpu_kernel_epilog_zgetrfsp_gemm( gpu_device_t* gpu_device,
                                      dague_gpu_context_t* this_task );

#if defined(DAGUE_PROF_TRACE)
static inline
void gpu_kernel_profile_zgetrfsp_gemm( gpu_device_t        *gpu_device,
                                       dague_gpu_context_t *this_task );
#endif

typedef struct dague_zgetrfsp_gemm_args_s {
    dague_gpu_context_t super;
    int pushout;
    my_tmp_int_t cblknum;
    my_tmp_int_t bloknum;
    my_tmp_int_t fcblknum;
    int M, N, K;
    size_t sizecblk, sizefcblk;
    dague_ddesc_t *ddesc;
} dague_zgetrfsp_gemm_args_t;

#include "gpu_scheduling.h"

static int ndevices = 0;

#if defined(DAGUE_PROF_TRACE)
static inline
void gpu_kernel_profile_zgetrfsp_gemm( gpu_device_t        *gpu_device,
                                       dague_gpu_context_t *this_task )
{
    if( dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_EXEC ) {
        dague_execution_context_t  *ec   = this_task->ec;
        dague_zgetrfsp_gemm_args_t *args = (dague_zgetrfsp_gemm_args_t*)this_task;
        dague_ddesc_t *ddesc = (dague_ddesc_t*)(args->ddesc);
        int data_id =
            ddesc->data_key(ddesc,
                            args->fcblknum );

        uint64_t task_id =
            ec->function->key( ec->dague_object, ec->locals );

        dague_profile_ddesc_info_t info;
        info.desc = ddesc;
        info.id = data_id;
        dague_profiling_trace( gpu_device->profiling,
                               DAGUE_PROF_FUNC_KEY_START(ec->dague_object,
                                                         ec->function->function_id),
                               task_id, ec->dague_object->object_id,
                               (void*)&info);
    }
}
#endif  /* defined(DAGUE_PROF_TRACE) */

int gpu_kernel_init_zgetrfsp_gemm( dague_context_t* dague_context )
{
    char *env;
    int i, dindex, nbgpus;
    (void)dague_context;

    nbgpus = dague_active_gpu();
    zgetrfsp_gemm_functions = calloc(nbgpus, sizeof(cuda_zgetrfsp_gemm_t));

    for( i = dindex = 0; i < nbgpus; i++ ) {
        gpu_device_t* gpu_device;
        CUresult status;
        void* fn;
        void* dlh;
        char library_name[FILENAME_MAX];
        char function_name[FILENAME_MAX];

        gpu_device = gpu_enabled_devices[i];
        fn = NULL;

        status = cuCtxPushCurrent( gpu_device->ctx );
        DAGUE_CUDA_CHECK_ERROR( "(INIT) cuCtxPushCurrent ", status, {continue;} );
        int major = gpu_device->major, minor = gpu_device->minor;

    retry_lesser_sm_version:
        snprintf(function_name, FILENAME_MAX, "zgemm_sparse_SM%d%d", major, minor);
        env = getenv("DAGUE_CUCORES_LIB");
        if(NULL == env) {
            snprintf(library_name,  FILENAME_MAX, "libdsparse_cucores_sm%d%d.so",  major, minor);
        }
        else {
            snprintf(library_name,  FILENAME_MAX, "%s", env);
        }

        dlh = dlopen(library_name, RTLD_NOW | RTLD_NODELETE );
        if(NULL == dlh) {
            if(env) ERROR(("Could not find %s library: %s\n"
                           "  It is derived from environment DAGUE_CUCORES_LIB=%s\n"
                           "  To resolve this issue, set this variable to the correct path\n"
                           "    ex: /path/libdsparse_cucores_sm20.so\n"
                           "  Or unset it to use the default GPU kernels\n"
                           , library_name, dlerror(), env));
            DEBUG3(("Could not find %s dynamic library (%s)\n", library_name, dlerror()));
        }
        else {
            fn = dlsym(dlh, function_name);
            dlclose(dlh);
        }

        /* Couldn't load from dynamic libs, try static */
        if(NULL == fn) {
            DEBUG3(("No dynamic function %s found, loading from statically linked\n", function_name));
            dlh = dlopen(NULL, RTLD_NOW | RTLD_NODELETE);
            if(NULL == dlh) ERROR(("Error parsing static libs: %s\n", dlerror()));
            fn = dlsym(dlh, function_name);
            if(env && fn) WARNING(("Internal static function %s used (because library %s didn't loaded correctly)\n", function_name, library_name));
            dlclose(dlh);
        }

        /* Still not found?? skip this GPU */
        if(NULL == fn) {
            STATUS(("No function %s found for GPU %d\n", function_name, i));
            if(minor > 0) {
                minor--;
                goto retry_lesser_sm_version;
            } else
            {
                major--; minor = 9;
                if(major > 0) goto retry_lesser_sm_version;
            }
            status = cuCtxPopCurrent(NULL);
            continue;
        }

        status = cuCtxPopCurrent(NULL);
        DAGUE_CUDA_CHECK_ERROR( "(INIT) cuCtxPopCurrent ", status,
                                {continue;} );

        gpu_device->index = (uint8_t)dindex;
        zgetrfsp_gemm_functions[dindex] = (cuda_zgetrfsp_gemm_t)fn;
        gpu_enabled_devices[dindex++] = gpu_device;
    }

    /* Update the number of GPUs available */
    dague_data_enable_gpu( dindex );
    ndevices = dindex;
    assert( nbgpus == ndevices ); /* the code for when some devices can load some functions but not others is not yet correct, blanket protection against this */

    return 0;
}

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
gpu_kernel_push_zgetrfsp_gemm( gpu_device_t        *gpu_device,
                               dague_gpu_context_t *gpu_task,
                               CUstream             stream )
{
    int ret, move_data_count = 0;
    int sizeloc[MAX_PARAM_COUNT];
    dague_execution_context_t  *this_task = gpu_task->ec;
    dague_zgetrfsp_gemm_args_t *args = (dague_zgetrfsp_gemm_args_t*)gpu_task;

    moesi_get_master(args->ddesc->moesi_map, KERNEL_KEY( args->ddesc, 0, args->cblknum ),
                     &(this_task->data[0].moesi_master));
    if( NULL == (this_task->data[0].moesi_master)->device_copies[gpu_device->index])
        move_data_count++;

    moesi_get_master(args->ddesc->moesi_map, KERNEL_KEY( args->ddesc, 1, args->cblknum ),
                     &(this_task->data[1].moesi_master));
    if( NULL == (this_task->data[1].moesi_master)->device_copies[gpu_device->index])
        move_data_count++;

    moesi_get_master(args->ddesc->moesi_map, KERNEL_KEY( args->ddesc, 0, args->fcblknum ),
                     &(this_task->data[2].moesi_master));
    if( NULL == (this_task->data[2].moesi_master)->device_copies[gpu_device->index])
        move_data_count++;

    moesi_get_master(args->ddesc->moesi_map, KERNEL_KEY( args->ddesc, 1, args->fcblknum ),
                     &(this_task->data[3].moesi_master));
    if( NULL == (this_task->data[3].moesi_master)->device_copies[gpu_device->index])
        move_data_count++;

    sizeloc[0] = args->sizecblk;
    sizeloc[1] = args->sizecblk;
    sizeloc[2] = args->sizefcblk;
    sizeloc[3] = args->sizefcblk;
    if( 0 != move_data_count ) { /* Try to reserve enough room for all data */
        ret = dague_gpu_data_reserve_device_space( gpu_device,
                                                   this_task,
                                                   sizeloc,
                                                   move_data_count );
        if( ret < 0 ) {
            goto release_and_return_error;
        }
    }

    assert( NULL != gpu_elem_obtain_from_master(this_task->data[0].moesi_master, gpu_device->index) );
    assert( NULL != gpu_elem_obtain_from_master(this_task->data[1].moesi_master, gpu_device->index) );
    assert( NULL != gpu_elem_obtain_from_master(this_task->data[2].moesi_master, gpu_device->index) );
    assert( NULL != gpu_elem_obtain_from_master(this_task->data[3].moesi_master, gpu_device->index) );

#if defined(DAGUE_PROF_TRACE)
    if( dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_DATA_IN )
        dague_profiling_trace( gpu_device->profiling, dague_cuda_movein_key_start,
                               (unsigned long)this_task, this_task->dague_object->object_id,
                               NULL );
#endif  /* defined(DAGUE_PROF_TRACE) */

    DEBUG3(("GPU[%1d]:\tIN Data of %s(%d, %d, %d, %d, %d) on GPU\n",
            gpu_device->device_index, this_task->function->in[0]->name,
            args->bloknum, args->fcblknum, args->cblknum, args->prev, args->next));
    ret = dague_gpu_data_stage_in( gpu_device, this_task->function->in[0]->access_type,
                                   &(this_task->data[0]), sizeloc[0], stream );
    if( ret < 0 ) {
        goto release_and_return_error;
    }

    DEBUG3(("GPU[%1d]:\tIN Data of %s(%d, %d, %d, %d, %d) on GPU\n",
            gpu_device->device_index, this_task->function->in[1]->name,
            args->bloknum, args->fcblknum, args->cblknum, args->prev, args->next));
    ret = dague_gpu_data_stage_in( gpu_device, this_task->function->in[1]->access_type,
                                   &(this_task->data[1]), sizeloc[1], stream );
    if( ret < 0 ) {
        goto release_and_return_error;
    }

    DEBUG3(("GPU[%1d]:\tIN Data of %s(%d, %d, %d, %d, %d) on GPU\n",
            gpu_device->device_index, this_task->function->in[2]->name,
            args->bloknum, args->fcblknum, args->cblknum, args->prev, args->next));
    ret = dague_gpu_data_stage_in( gpu_device, this_task->function->in[2]->access_type,
                                   &(this_task->data[2]), sizeloc[2], stream );
    if( ret < 0 ) {
        goto release_and_return_error;
    }

    DEBUG3(("GPU[%1d]:\tIN Data of %s(%d, %d, %d, %d, %d) on GPU\n",
            gpu_device->device_index, this_task->function->in[3]->name,
            args->bloknum, args->fcblknum, args->cblknum, args->prev, args->next));
    ret = dague_gpu_data_stage_in( gpu_device, this_task->function->in[3]->access_type,
                                   &(this_task->data[3]), sizeloc[3], stream );
    if( ret < 0 ) {
        goto release_and_return_error;
    }

  release_and_return_error:
    return ret;
}

static inline int
gpu_kernel_submit_zgetrfsp_gemm( gpu_device_t        *gpu_device,
                                 dague_gpu_context_t *gpu_task,
                                 CUstream             stream )
{
    dague_execution_context_t  *this_task = gpu_task->ec;
    dague_zgetrfsp_gemm_args_t *args = (dague_zgetrfsp_gemm_args_t*)gpu_task;
    gpu_elem_t *gpu_elem_Al = NULL, *gpu_elem_Cl = NULL;
    gpu_elem_t *gpu_elem_Au = NULL, *gpu_elem_Cu = NULL;
    CUdeviceptr d_Al, d_Cl;
    CUdeviceptr d_Au, d_Cu;
#if defined(DAGUE_DEBUG_VERBOSE2)
    char tmp[MAX_TASK_STRLEN];
#endif
#if defined(PRECISION_z) || defined(PRECISION_c)
    cuDoubleComplex alpha = make_cuDoubleComplex(-1.0, 0.0);
    cuDoubleComplex beta  = make_cuDoubleComplex( 1.0, 0.0);
#else
    double alpha = -1.0;
    double beta  = 1.0;
#endif

    cuda_zgetrfsp_gemm_t cuda_zgetrfsp_gemm = zgetrfsp_gemm_functions[gpu_device->index];

    gpu_elem_Al = gpu_elem_obtain_from_master(this_task->data[0].moesi_master, gpu_device->index);
    gpu_elem_Au = gpu_elem_obtain_from_master(this_task->data[1].moesi_master, gpu_device->index);
    gpu_elem_Cl = gpu_elem_obtain_from_master(this_task->data[2].moesi_master, gpu_device->index);
    gpu_elem_Cu = gpu_elem_obtain_from_master(this_task->data[3].moesi_master, gpu_device->index);
    d_Al = gpu_elem_Al->gpu_mem_ptr;
    d_Au = gpu_elem_Au->gpu_mem_ptr;
    d_Cl = gpu_elem_Cl->gpu_mem_ptr;
    d_Cu = gpu_elem_Cu->gpu_mem_ptr;

    DEBUG2(( "GPU[%1d]:\tEnqueue on device %s priority %d\n", gpu_device->device_index,
             dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, this_task),
             this_task->priority ));

#if defined(DAGUE_PROF_TRACE)
    gpu_kernel_profile( gpu_device, gpu_task );
#endif  /* defined(DAGUE_PROF_TRACE) */

    {
        const sparse_matrix_desc_t *sdesc = (sparse_matrix_desc_t*)(args->ddesc);
        SolverMatrix *datacode = &(sdesc->pastix_data->solvmatr);
        int bloknum  = args->bloknum;
        int fcblknum = args->fcblknum;
        int cblknum  = args->cblknum;

        int bloknbr = symbol_get_cblk_bloknbr(datacode, cblknum);
        int fblknbr = symbol_get_cblk_bloknbr(datacode, fcblknum);
        int m = args->M;
        int n = args->N;
        int k = args->K;
        CUdeviceptr d_blok;
        CUdeviceptr d_blocktab, d_fbloktab;

        d_blocktab = sdesc->d_blocktab[gpu_device->index] + 2 * bloknum                * sizeof(my_tmp_int_t);
        d_fbloktab = sdesc->d_blocktab[gpu_device->index] + 2 * SYMB_BLOKNUM(fcblknum) * sizeof(my_tmp_int_t);

        d_blok = d_Al + symbol_get_blok_coefind(datacode, bloknum)*sizeof(dague_complex64_t);
        d_Cl   = d_Cl + sizeof(dague_complex64_t) * symbol_get_cblk_stride(datacode, fcblknum) *
            (symbol_get_blok_frownum(datacode, bloknum) - symbol_get_cblk_fcolnum(datacode, fcblknum));

        cuda_zgetrfsp_gemm('N', 'C', m, n, k,
                           alpha, (cuDoubleComplex*)d_blok, symbol_get_cblk_stride(datacode, cblknum),
                                  (cuDoubleComplex*)d_blok, symbol_get_cblk_stride(datacode, cblknum),
                           beta,  (cuDoubleComplex*)d_Cl,   symbol_get_cblk_stride(datacode, fcblknum),
                           bloknbr, (const int *)d_blocktab,
                           fblknbr, (const int *)d_fbloktab,
                           stream );

        d_blok = d_Au + symbol_get_blok_coefind(datacode, bloknum)*sizeof(dague_complex64_t);
        d_Cu   = d_Cu + sizeof(dague_complex64_t) * symbol_get_cblk_stride(datacode, fcblknum) *
            (symbol_get_blok_frownum(datacode, bloknum) - symbol_get_cblk_fcolnum(datacode, fcblknum));

        cuda_zgetrfsp_gemm('N', 'C', m, n, k,
                           alpha, (cuDoubleComplex*)d_blok, symbol_get_cblk_stride(datacode, cblknum),
                                  (cuDoubleComplex*)d_blok, symbol_get_cblk_stride(datacode, cblknum),
                           beta,  (cuDoubleComplex*)d_Cu,   symbol_get_cblk_stride(datacode, fcblknum),
                           bloknbr, (const int *)d_blocktab,
                           fblknbr, (const int *)d_fbloktab,
                           stream );
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
gpu_kernel_pop_zgetrfsp_gemm( gpu_device_t        *gpu_device,
                              dague_gpu_context_t *gpu_task,
                              CUstream             stream )
{
    dague_execution_context_t  *this_task = gpu_task->ec;
    dague_zgetrfsp_gemm_args_t *args = (dague_zgetrfsp_gemm_args_t*)gpu_task;
    gpu_elem_t *gpu_elem = NULL;
    int return_code = 0, how_many = 0, i;
    cudaError_t status;

    /* Generic */
    for( i = 0; NULL != this_task->function->in[i]; i++ ) {
        gpu_elem = gpu_elem_obtain_from_master(this_task->data[i].moesi_master, gpu_device->index);
        if( this_task->function->in[i]->access_type & ACCESS_READ ) {
            gpu_elem->moesi.readers--; assert(gpu_elem->moesi.readers >= 0);
            if( (0 == gpu_elem->moesi.readers) &&
                !(this_task->function->in[i]->access_type & ACCESS_WRITE) ) {
                dague_list_item_ring_chop((dague_list_item_t*)gpu_elem);
                DAGUE_LIST_ITEM_CONSTRUCT(gpu_elem); /* TODO: singleton instead? */
                dague_ulist_fifo_push(gpu_device->gpu_mem_lru, (dague_list_item_t*)gpu_elem);
            }
        }
        if( this_task->function->in[i]->access_type & ACCESS_WRITE ) {
            gpu_elem = gpu_elem_obtain_from_master(this_task->data[i].moesi_master, gpu_device->index);

            /* Stage the transfer of the data back to main memory */
            gpu_device->required_data_out += args->sizefcblk;
            assert( ((dague_list_item_t*)gpu_elem)->list_next == (dague_list_item_t*)gpu_elem );
            assert( ((dague_list_item_t*)gpu_elem)->list_prev == (dague_list_item_t*)gpu_elem );

            if( args->pushout ) { /* next == 0 */
                DEBUG3(("GPU[%1d]:\tOUT Data of %s key %d\n", gpu_device->device_index,
                        this_task->function->in[i]->name, this_task->data[i].moesi_master->key));
#if defined(DAGUE_PROF_TRACE)
                if( dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_DATA_OUT )
                    dague_profiling_trace( gpu_device->profiling, dague_cuda_moveout_key_start,
                                           (unsigned long)this_task, this_task->dague_object->object_id,
                                           NULL );
#endif  /* defined(DAGUE_PROF_TRACE) */
                /* Move the data back into main memory */
                status = (cudaError_t)cuMemcpyDtoHAsync( ADATA(this_task->data[i].data), gpu_elem->gpu_mem_ptr, args->sizefcblk, stream );
                DAGUE_CUDA_CHECK_ERROR( "cuMemcpyDtoHAsync from device ", status,
                                        { WARNING(("data %s <<%p>> -> <<%p>>\n", this_task->function->in[i]->name,
                                                  (void*)gpu_elem->gpu_mem_ptr, (void*)ADATA(this_task->data[i].data)));
                                          return_code = -2;
                                          goto release_and_return_error;} );
                gpu_device->transferred_data_out += args->sizefcblk; /* TODO: not hardcoded, use datatype size */
                how_many++;
            }
        }
    }

 release_and_return_error:
    return (return_code < 0 ? return_code : how_many);
}

static inline int
gpu_kernel_epilog_zgetrfsp_gemm( gpu_device_t        *gpu_device,
                                 dague_gpu_context_t *gpu_task )
{
    dague_execution_context_t  *this_task = gpu_task->ec;
    dague_zgetrfsp_gemm_args_t *args = (dague_zgetrfsp_gemm_args_t*)gpu_task;
    gpu_elem_t* gpu_elem;
    moesi_master_t* master;
    int i;

    for( i = 0; NULL != (master = this_task->data[i].moesi_master); i++ ) {
        if( !(this_task->function->in[i]->access_type & ACCESS_WRITE) ) continue;

        gpu_elem = gpu_elem_obtain_from_master(master, gpu_device->index);
        assert( MOESI_OWNED == gpu_elem->moesi.coherency_state );
        gpu_elem->moesi.coherency_state = MOESI_SHARED;
        master->version = gpu_elem->moesi.version;
        master->owner_device = -1;

#if defined(DAGUE_PROF_TRACE)
        if( dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_DATA_IN )
            dague_profiling_trace( gpu_device->profiling, dague_cuda_movein_key_end,
                                   (unsigned long)this_task, this_task->dague_object->object_id,
                                   NULL );
#endif  /* defined(DAGUE_PROF_TRACE) */
        if( args->pushout ) {  /* next == 0 */
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

/**
 * This version is based on 4 streams: one for transfers from the memory to
 * the GPU, 2 for kernel executions and one for tranfers from the GPU into
 * the main memory. The synchronization on each stream is based on CUDA events,
 * such an event indicate that a specific epoch of the lifetime of a task has
 * been completed. Each type of stream (in, exec and out) has a pending FIFO,
 * where tasks ready to jump to the respective step are waiting.
 */
int gpu_zgetrfsp_gemm( dague_execution_unit_t* eu_context,
                       dague_execution_context_t* this_task,
                       int pushout,
                       my_tmp_int_t cblknum, my_tmp_int_t bloknum, my_tmp_int_t fcblknum,
                       const sparse_matrix_desc_t *ddesc )
{
    int which_gpu;
    dague_zgetrfsp_gemm_args_t *gpu_task = (dague_zgetrfsp_gemm_args_t*)malloc(sizeof(dague_zgetrfsp_gemm_args_t));
    SolverMatrix *datacode = &(ddesc->pastix_data->solvmatr);

    DAGUE_LIST_ITEM_CONSTRUCT(gpu_task);
    gpu_task->super.ec = this_task;
    gpu_task->pushout  = pushout;
    gpu_task->cblknum  = cblknum;
    gpu_task->bloknum  = bloknum;
    gpu_task->fcblknum = fcblknum;
    gpu_task->M        = symbol_get_cblk_stride(datacode, cblknum) - symbol_get_blok_coefind(datacode, bloknum);
    gpu_task->N        = symbol_get_blok_height(datacode, bloknum);
    gpu_task->K        = symbol_get_cblk_width( datacode, cblknum);
    gpu_task->sizecblk = sizeof(dague_complex64_t) * (size_t)symbol_get_cblk_stride(datacode, cblknum)  * symbol_get_cblk_width(datacode, cblknum);
    gpu_task->sizefcblk= sizeof(dague_complex64_t) * (size_t)symbol_get_cblk_stride(datacode, fcblknum) * symbol_get_cblk_width(datacode, fcblknum);
    gpu_task->ddesc    = (dague_ddesc_t*)ddesc;

    /* We always schedule the task on the GPU owning the C tile. */
    which_gpu = moesi_locate_device_with_valid_copy( ddesc->super.moesi_map, KERNEL_KEY( ddesc, 0, fcblknum ) );
    if( which_gpu < 0 ) {  /* this is the first time we see this tile.
                            * Let's decide which GPU will work on it. */
        int best_index = -1;  /* cores */
        /* There are 3 types of GEMMs kernels: the ones waiting on the
         * execution contextes queues to be investigated, the current one
         * which is investigated for execution on the context of the current
         * execution context, and the ones already queued on the GPUs. The
         * decision regarding the status of the current GEMM should be therefore
         * based only on the number of pending tasks on the GPUs.
         */
        float weight, best_weight = device_load[0] + device_weight[0];
        for( which_gpu = 0; which_gpu < ndevices; which_gpu++ ) {
            weight = device_load[which_gpu+1] + device_weight[which_gpu+1];
            if( best_weight > weight ) {
                best_index = which_gpu;
                best_weight = weight;
            }
        }
        if( best_index == -1 ) {
            dague_atomic_inc_32b( &dague_cpu_counter );
            return -99;
        }
        which_gpu = best_index;
    }
    /* Update the load of the selected GPU */
    device_load[which_gpu+1] += device_weight[which_gpu+1];

    return gpu_kernel_scheduler_zgetrfsp_gemm( eu_context, (dague_gpu_context_t*)gpu_task, which_gpu );
}
