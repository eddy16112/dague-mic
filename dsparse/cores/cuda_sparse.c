/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
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

#include "cuda_sparse.h"

int sparse_register_bloktab( dague_context_t* dague_context, 
                             sparse_matrix_desc_t *sparseA )
{
    gpu_device_t* gpu_device;
    CUresult status;
    int i, ndevices;
    my_tmp_int_t  iterblock;
    my_tmp_int_t *blocktab;
    size_t        blocktab_size;
    SolverMatrix *datacode = &(sparseA->pastix_data->solvmatr);
    (void)dague_context;

    ndevices = dague_active_gpu();
    if ( ndevices <= 0 )
        return 0;

    blocktab_size = 2 * (SYMB_BLOKNBR) * sizeof(my_tmp_int_t);
    blocktab = (my_tmp_int_t*)malloc( blocktab_size );
    
    for (iterblock = 0; iterblock < SYMB_BLOKNBR; iterblock++) {
        blocktab[2*iterblock]   = SYMB_FROWNUM(iterblock);
        blocktab[2*iterblock+1] = SYMB_LROWNUM(iterblock);
    }
    
    sparseA->d_blocktab = (CUdeviceptr *)calloc(ndevices, sizeof(CUdeviceptr));

    for(i = 0; i < ndevices; i++) {
        if( NULL == (gpu_device = gpu_enabled_devices[i]) ) continue;

        status = cuCtxPushCurrent( gpu_device->ctx );
        DAGUE_CUDA_CHECK_ERROR( "(dague_gpu_fini) cuCtxPushCurrent ", status,
                                {continue;} );
        
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
        
        status = cuCtxPopCurrent(NULL);
        DAGUE_CUDA_CHECK_ERROR( "(INIT) cuCtxPopCurrent ", status,
                                {continue;} );
    }

    free(blocktab);

    return SYMB_CBLKNBR;
}

int sparse_unregister_bloktab( dague_context_t* dague_context, 
                             sparse_matrix_desc_t *sparseA )
{
    gpu_device_t* gpu_device;
    CUresult status;
    int i, ndevices;
    (void)dague_context;
    
    ndevices = dague_active_gpu();
    if ( ndevices <= 0 )
        return 0;

    for(i = 0; i < ndevices; i++) {
        if( NULL == (gpu_device = gpu_enabled_devices[i]) ) continue;

        status = cuCtxPushCurrent( gpu_device->ctx );
        DAGUE_CUDA_CHECK_ERROR( "(dague_gpu_fini) cuCtxPushCurrent ", status,
                                {continue;} );
        
        if (sparseA->d_blocktab[i] != 0)
            cuMemFree(sparseA->d_blocktab[i]);

        status = cuCtxPopCurrent(NULL);
        DAGUE_CUDA_CHECK_ERROR( "(INIT) cuCtxPopCurrent ", status,
                                {continue;} );
    }

    free(sparseA->d_blocktab);
    return 0;
}
