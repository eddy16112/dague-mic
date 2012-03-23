/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _gpu_gemm_h
#define _gpu_gemm_h

#include "dague_config.h"
#include "gpu_data.h"
#include "dague.h"
#include "execution_unit.h"
#include "scheduling.h"
#include "data_distribution.h"
#include "data_dist/sparse-matrix/pastix_internal/pastix_internal.h"
#include "data_dist/sparse-matrix/sparse-matrix.h"

int sparse_gpu_sgemm( dague_execution_unit_t* eu_context,
                      dague_execution_context_t* this_task );

/****************************************************
 ** GPU-DATA Specific Starts Here **
 ****************************************************/

typedef struct _memory_elem memory_elem_t;
typedef struct _gpu_elem gpu_elem_t;

struct _gpu_elem {
    dague_list_item_t item;
    int lock;
    int type;
    CUdeviceptr gpu_mem;
    memory_elem_t* memory_elem;
    int gpu_version;
};
 	
struct _memory_elem {
    int memory_version;
    int readers;
    int writer;
    int cblk;
    void* memory;
    gpu_elem_t* gpu_elems[1];
};

typedef struct __dague_gpu_data_map {
    sparse_matrix_desc_t*  tiled_matrix;
    memory_elem_t** data_map;
} dague_gpu_data_map_t;


typedef enum {
    DAGUE_READ,
    DAGUE_WRITE
} dague_data_usage_type_t;

#include "data_dist/matrix/matrix.h"

int sparse_sgemm_cuda_init( dague_context_t* context, sparse_matrix_desc_t *tileA );
int sparse_sgemm_cuda_fini( dague_context_t* dague_context );

int sparse_sgemm_cuda_ndevices(void);

int sparse_gpu_data_map_init( gpu_device_t* gpu_device,
                              sparse_matrix_desc_t* data,
                              dague_gpu_data_map_t* gpu_map);
int sparse_gpu_data_map_fini( dague_gpu_data_map_t* gpu_map );


int sparse_gpu_mark_data_usage( dague_gpu_data_map_t* gpu_map, int type, int cblk);

int sparse_gpu_data_cblk_write_owner( dague_gpu_data_map_t* gpu_map, 
                                      int cblk );

int sparse_gpu_data_get_cblk( dague_gpu_data_map_t* gpu_map,
                              int cblk,
                              memory_elem_t **pmem_elem );
int sparse_gpu_data_is_on_gpu( gpu_device_t* gpu_device,
                               dague_gpu_data_map_t* gpu_map,
                               int type,
                               int cblk,
                               gpu_elem_t **pgpu_elem);


void core_spotrfsp1d_gemm(dague_int_t cblknum,
                          dague_int_t bloknum,
                          dague_int_t fcblknum,
                          float *L,
                          float *C,
                          float *work,
                          SolverMatrix *datacode);


#endif
