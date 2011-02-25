/*
 * Copyright (c) 2011      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/*@FILE
 * dague tiled sparse shared memory matrix: TSSM
 */

#ifndef _SPARSE_SHM_MATRIX_H_ 
#define _SPARSE_SHM_MATRIX_H_ 

#include <stdarg.h>
#include "dague_config.h"
#include "linked_list.h"
#include "data_distribution.h"
#include "data_dist/sparse-matrix/si-to-tssm.h"

#define TILE_STATUS_UNPACKING  (1<<0)
#define TILE_STATUS_PACKING    (1<<1)
#define TILE_STATUS_DIRTY      (1<<2)

struct dague_tssm_desc;

typedef struct dague_tssm_tile_entry {
    dague_list_item_t       super;
    dague_linked_list_t    *current_list;
    struct dague_tssm_desc *desc;

    /* Those are values written only by create_tile: */
    struct dague_tssm_data_map *packed_ptr;
    unsigned int m;
    unsigned int n;

    /* This lock to be used only for very short times:
     *   never lock to pack / unpack the tile
     * Only purpose: ensure consistency between writer, nbreaders, status, and
     *   value of the tile pointer (not content of the tile pointer).
     */
    volatile uint32_t lock;

    /* -1 == writer -> no current writer. i.e. only readers (or none)
     *  0 <= writer -> thread id of the current writer, i.e. no readers
     */
    int32_t  writer;
    uint32_t nbreaders;
    uint32_t status;

    void    *tile;
    int32_t  tile_owner; /**< Thread rank of the thread that allocated the tile */
} dague_tssm_tile_entry_t;

typedef struct dague_tssm_desc {
    dague_ddesc_t super;

    /* Write once values */
    unsigned int mt;
    unsigned int nt;
    unsigned int mb;
    unsigned int nb;
    size_t data_size;

    dague_tssm_tile_entry_t **mesh; /* nt x mt LAPACK style storage */
} dague_tssm_desc_t;

/**
 * create a tile at coordinated n, m in mesh, of size 
 * tile_n x tile_m elements of size data_size
 * with a packed representation defined by packed_ptr
 * @return 0 if success
 * @return non zero otherwise
 */
int dague_tssm_mesh_create_tile(dague_tssm_desc_t *mesh, uint64_t m, uint64_t n, 
                                uint32_t mb, uint32_t nb, 
                                dague_tssm_data_map_t *packed_ptr);
void dague_tssm_init(int nbthreads);
void dague_tssm_thread_init(int threadid, int nbtilesperthread, size_t tile_size);

dague_ddesc_t *dague_tssm_create_matrix(uint64_t mt, uint64_t nt, uint32_t mb, uint32_t nb,
                                        size_t data_size, uint32_t cores, dague_sparse_input_symbol_matrix_t *sm);

#endif /* _SPARSE_SHM_MATRIX_H_ */

