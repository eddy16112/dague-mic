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
#include "data_dist/matrix/matrix.h"
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
    uint64_t m;
    uint64_t n;

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
    tiled_matrix_desc_t super;

    dague_tssm_tile_entry_t **mesh; /* nt x mt LAPACK style storage */
} dague_tssm_desc_t;


/**
 * Initialization function. Must be called once only, and always before any other dague_tssm
 * function.
 *
 * @param [IN] nbcores: the number of cores used (assuming that cores 0 to nbcores-1
 *                      are going to be used.
 * @param [IN] tile_size: number of bytes of the tile size. Large enough to store
 *                        any tile explicitely. Watch out to give enough space even
 *                        if multiple matrices of different mt, nt, mb, nb, data types
 *                        are created.
 * @param [IN] tilespercore: number of tiles to allocate per core.
 */
void dague_tssm_init( uint32_t nbcores, size_t tile_size, uint32_t nbtilespercore );

/**
 * main entry point: initializes an internal representation of the matrix desc
 * on a tiled representation of size lm x ln, each tile being of size mb x nb
 * @param desc matrix description structure, already allocated, that will be initialize
 * @param cores number of cores (must match value passed to dague_tssm_init)
 * @param mb number of row in a tile
 * @param nb number of column in a tile
 * @param lm number of rows of the entire matrix
 * @param ln number of column of the entire matrix
 * @param i starting row index for the computation on a submatrix -- must be 0
 * @param j starting column index for the computation on a submatrix -- must be 0
 * @param m number of rows of the entire submatrix -- must be lm
 * @param n numbr of column of the entire submatrix -- must be ln
 * @param sm preallocated / initialized sparse matrix representation
 */
void dague_tssm_matrix_init(dague_tssm_desc_t * desc, enum matrix_type mtype, unsigned int cores, 
                            unsigned int mb, unsigned int nb, 
                            unsigned int lm, unsigned int ln, unsigned int i, unsigned int j, 
                            unsigned int m, unsigned int n, 
                            dague_sparse_input_symbol_matrix_t *sm);

/**
 * Flushes all caches of a matrix representation inside its own packed representation.
 * This updates the values of the sparse_input_symbol_matrix passed as argument to
 * create_matrix.
 *
 * @return the number of errors (should be zero).
 */
int dague_tssm_flush_matrix(dague_ddesc_t *m);

/**
 * create a tile at coordinated n, m in mesh, of size 
 * tile_n x tile_m elements of size data_size
 * with a packed representation defined by packed_ptr
 *
 * Used by the load function itself called by the create_matrix function.
 */
void dague_tssm_mesh_create_tile(dague_tssm_desc_t *mesh, uint64_t m, uint64_t n, 
                                 uint32_t mb, uint32_t nb, 
                                 dague_tssm_data_map_t *packed_ptr);

#endif /* _SPARSE_SHM_MATRIX_H_ */

