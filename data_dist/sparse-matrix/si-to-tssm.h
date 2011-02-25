#ifndef _SI_TO_TSSM_H_
#define _SI_TO_TSSM_H_

#include "dague_config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "data_dist/sparse-matrix/precision.h"
#include "data_dist/sparse-matrix/sparse-input.h"

#if defined(ELEM_IS_INT)
# define ELEM_SIZE sizeof(int)
#elif defined(ELEM_IS_FLOAT)
# define ELEM_SIZE sizeof(float)
#elif defined(ELEM_IS_DOUBLE)
# define ELEM_SIZE sizeof(double)
#elif defined(ELEM_IS_SCOMPLEX)
# define ELEM_SIZE (2*sizeof(float))
#elif defined(ELEM_IS_DCOMPLEX)
# define ELEM_SIZE (2*sizeof(double))
#else
# error "UNKNOWN ELEMENT SIZE"
#endif

/*
 * #: NNZ in the sparse input representation
 * &: NNZ in the sparse input representation that belongs to this tile
 * @: first NNZ in the sparse input representation that belongs to this tile (@ is pointed by ptr)
 *
 ^  --------    -------- ^
 |  |#     |    |######| |
 |  |##    |    |######| |
 |  |###   |    |######| |
 |  |####  |    |######| |
 |  |##### |    --------ldA
 |  |######|    |######| |
ldA --------   ^|@&&&&#| |
 |  |###@&&|^  h|&&&&&#| |
 |  |###&&&||  v|&&&&&#| |
 |  |###&&&|h   -------- v
 |  |###&&&||    <-w->
 |  |###&&&|v
 |  |######|
 v  --------
        <w>
*/
typedef struct dague_tssm_data_map {
    void    *ptr;    /**< points to @ in the above drawing                       */
    uint64_t ldA;    /**< leading dimension of the sparse-input representation   */
    /* h, w, and offset are in the bounds of the tile, supposed to fit uint32_t  */
    uint32_t h;      /**< height of useful rectangle in this sparse-input block  */
    uint32_t w;      /**< width of useful rectangle in this sparse-input block   */
    uint32_t offset; /**< Position of this rectangle inside the tile             */
} dague_tssm_data_map_t;

struct dague_tssm_desc;

int  dague_tssm_sparse_tile_unpack(void *tile_ptr, uint64_t m, uint64_t n, uint64_t mb, uint64_t nb, dague_tssm_data_map_t *map);
void dague_tssm_sparse_tile_pack(void *tile_ptr, uint64_t m, uint64_t n, uint64_t mb, uint64_t nb, dague_tssm_data_map_t *map);
void dague_sparse_input_to_tiles_load(struct dague_tssm_desc *mesh, uint64_t mt, uint64_t nt, uint32_t mb, uint32_t nb, 
                                      dague_sparse_input_symbol_matrix_t *sm);

#endif /* _SI_TO_TSSM_H_ */
