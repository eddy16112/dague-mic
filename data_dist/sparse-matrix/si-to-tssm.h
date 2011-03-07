#ifndef _SI_TO_TSSM_H_
#define _SI_TO_TSSM_H_

#include "dague_config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "data_dist/sparse-matrix/precision.h"
#include "data_dist/sparse-matrix/sparse-input.h"

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
    void        *ptr;    /**< points to @ in the above drawing                       */
    dague_int_t  ldA;    /**< leading dimension of the sparse-input representation   */
    /* h, w, and offset are in the bounds of the tile, supposed to fit uint32_t  */
    uint32_t     h;      /**< height of useful rectangle in this sparse-input block  */
    uint32_t     w;      /**< width of useful rectangle in this sparse-input block   */
    uint32_t     offset; /**< Position of this rectangle inside the tile             */
} dague_tssm_data_map_t;

struct dague_tssm_desc;

int  dague_sparse_ztile_unpack(Dague_Complex64_t *tile_ptr, dague_int_t m, dague_int_t n, dague_int_t mb, dague_int_t nb, dague_tssm_data_map_t *map);
void dague_sparse_ztile_pack(Dague_Complex64_t *tile_ptr, dague_int_t m, dague_int_t n, dague_int_t mb, dague_int_t nb, dague_tssm_data_map_t *map);

void dague_sparse_input_to_tiles_load(struct dague_tssm_desc *mesh, dague_int_t mt, dague_int_t nt, uint32_t mb, uint32_t nb, 
                                      dague_sparse_input_symbol_matrix_t *sm);

#endif /* _SI_TO_TSSM_H_ */
