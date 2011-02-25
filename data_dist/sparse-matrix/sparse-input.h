#ifndef _SPARSE_INPUT_H_
#define _SPARSE_INPUT_H_

#include <stdint.h>

/** Sparse (Matrix) Input Structures */

typedef struct dague_sparse_input_symbol_cblk {
    uint64_t    fcolnum;              /**< First column index                   */
    uint64_t    lcolnum;              /**< Last column index (inclusive)        */
    uint64_t    bloknum;              /**< First block in column (diagonal)     */
    void *      cblkptr;              /**< Pointer to the column data           */
    uint64_t    stride;               /**< Leading dimension of the column data */
} dague_sparse_input_symbol_cblk_t;

typedef struct dague_sparse_input_symbol_blok {
    uint64_t frownum;                 /**< First row index                       */
    uint64_t lrownum;                 /**< Last row index (inclusive)            */
    uint64_t coefind;                 /**< Index of the first byte of this block */
} dague_sparse_input_symbol_blok_t;

typedef struct dague_sparse_input_symbol_matrix {
    uint64_t baseval;                 /**< Base value for numberings         */
    uint64_t cblknbr;                 /**< Number of column blocks           */
    uint64_t bloknbr;                 /**< Number of blocks                  */
    uint64_t nodenbr;                 /**< Number of nodes in matrix         */
    dague_sparse_input_symbol_cblk_t * restrict cblktab; /**< Array of column blocks (size:cblknbr+1,based to 0) */
    dague_sparse_input_symbol_blok_t * restrict bloktab; /**< Array of blocks (size:bloknbr,based to 0)          */
} dague_sparse_input_symbol_matrix_t;

#endif /* _SPARSE_INPUT_H_ */
