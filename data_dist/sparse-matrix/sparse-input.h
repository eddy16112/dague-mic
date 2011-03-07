#ifndef _SPARSE_INPUT_H_
#define _SPARSE_INPUT_H_

#include <stdint.h>

typedef int64_t dague_int_t;

#define SPARSE_LLT   1
#define SPARSE_LDLT  2 
#define SPARSE_LU    3

/** Sparse (Matrix) Input Structures */

typedef struct dague_sparse_input_symbol_cblk {
    dague_int_t    fcolnum;              /**< First column index                   */
    dague_int_t    lcolnum;              /**< Last column index (inclusive)        */
    dague_int_t    bloknum;              /**< First block in column (diagonal)     */
    void          *cblkptr;              /**< Pointer to the column data           */
    dague_int_t    stride;               /**< Leading dimension of the column data */
} dague_sparse_input_symbol_cblk_t;

typedef struct dague_sparse_input_symbol_blok {
    dague_int_t frownum;                 /**< First row index                       */
    dague_int_t lrownum;                 /**< Last row index (inclusive)            */
    dague_int_t coefind;                 /**< Index of the first byte of this block */
} dague_sparse_input_symbol_blok_t;

typedef struct dague_sparse_input_symbol_matrix {
    dague_int_t cblknbr;                 /**< Number of column blocks           */
    dague_int_t bloknbr;                 /**< Number of blocks                  */
    dague_int_t nodenbr;                 /**< Number of nodes in matrix         */
    dague_sparse_input_symbol_cblk_t * restrict cblktab; /**< Array of column blocks (size:cblknbr+1,based to 0) */
    dague_sparse_input_symbol_blok_t * restrict bloktab; /**< Array of blocks (size:bloknbr,based to 0)          */
} dague_sparse_input_symbol_matrix_t;


typedef struct dsp_context_s {
    int          format;     /* Matrix file format                         */
    int          factotype;
    char        *matrixname; /* Filename to get the matrix                 */
    char        *ordername;  /* Filename where the ordering is stored      */
    char        *symbname;   /* Filename where the symbol matrix is stored */
    char        *type;       /* Type of the matrix                         */
    char        *rhstype;    /* Type of the RHS                            */
    dague_int_t  n;          /* Number of unknowns/columns/rows            */
    dague_int_t  nnz;        /* Number of non-zero values in the input matrix */
    dague_int_t *colptr;    /* Vector of size N+1 storing the starting point of each column in the array rows */
    dague_int_t *rows;      /* Indices of the rows present in each column */
    void        *values;    /* Values of the matrix                       */
    void        *rhs;       /* Right Hand Side                            */ 
    dague_int_t *permtab;   /* vector of permutation                      */
    dague_int_t *peritab;   /* vector of inverse permutation              */
    dague_sparse_input_symbol_matrix_t *symbmtx; /* Pointer to symbol matrix structure */
} dsp_context_t;

#endif /* _SPARSE_INPUT_H_ */
