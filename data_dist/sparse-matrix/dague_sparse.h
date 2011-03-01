#ifndef _DAGUE_SPARSE_H_
#define _DAGUE_SPARSE_H_

typedef int64_t dague_int_t;

#define SPARSE_LLT   1
#define SPARSE_LDLT  2 
#define SPARSE_LU    3

typedef struct dsp_context_s {
    int         format;     /* Matrix file format                         */
    int         factotype;
    char       *matrixname; /* Filename to get the matrix                 */
    char       *ordername;  /* Filename where the ordering is stored      */
    char       *symbname;   /* Filename where the symbol matrix is stored */
    char       *type;       /* Type of the matrix                         */
    char       *rhstype;    /* Type of the RHS                            */
    dague_int_t n;          /* Number of unknowns/columns/rows            */
    dague_int_t nnz;        /* Number of non-zero values in the input matrix */
    dague_int_t *colptr;    /* Vector of size N+1 storing the starting point of each column in the array rows */
    dague_int_t *rows;      /* Indices of the rows present in each column */
    void        *values;    /* Values of the matrix                       */
    void        *rhs;       /* Right Hand Side                            */ 
    dague_int_t *permtab;   /* vector of permutation                      */
    dague_int_t *peritab;   /* vector of inverse permutation              */
    dague_sparse_input_symbol_matrix_t *symbmtx; /* Pointer to symbol matrix structure */
} dsp_context_t;


#endif /* _DAGUE_SPARSE_H_ */
