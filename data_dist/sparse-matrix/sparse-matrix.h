/**
 *
 * @file sparse-matrix.h
 *
 * @author Mathieu Faverge
 * @date 2011-03-01
 * @precisions normal z -> c d s
 *
 **/

#ifndef _SPARSE_MATRIX_H_ 
#define _SPARSE_MATRIX_H_ 

#include <stdint.h>
#if defined(HAVE_CUDA)
#include <cuda.h>
#endif

#include "dague_config.h"
#include "data_distribution.h"
#include "data_dist/matrix/precision.h"

typedef int64_t dague_int_t;

#if !defined(PASTIX_STR_H) && !defined(_PASTIX_H_)
struct pastix_data_t;
typedef struct pastix_data_t pastix_data_t;
#endif

/* WARNING: Has to follow Pastix enum API_FACT */
#define DSPARSE_LLT   0
#define DSPARSE_LDLT  1 
#define DSPARSE_LU    2
#define DSPARSE_LDLTH 3

enum spmtx_type {
    spmtx_RealFloat     = 0,  /**< float          */
    spmtx_RealDouble    = 1,  /**< double         */
    spmtx_ComplexFloat  = 2,  /**< complex float  */
    spmtx_ComplexDouble = 3,  /**< complex double */
};

static inline int sparse_matrix_size_of(enum spmtx_type type)
{
    switch ( type ) {
    case spmtx_RealFloat:
        return sizeof(float);
    case spmtx_RealDouble:
        return sizeof(double);
    case spmtx_ComplexFloat:
        return sizeof(dague_complex32_t);
    case spmtx_ComplexDouble:
        return sizeof(dague_complex64_t);
    default:
        return sizeof(float);
    }
}

typedef struct dague_symbol_blok {
    dague_int_t frownum;                 /**< First row index                       */
    dague_int_t lrownum;                 /**< Last row index (inclusive)            */
    dague_int_t coefind;                 /**< Index of the first byte of this block */
} dague_symbol_blok_t;

typedef struct dague_symbol_cblk {
    dague_int_t    fcolnum;              /**< First column index                      */
    dague_int_t    lcolnum;              /**< Last column index (inclusive)           */
    dague_int_t    bloknum;              /**< First block in column (diagonal)        */
    dague_int_t    stride;               /**< Leading dimension of the column data    */
    void          *cblkptr;              /**< Pointer to the column data (Lower part) */
    void          *ucblkptr;             /**< Pointer to the column data (Upper part) */
} dague_symbol_cblk_t;

typedef struct dague_symbol_matrix {
    dague_int_t cblknbr;                    /**< Number of column blocks           */
    dague_int_t bloknbr;                    /**< Number of blocks                  */
    dague_int_t nodenbr;                    /**< Number of nodes in matrix         */
    dague_symbol_cblk_t * restrict cblktab; /**< Array of column blocks (size:cblknbr+1,based to 0) */
    dague_symbol_blok_t * restrict bloktab; /**< Array of blocks (size:bloknbr,based to 0)          */
} dague_symbol_matrix_t;

typedef struct sparse_matrix_desc_t {
    dague_ddesc_t         super;
    enum spmtx_type       mtype;   /* Precision of the matrix             */
    int                   typesze; /* Type size                           */
    //    dague_symbol_matrix_t symbmtx; /* Pointer to symbol matrix structure  */
#if defined(HAVE_CUDA)
    CUdeviceptr          *d_blocktab;
#endif
    pastix_data_t        *pastix_data;
    //    size_t *cblksize;
} sparse_matrix_desc_t;

typedef struct sparse_vector_desc_t {
    dague_ddesc_t         super;
    enum spmtx_type       mtype;   /* Precision of the matrix             */
    int                   typesze; /* Type size                           */
    //    dague_symbol_matrix_t symbmtx; /* Pointer to symbol matrix structure  */
    pastix_data_t        *pastix_data;
} sparse_vector_desc_t;


typedef struct sparse_context_s {
    int          format;     /* Matrix file format                         */
    int          factotype;
    int          coresnbr;   /* Number of cores to use for Pastix          */
    int          verbose;    /* Level of verbose                           */
    char        *matrixname; /* Filename to get the matrix                 */
    char        *rhsname;    /* Filename to get the matrix                 */
    char        *ordername;  /* Filename where the ordering is stored      */
    char        *symbname;   /* Filename where the symbol matrix is stored */
    char        *type;       /* Type of the matrix                         */
    char        *rhstype;    /* Type of the RHS                            */
    dague_int_t  n;          /* Number of unknowns/columns/rows            */
    dague_int_t  nnz;        /* Number of non-zero values in the input matrix */
    dague_int_t *colptr;     /* Vector of size N+1 storing the starting point of each column in the array rows */
    dague_int_t *rows;       /* Indices of the rows present in each column */
    void        *values;     /* Values of the matrix                       */
    void        *rhs;        /* Right Hand Side                            */ 
    dague_int_t *permtab;    /* vector of permutation                      */
    dague_int_t *peritab;    /* vector of inverse permutation              */
    dague_int_t  iparm[IPARM_SIZE];
    double       dparm[DPARM_SIZE];
    sparse_matrix_desc_t *desc; /* Pointer to symbol matrix structure */    
    sparse_vector_desc_t *rhsdesc; /* Pointer to symbol matrix structure */    
} sparse_context_t;

void sparse_matrix_init( sparse_matrix_desc_t *desc, 
                         enum spmtx_type mtype, 
                         int nodes, int cores, int myrank);
void sparse_matrix_destroy( sparse_matrix_desc_t *desc );

void sparse_vector_init( sparse_vector_desc_t *desc, 
                         enum spmtx_type mtype, 
                         int nodes, int cores, int myrank);
void sparse_vector_destroy( sparse_vector_desc_t *desc );


dague_int_t sparse_matrix_get_lcblknum(sparse_matrix_desc_t *mat, dague_int_t bloknum );
dague_int_t sparse_matrix_get_listptr_prev(sparse_matrix_desc_t *mat, dague_int_t bloknum, dague_int_t fcblknum );
dague_int_t sparse_matrix_get_listptr_next(sparse_matrix_desc_t *mat, dague_int_t bloknum, dague_int_t fcblknum );

double sparse_matrix_zrdmtx( sparse_context_t *dspctxt );
double sparse_matrix_crdmtx( sparse_context_t *dspctxt );
double sparse_matrix_drdmtx( sparse_context_t *dspctxt );
double sparse_matrix_srdmtx( sparse_context_t *dspctxt );

void sparse_matrix_zcheck( sparse_context_t *dspctxt );
void sparse_matrix_ccheck( sparse_context_t *dspctxt );
void sparse_matrix_dcheck( sparse_context_t *dspctxt );
void sparse_matrix_scheck( sparse_context_t *dspctxt );

void sparse_matrix_zclean( sparse_context_t *dspctxt );
void sparse_matrix_cclean( sparse_context_t *dspctxt );
void sparse_matrix_dclean( sparse_context_t *dspctxt );
void sparse_matrix_sclean( sparse_context_t *dspctxt );

void sparse_vector_zinit( sparse_context_t *dspctxt );
void sparse_vector_cinit( sparse_context_t *dspctxt );
void sparse_vector_dinit( sparse_context_t *dspctxt );
void sparse_vector_sinit( sparse_context_t *dspctxt );
void sparse_vector_zfinalize( sparse_context_t *dspctxt );
void sparse_vector_cfinalize( sparse_context_t *dspctxt );
void sparse_vector_dfinalize( sparse_context_t *dspctxt );
void sparse_vector_sfinalize( sparse_context_t *dspctxt );

#endif /* _SPARSE_MATRIX_H_ */
