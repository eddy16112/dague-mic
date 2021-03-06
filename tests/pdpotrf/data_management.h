/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#ifndef __DATA_MANAGEMENT__
#define __DATA_MANAGEMENT__


#include "dague.h"
#if defined(USE_MPI)
#include <mpi.h>
#endif  /* defined(USE_MPI) */
/*
 * General distribution of data. Suppose exists a matrix in process of mpi rank 0
 */


/*******************************************************************
 * distributed data structure and basic functionalities
 *******************************************************************/

/* structure equivalent to PLASMA_desc, but for distributed matrix data
 */
typedef struct dague_desc_t {
    void *mat;          // pointer to the beginning of the matrix
    PLASMA_enum dtyp;   // precision of the matrix
    int mb;             // number of rows in a tile
    int nb;             // number of columns in a tile
    int bsiz;           // size in elements including padding
    int lm;             // number of rows of the entire matrix
    int ln;             // number of columns of the entire matrix
    int lmt;            // number of tile rows of the entire matrix - derived parameter
    int lnt;            // number of tile columns of the entire matrix - derived parameter
    int i;              // row index to the beginning of the submatrix
    int j;              // column indes to the beginning of the submatrix
    int m;              // number of rows of the submatrix
    int n;              // number of columns of the submatrix
    int mt;             // number of tile rows of the submatrix - derived parameter
    int nt;             // number of tile columns of the submatrix - derived parameter
    int nrst;           // max number of tile rows in a super-tile
    int ncst;           // max number of tile columns in a super tiles
    int mpi_rank;       // well... mpi rank...
    int GRIDrows;       // number of processes rows in the process grid
    int GRIDcols;       // number of processes cols in the process grid
    int cores;          // number of cores used for computation per node
    int nodes;          // number of nodes involved in the computation
    int colRANK;        // process column rank in the process grid - derived parameter
    int rowRANK;        // process row rank in the process grid - derived parameter
    int nb_elem_r;      // number of row tiles  handled by this process
    int nb_elem_c;      // number of column tiles handled by this process
} DAGUE_desc;

typedef struct tile_coordinate{
    int row;
    int col;
} tile_coordinate_t;


/************************************************
 *   mpi ranks distribution in the process grid
 *   -----------------
 *   | 0 | 1 | 2 | 3 |
 *   |---------------|
 *   | 4 | 5 | 6 | 7 |
 *   -----------------
 ************************************************/

// #define A(m,n) &((double*)descA.mat)[descA.bsiz*(m)+descA.bsiz*descA.lmt*(n)]
/* initialize Ddesc from Pdesc */
int dague_desc_init(const PLASMA_desc * Pdesc, DAGUE_desc * Ddesc);

/* allocate the inner storage */
int dague_desc_workspace_allocate( DAGUE_desc * Ddesc );


/* initialize and bcast Ddesc from Pdesc */
int dague_desc_bcast(const PLASMA_desc * Pdesc, DAGUE_desc *Ddesc);

/* computing the mpi process rank that should handle tile A(m,n) */

int dague_get_rank_for_tile(DAGUE_desc * Ddesc, int m, int n);

#if 0
/* get a pointer to a specific LOCAL tile */
static inline void* dague_get_local_tile(DAGUE_desc* Ddesc, int m, int n)
{    
    return &((double*)Ddesc->mat)[Ddesc->bsiz * (m + Ddesc->lmt * n)];
}
#else 
#define dague_get_local_tile(d, m, n) dague_get_local_tile_s(d, m, n)
#endif

/* get a pointer to a specific tile 
 * if the tile is remote, it is downloaded first */
/* void * dague_get_tile(DAGUE_desc * Ddesc, int m, int n); */

/* get a pointer to a specific LOCAL tile, with supertile management. */
void * dague_get_local_tile_s(DAGUE_desc * Ddesc, int m, int n);


/* set new data to tile
 * return 0 if success, >0 if not
 */
int dague_set_tile(DAGUE_desc * Ddesc, int m, int n, void * buff);

/* translate a position in the matrix buffer to the tile it belongs */
void pos_to_coordinate(DAGUE_desc * Ddesc, double * position, tile_coordinate_t * tile);


/************************************************************************
 * matrix distribution (single tiled matrix to distributed tiled matrix)
 ************************************************************************/

/* distribute the matrix to the different mpi ranks 
 * 
 */
int distribute_data(PLASMA_desc * Pdesc , DAGUE_desc * Ddesc);

/* regroup the distributed tiles to the rank 0 */
/* Pdesc is NULL except on rank 0 */
int gather_data(PLASMA_desc * Pdesc, DAGUE_desc *Ddesc);

/*********************************************************************
 *  matrix generation, Lapack/Plasma format conversion, shared memory
 ********************************************************************/

/* generate a random matrix  */
int generate_matrix(int N, double * A1, double * A2, double * B1, double * B2, int LDA, int NRHS, int LDB);

/* Convert Lapack format to Plasma tiled description format (shared memory) */
int tiling(PLASMA_enum * uplo, int N, double *A, int LDA, int NRHS, PLASMA_desc * descA);

/* Convert Plasma tiled description format to Lapack format (shared memory) */
int untiling(PLASMA_enum * uplo, int N, double *A, int LDA, PLASMA_desc * descA);

/**********************************************************************
 * Distributed matrix generation
 **********************************************************************/

/* set values of the DAGUE_desc structure and allocate memory for matrix data
 */
int dague_description_init(DAGUE_desc * Ddesc, int LDA, int LDB, int NRHS, PLASMA_enum uplo);

/* affecting the complete local view of a distributed matrix with random values */
int rand_dist_matrix(DAGUE_desc * Ddesc);


/*********************************************************************
 * Debugging functions
 *********************************************************************/
/* debugging print of blocks */
void data_dist_verif(PLASMA_desc * Pdesc, DAGUE_desc * Ddesc );
int data_dump(DAGUE_desc * Ddesc);
int plasma_dump(PLASMA_desc * Pdesc);
int compare_matrices(DAGUE_desc * A, DAGUE_desc * B, double precision);
int compare_distributed_tiles(DAGUE_desc * A, DAGUE_desc * B, int row, int col, double precision);
int compare_plasma_matrices(PLASMA_desc * A, PLASMA_desc * B, double precision);


#endif
