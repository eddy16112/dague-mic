/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#ifndef __TWO_DIM_RECTANGLE_CYCLIC_H__
#define __TWO_DIM_RECTANGLE_CYCLIC_H__

#ifdef HAVE_MPI
#include <mpi.h>
#endif /* HAVE_MPI */

#include "dague_config.h"
#include "data_dist/matrix/matrix.h"
#include "data_dist/matrix/grid_2Dcyclic.h"

/*
 * General distribution of data. Suppose exists a matrix in process of mpi rank 0
 */

/*******************************************************************
 * distributed data structure and basic functionalities
 *******************************************************************/

/* structure equivalent to PLASMA_desc, but for distributed matrix data
 */
typedef struct two_dim_block_cyclic {
    tiled_matrix_desc_t super;
    grid_2Dcyclic_t     grid;
    void *mat;      /**< pointer to the beginning of the matrix */
    int nb_elem_r;  /**< number of row of tiles  handled by this process - derived parameter */
    int nb_elem_c;  /**< number of column of tiles handled by this process - derived parameter */
} two_dim_block_cyclic_t;

/************************************************
 *   mpi ranks distribution in the process grid
 *   -----------------
 *   | 0 | 1 | 2 | 3 |
 *   |---------------|
 *   | 4 | 5 | 6 | 7 |
 *   -----------------
 ************************************************/

// #define A(m,n) &((double*)descA.mat)[descA.bsiz*(m)+descA.bsiz*descA.lmt*(n)]

/**
 * Initialize the description of a  2-D block cyclic distributed matrix.
 * @param Ddesc matrix description structure, already allocated, that will be initialize
 * @param nodes number of nodes
 * @param cores number of cores per node
 * @param myrank rank of the local node (as of mpi rank)
 * @param mb number of row in a tile
 * @param nb number of column in a tile
 * @param lm number of rows of the entire matrix
 * @param ln number of column of the entire matrix
 * @param i starting row index for the computation on a submatrix
 * @param j starting column index for the computation on a submatrix
 * @param m number of rows of the entire submatrix
 * @param n numbr of column of the entire submatrix
 * @param nrst number of rows of tiles for block distribution
 * @param ncst number of column of tiles for block distribution
 * @param process_GridRows number of row of processes of the process grid (has to divide nodes)
 */
void two_dim_block_cyclic_init(two_dim_block_cyclic_t * twoDBCdesc,
                               enum matrix_type mtype,
                               enum matrix_storage storage,
                               int nodes, int cores, int myrank,
			       int mb,   int nb,   /* Tile size */
			       int lm,   int ln,   /* Global matrix size (what is stored)*/
                               int i,    int j,    /* Staring point in the global matrix */
			       int m,    int n,    /* Submatrix size (the one concerned by the computation */
                               int nrst, int ncst, /* Super-tiling size */
                               int process_GridRows );

int twoDBC_tolapack( two_dim_block_cyclic_t *Mdesc, void* A, int lda);
int twoDBC_ztolapack(two_dim_block_cyclic_t *Mdesc, Dague_Complex64_t* A, int lda);
int twoDBC_ctolapack(two_dim_block_cyclic_t *Mdesc, Dague_Complex32_t* A, int lda);
int twoDBC_dtolapack(two_dim_block_cyclic_t *Mdesc, double* A, int lda);
int twoDBC_stolapack(two_dim_block_cyclic_t *Mdesc, float* A, int lda);


void two_dim_block_cyclic_supertiled_view( two_dim_block_cyclic_t* target, 
                                           two_dim_block_cyclic_t* origin, 
                                           int rst, int cst );


#ifdef HAVE_MPI

int open_matrix_file(char * filename, MPI_File * handle, MPI_Comm comm);

int close_matrix_file(MPI_File * handle);

#endif /* HAVE_MPI */


#endif /* __TWO_DIM_RECTANGLE_CYCLIC_H__*/
