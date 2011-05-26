/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#ifndef __GRID_2DCYCLIC_H__
#define __GRID_2DCYCLIC_H__

#ifdef HAVE_MPI
#include <mpi.h>
#endif /* HAVE_MPI */

#include "dague_config.h"

/*******************************************************************
 * 2D (or 1D) cyclic distribution 
 *******************************************************************/

/* Placeholder for all relevant 2D distribution parameters */
typedef struct grid_2Dcyclic {
    unsigned int rank;       /**< Sequential rank of this processor */
    unsigned int rows;       /**< number of processes rows in the process grid */
    unsigned int cols;       /**< number of processes cols in the process grid - derived parameter */
    unsigned int strows;     /**< max number of tile rows in a super-tile */
    unsigned int stcols;     /**< max number of tile columns in a super tiles */
    unsigned int crank;      /**< process column rank in the process grid - derived parameter */
    unsigned int rrank;      /**< process row rank in the process grid - derived parameter */
    unsigned int rloc;       /**< number of row of tiles  handled by this process - derived parameter */
    unsigned int cloc;       /**< number of column of tiles handled by this process - derived parameter */
} grid_2Dcyclic_t;

/************************************************
 *   sequential ranks distribution 
 *   in a 2x4 process grid
 *   -----------------
 *   | 0 | 1 | 2 | 3 |
 *   |---------------|
 *   | 4 | 5 | 6 | 7 |
 *   -----------------
 ************************************************/

/**
 * Initialize the description of a  2-D cyclic distribution.
 * @param rank: sequential rank of the local node (as of mpi rank)
 * @param P: number of row of processes of the process grid
 * @param Q: number of colums of the processes of the process grid 
 * @param nrst: number of consecutive tiles along rows held by the same processor
 * @param ncst: number of consecutive tiles along columns held by the same processor
 */
void grid_2Dcyclic_init(grid_2Dcyclic_t* grid, unsigned int rank, unsigned int P, unsigned int Q, unsigned int nrst, unsigned int ncst);

#endif /* __GRID_2DCYCLIC_H__*/