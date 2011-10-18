/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 */
#ifndef _TESTSCOMMON_H
#define _TESTSCOMMON_H

/* includes used by the testing_*.c */
#include "dague_config.h"

/* system and io */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
/* Plasma and math libs */
#include <math.h>
#include <cblas.h>
#include <plasma.h>
#include <lapacke.h>
#include <core_blas.h>
/* PaStiX lib */ 
#include <pastix.h>
#include <read_matrix.h>
/* dague things */
#include "dague.h"
#include "scheduling.h"
#include "profiling.h"
/*#include "dsparse.h"*/
/* timings */
#include "common_timing.h"
#ifdef DAGUE_VTRACE
#include "vt_user.h"
#endif

#include "flops.h"

enum iparam_t {
  IPARAM_RANK,         /* Rank                              */
  IPARAM_NNODES,       /* Number of nodes                   */
  IPARAM_NCORES,       /* Number of cores                   */
  IPARAM_SCHEDULER,    /* What scheduler do we choose */
  IPARAM_NGPUS,        /* Number of GPUs                    */
  IPARAM_PRIO,         /* Switchpoint for priority DAG      */
//  IPARAM_P,            /* Rows in the process grid          */
//  IPARAM_Q,            /* Columns in the process grid       */
//  IPARAM_M,            /* Number of rows of the matrix      */
//  IPARAM_N,            /* Number of columns of the matrix   */
//  IPARAM_K,            /* RHS or K                          */
//  IPARAM_LDA,          /* Leading dimension of A            */
//  IPARAM_LDB,          /* Leading dimension of B            */
//  IPARAM_LDC,          /* Leading dimension of C            */
//  IPARAM_IB,           /* Inner-blocking size               */
//  IPARAM_NB,           /* Number of columns in a tile       */
//  IPARAM_MB,           /* Number of rows in a tile          */
//  IPARAM_SNB,          /* Number of columns in a super-tile */
//  IPARAM_SMB,          /* Number of rows in a super-tile    */
  IPARAM_CHECK,        /* Checking activated or not         */
  IPARAM_VERBOSE,      /* How much noise do we want?        */
//  IPARAM_LOWLVL_TREE,  /* Tree used for reduction inside nodes  (specific to xgeqrf_param) */
//  IPARAM_HIGHLVL_TREE, /* Tree used for reduction between nodes (specific to xgeqrf_param) */
//  IPARAM_QR_TS_SZE,    /* Size of TS domain                     (specific to xgeqrf_param) */
//  IPARAM_QR_HLVL_SZE,  /* Size of the high level tree           (specific to xgeqrf_param) */
//  IPARAM_QR_DOMINO,    /* Enable/disable the domino between the upper and the lower tree (specific to xgeqrf_param) */
  IPARAM_DOT,          /* Do we require to output the DOT file? */
  IPARAM_SIZEOF
};

enum sparam_t {
  SPARAM_MATRIX,    /* Matrix filename                */
  SPARAM_RHS,       /* RHS filename                   */
  SPARAM_ORDERING,  /* Ordering filename              */
  SPARAM_SYMBOL,    /* Symbol filename                */
  SPARAM_SIZEOF
};

#define PASTE_CODE_IPARAM_LOCALS(iparam) \
  int rank  = iparam[IPARAM_RANK];\
  int nodes = iparam[IPARAM_NNODES];\
  int cores = iparam[IPARAM_NCORES];\
  int gpus  = iparam[IPARAM_NGPUS];\
  int prio  = iparam[IPARAM_PRIO];\
  int check = iparam[IPARAM_CHECK];\
  int loud  = iparam[IPARAM_VERBOSE];\
  (void)rank;(void)nodes;(void)cores;(void)gpus;(void)prio;(void)check;(void)loud;

/* Define a double type which not pass through the precision generation process */
typedef double DagDouble_t;
#define PASTE_CODE_FLOPS( FORMULA, PARAMS ) \
  double gflops, flops = FORMULA PARAMS;
  
#if defined(PRECISIONS_z) || defined(PRECISIONS_c)
#define PASTE_CODE_FLOPS_COUNT(FADD,FMUL,PARAMS) \
  double gflops, flops = (2. * FADD PARAMS + 6. * FMUL PARAMS);
#else 
#define PASTE_CODE_FLOPS_COUNT(FADD,FMUL,PARAMS) \
  double gflops, flops = (FADD PARAMS + FMUL PARAMS);
#endif

/*******************************
 * globals values
 *******************************/

#if defined(HAVE_MPI)
extern MPI_Datatype SYNCHRO;
#endif  /* HAVE_MPI */

void print_usage(void);

dague_context_t *setup_dague(int argc, char* argv[], int *iparam, char **sparam);
void cleanup_dague(dague_context_t* dague, int *iparam, char **sparam);
void param_default(int* iparam, char **sparam);

/**
 * No macro with the name max or min is acceptable as there is
 * no way to correctly define them without borderline effects.
 */
#undef max
#undef min 
static inline int max(int a, int b) { return a > b ? a : b; }
static inline int min(int a, int b) { return a < b ? a : b; }

/* Paste code to allocate a matrix in desc if cond_init is true */
#define PASTE_CODE_ALLOCATE_MATRIX(DDESC, COND, TYPE, INIT_PARAMS)      \
    TYPE##_t DDESC;                                                     \
    if(COND) {                                                          \
        TYPE##_init INIT_PARAMS;                                        \
        DDESC.mat = dague_data_allocate((size_t)DDESC.super.nb_local_tiles * \
                                        (size_t)DDESC.super.bsiz *      \
                                        (size_t)DDESC.super.mtype);     \
        dague_ddesc_set_key((dague_ddesc_t*)&DDESC, #DDESC);            \
    }

#define PASTE_CODE_ENQUEUE_KERNEL(DAGUE, KERNEL, PARAMS) \
  SYNC_TIME_START(); \
  dague_object_t* DAGUE_##KERNEL = dsparse_##KERNEL##_New PARAMS; \
  dague_enqueue(DAGUE, DAGUE_##KERNEL); \
  if(loud) SYNC_TIME_PRINT(rank, ( #KERNEL " DAG creation: %u local tasks enqueued\n", DAGUE->taskstodo));


#define PASTE_CODE_PROGRESS_KERNEL(DAGUE, KERNEL) \
  SYNC_TIME_START(); \
  TIME_START(); \
  dague_progress(DAGUE); \
  if(loud) TIME_PRINT(rank, (#KERNEL " computed %u tasks,\trate %f task/s\n", \
              DAGUE_##KERNEL->nb_local_tasks, \
              DAGUE_##KERNEL->nb_local_tasks/time_elapsed)); \
  SYNC_TIME_PRINT(rank, (#KERNEL " computation N= %d NB= %d : %f gflops\n", N, NB, \
                   gflops = (flops/1e9)/(sync_time_elapsed))); \
  (void)gflops;


#define PASTE_CODE_INIT_CONTEXT( _dspctxt, _factotype)                \
    _dspctxt.format     = RSA;                     /* Matrix file format                         */ \
    _dspctxt.factotype  = _factotype;                                                               \
    _dspctxt.matrixname = sparam[SPARAM_MATRIX]  ; /* Filename to get the matrix                 */ \
    _dspctxt.rhsname    = sparam[SPARAM_RHS]     ; /* Filename where the ordering is stored      */ \
    _dspctxt.ordername  = sparam[SPARAM_ORDERING]; /* Filename where the symbol matrix is stored */ \
    _dspctxt.symbname   = sparam[SPARAM_ORDERING]; /* Filename where the symbol matrix is stored */ \
    _dspctxt.type       = NULL; /* Type of the matrix                         */    \
    _dspctxt.rhstype    = NULL; /* Type of the RHS                            */    \
    _dspctxt.n          = 0;    /* Number of unknowns/columns/rows            */    \
    _dspctxt.nnz        = 0;    /* Number of non-zero values in the input matrix */ \
    _dspctxt.colptr     = NULL; /* Vector of size N+1 storing the starting point of each column in the array rows */ \
    _dspctxt.rows       = NULL; /* Indices of the rows present in each column */    \
    _dspctxt.values     = NULL; /* Values of the matrix                       */    \
    _dspctxt.rhs        = NULL; /* Right Hand Side                            */    \
    _dspctxt.permtab    = NULL; /* vector of permutation                      */    \
    _dspctxt.peritab    = NULL; /* vector of inverse permutation              */    \

#endif /* _TESTSCOMMON_H */
