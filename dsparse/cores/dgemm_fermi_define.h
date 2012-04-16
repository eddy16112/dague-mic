/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011
*/

#ifndef _DGEMM_FERMI_DEFINE_H_
#define _DGEMM_FERMI_DEFINE_H_

#define PRECISION_d

#include "gemm_stencil_defs.cu"

///////////////////////////////////////////////////////////////////////////////////////////////////
// Common parameters

// size of work for a thread block
#define BLK_M_nn 64
#define BLK_N_nn 64
#define BLK_K_nn 16

#define BLK_M_nt 64
#define BLK_N_nt 64
#define BLK_K_nt 16

#define BLK_M_tt 64
#define BLK_N_tt 64
#define BLK_K_tt 16

#define BLK_M_tn 64
#define BLK_N_tn 64
#define BLK_K_tn 16

// size of work for a thread block
#define BLK_M BLK_M_nn
#define BLK_N BLK_N_nn 
#define BLK_K BLK_K_nn

// size of thread block for calculating C (innermost loop)
#define DIM_X 16
#define DIM_Y 16

// size of thread block for reading A (dev->regs->shmem)
#define DIM_XA 16
#define DIM_YA 16
  
// size of thread block for reading B (dev->regs->shmem)
#define DIM_XB 16
#define DIM_YB 16

///////////////////////////////////////////////////////////////////////////////////////////////////
//

#define version trans_nn
#include "gemm_stencil.cu"

#define version trans_nt
#include "gemm_stencil.cu"

#define version trans_tn
#include "gemm_stencil.cu"

#define version trans_tt
#include "gemm_stencil.cu"
 
#endif /* _DGEMM_FERMI_DEFINE_H_ */
