/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011
*/

#ifndef _SGEMM_FERMI_DEFINE_H_
#define _SGEMM_FERMI_DEFINE_H_

#define PRECISION_s

#include "gemm_stencil_defs.cu"

///////////////////////////////////////////////////////////////////////////////////////////////////
// Common parameters

// size of work for a thread block
#define BLK_M_nn 96
#define BLK_N_nn 96
#define BLK_K_nn 16

#define BLK_M_nt 96
#define BLK_N_nt 96
#define BLK_K_nt 16

#define BLK_M_tt 96
#define BLK_N_tt 96
#define BLK_K_tt 16

#define BLK_M_tn 96
#define BLK_N_tn 96
#define BLK_K_tn 16

// size of work for a thread block
#define BLK_M BLK_M_nn
#define BLK_N BLK_N_nn 
#define BLK_K BLK_K_nn

// size of thread block for calculating C (innermost loop)
#define DIM_X 16
#define DIM_Y 16

///////////////////////////////////////////////////////////////////////////////////////////////////
//
//  NoTrans - NoTrans
//

// size of thread block for reading A (dev->regs->shmem)
#define DIM_XA 32
#define DIM_YA  8
  
// size of thread block for reading B (dev->regs->shmem)
#define DIM_XB  8
#define DIM_YB 32

#define version trans_nn
#include "gemm_stencil.cu"
 
#undef DIM_XA
#undef DIM_YA

#undef DIM_XB
#undef DIM_YB

///////////////////////////////////////////////////////////////////////////////////////////////////
//
//  NoTrans - Trans
//
 
// size of thread block for reading A (dev->regs->shmem)
#define DIM_XA 32
#define DIM_YA  8

// size of thread block for reading B (dev->regs->shmem)
#define DIM_XB 32
#define DIM_YB  8

#define version trans_nt
#include "gemm_stencil.cu"

#undef DIM_XA
#undef DIM_YA

#undef DIM_XB
#undef DIM_YB

///////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Trans - Trans
//
 
// size of thread block for reading A (dev->regs->shmem)
#define DIM_XA 16
#define DIM_YA 16

// size of thread block for reading B (dev->regs->shmem)
#define DIM_XB 32
#define DIM_YB  8

#define version trans_tt
#include "gemm_stencil.cu"

#undef DIM_XA
#undef DIM_YA

#undef DIM_XB
#undef DIM_YB

///////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Trans - NoTrans
//
 
// size of thread block for reading A (dev->regs->shmem)
#define DIM_XA 16
#define DIM_YA 16

// size of thread block for reading B (dev->regs->shmem)
#define DIM_XB 16
#define DIM_YB 16

#define version trans_tn
#include "gemm_stencil.cu"

#endif /* _SGEMM_FERMI_DEFINE_H_ */
