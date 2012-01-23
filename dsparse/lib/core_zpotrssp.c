/**
 *
 * @file core_zpotrfsp.c
 *
 *  PLASMA core_blas kernel
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver
 *
 * @version 1.0.0
 * @author Mathieu Faverge
 * @author Pierre Ramet
 * @author Xavier Lacoste
 * @date 2011-11-11
 * @precisions normal z -> c d s
 *
 **/
#include <plasma.h>
#include <core_blas.h>
#include <cblas.h>

#include "dague.h"
#include "data_distribution.h"
#include "dplasma/lib/dplasmajdf.h"

#include "data_dist/sparse-matrix/pastix_internal/pastix_internal.h"
#include "data_dist/sparse-matrix/sparse-matrix.h"

#define min( _a, _b ) ( (_a) < (_b) ? (_a) : (_b) )

/*
  Constant: MAXSIZEOFBLOCKS
  Maximum size of blocks given to blas in factorisation
*/
#define MAXSIZEOFBLOCKS 64 /*64 in LAPACK*/

static Dague_Complex64_t zone  = 1.;
static Dague_Complex64_t mzone = -1.;

/* 
   Function: diagonal Lkk solve
   
   Compute op(L^{-1}) b => b BLAS2 3 terms
   L is nxn lower-triangular (with leading dimenstion = stride).
   B is nx1 (for now :).

*/

#if 0
static void core_zpotrsm2sp(CBLAS_TRANSPOSE Trans, 
                            dague_int_t  n, 
                            Dague_Complex64_t *L,
                            Dague_Complex64_t *B, 
                            dague_int_t  stride )
{
    dague_int_t k;
    Dague_Complex64_t *l1, *l2, *b1, *b2, alpha;

    if( Trans == CblasNoTrans ) {
      /* forward-substitution */
      for (k=0; k<n; k++){
        l1 = L + k*(stride+1);  // diagonal element
        b1 = B + k;             // corresponding b element
        *b1 /= (*l1);

        if( k+1 < n ) { // off-diagonal updates
          l2 = l1+1; 
          b2 = b1+1;
          alpha = -(*b1);
          cblas_zaxpy(n-k-1,
                      alpha, l2, 1, 
                             b2, 1);
        }
      }
    } else {
      /* backward-substitution */
      for (k=n-1; k>=0; k--){
        l1 = L + k*(stride+1);  // diagonal element
        b1 = B + k;             // corresponding b(k)
        if( k+1 < n ) { // update b(k) with b(k+1:n)
          l2 = l1+1;              
          b2 = b1+1;
          alpha = cblas_ddot(n-k-1,
                             l2, 1, 
                             b2, 1);
          *b1 -= alpha;
        }
        *b1 /= (*l1);
      }
    }
}

/*
  Function: triangular solves
  Compute L^{-1} b => b BLAS2 3 terms
*/
static void core_zpotrsmsp(CBLAS_TRANSPOSE Trans,
                           Dague_Complex64_t *L,
                           Dague_Complex64_t *B,
                           SolverMatrix *datacode, 
                           dague_int_t c )
{
    dague_int_t dima, stride, k, blocknbr, blocksize, matrixsize;
    Dague_Complex64_t *L11, *L21, *B1, *B2;

    /* check if diagonal block */
    assert( SYMB_FCOLNUM(c) == SYMB_FROWNUM(SYMB_BLOKNUM(c)) );

    /* Initialisation des pointeurs de blocs */
    dima   = SYMB_LCOLNUM(c)-SYMB_FCOLNUM(c)+1;
    stride = SOLV_STRIDE(c);

    blocknbr = (dague_int_t) ceil( (double)dima/(double)MAXSIZEOFBLOCKS );

    if( Trans == CblasNoTrans ) { /* forward substitution */
      for (k=0; k<blocknbr; k++) {
        blocksize = min(MAXSIZEOFBLOCKS, n-k*MAXSIZEOFBLOCKS);
        L11 = A+(k*MAXSIZEOFBLOCKS)*(stride+1); /* L(k,k) */
        B1  = B  + (k*MAXSIZEOFBLOCKS);
        
        /* solve with the diagonal block L(k,k) */
        core_zpotrsm2sp(Trans, blocksize, L11, B1, stride);
 
        /* update the remaining B */       
        if ((k*MAXSIZEOFBLOCKS+blocksize) < n) {
            matrixsize = n-(k*MAXSIZEOFBLOCKS+blocksize);
            L21 = L11  + blocksize;            /* L(k+1,k) */
            B2  = B1 + blocksize;
            cblas_zgemv(CblasColMajor, CblasNoTrans,
                        matrixsize, blocksize,
                        CBLAS_SADDR(mzone), L21, stride,
                        B1, 1, B2, 1);
	}
      }
    } else { /* backward substitution */
      for (k=blocknbr-1; k>=0; k--) {
        blocksize = min(MAXSIZEOFBLOCKS, n-k*MAXSIZEOFBLOCKS);
        L11 = A+(k*MAXSIZEOFBLOCKS)*(stride+1); /* L(k,k) */
        B1  = B  + (k*MAXSIZEOFBLOCKS);
        
        /* update with the previous B */
        if ((k*MAXSIZEOFBLOCKS+blocksize) < n) {
            matrixsize = n-(k*MAXSIZEOFBLOCKS+blocksize);
            L21 = L11  + blocksize;            /* L(k+1,k) */
            B2  = B1 + blocksize;
            cblas_zgemv(CblasColMajor, CblasConjTrans,
                        matrixsize, blocksize,
                        CBLAS_SADDR(mzone), L21, stride,
                        B2, 1, B1, 1);
	}

        /* solve with the diagonal block Akk */
        core_zpotrsm2sp(Trans, blocksize, L11, B1, stride);
    }
}
#endif

static void core_zpotrsmsp(CBLAS_TRANSPOSE Trans,
                           dague_int_t nrhs,
                           Dague_Complex64_t *L,
                           Dague_Complex64_t *B,
                           SolverMatrix *datacode, 
                           dague_int_t c )
{
    dague_int_t dima, ldl;

    /* check if diagonal block */
    assert( SYMB_FCOLNUM(c) == SYMB_FROWNUM(SYMB_BLOKNUM(c)) );

    dima = SYMB_LCOLNUM(c)-SYMB_FCOLNUM(c)+1;
    ldl  = SOLV_STRIDE(c);
    cblas_ztrsm(CblasColMajor, CblasLeft, CblasLower, Trans, CblasNonUnit,
                dima, nrhs, CBLAS_SADDR(zone), 
                L, ldl, B, ldl);
}

/*
 * updating the remaing blocks of B
 */
void core_zpotrfsp1d(CBLAS_TRANSPOSE Trans,
                     dague_int_t cblknum,   /* block-column index */
                     dague_int_t bloknum,   /* block number       */
                     dague_int_t fcblknum,  /* block-row index    */
                     Dague_Complex64_t *L,
                     Dague_Complex64_t *B,
                     SolverMatrix *datacode) 
{
    Dague_Complex64_t *Lik, *B1, *B2;
    dague_int_t indblok, ldl, dimi, dimj;

    indblok = SOLV_COEFIND(bloknum); /* offset to my block   */
    ldl     = SOLV_STRIDE(cblknum);  /* my leading dimension */

    dimj = SYMB_LCOLNUM(cblknum) - SYMB_FCOLNUM(cblknum) + 1;  /* number of columns */
    dimi = SYMB_LROWNUM(bloknum) - SYMB_FROWNUM(bloknum) + 1;  /* number of rows    */

    /* Matrix L(i,k) */
    Lik = L + indblok;
    if( Trans == CblasNoTrans ) { 
      /* forward-substitution */
      B1 = B + SYMB_FCOLNUM(  cblknum );
      B2 = B + SYMB_FCOLNUM( fcblknum );
    } else {  
      /* backward substitution */
      B1 = B + SYMB_FCOLNUM( fcblknum );
      B2 = B + SYMB_FCOLNUM(  cblknum );
    }
    cblas_zgemv(CblasColMajor, Trans,
                dimi, dimj,
                CBLAS_SADDR(mzone), Lik, ldl, B1, 1, 
                CBLAS_SADDR(zone), B2, 1);
}


