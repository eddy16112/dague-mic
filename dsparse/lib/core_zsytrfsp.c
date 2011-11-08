/**
 *
 * @file core_zsytrfsp.c
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

#include "dague.h"
#include "data_distribution.h"
#include "dplasma/lib/dplasmajdf.h"

#include "data_dist/sparse-matrix/pastix_internal/pastix_internal.h"
#include "data_dist/sparse-matrix/sparse-matrix.h"


/*
  Constant: MAXSIZEOFBLOCKS
  Maximum size of blocks given to blas in factorisation
*/
#define MAXSIZEOFBLOCKS 64 /*64 in LAPACK*/

static Dague_Complex64_t zone  = 1.;
static Dague_Complex64_t mzone = -1.;


/* 
   Function: FactorizationLDLT
   
   Factorisation LDLt BLAS2 3 terms

  > A = LDL^T

  Parameters: 
     A       - Matrix to factorize
     n       - Size of A
     stride  - Stide between 2 columns of the matrix
     nbpivot - IN/OUT pivot number.
     critere - Pivoting threshold.

*/
static void core_zsytf2sp(dague_int_t  n, 
                          Dague_Complex64_t * A, 
                          dague_int_t  stride, 
                          dague_int_t *nbpivot, 
                          double criteria )
{
    dague_int_t k;
    Dague_Complex64_t *tmp, *tmp1;

    for (k=0; k<n; k++){
        tmp = A + k*(stride+1);

        if ( cabs(*tmp) < criteria ) {
            (*tmp) = (Dague_Complex64_t)criteria;
            (*nbpivot)++;
	}

        tmp1 = tmp+1;
        
        alpha = 1. / (*tmp);
        cblas_zscal(n-k-1, CBLAS_SADDR( alpha ), tmp1, 1 );
        cblas_zsyr((CBLAS_UPLO)CblasLower, n-k-1, (double)(-(*tmp)), 
                   tmp1, 1, tmp1+stride, stride);
    }
}

/*
  Function: FactorizationLDLT_block

  Computes the block LDL^T factorisation of the
  matrix A.

  > A = LDL^T

  Parameters: 
     A       - Matrix to factorize
     n       - Size of A
     stride  - Stide between 2 columns of the matrix
     nbpivot - IN/OUT pivot number.
     critere - Pivoting threshold.

*/
static void core_zsytrfsp(dague_int_t  n, 
                          Dague_Complex64_t * A, 
                          dague_int_t  stride, 
                          dague_int_t *nbpivot, 
                          double       criteria,
                          Dague_Complex64_t *work)
{
    dague_int_t k, blocknbr, blocksize, matrixsize, col;
    Dague_Complex64_t *tmp,*tmp1,*tmp2;
    Dague_Complex64_t alpha;

    blocknbr = (dague_int_t) ceil( (double)n/(double)MAXSIZEOFBLOCKS );

    for (k=0; k<blocknbr; k++) {
      
        blocksize = min(MAXSIZEOFBLOCKS, n-k*MAXSIZEOFBLOCKS);
        tmp  = A+(k*MAXSIZEOFBLOCKS)*(stride+1); /* Lk,k     */
        tmp1 = tmp+ blocksize;                   /* Lk+1,k   */
        tmp2 = tmp1 + stride* blocksize;         /* Lk+1,k+1 */
        
        /* Factorize the diagonal block Akk*/
        core_zsytf2sp(blocksize, tmp, stride, nbpivot, critere);
        
        if ((k*MAXSIZEOFBLOCKS+blocksize) < n) {
            
            matrixsize = n-(k*MAXSIZEOFBLOCKS+blocksize);
            
            /* Compute the column Lk+1k */
            /** Compute Dk,k*Lk+1,k      */
            cblas_ztrsm(CblasColMajor,
                        (CBLAS_SIDE)CblasRight, (CBLAS_UPLO)CblasLower,
                        (CBLAS_TRANSPOSE)CblasTrans, (CBLAS_DIAG)CblasUnit,
                        matrixsize, blocksize,
                        CBLAS_SADDR(zone), tmp,  stride,
                                           tmp1, stride);

            for(col = 0; col < blocksize; col++) {
                /** Copy Dk,k*Lk+1,k and compute Lk+1,k */
                cblas_zcopy(matrixsize, tmp1+col*stride,     1, 
                                        work+col*matrixsize, 1);

                alpha = 1. / *(tmp + col*(stride+1));
                
                cblas_zscal(matrixsize, CBLAS_SADDR(alpha), 
                            tmp1+col*stride, 1);
            }
            
            /* Update Ak+1k+1 = Ak+1k+1 - Lk+1k*Dk,k*Lk+1kT */
            cblas_zgemm(CblasColMajor,
                        (CBLAS_TRANSPOSE)CblasNoTrans, (CBLAS_TRANSPOSE)CblasTrans,
                        matrixsize, matrixsize, blocksize,
                        CBLAS_SADDR(mzone), work, matrixsize,
                                            tmp1, stride,
                        CBLAS_SADDR(zone),  tmp2, stride);
	}
    }
}


/*
 * Factorization of diagonal block 
 */
void core_zsytrfsp1d(Dague_Complex64_t *L,
                     Dague_Complex64_t *U,
                     Dague_Complex64_t *work,
                     SolverMatrix *datacode, 
                     dague_int_t c)
{
    double         criteria = LAPACKE_dlamch_work('e'); /* TODO */ 
    dague_int_t    dima, dimb, stride;
    dague_int_t    fblknum, lblknum;
    dague_int_t    nbpivot = 0; /* TODO: return to higher level */

    /* check if diagonal column block */
    assert( SYMB_FCOLNUM(c) == SYMB_FROWNUM(SYMB_BLOKNUM(c)) );

    /* Initialisation des pointeurs de blocs */
    dima   = SYMB_LCOLNUM(c)-SYMB_FCOLNUM(c)+1;
    stride = SOLV_STRIDE(c);

    /* Factorize diagonal block (two terms version with workspace) */
    core_zsytrfsp(dima, L, stride, &nbpivot, criteria, work);

    /* Copy diagonal for updates (required by ESP or 1d+) */
    cblas_zcopy(dima, L, stride+1, U, 1);

    fblknum = SYMB_BLOKNUM(c);
    lblknum = SYMB_BLOKNUM(c + 1);

    /* vertical dimension */
    dimb = stride - dima;

    /* if there is an extra-diagonal bloc in column block */
    if ( fblknum+1 < lblknum ) 
    {
        /* first extra-diagonal bloc in column block address */
        fL = L + SOLV_COEFIND(fblknum+1);
        
        /* Three terms version, no need to keep L and L*D */
        cblas_ztrsm(CblasColMajor,
                    (CBLAS_SIDE)CblasRight, (CBLAS_UPLO)CblasLower,
                    (CBLAS_TRANSPOSE)CblasTrans, (CBLAS_DIAG)CblasNonUnit,
                    dimb, dima,
                    CBLAS_SADDR(zone), L,  stride,
                                       fL, stride);
    }
}

