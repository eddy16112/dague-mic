/**
 *
 * @file core_zgetrfsp.c
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

#define min( _a, _b ) ( (_a) < (_b) ? (_a) : (_b) )

/*
  Constant: MAXSIZEOFBLOCKS
  Maximum size of blocks given to blas in factorisation
*/
#define MAXSIZEOFBLOCKS 64 /*64 in LAPACK*/

static dague_complex64_t zone  = 1.;
static dague_complex64_t mzone = -1.;

void CORE_zgetro(int m, int n, 
                 PLASMA_Complex64_t *A, int lda,
                 PLASMA_Complex64_t *B, int ldb);

void CORE_zaxpyt(int m, int n, PLASMA_Complex64_t alpha,
                 PLASMA_Complex64_t *A, int lda,
                 PLASMA_Complex64_t *B, int ldb);

/* 
   Function: FactorizationLU
   
   LU Factorisation of one (diagonal) block 
   $A = LU$

   For each column : 
     - Divide the column by the diagonal element.
     - Substract the product of the subdiagonal part by
       the line after the diagonal element from the 
       matrix under the diagonal element.

  Parameters: 
     A       - Matrix to factorize
     m       - number of rows of the Matrix A
     n       - number of cols of the Matrix A
     stride  - Stide between 2 columns of the matrix
     nbpivot - IN/OUT pivot number.
     critere - Pivoting threshold.
*/
static void core_zgetf2sp(dague_int_t  m, 
                          dague_int_t  n, 
                          dague_complex64_t * A, 
                          dague_int_t  stride, 
                          dague_int_t *nbpivot, 
                          double criteria )
{
    dague_int_t k, minMN;
    dague_complex64_t *Akk, *Aik, alpha;

    minMN = min( m, n );

    Akk = A;
    for (k=0; k<minMN; k++) {
        Aik = Akk + 1;

        if ( cabs(*Akk) < criteria ) {
            (*Akk) = (dague_complex64_t)criteria;
            (*nbpivot)++;
	}

        /* A_ik = A_ik / A_kk, i = k+1 .. n */
        alpha = 1. / (*Akk);
        cblas_zscal(m-k-1, CBLAS_SADDR( alpha ), Aik, 1 );

        if ( k+1 < minMN ) {

            /* A_ij = A_ij - A_ik * A_kj, i,j = k+1..n */
            cblas_zgeru(CblasColMajor, m-k-1, n-k-1, 
                        CBLAS_SADDR(mzone), 
                        Aik,        1, 
                        Akk+stride, stride, 
                        Aik+stride, stride);
        }

        Akk += stride+1;
    }
}

/* 
   Function: FactorizationLU_block
   
   Block LU Factorisation of one (diagonal) big block 
   > A = LU

  Parameters: 
     A       - Matrix to factorize
     n       - Size of A
     stride  - Stide between 2 columns of the matrix
     nbpivot - IN/OUT pivot number.
     critere - Pivoting threshold.
*/
static void core_zgetrfsp(dague_int_t  n, 
                          dague_complex64_t *A, 
                          dague_int_t  stride, 
                          dague_int_t *nbpivot, 
                          double       criteria)
{
    dague_int_t k, blocknbr, blocksize, matrixsize, tempm;
    dague_complex64_t *Akk, *Lik, *Ukj, *Aij;

    blocknbr = (dague_int_t) ceil( (double)n/(double)MAXSIZEOFBLOCKS );

    Akk = A; /* Lk,k     */

    for (k=0; k<blocknbr; k++) {
      
        tempm = n - k * MAXSIZEOFBLOCKS;
        blocksize = min(MAXSIZEOFBLOCKS, tempm);
        Lik = Akk + blocksize;
        Ukj = Akk + blocksize*stride;
        Aij = Ukj + blocksize;
        
        /* Factorize the diagonal block Akk*/
        core_zgetf2sp( tempm, blocksize, Akk, stride, nbpivot, criteria );
        
        matrixsize = tempm - blocksize;
        if ( matrixsize > 0 ) {

            /* Compute the column Ukk+1 */
            cblas_ztrsm(CblasColMajor,
                        CblasLeft, CblasLower,
                        CblasNoTrans, CblasUnit,
                        blocksize, matrixsize,
                        CBLAS_SADDR(zone), Akk, stride,
                                           Ukj, stride);

            /* Update Ak+1,k+1 = Ak+1,k+1 - Lk+1,k*Uk,k+1 */
            cblas_zgemm(CblasColMajor,
                        CblasNoTrans, CblasNoTrans,
                        matrixsize, matrixsize, blocksize,
                        CBLAS_SADDR(mzone), Lik, stride,
                                            Ukj, stride,
                        CBLAS_SADDR(zone),  Aij, stride);
	}

        Akk += blocksize * (stride+1);
    }
}


/*
 * Factorization of diagonal block 
 */
void core_zgetrfsp1d(dague_complex64_t *L,
                     dague_complex64_t *U,
                     SolverMatrix *datacode, 
                     dague_int_t c,
                     double criteria)
{
    dague_complex64_t *fL, *fU;
    dague_int_t    dima, dimb, stride;
    dague_int_t    fblknum, lblknum;
    dague_int_t    nbpivot = 0; /* TODO: return to higher level */

    /* check if diagonal column block */
    assert( SYMB_FCOLNUM(c) == SYMB_FROWNUM(SYMB_BLOKNUM(c)) );

    /* Initialisation des pointeurs de blocs */
    dima   = SYMB_LCOLNUM(c) - SYMB_FCOLNUM(c) + 1;
    stride = SOLV_STRIDE(c);

    /* Factorize diagonal block (two terms version with workspace) */
    core_zgetrfsp(dima, L, stride, &nbpivot, criteria);

    /* Transpose Akk in ucoeftab */
    CORE_zgetro(dima, dima, L, stride, U, stride);

    fblknum = SYMB_BLOKNUM(c);
    lblknum = SYMB_BLOKNUM(c + 1);

    /* vertical dimension */
    dimb = stride - dima;

    /* if there is an extra-diagonal bloc in column block */
    if ( fblknum+1 < lblknum ) 
    {
        /* first extra-diagonal bloc in column block address */
        fL = L + SOLV_COEFIND(fblknum+1);
        fU = U + SOLV_COEFIND(fblknum+1);
        
        cblas_ztrsm(CblasColMajor,
                    (CBLAS_SIDE)CblasRight, (CBLAS_UPLO)CblasUpper,
                    (CBLAS_TRANSPOSE)CblasNoTrans, (CBLAS_DIAG)CblasNonUnit,
                    dimb, dima,
                    CBLAS_SADDR(zone), L,  stride,
                                       fL, stride);
        
        cblas_ztrsm(CblasColMajor,
                    (CBLAS_SIDE)CblasRight, (CBLAS_UPLO)CblasUpper,
                    (CBLAS_TRANSPOSE)CblasNoTrans, (CBLAS_DIAG)CblasUnit,
                    dimb, dima,
                    CBLAS_SADDR(zone), U,  stride,
                                       fU, stride);
    }
}


void core_zgetrfsp1d_gemm(dague_int_t cblknum,
                          dague_int_t bloknum,
                          dague_int_t fcblknum,
                          dague_complex64_t *L,
                          dague_complex64_t *U,
                          dague_complex64_t *Cl,
                          dague_complex64_t *Cu,
                          dague_complex64_t *work,
                          SolverMatrix *datacode)
{
    dague_complex64_t *Aik, *Akj, *Aij, *C;
    dague_complex64_t *wtmp;
    dague_int_t fblknum, lblknum, frownum;
    dague_int_t stride, stridefc, indblok;
    dague_int_t b, j;
    dague_int_t dimi, dimj, dima, dimb;

    fblknum = SYMB_BLOKNUM(cblknum);
    lblknum = SYMB_BLOKNUM(cblknum + 1);

    indblok = SOLV_COEFIND(bloknum);
    stride  = SOLV_STRIDE(cblknum);

    dimi = stride - indblok;
    dimj = SYMB_LROWNUM(bloknum) - SYMB_FROWNUM(bloknum) + 1;
    dima = SYMB_LCOLNUM(cblknum) - SYMB_FCOLNUM(cblknum) + 1;  

    /* Matrix A = Aik */
    Aik = L + indblok;
    Akj = U + indblok;

    /*
     * Compute update on L 
     */
    wtmp = work;
    CORE_zgemm( PlasmaNoTrans, PlasmaTrans, 
                dimi, dimj, dima,
                1.,  Aik,  stride, 
                     Akj,  stride,
                0.,  wtmp, dimi  );

    /*
     * Add contribution to facing cblk
     */
    b = SYMB_BLOKNUM( fcblknum );
    stridefc = SOLV_STRIDE(fcblknum);
    C = Cl + (SYMB_FROWNUM(bloknum) - SYMB_FCOLNUM(fcblknum)) * stridefc;
        
    /* for all following blocks in block column */
    for (j=bloknum; j<lblknum; j++) {
        frownum = SYMB_FROWNUM(j);

        /* Find facing bloknum */
#ifdef NAPA_SOPALIN /* ILU(k) */
        while (!(((SYMB_FROWNUM(j)>=SYMB_FROWNUM(b)) && 
                  (SYMB_LROWNUM(j)<=SYMB_LROWNUM(b))) ||
                 ((SYMB_FROWNUM(j)<=SYMB_FROWNUM(b)) && 
                  (SYMB_LROWNUM(j)>=SYMB_LROWNUM(b))) ||
                 ((SYMB_FROWNUM(j)<=SYMB_FROWNUM(b)) && 
                  (SYMB_LROWNUM(j)>=SYMB_FROWNUM(b))) ||
                 ((SYMB_FROWNUM(j)<=SYMB_LROWNUM(b)) && 
                  (SYMB_LROWNUM(j)>=SYMB_LROWNUM(b)))))
#else
        while (!((SYMB_FROWNUM(j)>=SYMB_FROWNUM(b)) && 
                 (SYMB_LROWNUM(j)<=SYMB_LROWNUM(b))))
#endif
            {
                b++;
                assert( b < SYMB_BLOKNUM( fcblknum+1 ) );
            }
        

        Aij = C + SOLV_COEFIND(b) + frownum - SYMB_FROWNUM(b);
        dimb = SYMB_LROWNUM(j) - frownum + 1;

        CORE_zgeadd( dimb, dimj, -1.0,
                    wtmp, dimi,
                    Aij,  stridefc );

        /* Displacement to next block */
        wtmp += dimb;
    }

 
    /* 
     * Compute update on U 
     */
    
    Aik = U + indblok;
    Akj = L + indblok;
    wtmp = work;
    
    CORE_zgemm( PlasmaNoTrans, PlasmaTrans, 
                dimi, dimj, dima,
                1.,  Aik,  stride, 
                     Akj,  stride,
                0.,  wtmp, dimi  );
  
    wtmp += SYMB_LROWNUM(bloknum) - SYMB_FROWNUM(bloknum) + 1;

    /*
     * Add contribution to facing cblk
     */
    b = SYMB_BLOKNUM( fcblknum );
    C = Cl + (SYMB_FROWNUM(bloknum) - SYMB_FCOLNUM(fcblknum));
        
    /* for all following blocks in block column */
    for (j=bloknum+1; j<lblknum; j++) {
        frownum = SYMB_FROWNUM(j);

        /* Find facing bloknum */
#ifdef NAPA_SOPALIN /* ILU(k) */
        /* WARNING: may not work for NAPA */
        if (!(((SYMB_FROWNUM(j)>=SYMB_FROWNUM(b)) && 
               (SYMB_LROWNUM(j)<=SYMB_LROWNUM(b))) ||
              ((SYMB_FROWNUM(j)<=SYMB_FROWNUM(b)) && 
               (SYMB_LROWNUM(j)>=SYMB_LROWNUM(b))) ||
              ((SYMB_FROWNUM(j)<=SYMB_FROWNUM(b)) && 
               (SYMB_LROWNUM(j)>=SYMB_FROWNUM(b))) ||
              ((SYMB_FROWNUM(j)<=SYMB_LROWNUM(b)) && 
               (SYMB_LROWNUM(j)>=SYMB_LROWNUM(b)))))
#else
        if (!((SYMB_FROWNUM(j)>=SYMB_FROWNUM(b)) && 
              (SYMB_LROWNUM(j)<=SYMB_LROWNUM(b))))
#endif
            break;
        

        Aij = C + (frownum - SYMB_FROWNUM(b))*stridefc;
        dimb = SYMB_LROWNUM(j) - frownum + 1;

        CORE_zaxpyt( dimj, dimb, -1.0,
                     wtmp, dimi,
                     Aij,  stridefc );

        /* Displacement to next block */
        wtmp += dimb;
    }


    C = Cu + (SYMB_FROWNUM(bloknum) - SYMB_FCOLNUM(fcblknum)) * stridefc;

    /* Keep updating on U */
    for (; j<lblknum; j++) {
        frownum = SYMB_FROWNUM(j);

        /* Find facing bloknum */
#ifdef NAPA_SOPALIN /* ILU(k) */
        while (!(((SYMB_FROWNUM(j)>=SYMB_FROWNUM(b)) && 
                  (SYMB_LROWNUM(j)<=SYMB_LROWNUM(b))) ||
                 ((SYMB_FROWNUM(j)<=SYMB_FROWNUM(b)) && 
                  (SYMB_LROWNUM(j)>=SYMB_LROWNUM(b))) ||
                 ((SYMB_FROWNUM(j)<=SYMB_FROWNUM(b)) && 
                  (SYMB_LROWNUM(j)>=SYMB_FROWNUM(b))) ||
                 ((SYMB_FROWNUM(j)<=SYMB_LROWNUM(b)) && 
                  (SYMB_LROWNUM(j)>=SYMB_LROWNUM(b)))))
#else
        while (!((SYMB_FROWNUM(j)>=SYMB_FROWNUM(b)) && 
                 (SYMB_LROWNUM(j)<=SYMB_LROWNUM(b))))
#endif
            {
                b++;
                assert( b < SYMB_BLOKNUM( fcblknum+1 ) );
            }
        

        dimb = SYMB_LROWNUM(j) - frownum + 1;
        Aij = C + SOLV_COEFIND(b) + frownum - SYMB_FROWNUM(b);
        
        CORE_zgeadd( dimb, dimj, -1.0,
                    wtmp, dimi,
                    Aij,  stridefc );

        /* Displacement to next block */
        wtmp += dimb;
    }
    
}
