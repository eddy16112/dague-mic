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

#include "dague.h"
#include "data_distribution.h"
#include "dplasma/include/dplasma.h"
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

/*
   Function: FactorizationLDLT

   Factorisation LDLt BLAS2 3 terms
   this subroutine factors the diagonal block

  > A = LDL^T

  Parameters:
     A       - Matrix to factorize
     n       - Size of A
     stride  - Stide between 2 columns of the matrix
     nbpivot - IN/OUT pivot number.
     criteria - Pivoting threshold.

*/
static void core_zpotf2sp(dague_int_t  n,
                          dague_complex64_t * A,
                          dague_int_t  stride,
                          dague_int_t *nbpivot,
                          double criteria )
{
    dague_int_t k;
    dague_complex64_t *tmp, *tmp1, alpha;

    for (k=0; k<n; k++){
        tmp = A + k*(stride+1); // = A + k*stride + k = diagonal element

        if ( cabs(*tmp) < criteria ) {
            (*tmp) = (dague_complex64_t)criteria;
            (*nbpivot)++;
        }

        /* Hermitian matrices, so imaginary part should be 0 */
        if ( creal(*tmp) < 0. )
        {
            errorPrint("Negative diagonal term\n");
            EXIT(MOD_SOPALIN, INTERNAL_ERR);
        }

        *tmp = (dague_complex64_t)dplasma_zsqrt(*tmp);
        tmp1 = tmp+1;

        alpha = 1. / (*tmp); // scaling with the diagonal to compute L((k+1):n,k)
        cblas_zscal(n-k-1, CBLAS_SADDR( alpha ), tmp1, 1 );

        cblas_zher(CblasColMajor, CblasLower,
                   n-k-1, -1.,
                   tmp1,        1,
                   tmp1+stride, stride);
    }
}

/*
  Function: FactorizationLDLT_block

  Computes the block LDL^T factorisation of the
  matrix A.
  this subroutine factors the diagonal supernode

  > A = LDL^T

  Parameters:
     A       - Matrix to factorize
     n       - Size of A
     stride  - Stide between 2 columns of the matrix
     nbpivot - IN/OUT pivot number.
     criteria - Pivoting threshold.

*/
static void core_zpotrfsp(dague_int_t  n,
                          dague_complex64_t * A,
                          dague_int_t  stride,
                          dague_int_t *nbpivot,
                          double       criteria)
{
    dague_int_t k, blocknbr, blocksize, matrixsize;
    dague_complex64_t *tmp,*tmp1,*tmp2;

        /* diagonal supernode is divided into MAXSIZEOFBLOCK-by-MAXSIZEOFBLOCKS blocks */
    blocknbr = (dague_int_t) ceil( (double)n/(double)MAXSIZEOFBLOCKS );

    for (k=0; k<blocknbr; k++) {

        blocksize = min(MAXSIZEOFBLOCKS, n-k*MAXSIZEOFBLOCKS);
        tmp  = A+(k*MAXSIZEOFBLOCKS)*(stride+1); /* Lk,k     */
        tmp1 = tmp  + blocksize;                 /* Lk+1,k   */
        tmp2 = tmp1 + blocksize * stride;        /* Lk+1,k+1 */

        /* Factorize the diagonal block Akk*/
        core_zpotf2sp(blocksize, tmp, stride, nbpivot, criteria);

        if ((k*MAXSIZEOFBLOCKS+blocksize) < n) {

            matrixsize = n-(k*MAXSIZEOFBLOCKS+blocksize);

            /* Compute the column L(k+1:n,k) = (L(k,k)D(k,k))^{-1}A(k+1:n,k)    */
            /* 1) Compute A(k+1:n,k) = A(k+1:n,k)L(k,k)^{-T} = D(k,k)L(k+1:n,k) */
                        /* input: L(k,k) in tmp, A(k+1:n,k) in tmp1   */
                        /* output: A(k+1:n,k) in tmp1                 */
            cblas_ztrsm(CblasColMajor,
                        CblasRight, CblasLower,
                        CblasConjTrans, CblasNonUnit,
                        matrixsize, blocksize,
                        CBLAS_SADDR(zone), tmp,  stride,
                                           tmp1, stride);

            /* Update Ak+1k+1 = Ak+1k+1 - Lk+1k * Lk+1kT */
            cblas_zherk(CblasColMajor, CblasLower, CblasNoTrans,
                        matrixsize, blocksize,
                        (double)mzone, tmp1, stride,
                        (double)zone,  tmp2, stride);
        }
    }
}


/*
 * Factorization of diagonal block
 * entry point from DAGue: to factor c-th supernodal column
 */
void core_zpotrfsp1d(dague_complex64_t *L,
                     SolverMatrix *datacode,
                     dague_int_t c,
                     double criteria)
{
    dague_complex64_t *fL;
    dague_int_t    dima, dimb, stride;
    dague_int_t    fblknum, lblknum;
    dague_int_t    nbpivot = 0; /* TODO: return to higher level */

    /* check if diagonal column block */
    assert( SYMB_FCOLNUM(c) == SYMB_FROWNUM(SYMB_BLOKNUM(c)) );

    /* Initialisation des pointeurs de blocs */
    dima   = SYMB_LCOLNUM(c)-SYMB_FCOLNUM(c)+1; /* (last column in this block-column)-(first column in this block-column)+1 */
    stride = SOLV_STRIDE(c);                    /*  leading dimension of this block                                         */

    /* Factorize diagonal block (two terms version with workspace) */
    core_zpotrfsp(dima, L, stride, &nbpivot, criteria );

    fblknum = SYMB_BLOKNUM(c);      /* block number of this diagonal block     */
    lblknum = SYMB_BLOKNUM(c + 1);  /* block number of the next diagonal block */

    /* vertical dimension */
    dimb = stride - dima;

    /* if there are off-diagonal supernodes in the column */
    if ( fblknum+1 < lblknum )
    {
        /* the first off-diagonal block in column block address */
        fL = L + SOLV_COEFIND(fblknum+1);

        /* Three terms version, no need to keep L and L*D */
        cblas_ztrsm(CblasColMajor,
                    CblasRight, CblasLower,
                    CblasConjTrans, CblasNonUnit,
                    dimb, dima,
                    CBLAS_SADDR(zone), L,  stride,
                                       fL, stride);
    }
}


void core_zpotrfsp1d_gemm(dague_int_t cblknum,
                          dague_int_t bloknum,
                          dague_int_t fcblknum,
                          dague_complex64_t *L,
                          dague_complex64_t *C,
                          dague_complex64_t *work,
                          SolverMatrix *datacode)
{
    dague_complex64_t *Aik, *Aij;
    dague_int_t lblknum, frownum;
    dague_int_t stride, stridefc, indblok;
    dague_int_t b, j;
    dague_int_t dimi, dimj, dima, dimb;

    lblknum = SYMB_BLOKNUM(cblknum + 1);

    indblok = SOLV_COEFIND(bloknum);
    stride  = SOLV_STRIDE(cblknum);

    dimi = stride - indblok;
    dimj = SYMB_LROWNUM(bloknum) - SYMB_FROWNUM(bloknum) + 1;
    dima = SYMB_LCOLNUM(cblknum) - SYMB_FCOLNUM(cblknum) + 1;

    /* Matrix A = Aik */
    Aik = L + indblok;

    /* Compute the contribution */
    CORE_zgemm( PlasmaNoTrans, PlasmaConjTrans,
                 dimi, dimj, dima,
                 1.,  Aik,  stride,
                      Aik,  stride,
                 0.,  work, dimi );

    /*
     * Add contribution to facing cblk  *
         * A(i,i+1:n) += work1              */
    b = SYMB_BLOKNUM( fcblknum );
    stridefc = SOLV_STRIDE(fcblknum);
    C = C + (SYMB_FROWNUM(bloknum) - SYMB_FCOLNUM(fcblknum)) * stridefc;

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
                    work, dimi,
                    Aij,  stridefc );

        /* Displacement to next block */
        work += dimb;
    }
}
