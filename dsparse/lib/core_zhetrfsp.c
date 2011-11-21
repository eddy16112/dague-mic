/**
 *
 * @file core_zhetrfsp.c
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

static Dague_Complex64_t zone  = 1.;
static Dague_Complex64_t mzone = -1.;


int CORE_zgemdm(int transA, int transB,
                int M, int N, int K,
                PLASMA_Complex64_t alpha, PLASMA_Complex64_t *A, int LDA,
                PLASMA_Complex64_t *B, int LDB,
                PLASMA_Complex64_t beta, PLASMA_Complex64_t *C, int LDC,
                PLASMA_Complex64_t *D, int incD,
                PLASMA_Complex64_t *WORK, int LWORK);

/* 
   Function: FactorizationLDLT
   
   Factorisation LDLt BLAS2 3 terms

  > A = LDL^T

  Parameters: 
     A       - Matrix to factorize
     n       - Size of A
     stride  - Stide between 2 columns of the matrix
     nbpivot - IN/OUT pivot number.
     criteria - Pivoting threshold.

*/
static void core_zhetf2sp(dague_int_t  n, 
                          Dague_Complex64_t * A, 
                          dague_int_t  stride, 
                          dague_int_t *nbpivot, 
                          double criteria )
{
    dague_int_t k;
    Dague_Complex64_t *tmp, *tmp1, alpha;

    for (k=0; k<n; k++){
        tmp = A + k*(stride+1);

        if ( cabs(*tmp) < criteria ) {
            (*tmp) = (Dague_Complex64_t)criteria;
            (*nbpivot)++;
	}

        tmp1 = tmp+1;
        
        alpha = 1. / (*tmp);
        cblas_zscal(n-k-1, CBLAS_SADDR( alpha ), tmp1, 1 );
        alpha = -(*tmp);

        cblas_zher(CblasColMajor, (CBLAS_UPLO)CblasLower, 
                   n-k-1, (double)alpha, 
                   tmp1,        1, 
                   tmp1+stride, stride);
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
     criteria - Pivoting threshold.

*/
static void core_zhetrfsp(dague_int_t  n, 
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
        core_zhetf2sp(blocksize, tmp, stride, nbpivot, criteria);
        
        if ((k*MAXSIZEOFBLOCKS+blocksize) < n) {
            
            matrixsize = n-(k*MAXSIZEOFBLOCKS+blocksize);
            
            /* Compute the column Lk+1k */
            /** Compute Lk+1,k*Dk,k      */
            cblas_ztrsm(CblasColMajor,
                        CblasRight, CblasLower,
                        CblasConjTrans, CblasUnit,
                        matrixsize, blocksize,
                        CBLAS_SADDR(zone), tmp,  stride,
                                           tmp1, stride);

            for(col = 0; col < blocksize; col++) {
                /** Copy Lk+1,k*Dk,k and compute Lk+1,k */
                cblas_zcopy(matrixsize, tmp1+col*stride,     1, 
                                        work+col*matrixsize, 1);

                alpha = 1. / *(tmp + col*(stride+1));
                
                cblas_zscal(matrixsize, CBLAS_SADDR(alpha), 
                            tmp1+col*stride, 1);
            }
            
            /* Update Ak+1k+1 = Ak+1k+1 - (Lk+1k*Dk,k)*Lk+1kT */
            cblas_zgemm(CblasColMajor,
                        CblasNoTrans, CblasConjTrans,
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
void core_zhetrfsp1d(Dague_Complex64_t *L,
                     Dague_Complex64_t *work,
                     SolverMatrix *datacode, 
                     dague_int_t c,
                     double criteria)
{
    Dague_Complex64_t *fL;
    dague_int_t    dima, dimb, stride;
    dague_int_t    fblknum, lblknum;
    dague_int_t    nbpivot = 0; /* TODO: return to higher level */

    /* check if diagonal column block */
    assert( SYMB_FCOLNUM(c) == SYMB_FROWNUM(SYMB_BLOKNUM(c)) );

    /* Initialisation des pointeurs de blocs */
    dima   = SYMB_LCOLNUM(c)-SYMB_FCOLNUM(c)+1;
    stride = SOLV_STRIDE(c);

    /* Factorize diagonal block (two terms version with workspace) */
    core_zhetrfsp(dima, L, stride, &nbpivot, criteria, work);

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
                    CblasRight, CblasLower,
                    CblasConjTrans, CblasUnit,
                    dimb, dima,
                    CBLAS_SADDR(zone), L,  stride,
                                       fL, stride);
        
        for (dague_int_t k=0; k<dima; k++)
        {
          Dague_Complex64_t alpha;
          alpha = 1. / L[k+k*stride];
          cblas_zscal(dimb, CBLAS_SADDR(alpha), &(fL[k*stride]), 1);
        }
    }
}


void core_zhetrfsp1d_gemm(dague_int_t cblknum,
                          dague_int_t bloknum,
                          dague_int_t fcblknum,
                          Dague_Complex64_t *L,
                          Dague_Complex64_t *C,
                          Dague_Complex64_t *work1,
                          Dague_Complex64_t *work2,
                          SolverMatrix *datacode)
{
    Dague_Complex64_t *Aik, *Aij;
    dague_int_t fblknum, lblknum, frownum;
    dague_int_t stride, stridefc, indblok;
    dague_int_t b, j;
    dague_int_t dimi, dimj, dima, dimb;
    dague_int_t ldw = SOLV_COEFMAX;

    fblknum = SYMB_BLOKNUM(cblknum);
    lblknum = SYMB_BLOKNUM(cblknum + 1);

    indblok = SOLV_COEFIND(bloknum);
    stride  = SOLV_STRIDE(cblknum);

    dimi = stride - indblok;
    dimj = SYMB_LROWNUM(bloknum) - SYMB_FROWNUM(bloknum) + 1;
    dima = SYMB_LCOLNUM(cblknum) - SYMB_FCOLNUM(cblknum) + 1;  

    /* Matrix A = Aik */
    Aik = L + indblok;

    /* Compute the contribution */
    CORE_zgemdm( PlasmaNoTrans, PlasmaConjTrans, 
                 dimi, dimj, dima,
                 1.,  Aik,   stride, 
                      Aik,   stride,
                 0.,  work1, dimi,
                      L,     stride+1, 
                      work2, ldw );
  
    /*
     * Add contribution to facing cblk
     */
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

        CORE_zaxpy( dimb, dimj, -1.0,
                    work1, dimi,
                    Aij,   stridefc );

        /* Displacement to next block */
        work1 += dimb;
    }
}
