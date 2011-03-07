/**
 *
 * @file core_zddtrf.c
 *
 *  PLASMA core_blas kernel
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver
 *
 * @version 2.3.1
 * @author Mathieu Faverge
 * @date 2010-11-15
 * @precisions normal z -> c d s
 *
 **/
#include "common.h"

/***************************************************************************//**
 *
 * @ingroup CORE_PLASMA_Complex64_t
 *
 *  CORE_zddtrf computes an LU factorization of a general diagonal dominant 
 *  M-by-N matrix A witout pivoting.
 *
 *  The factorization has the form
 *     A = L * U
 *  where L is lower triangular with unit
 *  diagonal elements (lower trapezoidal if m > n), and U is upper
 *  triangular (upper trapezoidal if m < n).
 *
 *  This is the right-looking Level 3 BLAS version of the algorithm.
 *
 *******************************************************************************
 *
 *  @param[in] M
 *          The number of rows of the matrix A.  M >= 0.
 *
 *  @param[in] N
 *          The number of columns of the matrix A.  N >= 0.
 *
 *  @param[in] NB
 *          The block size to switch between blocked and unblocked code
 *
 *  @param[in,out] A
 *          On entry, the M-by-N matrix to be factored.
 *          On exit, the factors L and U from the factorization
 *          A = P*L*U; the unit diagonal elements of L are not stored.
 *
 *  @param[in] LDA
 *          The leading dimension of the array A.  LDA >= max(1,M).
 *
 *  @param[out] INFO
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
 *                has been completed, but the factor U is exactly
 *                singular, and division by zero will occur if it is used
 *                to solve a system of equations.
 *
 *******************************************************************************
 *
 * @return
 *         \retval PLASMA_SUCCESS successful exit
 *         \retval <0 if INFO = -k, the k-th argument had an illegal value
 *         \retval >0 if INFO = k, U(k,k) is exactly zero. The factorization
 *              has been completed, but the factor U is exactly
 *              singular, and division by zero will occur if it is used
 *              to solve a system of equations.
 *
 ******************************************************************************/
int CORE_zddtrf(int M, int N, int IB,
                PLASMA_Complex64_t *A, int LDA, int *INFO)
{
    PLASMA_Complex64_t zone  = (PLASMA_Complex64_t)1.0;
    PLASMA_Complex64_t mzone = (PLASMA_Complex64_t)-1.0;
    int i, j, k, sb;
    int iinfo;

    /* Check input arguments */
    *INFO = 0;
    if (M < 0) {
        coreblas_error(1, "Illegal value of M");
        return -1;
    }
    if (N < 0) {
        coreblas_error(2, "Illegal value of N");
        return -2;
    }
    if (IB < 0) {
        coreblas_error(3, "Illegal value of IB");
        return -3;
    }
    if ((LDA < max(1,M)) && (M > 0)) {
        coreblas_error(5, "Illegal value of LDA");
        return -5;
    }

    /* Quick return */
    if ((M == 0) || (N == 0) || (IB == 0))
        return PLASMA_SUCCESS;

    k = min(M, N);
    for(i =0 ; i < k; i += IB) {
        sb = min(IB, k-i);
        /*
         * Factor diagonal and subdiagonal blocks and test for exact singularity.
         */
        iinfo = CORE_zddtf2( M-i, sb, &A[LDA*i+i], LDA );

        /*
         * Adjust INFO
         */
        if((*INFO == 0) && (iinfo > 0))
            *INFO = iinfo + i;

        if (i+sb < N) {
            /*  Compute block row of U */
            CORE_ztrsm( PlasmaLeft, PlasmaLower, 
                        PlasmaNoTrans, PlasmaUnit,
                        sb, N-(i+sb), 
                        zone, &A[LDA*i+i],      LDA,
                              &A[LDA*(i+sb)+i], LDA);

            if ( i+sb < M )
            {
                /* Update trailing submatrix */
                CORE_zgemm(
                    PlasmaNoTrans, PlasmaNoTrans,
                    M-(i+sb), N-(i+sb), sb,
                    mzone, &A[LDA*i     + i+sb], LDA,
                           &A[LDA*(i+sb)+ i   ], LDA,
                    zone,  &A[LDA*(i+sb)+ i+sb], LDA);
            }
        }
    }
    return PLASMA_SUCCESS;
}
