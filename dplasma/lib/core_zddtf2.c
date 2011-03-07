/**
 *
 * @file core_zddtf2.c
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
#include <lapacke.h>
#include <cblas.h>
#include <math.h>
#include <plasma.h>
#include <dague.h>
#include "dplasma.h"

#define coreblas_error(info, str) fprintf(stderr, "%s: %d - %s\n", __func__, info, str)
#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif
#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif

/***************************************************************************//**
 *
 * @ingroup CORE_Dague_Complex64_t
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
int CORE_zddtf2(int M, int N, 
                Dague_Complex64_t *A, int LDA, int *info)
{
    Dague_Complex64_t mzone = (Dague_Complex64_t)-1.0;
    Dague_Complex64_t alpha;
    double sfmin;
    int i, j, k;

    /* Check input arguments */
    *info = 0;
    if (M < 0) {
        coreblas_error(1, "Illegal value of M");
        return -1;
    }
    if (N < 0) {
        coreblas_error(2, "Illegal value of N");
        return -2;
    }
    if ((LDA < max(1,M)) && (M > 0)) {
        coreblas_error(5, "Illegal value of LDA");
        return -5;
    }

    /* Quick return */
    if ( (M == 0) || (N == 0) )
        return PLASMA_SUCCESS;

    sfmin = LAPACKE_dlamch_work('S');
    k = min(M, N);
    for(i=0 ; i < k; i++) {
        alpha = A[i*LDA+i];
        if ( alpha != (Dague_Complex64_t)0.0 ) {
            /* Compute elements J+1:M of J-th column. */
            if (i < M) {
                if ( cabs(alpha) > sfmin ) {
                    alpha = 1.0 / alpha;
                    cblas_zscal( M-i-1, CBLAS_SADDR(alpha), &(A[i*LDA+i+1]), 1);
                } else {
                    for(j=i+1; j<M; j++)
                        A[LDA*i+j] = A[LDA*i+j] / alpha;
                }
            }
        } else if ( *info == 0 ) {
            *info = i;
        }
        
        if ( i < k ) {
            /* Update trailing submatrix */
            cblas_zgeru(CblasColMajor, 
                        M-i-1, N-i-1, 
                        mzone, &A[LDA* i   +i+1], 1,
                               &A[LDA*(i+1)+i  ], LDA,
                               &A[LDA*(i+1)+i+1], LDA);
        }
    }
    return PLASMA_SUCCESS;
}
