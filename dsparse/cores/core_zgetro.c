/**
 *
 * @file core_zgetro.c
 *
 *  PLASMA InPlaceTransformation module
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver
 *
 *  This work is the implementation of an inplace transformation 
 *  based on the GKK algorithm by Gustavson, Karlsson, Kagstrom 
 *  and its fortran implementation.
 *
 * @version 2.4.2
 * @author Mathieu Faverge
 * @date 2010-11-15
 *
 * @precisions normal z -> c d s
 *
 **/

#include <plasma.h>
#include <core_blas.h>
#include <lapacke.h>
#include "dague.h"

/** ****************************************************************************
 *
 * @ingroup InPlaceTransformation
 * @ingroup CORE_PLASMA_Complex64_t
 *
 *  CORE_zgetro transposes a m-by-n matrix in place using an extra
 *      workspace of size m-by-n.  
 *      Note : For square tile, workspace is not used.
 *
 *******************************************************************************
 *
 * @param[in] m
 *         Number of lines of tile A
 *
 * @param[in] n
 *         Number of columns of tile A
 *
 * @param[in,out] A
 *         Tile of size m-by-n
 *         On exit, A = trans(A)
 *
 * @param[out] W
 *         Workspace of size n-by-m if n != m, NULL otherwise.
 *
 ******************************************************************************/
#if defined(PLASMA_HAVE_WEAK)
#pragma weak CORE_zgetro = PCORE_zgetro
#define CORE_zgetro PCORE_zgetro
#endif
void CORE_zgetro(int m, int n, 
                 PLASMA_Complex64_t *A, int lda,
                 PLASMA_Complex64_t *B, int ldb) 
{
    int i, j;
    
    /* rectangular transposition (use workspace) */
    for (i=0; i<m; i++) {
        for (j=0; j<n; j++) {
            B[j+i*ldb] = A[i+j*lda];
        }
    }
}

#if defined(PLASMA_HAVE_WEAK)
#pragma weak CORE_zaxpyt = PCORE_zaxpyt
#define CORE_zaxpyt PCORE_zaxpyt
#endif
void CORE_zaxpyt(int m, int n, PLASMA_Complex64_t alpha,
                 PLASMA_Complex64_t *A, int lda,
                 PLASMA_Complex64_t *B, int ldb) 
{
    int i, j;
    
    /* rectangular transposition (use workspace) */
    for (j=0; j<n; j++) {
        for (i=0; i<m; i++) {
            B[j*ldb+i] = B[j*ldb+i] + alpha * A[j+i*lda];
        }
    }
}
