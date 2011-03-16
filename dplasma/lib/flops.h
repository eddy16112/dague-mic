/**
 *
 * @file flops.h
 *
 *  File provided by Univ. of Tennessee,
 *
 * @version 1.0.0
 * @author Mathieu Faverge
 * @date 2010-12-20
 *
 **/
/*
 * This file provide the flops formula for all Level 3 BLAS and some
 * Lapack routines.  Each macro uses the same size parameters as the
 * function associated and provide one formula for additions and one
 * for multiplications. Ecample to use these macros:
 *  - In real:
 *    flops = FMULS_GEMM((double)m, (double(n), (double(k)) 
 *          + FADDS_GEMM((double)m, (double(n), (double(k));
 *
 *  - In complex:
 *    flops = 6.0 * FMULS_GEMM((double)m, (double(n), (double(k)) 
 *          + 2.0 * FADDS_GEMM((double)m, (double(n), (double(k));
 *
 * All the formula are reported in the LAPACK Lawn 41:
 *     http://www.netlib.org/lapack/lawns/lawn41.ps
 */
#ifndef _FLOPS_H_
#define _FLOPS_H_

/*
 * Level 2 BLAS 
 */  
#define FMULS_GEMV(__m, __n) ((__m) * (__n) + 2. * (__m))
#define FADDS_GEMV(__m, __n) ((__m) * (__n)           )

#define FMULS_SYMV(__n) ((__n) * (__n) + 2. * (__n))
#define FADDS_SYMV(__n) ((__n) * (__n)           )
#define FMULS_HEMV FMULS_SYMV
#define FADDS_HEMV FADDS_SYMV

/*
 * Level 3 BLAS 
 */
#define FMULS_GEMM(__m, __n, __k) ((__m) * (__n) * (__k))
#define FADDS_GEMM(__m, __n, __k) ((__m) * (__n) * (__k))

#define FMULS_SYMM_L(__m, __n) ((__m) * (__m) * (__n))
#define FADDS_SYMM_L(__m, __n) ((__m) * (__m) * (__n))
#define FMULS_HEMM_L FMULS_SYMM_L
#define FADDS_HEMM_L FADDS_SYMM_L

#define FMULS_SYMM_R(__m, __n) ((__m) * (__n) * (__n))
#define FADDS_SYMM_R(__m, __n) ((__m) * (__n) * (__n))
#define FMULS_HEMM_R FMULS_SYMM_R
#define FADDS_HEMM_R FADDS_SYMM_R

#define FMULS_SYRK(__k, __n) (0.5 * (__k) * (__n) * ((__n)+1))
#define FADDS_SYRK(__k, __n) (0.5 * (__k) * (__n) * ((__n)+1))
#define FMULS_HERK FMULS_SYRK
#define FADDS_HERK FADDS_SYRK

#define FMULS_SYR2K(__k, __n) ((__k) * (__n) * (__n)      )
#define FADDS_SYR2K(__k, __n) ((__k) * (__n) * (__n) + (__n))
#define FMULS_HER2K FMULS_SYR2K
#define FADDS_HER2K FADDS_SYR2K

#define FMULS_TRMM_L(__m, __n) (0.5 * (__n) * (__m) * ((__m)+1))
#define FADDS_TRMM_L(__m, __n) (0.5 * (__n) * (__m) * ((__m)-1))

#define FMULS_TRMM_R(__m, __n) (0.5 * (__m) * (__n) * ((__n)+1))
#define FADDS_TRMM_R(__m, __n) (0.5 * (__m) * (__n) * ((__n)-1))

#define FMULS_TRSM_L(__m, __n) (0.5 * (__n) * (__m) * ((__m)+1))
#define FADDS_TRSM_L(__m, __n) (0.5 * (__n) * (__m) * ((__m)-1))

#define FMULS_TRSM_R(__m, __n) (0.5 * (__m) * (__n) * ((__n)+1))
#define FADDS_TRSM_R(__m, __n) (0.5 * (__m) * (__n) * ((__n)-1))

/*
 * Lapack
 */
#define FMULS_GETRF(__m, __n) ( ((__m) < (__n)) ? (0.5 * (__m) * ((__m) * ((__n) - (1./3.) * (__m) - 1. ) + (__n)) + (2. / 3.) * (__m)) \
 			    :             (0.5 * (__n) * ((__n) * ((__m) - (1./3.) * (__n) - 1. ) + (__m)) + (2. / 3.) * (__n)) )
#define FADDS_GETRF(__m, __n) ( ((__m) < (__n)) ? (0.5 * (__m) * ((__m) * ((__n) - (1./3.) * (__m)      ) - (__n)) + (1. / 6.) * (__m)) \
			    :             (0.5 * (__n) * ((__n) * ((__m) - (1./3.) * (__n)      ) - (__m)) + (1. / 6.) * (__n)) )

#define FMULS_GETRI(__n) ( (__n) * ((5. / 6.) + (__n) * ((2. / 3.) * (__n) + 0.5)) )
#define FADDS_GETRI(__n) ( (__n) * ((5. / 6.) + (__n) * ((2. / 3.) * (__n) - 1.5)) )

#define FMULS_GETRS(__n, __nrhs) ((__nrhs) * (__n) *  (__n)      )
#define FADDS_GETRS(__n, __nrhs) ((__nrhs) * (__n) * ((__n) - 1 ))

#define FMULS_POTRF(__n) ((__n) * (((1. / 6.) * (__n) + 0.5) * (__n) + (1. / 3.)))
#define FADDS_POTRF(__n) ((__n) * (((1. / 6.) * (__n)      ) * (__n) - (1. / 6.)))

#define FMULS_POTRI(__n) ( (__n) * ((2. / 3.) + (__n) * ((1. / 3.) * (__n) + 1. )) )
#define FADDS_POTRI(__n) ( (__n) * ((1. / 6.) + (__n) * ((1. / 3.) * (__n) - 0.5)) )

#define FMULS_POTRS(__n, __nrhs) ((__nrhs) * (__n) * ((__n) + 1 ))
#define FADDS_POTRS(__n, __nrhs) ((__nrhs) * (__n) * ((__n) - 1 ))

//SPBTRF
//SPBTRS
//SSYTRF
//SSYTRI
//SSYTRS

#define FMULS_GEQRF(__m, __n) (((__m) > (__n)) ? ((__n) * ((__n) * (  0.5-(1./3.) * (__n) + (__m)) +    (__m) + 23. / 6.)) \
                                       : ((__m) * ((__m) * ( -0.5-(1./3.) * (__m) + (__n)) + 2.*(__n) + 23. / 6.)) )
#define FADDS_GEQRF(__m, __n) (((__m) > (__n)) ? ((__n) * ((__n) * (  0.5-(1./3.) * (__n) + (__m))          +  5. / 6.)) \
                                       : ((__m) * ((__m) * ( -0.5-(1./3.) * (__m) + (__n)) +    (__n) +  5. / 6.)) )

#define FMULS_GEQLF(__m, __n) FMULS_GEQRF(__m, __n)
#define FADDS_GEQLF(__m, __n) FADDS_GEQRF(__m, __n)

#define FMULS_GERQF(__m, __n) (((__m) > (__n)) ? ((__n) * ((__n) * (  0.5-(1./3.) * (__n) + (__m)) +    (__m) + 29. / 6.)) \
                                       : ((__m) * ((__m) * ( -0.5-(1./3.) * (__m) + (__n)) + 2.*(__n) + 29. / 6.)) )
#define FADDS_GERQF(__m, __n) (((__m) > (__n)) ? ((__n) * ((__n) * ( -0.5-(1./3.) * (__n) + (__m)) +    (__m) +  5. / 6.)) \
                                       : ((__m) * ((__m) * (  0.5-(1./3.) * (__m) + (__n)) +        +  5. / 6.)) )

#define FMULS_GELQF(__m, __n) FMULS_GERQF(__m, __n)
#define FADDS_GELQF(__m, __n) FADDS_GERQF(__m, __n)

#define FMULS_UNGQR(__m, __n, __k) ((__k) * (2.* (__m) * (__n) +  2. * (__n) - 5./3. + (__k) * ( 2./3. * (__k) - ((__m) + (__n)) - 1.)))
#define FADDS_UNGQR(__m, __n, __k) ((__k) * (2.* (__m) * (__n) + (__n) - (__m) + 1./3. + (__k) * ( 2./3. * (__k) - ((__m) + (__n))     )))
#define FMULS_UNGQL FMULS_UNGQR
#define FMULS_ORGQR FMULS_UNGQR
#define FMULS_ORGQL FMULS_UNGQR
#define FADDS_UNGQL FADDS_UNGQR
#define FADDS_ORGQR FADDS_UNGQR
#define FADDS_ORGQL FADDS_UNGQR

#define FMULS_UNGRQ(__m, __n, __k) ((__k) * (2.* (__m) * (__n) + (__m) + (__n) - 2./3. + (__k) * ( 2./3. * (__k) - ((__m) + (__n)) - 1.)))
#define FADDS_UNGRQ(__m, __n, __k) ((__k) * (2.* (__m) * (__n) + (__m) - (__n) + 1./3. + (__k) * ( 2./3. * (__k) - ((__m) + (__n))     )))
#define FMULS_UNGLQ FMULS_UNGRQ
#define FMULS_ORGRQ FMULS_UNGRQ
#define FMULS_ORGLQ FMULS_UNGRQ
#define FADDS_UNGLQ FADDS_UNGRQ
#define FADDS_ORGRQ FADDS_UNGRQ
#define FADDS_ORGLQ FADDS_UNGRQ

#define FMULS_GEQRS(__m, __n, __nrhs) ((__nrhs) * ((__n) * ( 2.* (__m) - 0.5 * (__n) + 2.5)))
#define FADDS_GEQRS(__m, __n, __nrhs) ((__nrhs) * ((__n) * ( 2.* (__m) - 0.5 * (__n) + 0.5)))

//UNMQR, UNMLQ, UNMQL, UNMRQ (Left)
//UNMQR, UNMLQ, UNMQL, UNMRQ (Right)

#define FMULS_TRTRI(__n) ((__n) * ((__n) * ( 1./6. * (__n) + 0.5 ) + 1./3.))
#define FADDS_TRTRI(__n) ((__n) * ((__n) * ( 1./6. * (__n) - 0.5 ) + 1./3.))

#define FMULS_GEHRD(__n) ( (__n) * ((__n) * (5./3. *(__n) + 0.5) - 7./6.) - 13. )
#define FADDS_GEHRD(__n) ( (__n) * ((__n) * (5./3. *(__n) - 1. ) - 2./3.) -  8. )

#define FMULS_SYTRD(__n) ( (__n) *  ( (__n) * ( 2./3. * (__n) + 2.5 ) - 1./6. ) - 15.)
#define FADDS_SYTRD(__n) ( (__n) *  ( (__n) * ( 2./3. * (__n) + 1.  ) - 8./3. ) -  4.)
#define FMULS_HETRD FMULS_SYTRD
#define FADDS_HETRD FADDS_SYTRD

#define FMULS_GEBRD(__m, __n) ( ((__m) >= (__n)) ? ((__n) * ((__n) * (2. * (__m) - 2./3. * (__n) + 2. )       + 20./3.)) \
                            :              ((__m) * ((__m) * (2. * (__n) - 2./3. * (__m) + 2. )       + 20./3.)) )
#define FADDS_GEBRD(__m, __n) ( ((__m) >= (__n)) ? ((__n) * ((__n) * (2. * (__m) - 2./3. * (__n) + 1. ) - (__m) +  5./3.)) \
                            :              ((__m) * ((__m) * (2. * (__n) - 2./3. * (__m) + 1. ) - (__n) +  5./3.)) )

#define FLOPS_GEMM(__m, __n, __k) (FMULS_GEMM((__m), (__n), (__k)) + FADDS_GEMM((__m), (__n), (__k)))
#define FLOPS_SYRK(__k, __n)    (FMULS_SYRK(__k, __n)    + FADDS_SYRK(__k, __n))
#define FLOPS_HERK FLOPS_SYRK

#define FLOPS_TRSM_R(__m, __n)  (FMULS_TRSM_R(__m, __n)  + FADDS_TRSM_R(__m, __n))
#define FLOPS_TRSM_L(__m, __n)  (FMULS_TRSM_L(__m, __n)  + FADDS_TRSM_L(__m, __n))

#define FLOPS_POTRF(__n)      (FMULS_POTRF(__n)      + FADDS_POTRF(__n))

#endif /* _FLOPS_H_ */
