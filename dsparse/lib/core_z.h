/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 */
#ifndef _CORE_Z_H_
#define _CORE_Z_H_

void core_zgetrfsp1d(Dague_Complex64_t *L,
                     Dague_Complex64_t *U,
                     SolverMatrix *datacode,
                     dague_int_t c,
                     double criteria);

void core_zgetrfsp1d_gemm(dague_int_t cblknum,
                          dague_int_t bloknum,
                          dague_int_t fcblknum,
                          Dague_Complex64_t *L,
                          Dague_Complex64_t *U,
                          Dague_Complex64_t *Cl,
                          Dague_Complex64_t *Cu,
                          Dague_Complex64_t *work,
                          SolverMatrix *datacode);


void core_zhetrfsp1d(Dague_Complex64_t *L,
                     Dague_Complex64_t *work,
                     SolverMatrix *datacode,
                     dague_int_t c,
                     double criteria);

void core_zhetrfsp1d_gemm(dague_int_t cblknum,
                          dague_int_t bloknum,
                          dague_int_t fcblknum,
                          Dague_Complex64_t *L,
                          Dague_Complex64_t *C,
                          Dague_Complex64_t *work1,
                          Dague_Complex64_t *work2,
                          SolverMatrix *datacode);

void core_zpotrfsp1d(Dague_Complex64_t *L,
                     SolverMatrix *datacode,
                     dague_int_t c,
                     double criteria);

void core_zpotrfsp1d_gemm(dague_int_t cblknum,
                          dague_int_t bloknum,
                          dague_int_t fcblknum,
                          Dague_Complex64_t *L,
                          Dague_Complex64_t *C,
                          Dague_Complex64_t *work,
                          SolverMatrix *datacode);

void core_zsytrfsp1d(Dague_Complex64_t *L,
                     Dague_Complex64_t *work,
                     SolverMatrix *datacode,
                     dague_int_t c,
                     double criteria);

void core_zsytrfsp1d_gemm(dague_int_t cblknum,
                          dague_int_t bloknum,
                          dague_int_t fcblknum,
                          Dague_Complex64_t *L,
                          Dague_Complex64_t *C,
                          Dague_Complex64_t *work1,
                          Dague_Complex64_t *work2,
                          SolverMatrix *datacode);

#endif /* _CORE_Z_H_ */
