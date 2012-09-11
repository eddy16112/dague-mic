/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 */
#ifndef _CORE_Z_H_
#define _CORE_Z_H_

void core_zgetrfsp1d(dague_complex64_t *L,
                     dague_complex64_t *U,
                     SolverMatrix *datacode,
                     dague_int_t c,
                     double criteria);

void core_zgetrfsp1d_gemm(dague_int_t cblknum,
                          dague_int_t bloknum,
                          dague_int_t fcblknum,
                          dague_complex64_t *L,
                          dague_complex64_t *U,
                          dague_complex64_t *Cl,
                          dague_complex64_t *Cu,
                          dague_complex64_t *work,
                          SolverMatrix *datacode);


void core_zhetrfsp1d(dague_complex64_t *L,
                     dague_complex64_t *work,
                     SolverMatrix *datacode,
                     dague_int_t c,
                     double criteria);

void core_zhetrfsp1d_gemm(dague_int_t cblknum,
                          dague_int_t bloknum,
                          dague_int_t fcblknum,
                          dague_complex64_t *L,
                          dague_complex64_t *C,
                          dague_complex64_t *work1,
                          dague_complex64_t *work2,
                          SolverMatrix *datacode);

void core_zpotrfsp1d(dague_complex64_t *L,
                     SolverMatrix *datacode,
                     dague_int_t c,
                     double criteria);

void core_zpotrfsp1d_gemm(dague_int_t cblknum,
                          dague_int_t bloknum,
                          dague_int_t fcblknum,
                          dague_complex64_t *L,
                          dague_complex64_t *C,
                          dague_complex64_t *work,
                          SolverMatrix *datacode);

void core_zsytrfsp1d(dague_complex64_t *L,
                     dague_complex64_t *work,
                     SolverMatrix *datacode,
                     dague_int_t c,
                     double criteria);

void core_zsytrfsp1d_gemm(dague_int_t cblknum,
                          dague_int_t bloknum,
                          dague_int_t fcblknum,
                          dague_complex64_t *L,
                          dague_complex64_t *C,
                          dague_complex64_t *work1,
                          dague_complex64_t *work2,
                          SolverMatrix *datacode);

#endif /* _CORE_Z_H_ */
