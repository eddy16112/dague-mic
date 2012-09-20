/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 */
#ifndef _DSPARSE_Z_H_
#define _DSPARSE_Z_H_

/***********************************************************
 *               Blocking interface 
 */
int dsparse_zcsc2cblk( dague_context_t *dague, sparse_matrix_desc_t *A );
int dsparse_zpotrf_sp( dague_context_t *dague, sparse_matrix_desc_t *A );
int dsparse_zgetrf_sp( dague_context_t *dague, sparse_matrix_desc_t *A );
int dsparse_zhetrf_sp( dague_context_t *dague, sparse_matrix_desc_t *A );
int dsparse_zsytrf_sp( dague_context_t *dague, sparse_matrix_desc_t *A );

int dsparse_zpotrs_sp( dague_context_t *dague, sparse_matrix_desc_t *A, sparse_vector_desc_t *B );
int dsparse_zgetrs_sp( dague_context_t *dague, sparse_matrix_desc_t *A, sparse_vector_desc_t *B );
int dsparse_zhetrs_sp( dague_context_t *dague, sparse_matrix_desc_t *A, sparse_vector_desc_t *B );
int dsparse_zsytrs_sp( dague_context_t *dague, sparse_matrix_desc_t *A, sparse_vector_desc_t *B );

/***********************************************************
 *             Non-Blocking interface
 */
dague_object_t* dsparse_zcsc2cblk_New( sparse_matrix_desc_t *A );
dague_object_t* dsparse_zpotrf_sp_New( sparse_matrix_desc_t *A );
dague_object_t* dsparse_zgetrf_sp_New( sparse_matrix_desc_t *A );
dague_object_t* dsparse_zhetrf_sp_New( sparse_matrix_desc_t *A );
dague_object_t* dsparse_zsytrf_sp_New( sparse_matrix_desc_t *A );

/***********************************************************
 *               Destruct functions
 */
void dsparse_zcsc2cblk_Destruct( dague_object_t *o );
void dsparse_zpotrf_sp_Destruct( dague_object_t *o );
void dsparse_zgetrf_sp_Destruct( dague_object_t *o );
void dsparse_zhetrf_sp_Destruct( dague_object_t *o );
void dsparse_zsytrf_sp_Destruct( dague_object_t *o );

#endif /* _DSPARSE_Z_H_ */