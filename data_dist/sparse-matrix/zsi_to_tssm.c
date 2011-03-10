/**
 *
 * @file zsi_to_tssm.c
 *
 * @author Anthony Danalis
 * @date 2011-03-01
 * @precisions normal z -> c d s
 *
 **/
#include "dague_config.h"

#include <assert.h>

#include "data_dist/matrix/precision.h"
#include "data_dist/sparse-matrix/sparse-input.h"
#include "data_dist/sparse-matrix/si-to-tssm.h"
#include "data_dist/sparse-matrix/sparse-shm-matrix.h"

void dague_zlacpy( char uplo, dague_int_t m, dague_int_t n, Dague_Complex64_t *A, dague_int_t lda, Dague_Complex64_t *B, dague_int_t ldb)
{
    dague_int_t i, j;
    
    /* For now, only general case is implemented, need to implement upper and 
     * lower and use it in scalapack/lapack convert instead of the actual function */
    uplo = 'U';

    for ( j=0; j<n; j++ )
    {
        for( i = 0; i<m; i++, A++, B++)
            *B = *A;
        A += lda - m;
        B += ldb - m;
    }
}

/*
 * unpack() copies data from the block based compresed representation of the
 * sparse matrix into the memory pointed to by the first argument "tile_ptr".
 * The caller is responsible for allocating/deallocating the memory pointed
 * to by this pointer.
 */
int dague_tssm_ztile_unpack(void *tile_ptr, dague_int_t m, dague_int_t n, dague_int_t mb, dague_int_t nb, dague_tssm_data_map_t *map)
{
    Dague_Complex64_t *data = (Dague_Complex64_t*)tile_ptr;
    dague_int_t i=0;

    (void)m;
    (void)n;
    (void)nb;

    assert( map );
    do {
        dague_tssm_data_map_elem_t *mp = &map->elements[i++];
        dague_zlacpy('A', mp->h, mp->w, (Dague_Complex64_t*)mp->ptr, mp->ldA, data + mp->offset, mb);
    } while( NULL != map->elements[i].ptr );

    return i;
}

/*
 * pack() copies data from the memory pointed to by the first argument
 * "tile_ptr", into the block based, compresed representation of the
 * sparse matrix. The caller is responsible for allocating/deallocating
 * the memory pointed to by "tile_ptr".
 */
void dague_tssm_ztile_pack(void *tile_ptr, dague_int_t m, dague_int_t n, dague_int_t mb, dague_int_t nb, dague_tssm_data_map_t *map)
{
    Dague_Complex64_t *data = (Dague_Complex64_t*)tile_ptr;
    dague_int_t i=0;

    (void)m;
    (void)n;
    (void)nb;

    assert( map );
    do {
        dague_tssm_data_map_elem_t *mp = &map->elements[i++];
        dague_zlacpy('A', mp->h, mp->w, data + mp->offset, mb, (Dague_Complex64_t*)mp->ptr, mp->ldA);
    } while( NULL != map->elements[i].ptr );

    return;
}

