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
#include <string.h> // for memcpy()


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
    (void)uplo;

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

typedef struct dague_tssm_dope_vector{
    uint64_t extent;
    uint64_t map_elem_count;
    dague_tssm_data_map_elem_t map_elems[1];
}dague_tssm_dope_vector_t;


/*
 * At the end of this function the buffer pointed to by the first parameter (compr_tile_ptr) will contain:
 * - A uint64_t holding the extent of the buffer, in bytes.
 * - A uint64_t holding the number of data rectangles in the buffer (say N).
 * - N structs of type dague_tssm_data_map_elem_t "describing" the N rectangles. However the "ptr" element of
 *   each map element will be the offset into the data part of the compressed buffer (cumulative size).
 * - The data of the N rectangles.
 */
void dague_tssm_ztile_compress(void **compr_tile_ptr, dague_tssm_data_map_t *map)
{
    Dague_Complex64_t *cmprsd_buffer;
    dague_int_t i=0;
    uint64_t compressed_size, cumulative_size, data_offset, dope_vector_size;
    dague_tssm_dope_vector_t *dope_vector;

    assert( map );

    /* The "-1" is because there is already room for 1 map element in the struct. */
    dope_vector_size = sizeof(dague_tssm_dope_vector_t) + (map->map_elem_count-1)*sizeof(dague_tssm_data_map_elem_t);

    /* Allocate the compressed buffer */
    compressed_size = map->filled_data*sizeof(Dague_Complex64_t) + dope_vector_size;
    cmprsd_buffer = (Dague_Complex64_t*)calloc(compressed_size, 1);
    assert( cmprsd_buffer );

    dope_vector = (dague_tssm_dope_vector_t *)cmprsd_buffer;

    /* Pass the pointer to the compressed buffer to the caller */
    *compr_tile_ptr = cmprsd_buffer;

    dope_vector->extent = compressed_size;
    dope_vector->map_elem_count = map->map_elem_count;

    cumulative_size = 0;
    do {
        intptr_t dst_ptr;
        dague_tssm_data_map_elem_t *mp = &(map->elements[i]);

        /* Copy the data of this rectangle into the buffer. */
        dst_ptr = (intptr_t)cmprsd_buffer + dope_vector_size + cumulative_size;
        dague_zlacpy('A', mp->h, mp->w, (Dague_Complex64_t*)mp->ptr, mp->ldA, (void *)dst_ptr, mp->h);

        /* Copy the map element into the compressed buffer and overwrite the pointer stored
	 * in the map element to point to where we just copied the data in the buffer.
	 */
        memcpy( &(dope_vector->map_elems[i]), mp, sizeof(dague_tssm_data_map_elem_t) );
        dope_vector->map_elems[i].ptr = (void *)cumulative_size;

        cumulative_size += mp->h*mp->w*sizeof(Dague_Complex64_t);

        ++i;
    } while( NULL != map->elements[i].ptr );

    return;
}
