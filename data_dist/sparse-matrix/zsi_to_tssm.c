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


/* At the end of this function the pointer pointed to by the first parameter "compr_tile_ptr" will contain:
 * - A uint64_t holding the number of rectangles in the buffer (say N).
 * - N structs of type dague_tssm_data_map_elem_t "describing" the N rectangles.
 * - The data of the N rectangles.
 * The second parameter "extent" will
 */
void dague_tssm_ztile_compress(void **compr_tile_ptr, dague_int_t *extent, dague_tssm_data_map_t *map)
{
    Dague_Complex64_t *cmprsd_buffer;
    dague_int_t i=0;
    uint64_t compressed_size, cumulative_size, data_offset, counter_offset;

    assert( map );

    counter_offset = sizeof(uint64_t);
    compressed_size = map->filled_data*sizeof(Dague_Complex64_t) + map->map_elem_count*sizeof(dague_tssm_data_map_elem_t) + counter_offset;

    // Pass the extent of this buffer to the caller
    *extent = compressed_size;

    cmprsd_buffer = (Dague_Complex64_t*)calloc(compressed_size, 1);
    assert( cmprsd_buffer );

    // Pass the pointer to the compressed buffer to the caller
    *compr_tile_ptr = cmprsd_buffer;

    // The first "data_offset" bytes will be occupied by meta-data
    data_offset = map->map_elem_count*sizeof(dague_tssm_data_map_elem_t);

    cumulative_size = 0;
    do {
        intptr_t dst_ptr;
        dague_tssm_data_map_elem_t mp = map->elements[i];

        /* Copy the data of this rectangle into the buffer. */
        dst_ptr = (intptr_t)cmprsd_buffer + counter_offset + data_offset + cumulative_size;
#warning "Need to verify this with someone. Especially that 'mp.h' at the end."
        dague_zlacpy('A', mp.h, mp.w, (Dague_Complex64_t*)mp.ptr, mp.ldA, (void *)dst_ptr, mp.h);

        cumulative_size += mp.h*mp.w*sizeof(Dague_Complex64_t);

        /* Overwrite the pointer stored in the map element to point to where we just copied the data in the buffer. */
        mp.ptr = (void *)dst_ptr;
        /* Copy the meta-data of this map element into the buffer. */
        dst_ptr = (intptr_t)cmprsd_buffer + counter_offset + i*sizeof(dague_tssm_data_map_elem_t);
        memcpy((void *)dst_ptr, &mp, sizeof(dague_tssm_data_map_elem_t));

        ++i;
    } while( NULL != map->elements[i].ptr );
    *((uint64_t *)cmprsd_buffer) = i;

    return;
}



