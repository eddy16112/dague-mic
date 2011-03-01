#include "dague_config.h"

#include <assert.h>

#include "data_dist/sparse-matrix/si-to-tssm.h"
#include "data_dist/sparse-matrix/sparse-shm-matrix.h"

#ifdef GEN_DEBUG_PIXMAP
# include "data_dist/sparse-matrix/debug-png-generation.h"
#endif

/** TODO: include Mathieu's definition file here */
extern int zlacpy(const char *name, int h, int w, void *ptrA, int ldA, void *ptrB, int ldB);

#undef MIN
#define MIN(_X , _Y ) (( (_X) < (_Y) ) ? (_X) : (_Y))
#undef MAX
#define MAX(_X , _Y ) (( (_X) > (_Y) ) ? (_X) : (_Y))

/*
 * unpack() copies data from the block based compresed representation of the
 * sparse matrix into the memory pointed to by the first argument "tile_ptr".
 * The caller is responsible for allocating/deallocating the memory pointed
 * to by this pointer.
 */
int dague_tssm_sparse_tile_unpack(void *tile_ptr, uint64_t m, uint64_t n, uint64_t mb, uint64_t nb, dague_tssm_data_map_t *map)
{
    uint64_t i=0;

    (void)m;
    (void)n;
    (void)nb;

    assert( map );
    do {
        dague_tssm_data_map_t *mp = &map[i++];
        zlacpy("A", mp->h, mp->w, mp->ptr, mp->ldA, (void*)(((uintptr_t)tile_ptr)+mp->offset*ELEM_SIZE), mb);
    } while( NULL != map[i].ptr );

    return i;
}

/*
 * pack() copies data from the memory pointed to by the first argument
 * "tile_ptr", into the block based, compresed representation of the
 * sparse matrix. The caller is responsible for allocating/deallocating
 * the memory pointed to by "tile_ptr".
 */
void dague_tssm_sparse_tile_pack(void *tile_ptr, uint64_t m, uint64_t n, uint64_t mb, uint64_t nb, dague_tssm_data_map_t *map)
{
    uint64_t i=0;

    (void)m;
    (void)n;
    (void)nb;

    assert( map );
    do {
        dague_tssm_data_map_t *mp = &map[i++];
        zlacpy("A", mp->h, mp->w, (void*)(((uintptr_t)tile_ptr)+mp->offset*ELEM_SIZE), mb, mp->ptr, mp->ldA);
    } while( NULL != map[i].ptr );

    return;
}

/*
 * dague_pastix_to_tiles_load() reads a (slightly modified) pastix structure and creates
 * a mapping from the blocked columns to tiles so that the pack() and unpack() functions
 * can read/write the data that corresponds to a single tile from/to the pastix data.
 */
void dague_sparse_input_to_tiles_load(dague_tssm_desc_t *mesh, uint64_t mt, uint64_t nt, uint32_t mb, uint32_t nb, 
                                      dague_sparse_input_symbol_matrix_t *sm)
{
    dague_int_t cblknbr = sm->cblknbr; /* Number of column blocks */ 
    dague_sparse_input_symbol_cblk_t * restrict cblktab = sm->cblktab; /* Array of column blocks [+1, based] */
    dague_sparse_input_symbol_blok_t * restrict bloktab = sm->bloktab; /* Array of blocks [based] */
    dague_int_t fbc=0; /* we start from zero, alhthough cblktab's comment says it is "+1 based", because Mathieu said so */
    dague_int_t lbc=0;
    dague_int_t i, j, bc, b;
 
#ifdef GEN_DEBUG_PIXMAP
    dague_int_t bc;
    for(bc=0; bc < cblknbr; bc++){
        dague_int_t b, strCol, endCol, fb, lb;

        strCol = cblktab[bc].fcolnum;
        endCol = cblktab[bc].lcolnum;

        fb=cblktab[bc].bloknum;
        /* The first block in the next column is just one past my last block */
        lb=cblktab[bc+1].bloknum;

        for(b=fb; b<lb; b++){
            dague_int_t endRow, strRow, ptr_offset;
            strRow = bloktab[b].frownum;
            endRow = bloktab[b].lrownum;

            dague_pxmp_si_color_rectangle(strCol, endCol, strRow, endRow, mt*mb, nt*nb);
        }

    }
    dague_pxmp_si_dump_image("test.png");
#endif /* GEN_PIXMAP */

    /* "tmp_map_buf" is guaranteed to fit the maximum number of meta-data
     * entries, since at maximum we can only have an entry per element of the
     * tile. We will work in this buffer for every tile and when we are done
     * with a tile we will allocate the minimum necessary buffer and copy the
     * entries there.
     */
    dague_tssm_data_map_t *tmp_map_buf = (dague_tssm_data_map_t *)calloc(nb*mb, sizeof(dague_tssm_data_map_t));

    for(i=0; i<nt; i++){
       /* We are filling up the meta-data structures of the tiles that are in
        * the i-th column. So, we need to find the first block column that is
        * not after the begining of the i-th tile and the last block column
        * that is not before the end of the i-th tile.
        */
        fbc = lbc; /* start from where we left before */
        for(; (fbc < cblknbr) && (cblktab[fbc].fcolnum < i*nb); fbc++)
            ;
        for(lbc=fbc; (lbc < cblknbr) && (cblktab[lbc].fcolnum < (i+1)*nb); lbc++)
            ;

        /* Now for each tile "j" of this column, populare the map data structure,
         * by looking at every block column from fbc to lbc (inclusive) to see if
         * it has data that belongs to the j-the tile 
         */
        for(j=0; j<mt; j++){
            dague_int_t blocksInTile = 0;
            for(bc=fbc; bc<=lbc; bc++){
                dague_int_t endCol, strCol;
                dague_int_t dx, dy, off_x, off_y, ldA;

                /* Find the edges of the intersection of the i-th column of
                 * tiles and the "bc"-th block column.
                 */
                endCol = MIN( cblktab[bc].lcolnum , (i+1)*nb-1 );
                strCol = MAX( cblktab[bc].fcolnum , i*nb );
                /* The offset from the beginning of this block to the beginning of the tile */
                dx = strCol - cblktab[bc].fcolnum;
                /* The offset from the beginning of the tile to the data of this block */
                off_x = strCol - i*nb;
                ldA = cblktab[bc].stride;

                /* For each block column iterate over all the blocks it contains
                 * and see if they have rows that map to the j-the tile.
                 */
                dague_int_t fb=cblktab[bc].bloknum;
                /* The first block in the next column is just one past my last block */
                dague_int_t lb=cblktab[bc+1].bloknum;

                for(b=fb; b<lb; b++){
                    dague_int_t endRow, strRow, ptr_offset;
                    /* check if the b-th block has a first row that is not bigger than
                     * my last row and a last row that is not smaller than my first row
                     */
                    if( bloktab[b].frownum > (j+1)*mb )
                        continue;
                    if( bloktab[b].lrownum < j*mb )
                        continue;

                    /* Find the edges of the intersection of the j-th row of
                     * tiles and the "b"-th block of the "bc"-th block column.
                     */
                    endRow = MIN( bloktab[b].lrownum , (j+1)*mb-1 );
                    strRow = MAX( bloktab[b].frownum , j*mb );
                    dy = strRow - bloktab[b].frownum;
                    off_y = strRow - j*mb;

                    ptr_offset = dx*ldA + bloktab[b].coefind + dy;
                    tmp_map_buf[blocksInTile].ptr = (void*)( ((uintptr_t)cblktab[bc].cblkptr) + ptr_offset*ELEM_SIZE);
                    tmp_map_buf[blocksInTile].ldA = ldA;
                    tmp_map_buf[blocksInTile].h = endRow - strRow + 1;
                    tmp_map_buf[blocksInTile].w = endCol - strCol + 1;
                    /* this offset is in elements, not in bytes */
                    tmp_map_buf[blocksInTile].offset = off_x*mb + off_y;
#ifdef GEN_PIXMAP
//                    dague_color_rectangle(&tmp_map_buf[blocksInTile], j, i, mt*mb, nt*nb, ELEM_SIZE);
#endif /* GEN_PIXMAP */
                    ++blocksInTile;
                }
            }
            /* Put the meta-data in a buffer with just one extra element */
            dague_tssm_data_map_t *mapEntry = (dague_tssm_data_map_t *)calloc(1+blocksInTile, sizeof(dague_tssm_data_map_t));
            memcpy(mapEntry, tmp_map_buf, blocksInTile*sizeof(dague_tssm_data_map_t));
            mapEntry[blocksInTile].ptr = NULL; /* Just being ridiculous */

            /* Pass the meta-data to the LRU handling code */
            dague_tssm_mesh_create_tile(mesh, j, i, mb, nb, mapEntry);
        }
    }
    return;
}

