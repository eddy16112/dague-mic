#include "dague_config.h"

#include <assert.h>

#include "data_dist/matrix/precision.h"
#include "data_dist/sparse-matrix/sparse-input.h"
#include "data_dist/sparse-matrix/si-to-tssm.h"
#include "data_dist/sparse-matrix/sparse-shm-matrix.h"

//#define GEN_DEBUG_PIXMAP
#ifdef GEN_DEBUG_PIXMAP
# include "data_dist/sparse-matrix/debug-png-generation.h"
#endif

#undef MIN
#define MIN(_X , _Y ) (( (_X) < (_Y) ) ? (_X) : (_Y))
#undef MAX
#define MAX(_X , _Y ) (( (_X) > (_Y) ) ? (_X) : (_Y))

/*
 * dague_pastix_to_tiles_load() reads a (slightly modified) pastix structure and creates
 * a mapping from the blocked columns to tiles so that the pack() and unpack() functions
 * can read/write the data that corresponds to a single tile from/to the pastix data.
 */
void dague_sparse_input_to_tiles_load(dague_tssm_desc_t *mesh, dague_int_t mt, dague_int_t nt, uint32_t mb, uint32_t nb, 
                                      dague_sparse_input_symbol_matrix_t *sm)
{
    dague_int_t cblknbr = sm->cblknbr; /* Number of column blocks */ 
    dague_sparse_input_symbol_cblk_t * restrict cblktab = sm->cblktab; /* Array of column blocks [+1, based] */
    dague_sparse_input_symbol_blok_t * restrict bloktab = sm->bloktab; /* Array of blocks [based] */
    dague_int_t fbc=0; /* we start from zero, alhthough cblktab's comment says it is "+1 based", because Mathieu said so */
    dague_int_t lbc=0;
    dague_int_t i, j, bc, b;
    int elem_size = sm->elemsze;
 
#ifdef GEN_DEBUG_PIXMAP
    for(bc=0; bc < cblknbr; bc++){
        dague_int_t strCol, endCol, fb, lb;

        strCol = cblktab[bc].fcolnum;
        endCol = cblktab[bc].lcolnum;

        fb=cblktab[bc].bloknum;
        /* The first block in the next column is just one past my last block */
        lb=cblktab[bc+1].bloknum;

        for(b=fb; b<lb; b++){
            dague_int_t endRow, strRow;
            strRow = bloktab[b].frownum;
            endRow = bloktab[b].lrownum;

            dague_pxmp_si_color_rectangle(strCol, endCol, strRow, endRow, mt*mb, nt*nb);
        }

    }
    dague_pxmp_si_dump_image("test.png", mt*mb*nt*nb);
#endif /* GEN_PIXMAP */

    /* "tmp_map_buf" is guaranteed to fit the maximum number of meta-data
     * entries, since at maximum we can only have an entry per element of the
     * tile. We will work in this buffer for every tile and when we are done
     * with a tile we will allocate the minimum necessary buffer and copy the
     * entries there.
     */
    dague_tssm_data_map_t *tmp_map_buf = (dague_tssm_data_map_t *)calloc(nb*mb, sizeof(dague_tssm_data_map_t));

    fprintf(stderr, "cblknbr lcolnum = %ld, fcolnum = %ld, stride = %ld\n",
	    cblktab[cblknbr-1].lcolnum, cblktab[cblknbr-1].fcolnum,
	    cblktab[cblknbr-1].stride);

    for(i=0; i<nt; i++){
//        fbc = 0;
//        lbc = 0;
        
       /* We are filling up the meta-data structures of the tiles that are in
        * the i-th column. So, we need to find 
        * a) fbc = the first block column whose last column is >= than the
        *    begining of the i-th tile and 
        * b) lbc = the last block column whose first column is < than the
        *    begining of the (i+1)-th tile (or <= than the end of the i-th tile).
        */
        fbc = lbc; /* start from where we left before */
        for(; (fbc < cblknbr) && (cblktab[fbc].lcolnum < i*nb); fbc++)
            ;
   
        for(lbc=fbc; (lbc < cblknbr) && (cblktab[lbc].fcolnum < (i+1)*nb); lbc++)
            ;
        --lbc;

        // If there are no block columns for this tile column, skip it
        if( cblktab[fbc].fcolnum >= (i+1)*nb )
            continue;

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
                assert( endCol >= strCol );

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
                    if( bloktab[b].frownum >= (j+1)*mb )
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
                    tmp_map_buf[blocksInTile].ptr = (void*)( ((uintptr_t)cblktab[bc].cblkptr) + ptr_offset*elem_size);
                    tmp_map_buf[blocksInTile].ldA = ldA;
                    tmp_map_buf[blocksInTile].h = endRow - strRow + 1;
                    assert( tmp_map_buf[blocksInTile].h > 0 );
                    tmp_map_buf[blocksInTile].w = endCol - strCol + 1;
                    assert( tmp_map_buf[blocksInTile].w > 0 );
                    /* this offset is in elements, not in bytes */
                    tmp_map_buf[blocksInTile].offset = off_x*mb + off_y;
                    ++blocksInTile;
                }
            }
            if( blocksInTile > 0 ){
                /* Put the meta-data in a buffer with just one extra element */
                dague_tssm_data_map_t *mapEntry = (dague_tssm_data_map_t *)calloc(1+blocksInTile, sizeof(dague_tssm_data_map_t));
                memcpy(mapEntry, tmp_map_buf, blocksInTile*sizeof(dague_tssm_data_map_t));
                mapEntry[blocksInTile].ptr = NULL; /* Just being ridiculous */

                /* Pass the meta-data to the LRU handling code */
                dague_tssm_mesh_create_tile(mesh, j, i, mb, nb, mapEntry);
            }
        }
    }
#ifdef GEN_DEBUG_PIXMAP
    for(j=0; j<mt; j++){
        for(i=0; i<nt; i++){
           dague_int_t elemcount=0;
           dague_tssm_tile_entry_t *meta_data = mesh->mesh[j*nt+i];
           if( NULL == meta_data )
               continue;
           dague_tssm_data_map_t *mapEntries = meta_data->packed_ptr;

           do {
               dague_int_t strCol, endCol;
               dague_int_t endRow, strRow;
               dague_tssm_data_map_t *mp = &mapEntries[elemcount++];


               strCol = i*nb+(mp->offset)/mb;
               endCol = strCol+(mp->w)-1;
               assert(endCol>=strCol);
               assert(strCol>=0);
               strRow = j*mb+(mp->offset)%mb;
               endRow = strRow+(mp->h)-1;
               assert(endRow>=strRow);
               assert(strRow>=0);

               dague_pxmp_si_color_rectangle(strCol, endCol, strRow, endRow, mt*mb, nt*nb);
           } while( NULL != mapEntries[elemcount].ptr );
       }
    }
    dague_pxmp_si_dump_image("test2.png", mt*mb*nt*nb);
#endif

    return;
}

