/*
 * Copyright (c) 2011-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague.h"
#include "execution_unit.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"

static int
dague_operator_print_id( struct dague_execution_unit *eu,
                         const void* src,
                         void* dest,
                         void* op_data, ... )
{
    va_list ap;
    int k, n;

    va_start(ap, op_data);
    k = va_arg(ap, int);
    n = va_arg(ap, int);
    va_end(ap);
    printf( "tile (%d, %d) -> %p:%p thread %d\n",
            k, n, src, dest, eu->eu_id );
    return 0;
}

int main( int argc, char* argv[] )
{
    dague_context_t* dague;
    struct dague_object_t* object;
    two_dim_block_cyclic_t ddescA;
    int cores = 4, world = 1, rank = 0;
    int mb = 100, nb = 100;
    int lm = 1000, ln = 1000;
    int rows = 1;

#if defined(HAVE_MPI)
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    dague = dague_init(cores, &argc, &argv);
    
    two_dim_block_cyclic_init( &ddescA, matrix_RealFloat, matrix_Tile,
                               world, cores, rank, mb, nb, lm, ln, 0, 0, lm, ln, 1, 1, rows );
    ddescA.mat = dague_data_allocate((size_t)ddescA.super.nb_local_tiles *
                                     (size_t)ddescA.super.bsiz *
                                     (size_t)dague_datadist_getsizeoftype(ddescA.super.mtype));

    dague_ddesc_set_key(&ddescA.super.super, "A");
    object = dague_map_operator_New((tiled_matrix_desc_t*)&ddescA,
                                    NULL,
                                    dague_operator_print_id,
                                    "A");
    dague_enqueue(dague, (dague_object_t*)object);

    dague_progress(dague);

    dague_map_operator_Destruct( object );

    dague_fini(&dague);

    return 0;
}
