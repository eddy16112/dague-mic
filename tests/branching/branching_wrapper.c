#include "dague.h"
#include <data_distribution.h>
#include <arena.h>

#if defined(HAVE_MPI)
#include <mpi.h>
static MPI_Datatype block;
#endif
#include <stdio.h>

#include "branching.h"
#include "branching_wrapper.h"

/**
 * @param [IN] A    the data, already distributed and allocated
 * @param [IN] size size of each local data element
 * @param [IN] nb   number of iterations
 *
 * @return the dague object to schedule.
 */
dague_object_t *branching_new(dague_ddesc_t *A, int size, int nb)
{
    dague_branching_object_t *o = NULL;

    if( nb <= 0 || size <= 0 ) {
        fprintf(stderr, "To work, BRANCHING nb and size must be > 0\n");
        return (dague_object_t*)o;
    }

    o = dague_branching_new(A, nb);

#if defined(HAVE_MPI)
    {
    	MPI_Type_vector(1, size, size, MPI_BYTE, &block);
        MPI_Type_commit(&block);
        dague_arena_construct(o->arenas[DAGUE_branching_DEFAULT_ARENA],
                              size * sizeof(char), size * sizeof(char), 
                              block);
    }
#endif

    return (dague_object_t*)o;
}

/**
 * @param [INOUT] o the dague object to destroy
 */
void branching_destroy(dague_object_t *o)
{
#if defined(HAVE_MPI)
    MPI_Type_free( &block );
#endif

    dague_branching_destroy( (dague_branching_object_t*)o );
}
