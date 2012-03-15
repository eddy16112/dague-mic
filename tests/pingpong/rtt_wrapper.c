#include "dague.h"
#include <data_distribution.h>
#include <arena.h>

#if defined(HAVE_MPI)
#include <mpi.h>
static MPI_Datatype block;
#endif
#include <stdio.h>

#include "rtt.h"
#include "rtt_wrapper.h"

/**
 * @param [IN] A    the data, already distributed and allocated
 * @param [IN] size size of each local data element
 * @param [IN] nb   number of iterations
 *
 * @return the dague object to schedule.
 */
dague_object_t *rtt_new(dague_ddesc_t *A, int size, int nb)
{
    dague_rtt_object_t *o = NULL;

    if( nb <= 0 || size <= 0 ) {
        fprintf(stderr, "To work, RTT must do at least one round time trip of at least one byte\n");
        return (dague_object_t*)o;
    }

    o = dague_rtt_new(A, nb);

#if defined(HAVE_MPI)
    {
    	MPI_Type_vector(1, size, size, MPI_BYTE, &block);
        MPI_Type_commit(&block);
        dague_arena_construct(o->arenas[DAGUE_rtt_DEFAULT_ARENA],
                              size * sizeof(char), size * sizeof(char), 
                              block);
    }
#endif

    return (dague_object_t*)o;
}

/**
 * @param [INOUT] o the dague object to destroy
 */
void rtt_destroy(dague_object_t *o)
{
#if defined(HAVE_MPI)
    MPI_Type_free( &block );
#endif

    dague_rtt_destroy( (dague_rtt_object_t*)o );
}
