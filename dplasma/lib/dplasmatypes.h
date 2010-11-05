#ifndef DPLASMA_DATATYPE_H_HAS_BEEN_INCLUDED
#define DPLASMA_DATATYPE_H_HAS_BEEN_INCLUDED

/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include <dague.h>
#include "dplasma.h"

#if defined(HAVE_MPI)
int dplasma_datatype_define_rectangle( dague_remote_dep_datatype_t oldtype,
                                       unsigned int tile_mb,
                                       unsigned int tile_nb,
                                       int resized,
                                       dague_remote_dep_datatype_t* newtype );
int dplasma_datatype_define_tile( dague_remote_dep_datatype_t oldtype,
                                  unsigned int tile_nb,
                                  dague_remote_dep_datatype_t* newtype );
int dplasma_datatype_define_upper( dague_remote_dep_datatype_t oldtype,
                                   unsigned int tile_nb, int diag,
                                   dague_remote_dep_datatype_t* newtype );
int dplasma_datatype_define_lower( dague_remote_dep_datatype_t oldtype,
                                   unsigned int tile_nb, int diag,
                                   dague_remote_dep_datatype_t* newtype );
#else
# define MPI_DOUBLE_COMPLEX NULL
# define MPI_COMPLEX        NULL
# define MPI_DOUBLE         NULL
# define MPI_FLOAT          NULL

# define dplasma_datatype_define_rectangle( oldtype, tile_mb, tile_nb, resized, newtype) (*(newtype) = NULL)
# define dplasma_datatype_define_tile(      oldtype, tile_nb, newtype ) (*(newtype) = NULL)
# define dplasma_datatype_define_upper(     oldtype, tile_nb, diag, newtype) (*(newtype) = NULL)
# define dplasma_datatype_define_lower(     oldtype, tile_nb, diag, newtype) (*(newtype) = NULL)
#endif

int dplasma_add2arena_rectangle( dague_arena_t *arena, size_t elem_size, size_t alignment,
                                 dague_remote_dep_datatype_t oldtype, 
                                 unsigned int tile_mb, unsigned int tile_nb, int resized );
int dplasma_add2arena_tile( dague_arena_t *arena, size_t elem_size, size_t alignment,
                            dague_remote_dep_datatype_t oldtype, unsigned int tile_mb );
int dplasma_add2arena_upper( dague_arena_t *arena, size_t elem_size, size_t alignment,
                             dague_remote_dep_datatype_t oldtype, unsigned int tile_mb, int diag );
int dplasma_add2arena_lower( dague_arena_t *arena, size_t elem_size, size_t alignment,
                             dague_remote_dep_datatype_t oldtype, unsigned int tile_mb, int diag );

#endif  /* DPLASMA_DATATYPE_H_HAS_BEEN_INCLUDED */
