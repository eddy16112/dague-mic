#ifndef _zgemm_NN_bcast_h_
#define _zgemm_NN_bcast_h_
#include <dague.h>
#include <debug.h>
#include <assert.h>

#define DAGUE_zgemm_NN_bcast_DEFAULT_ARENA    0
#define DAGUE_zgemm_NN_bcast_ARENA_INDEX_MIN 1

typedef struct dague_zgemm_NN_bcast_object {
  dague_object_t super;
  /* The list of globals */
  int transA;
  int transB;
  Dague_Complex64_t alpha;
  Dague_Complex64_t beta;
  tiled_matrix_desc_t descA;
  dague_ddesc_t * A /* data A */;
  tiled_matrix_desc_t descB;
  dague_ddesc_t * B /* data B */;
  tiled_matrix_desc_t descC;
  dague_ddesc_t * C /* data C */;
  /* The array of datatypes DEFAULT and the others */
  dague_arena_t** arenas;
  int arenas_size;
} dague_zgemm_NN_bcast_object_t;

extern dague_zgemm_NN_bcast_object_t *dague_zgemm_NN_bcast_new(int transA, int transB, Dague_Complex64_t alpha, Dague_Complex64_t beta, tiled_matrix_desc_t descA, dague_ddesc_t * A /* data A */, tiled_matrix_desc_t descB, dague_ddesc_t * B /* data B */, tiled_matrix_desc_t descC, dague_ddesc_t * C /* data C */);
#endif /* _zgemm_NN_bcast_h_ */ 
