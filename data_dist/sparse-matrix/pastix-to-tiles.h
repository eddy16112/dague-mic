#ifndef _PASTIX_TO_TILES_H_
#define _PASTIX_TO_TILES_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sparse-shm-matrix.h"
#include "Pastix_structs.h"

#define MIN(_X , _Y ) (( (_X) < (_Y) ) ? (_X) : (_Y))
#define MAX(_X , _Y ) (( (_X) > (_Y) ) ? (_X) : (_Y))

#ifdef ELEM_IS_INT
 #define ELEM_SIZE sizeof(int)
#elif ELEM_IS_FLOAT
 #define ELEM_SIZE sizeof(float)
#elif ELEM_IS_DOUBLE
 #define ELEM_SIZE sizeof(double)
#elif ELEM_IS_SCOMPLEX
 #define ELEM_SIZE 2*sizeof(float)
#elif ELEM_IS_DCOMPLEX
 #define ELEM_SIZE 2*sizeof(double)
#else
 #error "UNKNOWN ELEMENT SIZE"
#endif

typedef struct dataMap_t{
    void *ptr;
    int ldA;
    int h;
    int w;
    int offset;
} dataMap_t;


int dague_sparse_tile_unpack(void *tile_ptr, int mb, dataMap_t *map);
void dague_sparse_tile_pack(void *tile_ptr, int mb, dataMap_t *map);
void dague_pastix_to_tiles_load(dague_tssm_desc_t *mesh, unsigned int M, unsigned int N, unsigned int mb, unsigned int nb, SymbolMatrix *sm);

#endif
