#include "ctlgat_data.h"
#include "stdarg.h"
#include "data_distribution.h"

#include <assert.h>

typedef struct {
    dague_ddesc_t super;
    int   seg;
    int   size;
    uint32_t* data;
} my_datatype_t;

static uint32_t rank_of(dague_ddesc_t *desc, ...)
{
    int k;
    va_list ap;
    my_datatype_t *dat = (my_datatype_t*)desc;

    va_start(ap, desc);
    k = va_arg(ap, int);
    va_end(ap);

    assert( k < dat->size && k >= 0 );

    return k;
}

static int32_t vpid_of(dague_ddesc_t *desc, ...)
{
    int k;
    va_list ap;
    my_datatype_t *dat = (my_datatype_t*)desc;

    va_start(ap, desc);
    k = va_arg(ap, int);
    va_end(ap);

    assert( k < dat->size && k >= 0 );

    return 0;
}

static void *data_of(dague_ddesc_t *desc, ...)
{
    int k;
    va_list ap;
    my_datatype_t *dat = (my_datatype_t*)desc;

    va_start(ap, desc);
    k = va_arg(ap, int);
    va_end(ap);

    assert( k < dat->size && k >= 0 );

    return (void*)dat->data;
} 

#if defined(DAGUE_PROF_TRACE)
static uint32_t data_key(struct dague_ddesc *desc, ...)
{
    int k;
    va_list ap;
    my_datatype_t *dat = (my_datatype_t*)desc;

    va_start(ap, desc);
    k = va_arg(ap, int);
    va_end(ap);

    assert( k < dat->size && k >= 0 );

    return (uint32_t)k;
}
#endif

dague_ddesc_t *create_and_distribute_data(int rank, int world, int cores, int size, int seg)
{
    my_datatype_t *m = (my_datatype_t*)calloc(1, sizeof(my_datatype_t));
    dague_ddesc_t *d = &(m->super);

    d->myrank = rank;
    d->cores  = cores;
    d->nodes  = world;
    d->rank_of = rank_of;
    d->data_of = data_of;
    d->vpid_of = vpid_of;
#if defined(DAGUE_PROF_TRACE)
    asprintf(&d->key_dim, "(%d)", size);
    d->key = strdup("A");
    d->data_key = data_key;
#endif

    m->size = size;
    m->seg  = seg;
    m->data = (uint32_t*)calloc(seg * size, sizeof(uint32_t) );

    return d;
}

void free_data(dague_ddesc_t *d)
{
    my_datatype_t *m = (my_datatype_t*)d;
    free(m->data);
    dague_ddesc_destroy(d);
    free(d);
}
