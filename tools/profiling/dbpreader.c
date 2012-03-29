#include "dague_config.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <inttypes.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "profiling.h"
#include "dbp.h"
#include "dbpreader.h"

#if defined(DAGUE_DEBUG_VERBOSE1)
#define DEBUG(toto) output toto
#else
#define DEBUG(toto) do {} while(0)
#endif
#define WARNING(toto) output toto

static void output(const char *format, ...)
{
    va_list ap;
    va_start(ap, format);
    vfprintf(stderr, format, ap);
    va_end(ap);
}

/* Buffer constants */
static int event_buffer_size = 0;
static int event_avail_space = 0;

struct dbp_event {
    dague_profiling_output_t *native;
};

int dbp_event_get_key(const dbp_event_t *e)
{
    return e->native->event.key;
}

int dbp_event_get_flags(const dbp_event_t *e)
{
    return e->native->event.flags;
}

int dbp_event_get_id(const dbp_event_t *e)
{
    return e->native->event.id;
}

dague_time_t dbp_event_get_timestamp(const dbp_event_t *e)
{
    return e->native->event.timestamp;
}

void *dbp_event_get_info(const dbp_event_t *e)
{
    if( EVENT_HAS_INFO( e->native ) ) {
        return e->native->info;
    } else {
        return NULL;
    }
}

int dbp_event_info_len(const dbp_event_t *e, const dbp_multifile_reader_t *dbp)
{
    if( e->native->event.flags & DAGUE_PROFILING_EVENT_HAS_INFO ) {
        return dbp_dictionary_keylen(dbp_reader_get_dictionary(dbp, BASE_KEY(dbp_event_get_key(e))));
    } else {
        return 0;
    }
}

struct dbp_event_iterator {
    const dbp_thread_t              *thread;
    dbp_event_t                      current_event;
    dague_profiling_buffer_t        *current_events_buffer;
    int64_t                          current_event_position;
    int64_t                          current_event_index;
    int64_t                          current_buffer_position;
};

struct dbp_info {
    char *key;
    char *value;
};

struct dbp_file {
    dbp_multifile_reader_t *parent;
    char *hr_id;
    int fd;
    dague_time_t min_date;
    char *filename;
    int rank;
    int nb_infos;
    int  nb_threads;
    dbp_info_t  **infos;
    dbp_thread_t *threads;
};

struct dbp_dictionary {
    int keylen;
    char *name;
    char *attributes;
    char *convertor;
};

char *dbp_dictionary_name(const dbp_dictionary_t *dico)
{
    return dico->name;
}

char *dbp_dictionary_attributes(const dbp_dictionary_t *dico)
{
    return dico->attributes;
}

char *dbp_dictionary_convertor(const dbp_dictionary_t *dico)
{
    return dico->convertor;
}

int dbp_dictionary_keylen(const dbp_dictionary_t *dico)
{
    return dico->keylen;
}

struct dbp_multifile_reader {
    int nb_files;
    int dico_size;
    int worldsize;
    int nb_infos;
    dague_time_t min_date;
    dbp_info_t *infos;
    dbp_dictionary_t *dico_keys;
    dbp_file_t *files;
};

dbp_dictionary_t *dbp_reader_get_dictionary(const dbp_multifile_reader_t *dbp, int did)
{
    assert( did >= 0 && did < dbp->dico_size );
    return &(dbp->dico_keys[did]);
}

#define DBP_EVENT_LENGTH(dbp_event, dbp_main_object) \
    (sizeof(dague_profiling_output_base_event_t) +                      \
     (EVENT_HAS_INFO((dbp_event)->native) ?                             \
      (dbp_main_object)->dico_keys[BASE_KEY((dbp_event)->native->event.key)].keylen : 0))

struct dbp_thread {
    const dague_thread_profiling_t  *profile;
    dbp_file_t                      *file;
    int                              nb_infos;
    dbp_info_t                      *infos;
};

static void release_events_buffer(dague_profiling_buffer_t *buffer)
{
    if( NULL == buffer )
        return;
    if( munmap(buffer, event_buffer_size) == -1 ) {
        WARNING(("Warning profiling system: unmap of the events backend file at %p failed: %s\n",
                 buffer, strerror(errno)));
    }
}

static dague_profiling_buffer_t *refer_events_buffer( int fd, int64_t offset )
{
    dague_profiling_buffer_t *res;
    res = mmap(NULL, event_buffer_size, PROT_READ, MAP_SHARED, fd, offset);
    if( MAP_FAILED == res )
        return NULL;
    return res;
}

dbp_event_iterator_t *dbp_iterator_new_from_thread(const dbp_thread_t *th)
{
    dbp_event_iterator_t *res = (dbp_event_iterator_t*)malloc(sizeof(dbp_event_iterator_t));
    res->thread = th;
    res->current_event.native = NULL;
    res->current_event_position = 0;
    res->current_event_index = 0;
    res->current_buffer_position = (off_t)-1;
    res->current_events_buffer  = NULL;
    (void)dbp_iterator_first(res);
    return res;
}

dbp_event_iterator_t *dbp_iterator_new_from_iterator(const dbp_event_iterator_t *it)
{
    dbp_event_iterator_t *res = (dbp_event_iterator_t*)malloc(sizeof(dbp_event_iterator_t));
    res->thread = it->thread;
    res->current_event.native = it->current_event.native;
    res->current_event_position = it->current_event_position;
    res->current_event_index = it->current_event_index;
    res->current_buffer_position = it->current_buffer_position;    
    res->current_events_buffer = refer_events_buffer( it->thread->file->fd, res->current_buffer_position );
    return res;
}

const dbp_event_t *dbp_iterator_current(const dbp_event_iterator_t *it)
{
    if( it->current_events_buffer == NULL ||
        it->current_event.native == NULL )
        return NULL;
    return &it->current_event;
}

const dbp_event_t *dbp_iterator_first(dbp_event_iterator_t *it)
{
    if( it->current_events_buffer != NULL ) {
        release_events_buffer( it->current_events_buffer );
        it->current_events_buffer = NULL;
        it->current_event.native = NULL;
    }

    it->current_events_buffer = refer_events_buffer( it->thread->file->fd, it->thread->profile->first_events_buffer_offset );
    it->current_buffer_position = it->thread->profile->first_events_buffer_offset;
    it->current_event_position = 0;
    if( it->current_events_buffer != NULL ) 
        it->current_event.native = (dague_profiling_output_t*)&(it->current_events_buffer->buffer[it->current_event_position]);
    else
        it->current_event.native = NULL;
    return dbp_iterator_current(it);
}

const dbp_event_t *dbp_iterator_next(dbp_event_iterator_t *it)
{
    size_t elen;
    dague_profiling_output_t *current;
    off_t next_off;
    
    current = it->current_event.native;
    if( NULL == current )
        return NULL;
    elen = DBP_EVENT_LENGTH(&it->current_event, it->thread->file->parent);
    assert( it->current_events_buffer->buffer_type == PROFILING_BUFFER_TYPE_EVENTS );
    if( it->current_event_index+1 >= it->current_events_buffer->this_buffer.nb_events ) {
        next_off = it->current_events_buffer->next_buffer_file_offset;
        release_events_buffer( it->current_events_buffer );
        it->current_event_position = 0;
        it->current_event_index = 0;
        it->current_events_buffer = refer_events_buffer( it->thread->file->fd, next_off );
        it->current_buffer_position = next_off;

        if( NULL == it->current_events_buffer ) {
            it->current_event.native = NULL;
            return NULL;
        } else {
            it->current_event.native = (dague_profiling_output_t*)&(it->current_events_buffer->buffer[it->current_event_position]);
        }
    } else {
        it->current_event_position += elen;
        it->current_event.native = (dague_profiling_output_t*)&(it->current_events_buffer->buffer[it->current_event_position]);
        it->current_event_index++;
    }
    assert( it->current_event_position <= event_avail_space );
    assert( it->current_events_buffer->buffer_type == PROFILING_BUFFER_TYPE_EVENTS );

    current = it->current_event.native;
    assert((current == NULL) ||
           (current->event.timestamp.tv_sec != 0));

    return dbp_iterator_current(it);
}

const dbp_thread_t *dbp_iterator_thread(const dbp_event_iterator_t *it)
{
    return it->thread;
}

void dbp_iterator_delete(dbp_event_iterator_t *it)
{
    if( NULL != it->current_events_buffer )
        release_events_buffer(it->current_events_buffer);
    free(it);
}

int dbp_iterator_move_to_matching_event(dbp_event_iterator_t *pos,
                                        const dbp_event_t *ref)
{
    const dbp_event_t *e;
    
    e = dbp_iterator_current( pos );
    while( NULL != e ) {
        if( (dbp_event_get_id(e) == dbp_event_get_id(ref)) &&
            (dbp_event_get_key(e) == END_KEY(BASE_KEY(dbp_event_get_key(ref)))) ) {
            if( dbp_event_get_id(e) != 0 ||
                time_less( dbp_event_get_timestamp(ref), dbp_event_get_timestamp(e)) ||
                (diff_time(dbp_event_get_timestamp(ref), dbp_event_get_timestamp(e)) == 0) ) {
                return 1;
            } else if ( dbp_event_get_id(e) != 0 ) {
                WARNING(("Event with ID %d appear in reverse order\n",
                         dbp_event_get_id(e)));
            }
        }
        e = dbp_iterator_next( pos );
    }
    return 0;
}

dbp_event_iterator_t *dbp_iterator_find_matching_event_all_threads(const dbp_event_iterator_t *pos)
{
    dbp_event_iterator_t *it;
    const dbp_event_t *ref;
    dbp_file_t *dbp_file;
    int th;

    ref = dbp_iterator_current(pos);
    it = dbp_iterator_new_from_iterator(pos);
    if( dbp_iterator_move_to_matching_event(it, ref) )
        return it;
    dbp_iterator_delete(it);

    dbp_file = pos->thread->file;

    for(th = 0; th < dbp_file_nb_threads(dbp_file); th++) {
        if( pos->thread == dbp_file_get_thread(dbp_file, th) )
            continue;
        it = dbp_iterator_new_from_thread( dbp_file_get_thread(dbp_file, th) );
        if( dbp_iterator_move_to_matching_event(it, ref) )
            return it;
        dbp_iterator_delete(it);
    }
    
    return NULL;
}

char *dbp_info_get_key(const dbp_info_t *info)
{
    return info->key;
}

char *dbp_info_get_value(const dbp_info_t *info)
{
    return info->value;
}

int dbp_thread_nb_events(const dbp_thread_t *th)
{
    return th->profile->nb_events;
}

int dbp_thread_nb_infos(const dbp_thread_t *th)
{
    return th->nb_infos;
}

dbp_info_t *dbp_thread_get_info(const dbp_thread_t *th, int iid)
{
    assert( iid >= 0 && iid < th->nb_infos );
    return &th->infos[iid];
}

char *dbp_thread_get_hr_id(const dbp_thread_t *th)
{
    return th->profile->hr_id;
}

dbp_thread_t *dbp_file_get_thread(const dbp_file_t *file, int tid)
{
    assert( tid >= 0 && tid < file->nb_threads );
    return &file->threads[tid];
}

char *dbp_file_hr_id(const dbp_file_t *file)
{
    return file->hr_id;
}

int dbp_file_get_rank(const dbp_file_t *file)
{
    return file->rank;
}

dague_time_t dbp_file_get_min_date(const dbp_file_t *file)
{
    return file->min_date;
}

int dbp_file_nb_threads(const dbp_file_t *file)
{
    return file->nb_threads;
}

int dbp_file_nb_infos(const dbp_file_t *file)
{
    return file->nb_infos;
}

dbp_info_t *dbp_file_get_info(const dbp_file_t *file, int iid)
{
    assert( iid >= 0 && iid < file->nb_infos );
    return file->infos[iid];
}

dbp_file_t *dbp_reader_get_file(const dbp_multifile_reader_t *dbp, int fid)
{
    assert(fid >= 0 && fid < dbp->nb_files );
    return &dbp->files[fid];
}

int dbp_reader_nb_files(const dbp_multifile_reader_t *dbp)
{
    return dbp->nb_files;
}

int dbp_reader_nb_dictionary_entries(const dbp_multifile_reader_t *dbp)
{
    return dbp->dico_size;
}

dague_time_t dbp_reader_min_date(const dbp_multifile_reader_t *dbp)
{
    return dbp->min_date;
}

int dbp_reader_worldsize(const dbp_multifile_reader_t *dbp)
{
    return dbp->worldsize;
}

void dbp_reader_close_files(dbp_multifile_reader_t *dbp)
{
    (void)dbp;
}

static void read_infos(dbp_multifile_reader_t *dbp, int n, dague_profiling_binary_file_header_t *head)
{
    (void)n;
    (void)head;
    dbp->nb_infos = 0;
    dbp->infos = NULL;
}

static int read_dictionnary(dbp_multifile_reader_t *dbp, int fd, const dague_profiling_binary_file_header_t *head)
{
    dague_profiling_buffer_t *dico, *next;
    dague_profiling_key_buffer_t *a;
    int nb, nbthis, pos;

    dbp->dico_size = 0;

    /* Dictionaries match: take the first in memory */
    dico = refer_events_buffer( fd, head->dictionary_offset );
    if( NULL == dico ) {
        fprintf(stderr, "Unable to read entire dictionary entry\n");
        return -1;
    }
    assert( PROFILING_BUFFER_TYPE_DICTIONARY == dico->buffer_type );

    dbp->dico_size = head->dictionary_size;
    dbp->dico_keys = (dbp_dictionary_t *)calloc(head->dictionary_size, sizeof(dbp_dictionary_t));
    nb = dbp->dico_size;
    nbthis = dico->this_buffer.nb_dictionary_entries;
    pos = 0;
    while( nb > 0 ) {
        a = (dague_profiling_key_buffer_t*)&dico->buffer[pos];

        dbp->dico_keys[ dbp->dico_size - nb ].name = malloc( 64 );
        strncpy(dbp->dico_keys[ dbp->dico_size - nb ].name, a->name, 64);
        assert( strlen(a->attributes) > 6 );
        dbp->dico_keys[ dbp->dico_size - nb ].attributes = malloc( 128 );
        strncpy(dbp->dico_keys[ dbp->dico_size - nb ].attributes, ((char*)a->attributes) + strlen(a->attributes) - 6, 128 );
        dbp->dico_keys[ dbp->dico_size - nb ].convertor = (char*)malloc(a->keyinfo_convertor_length+1);
        memcpy(dbp->dico_keys[ dbp->dico_size - nb ].convertor,
               a->convertor,
               a->keyinfo_convertor_length);
        dbp->dico_keys[ dbp->dico_size - nb ].convertor[a->keyinfo_convertor_length] = '\0';
        dbp->dico_keys[ dbp->dico_size - nb ].keylen = a->keyinfo_length;

        pos += a->keyinfo_convertor_length - 1 + sizeof(dague_profiling_key_buffer_t);
        nb--;
        nbthis--;

        if( nb > 0 && nbthis == 0 ) {
            next = refer_events_buffer( fd, dico->next_buffer_file_offset );
            if( NULL == next ) {
                fprintf(stderr, "Dictionary entry %d is broken. Dictionary broken.\n", dbp->dico_size - nb);
                release_events_buffer( dico );
                return -1;
            }
            assert( PROFILING_BUFFER_TYPE_DICTIONARY == dico->buffer_type );
            release_events_buffer( dico );
            dico = next;

            pos = 0;
        }
    }
    release_events_buffer( dico );
    return 0;
}

static int check_dictionnary(const dbp_multifile_reader_t *dbp, int fd, const dague_profiling_binary_file_header_t *head)
{
    dague_profiling_buffer_t *dico, *next;
    dague_profiling_key_buffer_t *a;
    int nb, nbthis, pos;

    /* Dictionaries match: take the first in memory */
    dico = refer_events_buffer( fd, head->dictionary_offset );
    if( NULL == dico ) {
        fprintf(stderr, "Unable to read entire dictionary entry.\n");
        return -1;
    }
    assert( PROFILING_BUFFER_TYPE_DICTIONARY == dico->buffer_type );

    if(dbp->dico_size != head->dictionary_size) {
        fprintf(stderr, "Dictionary sizes do not match.\n");
        goto error;
    }

    nb = dbp->dico_size;
    nbthis = dico->this_buffer.nb_dictionary_entries;
    pos = 0;
    while( nb > 0 ) {
        a = (dague_profiling_key_buffer_t*)&dico->buffer[pos];

        if( strncmp(dbp->dico_keys[ dbp->dico_size - nb ].name, a->name, 64) ) {
            fprintf(stderr, "Dictionary entry %d has a name of %s in the reference dictionary, and %s in the new file dictionary.\n",
                    dbp->dico_size - nb, dbp->dico_keys[ dbp->dico_size - nb ].name, a->name);
            goto error;
        }

        assert( strlen(a->attributes) > 6 );
        if( strncmp(dbp->dico_keys[ dbp->dico_size - nb ].attributes, ((char*)a->attributes) + strlen(a->attributes) - 6, 128 ) ) {
            fprintf(stderr, "Dictionary entry %d has a name of %s in the reference dictionary, and %s in the new file dictionary.\n",
                    dbp->dico_size - nb, dbp->dico_keys[ dbp->dico_size - nb ].attributes, a->attributes);
            goto error;
        }

        if( strlen(dbp->dico_keys[ dbp->dico_size - nb ].convertor) != a->keyinfo_convertor_length ) {
            fprintf(stderr, "Dictionary entry %d has a convertor of %d bytes in the reference dictionary, and %d in the new file dictionary.\n",
                    dbp->dico_size - nb, (int)strlen(dbp->dico_keys[ dbp->dico_size - nb ].convertor), a->keyinfo_convertor_length);
            goto error;
        }

        if( strncmp(dbp->dico_keys[ dbp->dico_size - nb ].convertor,
                    a->convertor,
                    a->keyinfo_convertor_length) ) {
            fprintf(stderr, "Dictionary entry %d has a convertor in the reference dictionary, that is different from the convertor for the same entry in the new file dictionary.\n",
                    dbp->dico_size - nb);
            goto error;
        }

        if( dbp->dico_keys[ dbp->dico_size - nb ].keylen != a->keyinfo_length ) {
            fprintf(stderr, "Dictionary entry %d has an info length in the reference dictionary of %d bytes, that is different from the key info length of %d bytes for the same entry in the new file dictionary.\n",
                    dbp->dico_size - nb, dbp->dico_keys[ dbp->dico_size - nb ].keylen, a->keyinfo_length);
             goto error;
        }

        pos += a->keyinfo_convertor_length - 1 + sizeof(dague_profiling_key_buffer_t);
        nb--;
        nbthis--;

        if( nb > 0 && nbthis == 0 ) {
            next = refer_events_buffer( fd, dico->next_buffer_file_offset );
            if( NULL == next ) {
                fprintf(stderr, "Dictionary entry %d is broken. Dictionary broken.\n", dbp->dico_size - nb);
                goto error;
            }
            assert( PROFILING_BUFFER_TYPE_DICTIONARY == dico->buffer_type );
            release_events_buffer( dico );
            dico = next;
            
            pos = 0;
        }
    }
    release_events_buffer( dico );
    return 0;

 error:
    release_events_buffer( dico );
    return -1;    
}

static size_t read_thread_infos(dague_thread_profiling_t * res,
                                dbp_thread_t *th,
                                int nb_infos, const char *br)
{
    dague_profiling_info_buffer_t *ib;
    dbp_info_t *nfo;
    int i, pos = 0;

    res->infos = NULL;
    th->infos = (dbp_info_t *)calloc(nb_infos, sizeof(dbp_info_t));
    for(i = 0; i < nb_infos; i++) {
        ib = (dague_profiling_info_buffer_t*)&(br[pos]);
        pos += ib->info_size + ib->value_size;
        nfo = (dbp_info_t*)&th->infos[i];
        nfo->key = (char*)malloc(ib->info_size+1);
        memcpy(nfo->key, ib->info_and_value, ib->info_size);
        nfo->key[ib->info_size] = '\0';
        nfo->value = (char*)malloc(ib->value_size+1);
        memcpy(nfo->value, ib->info_and_value + ib->info_size, ib->value_size);
        nfo->value[ib->value_size] = '\0';
    }

    return pos;
}

static int read_threads(dbp_multifile_reader_t *dbp, int n, int fd, const dague_profiling_binary_file_header_t *head)
{
    dague_thread_profiling_t *res;
    dague_profiling_thread_buffer_t *br;
    dague_profiling_buffer_t *b, *next;
    dague_profiling_output_t *fevent;
    int nb, nbthis, pos;
    dague_time_t zero = ZERO_TIME;

    dbp->files[n].nb_threads = head->nb_threads;
    dbp->files[n].threads = (dbp_thread_t*)calloc(head->nb_threads, sizeof(dbp_thread_t));
    dbp->files[n].min_date = zero;

    pos = 0;
    nb = head->nb_threads;
    b = refer_events_buffer(fd, head->thread_offset);
    nbthis = b->this_buffer.nb_threads;
    while( nb > 0 ) {
        assert(PROFILING_BUFFER_TYPE_THREAD == b->buffer_type);
        assert(nbthis > 0);

        br = (dague_profiling_thread_buffer_t*)&(b->buffer[pos]);
        res = (dague_thread_profiling_t*)malloc( sizeof(dague_thread_profiling_t) );
        res->next_event_position = -1; /* No need for a next event position */
        res->nb_events = br->nb_events;
        res->hr_id = (char*)malloc(128);
        strncpy(res->hr_id, br->hr_id, 128);
        res->first_events_buffer_offset = br->first_events_buffer_offset;
        res->current_events_buffer = refer_events_buffer(fd, br->first_events_buffer_offset);
        if( NULL != res->current_events_buffer ) {
            assert( res->current_events_buffer->buffer_type == PROFILING_BUFFER_TYPE_EVENTS );
            fevent = (dague_profiling_output_t *)res->current_events_buffer->buffer;
            if( time_less( fevent->event.timestamp, dbp->min_date ) ||
                (diff_time(dbp->min_date, zero) == 0) ) {
                dbp->min_date = fevent->event.timestamp;
            }
            if( time_less( fevent->event.timestamp, dbp->files[n].min_date ) ||
                (diff_time(dbp->files[n].min_date, zero) == 0) ) {
                dbp->files[n].min_date = fevent->event.timestamp;
            }
            release_events_buffer( res->current_events_buffer );
            res->current_events_buffer = NULL;
        }
        DAGUE_LIST_ITEM_CONSTRUCT( res );
        
        dbp->files[n].threads[head->nb_threads - nb].file = &(dbp->files[n]);
        dbp->files[n].threads[head->nb_threads - nb].profile = res;
        
        nbthis--;
        nb--;
        assert(nb >= 0);
        pos += sizeof(dague_profiling_thread_buffer_t) - sizeof(dague_profiling_info_buffer_t);
        pos += read_thread_infos( res, &dbp->files[n].threads[head->nb_threads-nb],
                                  br->nb_infos, (char*)br->infos );

        if( nbthis == 0 && nb > 0 ) {
            assert( b->next_buffer_file_offset != -1 );
            next = refer_events_buffer(fd, b->next_buffer_file_offset);
             if( NULL == next ) {
                fprintf(stderr, "Unable to read thread entry: Profile file broken\n");
                release_events_buffer( b );
                return -1;
            }
            assert( PROFILING_BUFFER_TYPE_THREAD == next->buffer_type );
            release_events_buffer( b );
            b = next;

            nbthis = b->this_buffer.nb_threads;

            pos = 0;
        }
    }

    release_events_buffer( b );
    return 0;
}

static dbp_multifile_reader_t *open_files(int argc, char *argv[])
{
    int fd, i, j, p, n;
    dague_profiling_buffer_t dummy_events_buffer;
    dague_profiling_binary_file_header_t head;
    dbp_multifile_reader_t *dbp;
    dague_time_t zero = ZERO_TIME;

    dbp = (dbp_multifile_reader_t*)malloc(sizeof(dbp_multifile_reader_t));
    dbp->files = (dbp_file_t*)malloc((argc-1) * sizeof(dbp_file_t));
    dbp->min_date = zero;

    n = 0;
    for(i = 1; i < argc; i++) {
        fd = open(argv[i], O_RDONLY);
        if( fd == -1 ) {
            fprintf(stderr, "Unable to open %s: %s -- skipped\n", argv[i], strerror(errno));
            continue;
        }
        dbp->files[n].parent = dbp;
        dbp->files[n].fd = fd;
        if( read( fd, &head, sizeof(dague_profiling_binary_file_header_t) ) != sizeof(dague_profiling_binary_file_header_t) ) {
            fprintf(stderr, "File %s does not seem to be a correct DAGUE Binary Profile, ignored\n",
                    argv[i]);
            close(fd);
            continue;
        }
        if( strncmp( head.magick, DAGUE_PROFILING_MAGICK, 24 ) ) {
            fprintf(stderr, "File %s does not seem to be a correct DAGUE Binary Profile, ignored\n",
                    argv[i]);
            close(fd);
            continue;
        }
        if( head.byte_order != 0x0123456789ABCDEF ) {
            fprintf(stderr, "The profile in file %s has been generated with a different byte ordering. File ignored\n",
                    argv[i]);
            close(fd);
            continue;
        }
        dbp->files[n].filename = strdup(argv[i]);
        if( n > 0 ) {
            if( strncmp(head.hr_id, dbp->files[0].hr_id, 128) ) {
                fprintf(stderr, "The profile in file %s has unique id %s, which is not compatible with id %s of file %s. File ignored.\n",
                        dbp->files[n].filename, head.hr_id,
                        dbp->files[0].hr_id, dbp->files[0].filename);
                close(fd);
                continue;
            }

            if( head.profile_buffer_size != event_buffer_size ) {
                fprintf(stderr, "The profile in file %s has a buffer size of %d, which is not compatible with the buffer size %d of file %s. File ignored.\n",
                        dbp->files[n].filename, head.profile_buffer_size,
                        event_buffer_size, dbp->files[0].filename);
                close(fd);
                continue;
            }

            if( head.worldsize != dbp->worldsize ) {
                fprintf(stderr, "The profile in file %s has a world size of %d, which is not compatible with the world size %d of file %s. File ignored.\n",
                        dbp->files[n].filename, head.worldsize,
                        dbp->worldsize, dbp->files[0].filename);
                close(fd);
                continue;
            }

            if( check_dictionnary(dbp, fd, &head) != 0 ) {
                fprintf(stderr, "The profile in file %s has a broken or unmatching dictionary. Dictionary ignored.\n",
                        dbp->files[n].filename);
            }
        } else {
            event_buffer_size = head.profile_buffer_size;
            if( read_dictionnary(dbp, fd, &head) != 0 ) {
                fprintf(stderr, "The profile in file %s has a broken dictionary. Trying to use the dictionary of next file. Ignoring the file.\n",
                        dbp->files[n].filename);
                close(fd);
                continue;
            }
            dbp->worldsize = head.worldsize;
            
        }
        
        dbp->files[n].hr_id = strdup(head.hr_id);
        dbp->files[n].rank = head.rank;
        
        read_infos(dbp, n, &head);

        if( read_threads(dbp, n, fd, &head) != 0 ) {
            fprintf(stderr, "unable to read all threads of profile %d in file %s. File ignored.\n",
                    n, dbp->files[n].filename);
            continue;
        }

        n++;
    }
    
    if( dbp->worldsize > n ) {
        fprintf(stderr, "The profile in file %s has a world size of %d, but only %d files can be read in input. The trace will be truncated\n",
                        dbp->files[0].filename, dbp->worldsize, n);
    } else if( dbp->worldsize < n ) {
        fprintf(stderr, "The profile in file %s has a world size of %d, but %d files should be read in input. The trace will be... Strange...\n",
                        dbp->files[0].filename, dbp->worldsize, n);
    } else {
        for(i = 0; i < n; i++) {
            p = 0;
            for(j = 0; j < n; j++) {
                if( dbp->files[j].rank == i )
                    p++;
            }
            if( p != 1 ) {
                fprintf(stderr, "The rank %d appears %d times in this collection of profiles... The trace will be... Strange...\n",
                        i, p);
            }
        }
    }

    dbp->nb_files = n;
    event_avail_space = event_buffer_size -
        ( (char*)&dummy_events_buffer.buffer[0] - (char*)&dummy_events_buffer);
    
    return dbp;
}

dbp_multifile_reader_t *dbp_reader_open_files(int nbfiles, char *files[])
{
    dbp_multifile_reader_t *dbp;

    (void)event_buffer_size;
    (void)event_avail_space;

    dbp = open_files(nbfiles, files);
    
    return dbp;
}
