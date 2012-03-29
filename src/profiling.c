/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <sys/mman.h>

#include "profiling.h"
#include "dbp.h"
#include "data_distribution.h"
#include "debug.h"
#include "fifo.h"

#define min(a, b) ((a)<(b)?(a):(b))

#define MINIMAL_EVENT_BUFFER_SIZE          4088

static dague_profiling_buffer_t *allocate_empty_buffer(int64_t *offset, char type);

/* Process-global dictionnary */
static unsigned int dague_prof_keys_count, dague_prof_keys_number;
static dague_profiling_key_t* dague_prof_keys;

static dague_time_t dague_start_time;

/* Process-global profiling list */
static dague_list_t threads;
static char *hr_id = NULL;
static dague_profiling_info_t *dague_profiling_infos = NULL;

static char *dague_profiling_last_error = NULL;

/* File backend globals. */
static pthread_mutex_t file_backend_lock = PTHREAD_MUTEX_INITIALIZER;
static off_t file_backend_next_offset = 0;
static int   file_backend_fd = -1;

/* File backend constants, computed at init time */
static size_t event_buffer_size = 0;
static size_t event_avail_space = 0;
static int file_backend_extendable;

static dague_profiling_binary_file_header_t *profile_head = NULL;
static char *bpf_filename = NULL;

char *dague_profiling_strerror(void)
{
    return dague_profiling_last_error;
}

void dague_profiling_add_information( const char *key, const char *value )
{
    dague_profiling_info_t *n;
    n = (dague_profiling_info_t *)calloc(1, sizeof(dague_profiling_info_t));
    n->key = strdup(key);
    n->value = strdup(value);
    n->next = dague_profiling_infos;
    dague_profiling_infos = n;
}

int dague_profiling_change_profile_attribute( const char *format, ... )
{
    va_list ap;

    if( hr_id != NULL ) {
        free(hr_id);
    }

    va_start(ap, format);
    vasprintf(&hr_id, format, ap);
    va_end(ap);

    return 0;
}

int dague_profiling_init( const char *format, ... )
{
    va_list ap;
    char *c, *hr_id_basename, *hr_id_dir;
    dague_profiling_buffer_t dummy_events_buffer;
    long ps;
    int rank = 0;
    int worldsize = 1;
    int64_t zero;

#if defined(HAVE_MPI)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
#endif

    if( hr_id != NULL ) {
        ERROR(("dague_profiling_init: profiling already initialized"));
        return -1;
    }

    va_start(ap, format);
    vasprintf(&hr_id, format, ap);
    va_end(ap);

    bpf_filename = (char*)malloc(strlen(hr_id) + 16);

    hr_id_dir = strdup(hr_id);
    hr_id_basename = hr_id_dir;
    for(c = hr_id_dir; *c != '\0'; c++) {
        if( *c == '/' )
            hr_id_basename = c+1;
    }
    if( hr_id_basename != hr_id_dir ) {
        *(hr_id_basename-1) = '\0';
    }

    sprintf(bpf_filename, "%s/.%s.prof-XXXXXX", hr_id_dir, hr_id_basename);
    free(hr_id_dir);
    hr_id_dir = NULL;
    hr_id_basename = NULL;

    file_backend_fd = mkstemp(bpf_filename);
    if( -1 == file_backend_fd ) {
        fprintf(stderr, "Warning profiling system: unable to create temporary backend file %s: %s. Events not logged.\n",
                bpf_filename, strerror(errno));
        free(bpf_filename);
        bpf_filename = NULL;
        file_backend_extendable = 0;
    } else {
        file_backend_extendable = 1;
        ps = sysconf(_SC_PAGESIZE);
        event_buffer_size = ps * ((MINIMAL_EVENT_BUFFER_SIZE + ps) / ps);
        event_avail_space = event_buffer_size - 
            ( (char*)&dummy_events_buffer.buffer[0] - (char*)&dummy_events_buffer);

        assert( sizeof(dague_profiling_binary_file_header_t) < event_buffer_size );
        profile_head = (dague_profiling_binary_file_header_t*)allocate_empty_buffer(&zero, PROFILING_BUFFER_TYPE_HEADER);
        if( NULL != profile_head ) {
            strcpy(profile_head->magick, DAGUE_PROFILING_MAGICK);
            profile_head->byte_order = 0x0123456789ABCDEF;
            profile_head->profile_buffer_size = event_buffer_size;
            strncpy(profile_head->hr_id, hr_id, 128);
            profile_head->rank = rank;
            profile_head->worldsize = worldsize;
        }
    }

    dague_list_construct( &threads );

    dague_prof_keys = (dague_profiling_key_t*)calloc(128, sizeof(dague_profiling_key_t));
    dague_prof_keys_count = 0;
    dague_prof_keys_number = 128;

#if defined(HAVE_MPI)
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    dague_start_time = take_time();

    return 0;
}

dague_thread_profiling_t *dague_profiling_thread_init( size_t length, const char *format, ...)
{
    va_list ap;
    dague_thread_profiling_t *res;

    /** Remark: maybe calloc would be less perturbing for the measurements,
     *  if we consider that we don't care about the _init phase, but only
     *  about the measurement phase that happens later.
     */
    res = (dague_thread_profiling_t*)malloc( sizeof(dague_thread_profiling_t) + length );
    if( NULL == res ) {
        ERROR(("dague_profiling_thread_init: unable to allocate %u bytes", length));
        return NULL;
    }

    va_start(ap, format);
    vasprintf(&res->hr_id, format, ap);
    va_end(ap);

    assert( event_buffer_size != 0 );
    /* To trigger a buffer allocation at first creation of an event */
    res->next_event_position = event_buffer_size;
    res->nb_events = 0;

    res->infos = NULL;

    res->first_events_buffer_offset = (off_t)-1;
    res->current_events_buffer = NULL;
    res->thread_owner = pthread_self();

    DAGUE_LIST_ITEM_CONSTRUCT( res );
    dague_list_fifo_push( &threads, (dague_list_item_t*)res );

    return res;
}

int dague_profiling_fini( void )
{
    dague_thread_profiling_t *t;
    
    while( (t = (dague_thread_profiling_t*)dague_ulist_fifo_pop(&threads)) ) {
        free(t->hr_id);
        free(t);
    }
    free(hr_id);
    dague_list_destruct(&threads);

    dague_profiling_dictionary_flush();
    free(dague_prof_keys);
    dague_prof_keys_number = 0;

    return 0;
}

int dague_profiling_reset( void )
{
    dague_thread_profiling_t *t;
    
    DAGUE_LIST_ITERATOR(&threads, it, {
        t = (dague_thread_profiling_t*)it;
        t->next_event_position = 0;
        /* TODO: should reset the backend file / recreate it */
    });

    return 0;
}

int dague_profiling_add_dictionary_keyword( const char* key_name, const char* attributes,
                                            size_t info_length, 
                                            const char* convertor_code,
                                            int* key_start, int* key_end )
{
    unsigned int i;
    int pos = -1;

    for( i = 0; i < dague_prof_keys_count; i++ ) {
        if( NULL == dague_prof_keys[i].name ) {
            if( -1 == pos ) {
                pos = i;
            }
            continue;
        }
        if( 0 == strcmp(dague_prof_keys[i].name, key_name) ) {
            *key_start = START_KEY(i);
            *key_end = END_KEY(i);
            return 0;
        }
    }
    if( -1 == pos ) {
        if( dague_prof_keys_count == dague_prof_keys_number ) {
            ERROR(("dague_profiling_add_dictionary_keyword: Number of keyword limits reached"));
            return -1;
        }
        pos = dague_prof_keys_count;
    }

    dague_prof_keys[pos].name = strdup(key_name);
    dague_prof_keys[pos].attributes = strdup(attributes);
    dague_prof_keys[pos].info_length = info_length;
    if( NULL != convertor_code ) 
        dague_prof_keys[pos].convertor = strdup(convertor_code);
    else
        dague_prof_keys[pos].convertor = NULL;

    *key_start = START_KEY(pos);
    *key_end = END_KEY(pos);
    dague_prof_keys_count++;
    return 0;
}

int dague_profiling_dictionary_flush( void )
{
    unsigned int i;

    for( i = 0; i < dague_prof_keys_count; i++ ) {
        if( NULL != dague_prof_keys[i].name ) {
            free(dague_prof_keys[i].name);
            free(dague_prof_keys[i].attributes);
        }
    }
    dague_prof_keys_count = 0;

    return 0;
}

static dague_profiling_buffer_t *allocate_empty_buffer(int64_t *offset, char type)
{
    dague_profiling_buffer_t *res;

    if( !file_backend_extendable ) {
        *offset = -1;
        return NULL;
    }

    if( ftruncate(file_backend_fd, file_backend_next_offset+event_buffer_size) == -1 ) {
        fprintf(stderr, "Warning profiling system: resize of the events backend file failed: %s. Events trace will be truncated.\n",
                strerror(errno));
        file_backend_extendable = 0;
        *offset = -1;
        return NULL;
    }

    res = mmap(NULL, event_buffer_size, PROT_READ | PROT_WRITE, MAP_SHARED, file_backend_fd, file_backend_next_offset);
    *offset = file_backend_next_offset;
    file_backend_next_offset += event_buffer_size;

    if( MAP_FAILED == res ) {
        fprintf(stderr, "Warning profiling system: remap of the events backend file failed: %s. Events trace will be truncated.\n",
                strerror(errno));
        file_backend_extendable = 0;
        *offset = -1;
        return NULL;
    }

    if(PROFILING_BUFFER_TYPE_HEADER != type ) {
        res->next_buffer_file_offset = (off_t)-1;

        res->buffer_type = type;
        switch( type ) {
        case PROFILING_BUFFER_TYPE_EVENTS:
            res->this_buffer.nb_events = 0;
            break;
        case PROFILING_BUFFER_TYPE_DICTIONARY:
            res->this_buffer.nb_dictionary_entries = 0;
            break;
        case PROFILING_BUFFER_TYPE_THREAD:
            res->this_buffer.nb_threads = 0;
            break;
        case PROFILING_BUFFER_TYPE_GLOBAL_INFO:
            res->this_buffer.nb_infos = 0;
            break;
        }
    } else {
        assert( *offset == 0 );
    }

    return res;
}

static void write_down_existing_buffer(dague_profiling_buffer_t *buffer)
{
    if( NULL == buffer )
        return;
    if( munmap(buffer, event_buffer_size) == -1 ) {
        fprintf(stderr, "Warning profiling system: unmap of the events backend file at %p failed: %s\n",
                buffer, strerror(errno));
    }
}

static int switch_event_buffer( dague_thread_profiling_t *context )
{
    dague_profiling_buffer_t *new_buffer;
    dague_profiling_buffer_t *old_buffer;
    off_t off;

    pthread_mutex_lock( &file_backend_lock );

    new_buffer = allocate_empty_buffer(&off, PROFILING_BUFFER_TYPE_EVENTS);

    if( NULL == new_buffer ) {
        pthread_mutex_unlock( &file_backend_lock );
        return -1;
    }

    old_buffer = context->current_events_buffer;
    if( NULL == old_buffer ) {
        context->first_events_buffer_offset = off;
    } else {
        old_buffer->next_buffer_file_offset = off;
    }
    context->current_events_buffer = new_buffer;
    context->current_events_buffer_offset = off;
    context->next_event_position = 0;

    write_down_existing_buffer( old_buffer );

    pthread_mutex_unlock( &file_backend_lock );

    return 0;
}

int dague_profiling_trace( dague_thread_profiling_t* context, int key, unsigned long id, void *info )
{
    dague_profiling_output_t *this_event;
    size_t this_event_length;

    if( -1 == file_backend_fd ) {
        return -1;
    }

    this_event_length = EVENT_LENGTH( key, (NULL != info) );
    assert( this_event_length < event_avail_space );
    if( context->next_event_position + this_event_length > event_avail_space ) {
        if( switch_event_buffer(context) == -1 ) {
            return -1;
        }
    }
    /*
    fprintf(stderr, "%s event of key %d (%s) id %lu is event %ld->%ld in buffer @%ld of profiling context %p\n",
            START_KEY(BASE_KEY(key)) == key ? "start" : "end",
            BASE_KEY(key),
            dague_prof_keys[ BASE_KEY(key) ].name,
            id,
            context->next_event_position, context->next_event_position+this_event_length,
            context->current_events_buffer_offset,
            context);
    */
    this_event = (dague_profiling_output_t *)&context->current_events_buffer->buffer[context->next_event_position];
    assert( context->current_events_buffer->buffer_type == PROFILING_BUFFER_TYPE_EVENTS );
    context->current_events_buffer->this_buffer.nb_events++;

    context->next_event_position += this_event_length;
    context->nb_events++;

    this_event->event.key   = (uint16_t)key;
    this_event->event.id    = id;
    this_event->event.flags = 0;

    if( NULL != info ) {
        memcpy(this_event->info, info, dague_prof_keys[ BASE_KEY(key) ].info_length);
        this_event->event.flags = DAGUE_PROFILING_EVENT_HAS_INFO;
    }
    this_event->event.timestamp = take_time();
    
    return 0;
}

static int64_t dump_global_infos(int *nbinfos)
{
    dague_profiling_buffer_t *b, *n;
    dague_profiling_info_buffer_t *ib;
    dague_profiling_info_t *i;
    int nb, nbthis, is, vs;
    int pos;
    int64_t first_off;

    if( NULL == dague_profiling_infos ) {
        *nbinfos = 0;
        return -1;
    }

    b = allocate_empty_buffer(&first_off, PROFILING_BUFFER_TYPE_GLOBAL_INFO);
    if( NULL == b ) {
        fprintf(stderr, "Profiling System Warning: Unable to dump the global infos\n");
        *nbinfos = 0;
        return -1;
    }

    pos = 0;    
    nb = 0;
    nbthis = 0;
    for(i = dague_profiling_infos; i != NULL; i = i->next) {
        is = strlen(i->key);
        vs = strlen(i->value);
        
        if( pos + sizeof(dague_profiling_info_buffer_t) + is + vs - 1 >= event_avail_space ) {
            b->this_buffer.nb_infos = nbthis;
            n = allocate_empty_buffer(&b->next_buffer_file_offset, PROFILING_BUFFER_TYPE_GLOBAL_INFO);
            if( NULL == n ) {
                fprintf(stderr, "Profiling System Warning: Global Infos will be truncated to %d infos only\n", nb);
                *nbinfos = nb;
                return first_off;
            }

            write_down_existing_buffer(b);

            b = n;
            pos = 0;
            nbthis = 0;

        }
        ib = (dague_profiling_info_buffer_t *)&(b->buffer[pos]);
        ib->info_size = is;
        ib->value_size = vs;        
        memcpy(ib->info_and_value, i->key, ib->info_size);
        memcpy(ib->info_and_value + ib->info_size, i->value, ib->value_size);
        nb++;
        nbthis++;
        pos += sizeof(dague_profiling_info_buffer_t) + is + vs - 1;
    }

    b->this_buffer.nb_infos = nbthis;
    write_down_existing_buffer(b);

    *nbinfos = nb;
    return first_off;
}

static int64_t dump_dictionary(int *nbdico)
{
    dague_profiling_buffer_t *b, *n;
    dague_profiling_key_buffer_t *kb;
    dague_profiling_key_t *k;
    int nb, nbthis, cs, i;
    int pos;
    int64_t first_off;

    if( 0 == dague_prof_keys_count ) {
        *nbdico = 0;
        return -1;
    }

    b = allocate_empty_buffer(&first_off, PROFILING_BUFFER_TYPE_DICTIONARY);
    if( NULL == b ) {
        fprintf(stderr, "Profiling System Warning: Unable to dump the dictionary\n");
        *nbdico = 0;
        return -1;
    }

    pos = 0;    
    nb = 0;
    nbthis = 0;
    for(i = 0; i < dague_prof_keys_count; i++) {
        k = &dague_prof_keys[i];
        if(NULL == k->convertor )
            cs = 0;
        else
            cs = strlen(k->convertor);
                
        if( pos + sizeof(dague_profiling_key_buffer_t) + cs - 1 >= event_avail_space ) {
            b->this_buffer.nb_dictionary_entries = nbthis;
            n = allocate_empty_buffer(&b->next_buffer_file_offset, PROFILING_BUFFER_TYPE_DICTIONARY);
            if( NULL == n ) {
                fprintf(stderr, "Profiling System Warning: Dictionnary will be truncated to %d entries only\n", nb);
                *nbdico = nb;
                return first_off;
            }

            write_down_existing_buffer(b);

            b = n;
            pos = 0;
            nbthis = 0;

        }
        kb = (dague_profiling_key_buffer_t *)&(b->buffer[pos]);
        strncpy(kb->name, k->name, 64);
        strncpy(kb->attributes, k->attributes, 128);
        kb->keyinfo_length = k->info_length;
        if( cs > 0 ) {
            memcpy(kb->convertor, k->convertor, cs);
        }
        nb++;
        nbthis++;
        pos += sizeof(dague_profiling_key_buffer_t) + cs - 1;
    }

    b->this_buffer.nb_dictionary_entries = nbthis;
    write_down_existing_buffer(b);

    *nbdico = nb;
    return first_off;
}

static size_t thread_size(dague_thread_profiling_t *thread)
{
    size_t s = 0;
    dague_profiling_info_t *i;
    int ks, vs;

    s += sizeof(dague_profiling_thread_buffer_t) - sizeof(dague_profiling_info_buffer_t);
    for(i = thread->infos; NULL!=i; i = i->next) {
        ks = strlen(i->key);
        vs = strlen(i->value);
        if( s + ks + vs + sizeof(dague_profiling_info_buffer_t) - 1 > event_avail_space ) {
            fprintf(stderr, "Profiling System Warning: unable to save info %s of thread %s, info ignored\n",
                    i->key, thread->hr_id);
            continue;
        }
        s += ks + vs + sizeof(dague_profiling_info_buffer_t) - 1;
    }
    return s;
}

static int64_t dump_thread(int *nbth)
{
    dague_profiling_buffer_t *b, *n;
    dague_profiling_thread_buffer_t *tb;
    int nb, nbthis;
    int nbinfos, ks, vs, pos;
    dague_profiling_info_t *i;
    dague_profiling_info_buffer_t *ib;
    int64_t off;
    size_t th_size;
    dague_list_item_t *it;
    dague_thread_profiling_t* thread;

    if( dague_list_is_empty(&threads) ) {
        *nbth = 0;
        return -1;
    }

    b = allocate_empty_buffer(&off, PROFILING_BUFFER_TYPE_THREAD);
    if( NULL == b ) {
        fprintf(stderr, "Profiling System Warning: Unable to dump some thread profiles\n");
        *nbth = 0;
        return -1;
    }

    pos = 0;    
    nb = 0;
    nbthis = 0;

    for(it = DAGUE_LIST_ITERATOR_FIRST( &threads );
        it != DAGUE_LIST_ITERATOR_END( &threads );
        it = DAGUE_LIST_ITERATOR_NEXT( it ) ) {
        thread = (dague_thread_profiling_t*)it;
        th_size = thread_size(thread);
        
        if( pos + th_size >= event_avail_space ) {
            b->this_buffer.nb_threads = nbthis;
            n = allocate_empty_buffer(&b->next_buffer_file_offset, PROFILING_BUFFER_TYPE_THREAD);
            if( NULL == n ) {
                fprintf(stderr, "Profiling System Warning: Threads will be truncated to %d tnreads only\n", nb);
                *nbth = nb;
                return off;
            }

            write_down_existing_buffer(b);

            b = n;
            pos = 0;
            nbthis = 0;
        }

        tb = (dague_profiling_thread_buffer_t *)&(b->buffer[pos]);
        tb->nb_events = thread->nb_events;
        strncpy(tb->hr_id, thread->hr_id, 128);
        tb->first_events_buffer_offset = thread->first_events_buffer_offset;

        nb++;
        nbthis++;

        nbinfos = 0;
        i = thread->infos;
        pos += sizeof(dague_profiling_thread_buffer_t) - sizeof(dague_profiling_info_buffer_t);
        while( NULL != i ) {
            ks = strlen(i->key);
            vs = strlen(i->value);
            if( pos + ks + vs + sizeof(dague_profiling_info_buffer_t) - 1 >= event_avail_space ) {
                continue;
            }
            ib = (dague_profiling_info_buffer_t*)&(b->buffer[pos]);
            ib->info_size = ks;
            ib->value_size = vs;
            memcpy(ib->info_and_value, i->key, ks);
            memcpy(ib->info_and_value + ks, i->value, vs);
            pos += ks + vs + sizeof(dague_profiling_info_buffer_t) - 1;
            i = i->next;
            nbinfos++;
        }
        tb->nb_infos = nbinfos;
    }

    b->this_buffer.nb_threads = nbthis;
    write_down_existing_buffer(b);

    *nbth = nb;
    return off;
}

int dague_profiling_dump_dbp( const char* filename )
{
    int nb_threads = 0;
    dague_thread_profiling_t *t;
    int nb_infos, nb_dico;

    /* Flush existing events buffer, inconditionnally */
    DAGUE_LIST_ITERATOR(&threads, it, {
        t = (dague_thread_profiling_t*)it;
        if( NULL != t->current_events_buffer ) {
            write_down_existing_buffer(t->current_events_buffer);
            t->current_events_buffer = NULL;
        }
    });

    if( rename(bpf_filename, filename) == -1 ) {
        fprintf(stderr, "Warning Profiling System: Unable to rename events file %s in %s: %s\n",
                bpf_filename, filename, strerror(errno));
        unlink(bpf_filename);
        free(bpf_filename);
    } else {
        profile_head->dictionary_offset = dump_dictionary(&nb_dico);
        profile_head->dictionary_size = nb_dico;

        profile_head->info_offset = dump_global_infos(&nb_infos);
        profile_head->info_size = nb_infos;

        profile_head->thread_offset = dump_thread(&nb_threads);
        profile_head->nb_threads = nb_threads;

        profile_head->start_time = dague_start_time;
    }

    /* The head is now complete. Last flush. */
    write_down_existing_buffer((dague_profiling_buffer_t *)profile_head);

    /* Close the backend file */
    pthread_mutex_lock(&file_backend_lock);
    close(file_backend_fd);
    file_backend_fd = -1;
    file_backend_extendable = 0;
    pthread_mutex_unlock(&file_backend_lock);

    return 0;
}

char *dague_profile_ddesc_key_to_string = "";
