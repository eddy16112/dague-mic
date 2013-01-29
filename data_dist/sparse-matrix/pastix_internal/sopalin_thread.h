/*
  File: sopalin_thread.h

  Structures, function declarations and macros used for thread management.
 */
#ifndef SOPALIN_THREAD_H
#define SOPALIN_THREAD_H

#include <sys/time.h>
#include <time.h>


/*
  Struct: sopthread_data

  Structure conotaining the thread number and a pointer to data given to thread in parameters.
 */
typedef struct sopthread_data {
  int    me;                   /*+ Thread number                    +*/
  void * data;                 /*+ Data given to thread as argument +*/
} sopthread_data_t;

/*
 * Function: sopalin_launch_thread
 *
 * Launch all PaStiX threads on wanted functions.
 *
 * Parameters:
 *   procnum       - MPI process rank
 *   procnbr       - Number of MPI process
 *   ptr           - Pointer the the bubble structure to use (if MARCEL version)
 *   verbose       - Verbosity level
 *   calc_thrdnbr  - Number of computing threads.
 *   calc_routine  - Computing function.
 *   calc_data     - Parameters for computing function.
 *   comm_thrdnbr  - Number of communicating threads.
 *   comm_routine  - communication function.
 *   comm_data     - Parameters for communication function.
 *   ooc_thrdnbr   - Number of out-of-core threads.
 *   ooc_routine   - Out-of-core function.
 *   ooc_data      - Parameters for *ooc_routine*.
 */
void sopalin_launch_thread(void *sopalin_data,
                           INT procnum, INT procnbr, void *ptr, INT verbose,
			   INT calc_thrdnbr, void * (*calc_routine)(void *), void *calc_data,
			   INT comm_thrdnbr, void * (*comm_routine)(void *), void *comm_data,
			   INT ooc_thrdnbr,  void * (*ooc_routine) (void *), void *ooc_data);

/*
  Function: sopalin_launch_comm

  Launch communication threads

  Parameters
    nbthrdcomm    - Number of threads to launch.
    comm_routine  - Communication function.
    data          - Data for communication function.
 */
void sopalin_launch_comm(int nbthrdcomm, void * (*comm_routine)(void *), void *data);


/*
  Function: sopalin_bindthread

  Bind threads onto processors.

  Parameters:
    cpu - Processor to bind to.
 */
INT  sopalin_bindthread(INT);

/* Version SMP */
#ifndef FORCE_NOSMP

#define MONOTHREAD_BEGIN if(me==0){
#define MONOTHREAD_END   }

/*
  Struct: sopthread_barrier

  Computing threads synchronisation barrier.
 */
typedef struct sopthread_barrier {
  int volatile    instance;         /*+ ID of the barrier                +*/
  int volatile    blocked_threads;  /*+ Number of threads in the barrier +*/
  pthread_mutex_t sync_lock;        /*+ mutex for the barrier            +*/
  pthread_cond_t  sync_cond;        /*+ cond for the barrier             +*/
} sopthread_barrier_t;

/*
  Macro: SYNCHRO_X_THREAD

  Synchronize *nbthread* threads.

  Parameters: 
    nbthread - Number of threads to synchronize.
    barrier  - sopthread_barrier structure associated with the synchronisation.
  
 */
#define SYNCHRO_X_THREAD(nbthread, barrier)                             \
  {                                                                     \
    int instance;                                                       \
    pthread_mutex_lock(&((barrier).sync_lock));                         \
    instance = (barrier).instance;                                      \
    (barrier).blocked_threads++;                                        \
    if ((barrier).blocked_threads == (nbthread))                        \
      {                                                                 \
        (barrier).blocked_threads = 0;                                  \
        (barrier).instance++;                                           \
        pthread_cond_broadcast(&((barrier).sync_cond));                 \
      }                                                                 \
    while (instance == (barrier).instance)                              \
      {                                                                 \
        pthread_cond_wait(&((barrier).sync_cond), &((barrier).sync_lock)); \
      }                                                                 \
    pthread_mutex_unlock(&((barrier).sync_lock));                       \
  }
 
/*
 * Définition de MUTEX_LOCK et COND_WAIT (avec ou sans compteurs)
 */
#ifdef TRYLOCK
/*
INT *ptbusy,*ptfree;
INT *ptwait;
*/
#define MUTEX_LOCK(x)   if (pthread_mutex_trylock(x)) {thread_data->ptbusy;pthread_mutex_lock(x);} \
                        else thread_data->ptfree++
#define COND_WAIT(x,y)  pthread_cond_wait(x,y); thread_data->ptwait++

/* Cond de 5ms */
#define COND_TIMEWAIT(x,y) {				\
    struct timeval  now;				\
    struct timespec timeout;				\
    gettimeofday(&now, NULL);				\
    timeout.tv_sec  = now.tv_sec;			\
    timeout.tv_nsec = now.tv_usec * 1000 + 5 * 1000000;	\
    pthread_cond_timedwait(x,y,&timeout);		\
    thread_data->ptwait++;				\
  }

#else

#define MUTEX_LOCK(x)      pthread_mutex_lock(x)
#define COND_WAIT(x,y)     pthread_cond_wait(x,y)
/* Cond de 5ms */
#define COND_TIMEWAIT(x,y) {				\
    struct timeval  now;				\
    struct timespec timeout;				\
    gettimeofday(&now, NULL);				\
    timeout.tv_sec  = now.tv_sec;			\
    timeout.tv_nsec = now.tv_usec * 1000 + 5 * 1000000;	\
    pthread_cond_timedwait(x,y,&timeout);		\
  }

#endif

#define MUTEX_UNLOCK(x) pthread_mutex_unlock(x)

/* #if (_POSIX_C_SOURCE >= 200112L || _XOPEN_SOURCE >= 600) */
/* #define SEM_TIMEWAIT(x, err) {						\ */
/*     struct timeval  now;						\ */
/*     struct timespec timeout;						\ */
/*     gettimeofday(&now, NULL);						\ */
/*     timeout.tv_sec  = now.tv_sec;					\ */
/*     timeout.tv_nsec = now.tv_usec * 1000 + 5 * 1000000;			\ */
/*     err = sem_timedwait(x,&timeout);					\ */
/*   } */
/* #else */
/* #define SEM_TIMEDWAIT(x, err) {err = sem_wait(x);} */
/* #endif */

/* Version Non-SMP */
#else /* FORCE_NOSMP */

#define pthread_mutex_lock(x)
#define pthread_mutex_unlock(x)

#define MUTEX_LOCK(x)    {}
#define MUTEX_UNLOCK(x)  {}

#define pthread_cond_signal(x)
#define pthread_cond_broadcast(x)
#define COND_WAIT(x,y)
#define COND_TIMEWAIT(x,y)

#define SYNCHRO_X_THREAD(nbthrd, barrier)
#define MONOTHREAD_BEGIN 
#define MONOTHREAD_END

#define sopthread_barrier_t int
#endif /* FORCE_NOSMP */

#endif /* SOPALIN_THREAD_H */

