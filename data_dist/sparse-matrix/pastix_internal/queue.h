/************************************************************/
/**                                                        **/
/**   NAME       : queue.h                                 **/
/**                                                        **/
/**   AUTHORS    : Pascal HENON                            **/
/**                                                        **/
/**   FUNCTION   : queue of INT that sorts elements        **/
/**                in ascending way according to a         **/
/**                FLOAT key                               **/
/**   DATES      : # Version 0.0  : from : 22 jul 1998     **/
/**                                 to     08 sep 1998     **/
/**                                                        **/
/************************************************************/

#ifndef QUEUE_H
#define QUEUE_H

/*
**  The type and structure definitions.
*/

typedef struct Queue_ {
  INT        size;                  /*+ Allocated memory size             +*/ 
  INT        used;                  /*+ Number of element in the queue    +*/
  INT    *   elttab;                /*+ Array of the element              +*/
  double *   keytab;                /*+ Array of keys                     +*/
  INT    *   keytab2;               /*+ Another array of keys             +*/
} Queue;


#define static

int     queueInit       (Queue *, INT size);
void    queueExit       (Queue *);
Queue * queueCopy       (Queue *dst, Queue *src);
void    queueAdd        (Queue *, INT, double);
void    queueAdd2       (Queue *, INT, double, INT);
INT     queueGet        (Queue *);
INT     queueSize       (Queue *);
void    queueClear      (Queue *);
INT     queueRead       (Queue *);
INT     queueGet2       (Queue *, double *, INT *);
int     queuePossess    (Queue *, INT);
void    queuePrint      (Queue *);

static INT compWith2keys(Queue *, INT, INT);
#undef static
#endif
