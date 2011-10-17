/*
  File: sopalin3d.h

  Contains structures used in sopalin and declarations of functions of <sopalin3d.h>
 */

#ifndef SOPALIN_3D_H
#define SOPALIN_3D_H

#ifndef SOPALIN_THREAD_H
#error "sopalin_thread.h must be included before sopalin3d.h"
#endif

#ifdef OOC
typedef struct ooc_thread_ ooc_thread_t;
typedef struct ooc_ ooc_t;
#endif

/*
  Enum: SOPALIN_TASK

  tasks wich will be computed in current PaStiX call

  constants:
  
  SOPALIN_ONLY       - Only runs factorisation.
  SOPALIN_UPDO       - Runs factorisation and up down.
  SOPALIN_UPDO_GMRES - Runs factorisation, up down and GMRES.
  SOPALIN_UPDO_GRAD  - Runs factorisation, up down and conjugate gradient.
  SOPALIN_UPDO_PIVOT - Runs factorisation, up down and simple iterative raffinement.
  UPDO_ONLY          - Only up down.
  RAFF_GMRES         - Only GMRES.
  RAFF_GRAD          - Only conjugate gradient.
  RAFF_PIVOT         - Only simple iterative raffinement.
  SOPALIN_NBTASKS    - Number of existing tasks.
*/
enum SOPALIN_TASK {
  SOPALIN_ONLY = 0,
  SOPALIN_UPDO,
  SOPALIN_UPDO_GMRES,
  SOPALIN_UPDO_GRAD,
  SOPALIN_UPDO_PIVOT,
  UPDO_ONLY,
  RAFF_GMRES,
  RAFF_GRAD,
  RAFF_PIVOT,
  SOPALIN_NBTASKS
};


/************************************************/
/*         Parametres de sopalin                */
/************************************************/
/*
  struct: SopalinParam_

  Parameters for factorisation, updown and reffinement. 
 */
typedef struct SopalinParam_ {
  double   epsilonraff;         /*+ epsilon to stop reffinement                      +*/
  double   rberror;             /*+ ||r||/||b||                                      +*/
  double   espilondiag;         /*+ epsilon critere for diag control                 +*/
  FLOAT   *b;                   /*+ b vector (RHS and solution)                      +*/
  FLOAT   *transcsc;            /*+ transpose csc                                    +*/
  INT      itermax;             /*+ max number of iteration                          +*/
  INT      diagchange;          /*+ number of change of diag                         +*/
  INT      gmresim;	        /*+ Krylov subspace size for GMRES                   +*/
  INT      fakefact;	        /*+ Flag indicating if we want fake factorisation    +*/  
  INT      usenocsc;	        /*+ Flag indicating if we want to use the intern CSC +*/
  int      factotype;           /*+ Type of factorization                            +*/
  int      symmetric;           /*+ Symmetric                                        +*/
  MPI_Comm pastix_comm;         /*+ MPI communicator                                 +*/
  int      type_comm;           /*+ Communication mode                               +*/
  int      nbthrdcomm;          /*+ Communication's thread number                    +*/
  INT     *iparm;               /*+ In/Out integer parameters                        +*/
  double  *dparm;               /*+ In/Out float parameters                          +*/
  int     *bindtab;             /*+ Define where to bin threads                      +*/
#ifdef THREAD_COMM
  int      stopthrd;
#endif
  int      schur;               /*+ If API_YES won't compute last diag               +*/
  INT      n;                   /*+ size of the matrix                               +*/
  INT      gN;
} SopalinParam;

/************************************************/
/*  Structure pour le AllReduce Funneled        */
/************************************************/
/*
  Struct: Pastix_Allreduce_

  Structure used for MPI_Allreduce operations in Thread Funneled mode.
*/
typedef struct Pastix_Allreduce_ {
  void *           sendbuf;        /*+ Sending buffer                  +*/
  void *           recvbuf;        /*+ receiving buffer                +*/
  int              count;	   /*+ Number of elements to reduce    +*/
  MPI_Datatype     datatype;	   /*+ MPI Datatype to reduce          +*/   
  MPI_Op           op;             /*+ MPI operation                   +*/
} Pastix_Allreduce_t;


/************************************************/
/*         Thread Data                          */
/*   Donn�es locales � chaque threads           */
/*   n�cessitant pas de mutex pour modification */
/************************************************/
/* 
   Struct: Thread_Data_

   Structure used to contain data local to each thread.
   These datas do not need to be protected by mutexes.
 */
typedef struct Thread_Data_ {
  Clock            sop_clk;                  /*+ Clock                                               +*/
  Clock            sop_clk_comm;             /*+ Communication clock                                 +*/
  INT              nbpivot;                  /*+ Number of pivoting performed                        +*/
  INT              flag_bind;                /*+ Indicate if threads are binded on processors        +*/
#ifdef TRYLOCK   			       
  INT              ptbusy;                   /*+ Number of mutexes in use                            +*/
  INT              ptfree;                   /*+ Nomber of free mutexes                              +*/
  INT              ptwait;                   /*+ Nombre de cond_wait appele                          +*/
#endif	         			       
  INT              firstbloktab;             /*+ First block index to compute stride in maxbloktab   +*/ 
  INT              stridebloktab;            /*+ Temporary tabulars maxblokttab's stride copy        +*/ 
  FLOAT           *maxbloktab1;              /*+ Temporary tabular to add contributions              +*/
  FLOAT           *maxbloktab2;              /*+ Temporary tabular to add contributions (for LU)     +*/
  MPI_Request     *send_block_requests;      /*+ sent blocks requests                                +*/
  INT             *send_block_target;        /*+ sent blocks targets                                 +*/
  MPI_Request     *send_fanin_requests;      /*+ sent fanins requests                                +*/
#if ((defined PASTIX_UPDO_ISEND) || (defined TEST_ISEND)) && (!defined NO_MPI_TYPE)
  MPI_Datatype    *send_fanin_mpitypes;      /*+ sent fanins mpi types                               +*/
#endif
#if (defined PASTIX_UPDO_ISEND) && (!defined NO_MPI_TYPE)
  INT            **send_fanin_infotab;       /*+ sent fanins mpi types                               +*/
#endif
  INT             *send_fanin_target;        /*+ sent fanins targets                                 +*/
  INT            **send_fanin_target_extra;  /*+ extra sent fanin targets                            +*/
#ifdef TEST_IRECV			       
  MPI_Request     *recv_fanin_request;       /*+ receiving fanin requests                            +*/
  MPI_Request     *recv_block_request;       /*+ receiving blocks requests                           +*/
  void           **recv_fanin_buffer;        /*+ fanin reception buffers                             +*/
  void           **recv_block_buffer;        /*+ blocks reception buffers                            +*/
#endif					       
  INT              maxsrequest_fanin;        /*+ Maximum number of send requests used                +*/
  INT              maxsrequest_block;        /*+ Maximum number of send requests used                +*/
#ifndef FORCE_NOMPI
  MPI_Status      *srteststatus;
  int             *srtestindices;
#endif
  void            *recv_buffer;              /*+ reception buffer                                    +*/
  int             *gtabsize;                 /*+ size of the data type to send                       +*/
#ifndef NO_MPI_TYPE			       						             
  MPI_Aint        *gtaboffs;                 /*+ offsize of the data type to send                    +*/
  MPI_Datatype    *gtabtype;                 /*+ Type of data to send                                +*/
#else 					       
  void           **gtaboffs;                 /*+ pointer to the data to send                         +*/
  int             *gtabtype;                 /*+ Size of the data to send                            +*/
  void           **send_fanin_buffer;        /*+ Fanins sending buffers                              +*/
  void           **send_block_buffer;        /*+ Blocks sending buffers                              +*/
  INT             *send_fanin_buffer_size;   /*+ Fanins sending buffers size                         +*/
  INT             *send_block_buffer_size;   /*+ Blocks sending buffers size                         +*/
#endif /* NO_MPI_TYPE */		       
#ifdef TRACE_SOPALIN 			       
  FILE            *tracefile;                /*+ Tracing file for the solver                         +*/
  int              traceactive;              /*+ Flag indicating if trace is active                  +*/
  int              traceid;                  /*+ Flag indicating if trace is active                  +*/
#endif
#ifdef COMPACT_SMX
  INT             stridebloktab2;
#endif /* COMPACT_SMX */
#ifdef PASTIX_DYNSCHED
  INT             esp;
#ifdef PASTIX_REVERTSTEAL
  faststack_t     stack;
  INT            *tabtravel;
#endif
#endif
} Thread_Data_t;

/************************************************/
/*          Sopalin Data                        */
/*   Donn�es Communes � tous les threads        */
/************************************************/
/* 
   Struct: Sopalin_Data_
   
   Data common to all threads.
*/
typedef struct Sopalin_Data_ {
  SolverMatrix    *datacode;                 /*+ Solver matrix                      +*/
  SopalinParam    *sopar;                    /*+ Sopalin parameters                 +*/
  Thread_Data_t  **thread_data;              /*+ Threads data                       +*/
  Queue           *fanintgtsendqueue;        /*+ Fanins to send queue               +*/
  Queue           *blocktgtsendqueue;        /*+ Blocks to send queue               +*/
  INT             *taskmark;                 /*+ Task marking for 2D                +*/
#ifdef TRACE_SOPALIN 								    
  FILE            *tracefile;                /*+ Tracing file for the solver        +*/
  double           timestamp;                /*+ Original time for trace            +*/
#endif										    
#if (defined COMPUTE_ALLOC) || (defined STATS_SOPALIN)				    
  INT              current_alloc;            /*+ Current allocated memory           +*/
#endif										    
#ifdef ALLOC_FTGT								    
  INT              max_alloc;                /*+ Maximum allocated memory           +*/
  INT              alloc_init;               /*+ Initial allocated memory           +*/
#ifdef STATS_SOPALIN
  pthread_mutex_t  mutex_alloc;		     /*+ Mutex on allocated memory variable +*/
#endif
#endif
#ifdef USE_CSC
  double volatile  critere;                  /*+ Stopping threshold for refinement   +*/
  double volatile  stop;		     /*+ Flag to stop threads on refinement  +*/
  double volatile  berr;		     /*+ Error in refinement                 +*/
  double volatile  lberr;		     /*+ Last error in refinement            +*/
  INT    volatile  raffnbr;		     /*+ Number of iterations in refinement  +*/
  INT    volatile  count_iter;		     /*+ Number of iterations in refinement  +*/
  INT    volatile  flag_gmres;		     /*+ Flag to continue in static pivoting +*/
  INT    volatile  gmresout_flag;	     /*+ Flag for GMRES outter loop          +*/
  INT    volatile  gmresin_flag;	     /*+ Flag for GMRES inner loop           +*/
  double volatile  gmresro;                  /*+ Norm of GMRES residue               +*/
#endif
#ifdef SMP_SOPALIN
  pthread_mutex_t *mutex_task;		     /*+ Mutex on each task                               +*/
  pthread_cond_t  *cond_task;		     /*+ Cond for each task                               +*/
  pthread_mutex_t *mutex_fanin;		     /*+ Mutex on each fanin                              +*/
  pthread_cond_t  *cond_fanin;		     /*+ Cond for each fanin                              +*/
  pthread_mutex_t *mutex_blok;		     /*+ Mutex on each block                              +*/
  pthread_mutex_t *mutex_queue_fanin;	     /*+ Mutex on the fanins queue                        +*/
  pthread_mutex_t *mutex_queue_block;	     /*+ Mutex on the blocks queue                        +*/
#else /* SMP_SOPALIN */								                            
  Queue            taskqueue;		     /*+ Task queue for NOSMP version                     +*/
#endif										                            
  sopthread_barrier_t barrier;               /*+ Threads synchronisation barrier                  +*/
#ifdef THREAD_COMM								                            
  pthread_mutex_t  mutex_comm;               /*+ Mutex on communication variables                 +*/
  pthread_cond_t   cond_comm;	             /*+ Condition on step_comm                           +*/
  int              step_comm;		     /*+ Current step indicator                           +*/
#ifdef PASTIX_FUNNELED								                  	       
  Pastix_Allreduce_t allreduce;              /*+ Data structure for MPi_Allreduce                 +*/
  Queue             *sendqueue;              /*+ Ready to send data queue                         +*/
#endif										                  	       
#endif /* THREAD_COMM */							                  	       
#ifdef STORAGE									                  	       
  FLOAT           *grhs;		     /*+ Data storage tabular                             +*/
  volatile INT    *flagtab;		     /*+ Indicate received cblk in up step                +*/
  pthread_mutex_t *mutex_flagtab;	     /*+ Mutex on flagtab                                 +*/
  pthread_cond_t  *cond_flagtab;             /*+ cond on flagtab                                  +*/
#endif										                  
  volatile void   *ptr_raff[10];             /*+ pointers used in refinement                      +*/
  void           **ptr_csc;                  /*+ pointer to data used by each threads in csc_code +*/
  double          *common_flt;               /*+ Common pointer to share a float between threads  +*/
  pthread_mutex_t  mutex_raff;               /*+ mutex on common tabulars during csc operations   +*/
  pthread_cond_t   cond_raff;                /*+ cond corresponding to mutex_raff                 +*/
#ifdef PASTIX_DYNSCHED
  pthread_mutex_t *tasktab_mutex;            /*+ +*/
  pthread_cond_t  *tasktab_cond;	     /*+ +*/
  volatile INT    *tasktab_indice;	     /*+ +*/
  volatile INT    *tasktab_nbthrd;	     /*+ +*/
  Queue           *taskqueue;                /*+ +*/
#endif
#ifdef OOC
  ooc_t           *ooc;                      /*+ Data structure needed for Out-of-core            +*/
#endif
} Sopalin_Data_t;


/************************************************/
/*     Fonctions publiques de sopalin3d         */
/************************************************/
/*
  Functions: <Sopalin3d.c> functions declarations. 
 */
void  Usopalin_launch           (SolverMatrix *, SopalinParam *, INT cas);
void  Usopalin_thread           (SolverMatrix *, SopalinParam *);
void  Usopalin_updo_thread      (SolverMatrix *, SopalinParam *);
void  Usopalin_updo_gmres_thread(SolverMatrix *, SopalinParam *);
void  Usopalin_updo_grad_thread (SolverMatrix *, SopalinParam *);
void  Usopalin_updo_pivot_thread(SolverMatrix *, SopalinParam *);
void  Uupdo_thread              (SolverMatrix *, SopalinParam *);
void  Upivot_thread             (SolverMatrix *, SopalinParam *);
void  Ugmres_thread             (SolverMatrix *, SopalinParam *);
void  Ugrad_thread              (SolverMatrix *, SopalinParam *);

void  Dsopalin_launch           (SolverMatrix *, SopalinParam *, INT cas);
void  Dsopalin_thread           (SolverMatrix *, SopalinParam *);
void  Dsopalin_updo_thread      (SolverMatrix *, SopalinParam *);
void  Dsopalin_updo_gmres_thread(SolverMatrix *, SopalinParam *);
void  Dsopalin_updo_grad_thread (SolverMatrix *, SopalinParam *);
void  Dsopalin_updo_pivot_thread(SolverMatrix *, SopalinParam *);
void  Dupdo_thread              (SolverMatrix *, SopalinParam *);
void  Dpivot_thread             (SolverMatrix *, SopalinParam *);
void  Dgmres_thread             (SolverMatrix *, SopalinParam *);
void  Dgrad_thread              (SolverMatrix *, SopalinParam *);

void  Lsopalin_launch           (SolverMatrix *, SopalinParam *, INT cas);
void  Lsopalin_thread           (SolverMatrix *, SopalinParam *);
void  Lsopalin_updo_thread      (SolverMatrix *, SopalinParam *);
void  Lsopalin_updo_gmres_thread(SolverMatrix *, SopalinParam *);
void  Lsopalin_updo_grad_thread (SolverMatrix *, SopalinParam *);
void  Lsopalin_updo_pivot_thread(SolverMatrix *, SopalinParam *);
void  Lupdo_thread              (SolverMatrix *, SopalinParam *);
void  Lpivot_thread             (SolverMatrix *, SopalinParam *);
void  Lgmres_thread             (SolverMatrix *, SopalinParam *);
void  Lgrad_thread              (SolverMatrix *, SopalinParam *);

#endif
