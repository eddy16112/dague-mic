/*
  File: debug.h

  Defines debugs flags and <print_debug> macro which use them.

  Authors:
    Mathieu Faverge - faverge@labri.fr
    Xavier  LACOSTE - lacoste@labri.fr

 */
/* DEBUG FLAGS for print_debug */
/* Scotch */
#define DBG_SCOTCH              0
#define DBG_STEP                0
#define DBG_CSCD                0
/* blend */
#define DBG_BUBBLESPLIT         0
#define DBG_BUBBLES             0

/* Sopalin */
#define DBG_SOPALIN_NAN         0
#define DBG_SOPALIN_INF         0
#define DBG_SOPALIN_RAFF        0
#define DBG_SOPALIN_DEBUG       0
#define DBG_SOPALIN_MAIN        0
#define DBG_SOPALIN_THREADCOMM  0
#define DBG_SOPALIN_ALLOC       0
#define DBG_SOPALIN_BLEND       0
#define DBG_SOPALIN_DRUNK       0
#define DBG_SOPALIN_COMM        0
#define DBG_SOPALIN_COMPUTE     0
#define DBG_SOPALIN_COMP1D      0
#define DBG_SOPALIN_DIAG        0
#define DBG_SOPALIN_E1          0
#define DBG_SOPALIN_E2          0
#define DBG_SOPALIN_NAPA        0
#define DBG_FUNNELED            0
#define DBG_THCOMM              0
#define DBG_UPDO                0
#define DBG_PASTIX_DYNSCHED     0
#define DBG_SOPALIN_TIME        0
#define DBG_CSC_LOG             0
#define DBG_PASTIX_REVERTSTEAL  0

/* Sopalin SendRecv */
#define DBG_SOPALIN_SEND        0
#define DBG_SOPALIN_RECV        0


/* updown */		        
#define DBG_SOPALIN_UPDO        0
#define DBG_SOPALIN_UP          0
#define DBG_SOPALIN_DOWN        0

/* OOC */		        
#define DBG_OOC_TRACE_V1        0
#define DBG_OOC_TRACE_V2        0
#define DBG_OOC_DEBUG           0
#define DBG_OOC_PREDICTION      0
#define DBG_OOC_SAVE            0
#define DBG_OOC_MUTEX_ALLOCATED 0
#define DBG_OOC_WAIT_FOR_FTGT   0
#define DBG_OOC_FTGT            0

/* Raff */
#define DBG_RAFF_PIVOT          0
#define DBG_RAFF_GMRES          0
#define DBG_RAFF_GRAD           0

/* MURGE */
#define DBG_MURGE               0

/*
  Macro: print_debug
  
  Prints debugging message if PASTIX_DEBUG is defined and if the 
  debug flag is the to 1.
 */
#ifdef PASTIX_DEBUG
#define print_debug(mod,...) {if (mod) fprintf(stderr, __VA_ARGS__);}
#else
#define print_debug(...)     {}
#endif
