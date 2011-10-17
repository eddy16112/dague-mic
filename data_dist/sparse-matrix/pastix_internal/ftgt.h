/************************************************************/
/**                                                        **/
/**   NAME       : ftgt.h                                  **/
/**                                                        **/
/**   AUTHORS    : David GOUDIN                            **/
/**                Pascal HENON                            **/
/**                Francois PELLEGRINI                     **/
/**                Pierre RAMET                            **/
/**                                                        **/
/**   FUNCTION   : Part of a parallel direct block solver. **/
/**                These lines are the data declarations   **/
/**                for the graph structure.                **/
/**                                                        **/
/**   DATES      : # Version 0.0  : from : 02 may 1998     **/
/**                                 to     16 oct 1998     **/
/**                                                        **/
/************************************************************/

#define FTGT_H

#ifdef CXREF_DOC
#include "common_pastix.h"
#endif /* CXREF_DOC */


/*
**  The type and structure definitions.
*/

/*
** WARNING : All structures should have a odd number of integer for memory alignment
*/

/*+ Fanintarget info type +*/

typedef enum {
  FTGT_CTRBNBR = 0,                               /*+ Number of contributions           +*/
  FTGT_CTRBCNT,                                   /*+ Number of contributions remaining +*/
  FTGT_PROCDST,                                   /*+ Destination for fanintarget       +*/
  FTGT_TASKDST,                                   /*+ Task  destination                 +*/
  FTGT_BLOKDST,                                   /*+ Block destination (->COMP_1D)     +*/
  FTGT_PRIONUM,                                   /*+ Fanintarget priority              +*/
  FTGT_FCOLNUM,                                   /*+ Fanintarget first column          +*/
  FTGT_LCOLNUM,                                   /*+ Fanintarget last column           +*/
  FTGT_FROWNUM,                                   /*+ Fanintarget first row             +*/
  FTGT_LROWNUM,                                   /*+ Fanintarget last row              +*/
#if (defined OOC) || (defined TRACE_SOPALIN)
  FTGT_GCBKDST,                                   /*+ Global Cblk destination(->COMP_1D)+*/
  FTGT_IDTRACE,                                   /*+ To have 12 integer in FanInTarget +*/
#endif
  MAXINFO
} FanInInfo;

typedef enum {
  BTAG_PRIONUM = 0,
  BTAG_TASKDST,
  BTAG_PROCDST,
  BTAG_TASKCNT,
#if (defined TRACE_SOPALIN)
  BTAG_NULL,                                      /*+ Global Cblk destination(->COMP_1D)+*/
  BTAG_IDTRACE,                                   /*+ To have 12 integer in FanInTarget +*/
#endif
  BTAGINFO
} BtagInfo;

typedef enum {
  BCOF_FROWNUM = 0,
  BCOF_LROWNUM,
  BCOF_FCOLNUM,
  BCOF_LCOLNUM,
  BCOFINFO
} BcofInfo;




/*+ Fanintarget structure +*/

typedef struct FanInTarget_ {
  INT                       infotab[MAXINFO];     /*+ Fanintarget descriptor (size MAXINFO) +*/
  FLOAT *                   coeftab;              /*+ Fanintarget vector access             +*/
} FanInTarget;

typedef struct BlockCoeff_ 
{
  INT              infotab[BCOFINFO];
  volatile FLOAT * coeftab; /* blocktarget coeff vector if != NULL envoi possible */
  INT              sendcnt; /* number of blocktarget send, if == 0 free coeftab */
} BlockCoeff;


/*+ BlockTarget structure +*/
typedef struct BlockTarget_ {
  INT           infotab[BTAGINFO];
  BlockCoeff*   bcofptr; /* index of the blockCoeff                        */
} BlockTarget;

