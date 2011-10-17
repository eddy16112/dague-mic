/************************************************************/
/**                                                        **/
/**   NAME       : dof.h                                   **/
/**                                                        **/
/**   AUTHORS    : David GOUDIN                            **/
/**                Pascal HENON                            **/
/**                Francois PELLEGRINI                     **/
/**                Pierre RAMET                            **/
/**                                                        **/
/**   FUNCTION   : Part of a parallel direct block solver. **/
/**                These lines are the data declarations   **/
/**                for the DOF handling structure.         **/
/**                                                        **/
/**   DATES      : # Version 0.0  : from : 07 oct 1998     **/
/**                                 to     16 oct 1998     **/
/**                                                        **/
/************************************************************/

#define DOF_H

#ifdef CXREF_DOC
#ifndef COMMON_H
#include "common_pastix.h"
#endif /* COMMON_H */
#ifndef GRAPH_H
#include "graph.h"
#endif /* GRAPH_H */
#endif /* CXREF_DOC */

/*
**  The type and structure definitions.
*/

/*+ The DOF structure. This structure is
    always associated to a Graph structure,
    which holds the base value.             +*/

typedef struct Dof_ {
  INT                       baseval;              /*+ Base value for indexing                                       +*/
  INT                       nodenbr;              /*+ Number of nodes in DOF array                                  +*/
  INT                       noddval;              /*+ DOF value for every node (if noddtab == NULL, 0 else)         +*/
  INT *                     noddtab;              /*+ Array of node->first DOF indexes (if noddval == 0) [+1,based] +*/
} Dof;

/*
**  The function prototypes.
*/

#ifndef DOF
#define static
#endif

INT                         dofInit             (Dof * const deofptr);
void                        dofExit             (Dof * const deofptr);
INT                         dofLoad             (Dof * const deofptr, FILE * const stream);
INT                         dofSave             (const Dof * const deofptr, FILE * const stream);
void                        dofConstant         (Dof * const deofptr, const INT baseval, const INT nodenbr, const INT noddval);
INT                         dofGraph            (Dof * const deofptr, const Graph * grafptr, const INT * const peritab);

#undef static

/*
**  The macro definitions.
*/

#ifdef DOF_CONSTANT
#define noddVal(deofptr,nodenum)    ((deofptr)->baseval + (deofptr)->noddval * ((nodenum) - (deofptr)->baseval))
#define noddDlt(deofptr,nodenum)    ((deofptr)->noddval)
#else /* DOF_CONSTANT */
#define noddVal(deofptr,nodenum)    (((deofptr)->noddtab != NULL) ? (deofptr)->noddtab[(deofptr)->baseval + (nodenum)] : ((deofptr)->baseval + (deofptr)->noddval * ((nodenum) - (deofptr)->baseval)))
#define noddDlt(deofptr,nodenum)    (((deofptr)->noddtab != NULL) ? ((deofptr)->noddtab[(deofptr)->baseval + (nodenum) + 1] - (deofptr)->noddtab[(deofptr)->baseval + (nodenum)]) : (deofptr)->noddval)
#endif /* DOF_CONSTANT */
