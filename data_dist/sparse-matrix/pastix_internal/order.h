/* Copyright INRIA 2004
**
** This file is part of the Scotch distribution.
**
** The Scotch distribution is libre/free software; you can
** redistribute it and/or modify it under the terms of the
** GNU Lesser General Public License as published by the
** Free Software Foundation; either version 2.1 of the
** License, or (at your option) any later version.
**
** The Scotch distribution is distributed in the hope that
** it will be useful, but WITHOUT ANY WARRANTY; without even
** the implied warranty of MERCHANTABILITY or FITNESS FOR A
** PARTICULAR PURPOSE. See the GNU Lesser General Public
** License for more details.
**
** You should have received a copy of the GNU Lesser General
** Public License along with the Scotch distribution; if not,
** write to the Free Software Foundation, Inc.,
** 59 Temple Place, Suite 330, Boston, MA 02111-1307, USA.
**
** $Id: order.h 2 2004-06-02 14:05:03Z ramet $
*/
/************************************************************/
/**                                                        **/
/**   NAME       : order.h                                 **/
/**                                                        **/
/**   AUTHORS    : Francois PELLEGRINI                     **/
/**                                                        **/
/**   FUNCTION   : Part of a parallel direct block solver. **/
/**                These lines are the data declarations   **/
/**                for the graph ordering routine.         **/
/**                                                        **/
/**   DATES      : # Version 0.0  : from : 22 aug 1998     **/
/**                                 to     01 may 1999     **/
/**                # Version 2.0  : from : 25 oct 2003     **/
/**                                 to     22 apr 2004     **/
/**                                                        **/
/************************************************************/

#define ORDER_H

/*
**  The type and structure definitions.
*/

/*+ Ordering structure. vnodbas holds the base
    value for node indexings. vnodbas is equal
    to baseval for graphs, and to vnodbas for
    meshes. The same holds for rangtab, with
    rangtab[0] = vnodbas.                      +*/

typedef struct Order_ {
  INT                       cblknbr;              /*+ Number of column blocks             +*/
  INT *                     rangtab;              /*+ Column block range array [based,+1] +*/
  INT *                     permtab;              /*+ Permutation array [based]           +*/
  INT *                     peritab;              /*+ Inverse permutation array [based]   +*/
} Order;

/*
**  The function prototypes.
*/

#ifndef ORDER
#define static
#endif

int                         orderInit           (Order * const ordeptr);
void                        orderExit           (Order * const ordeptr);
int                         orderLoad           (Order * const ordeptr, FILE * const stream);
int                         orderSave           (const Order * const ordeptr, FILE * const stream);
void                        orderBase           (Order * restrict const ordeptr, const INT baseval);

int                         orderCheck          (const Order * const ordeptr);

int                         orderGrid2          (Order * const ordeptr, const INT xnbr, const INT ynbr, const INT baseval, const INT xlim, const INT ylim);
int                         orderGrid2C         (Order * const ordeptr, const INT xnbr, const INT ynbr, const INT baseval, const INT xlim, const INT ylim);
int                         orderGrid3          (Order * const ordeptr, const INT xnbr, const INT ynbr, const INT znbr, const INT baseval, const INT xlim, const INT ylim, const INT zlim);
int                         orderGrid3C         (Order * const ordeptr, const INT xnbr, const INT ynbr, const INT znbr, const INT baseval, const INT xlim, const INT ylim, const INT zlim);

#if (defined SCOTCH_PTSCOTCH) || (defined SCOTCH_SEQSCOTCH)
int                         orderGraph          (Order * restrict const ordeptr, const SCOTCH_Graph * restrict const grafptr);
int                         orderGraphList      (Order * restrict const ordeptr, const SCOTCH_Graph * restrict const grafptr, const INT listnbr, const INT * restrict const listtab);
int                         orderGraphStrat     (Order * restrict const ordeptr, const SCOTCH_Graph * restrict const grafptr, const char * restrict const);
int                         orderGraphListStrat (Order * restrict const ordeptr, const SCOTCH_Graph * restrict const grafptr, const INT listnbr, const INT * restrict const listtab, const char * const);
#endif

#ifdef MESH_H
int                         orderMesh           (Order * restrict const ordeptr, const Mesh * restrict const meshptr);
int                         orderMeshList       (Order * restrict const ordeptr, const Mesh * restrict const meshptr, const INT listnbr, const INT * restrict const listtab);
int                         orderMeshStrat      (Order * restrict const ordeptr, const Mesh * restrict const meshptr, const char * const);
int                         orderMeshListStrat  (Order * restrict const ordeptr, const Mesh * restrict const meshptr, const INT listnbr, const INT * restrict const listtab, const char * const);
#endif /* MESH_H */

#undef static
