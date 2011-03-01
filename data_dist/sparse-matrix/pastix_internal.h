#ifndef _DAGUE_PASTIX_INTERNAL_H_
#define _DAGUE_PASTIX_INTERNAL_H_

/*************************************************************************
 *                  PaStiX structures and calls
 */

/* Ordering */
typedef struct Order_ {
    dague_int_t  cblknbr;              /*+ Number of column blocks             +*/
    dague_int_t *rangtab;              /*+ Column block range array [based,+1] +*/
    dague_int_t *permtab;              /*+ Permutation array [based]           +*/
    dague_int_t *peritab;              /*+ Inverse permutation array [based]   +*/
} Order;

int orderLoad (Order * const ordeptr, FILE * const stream);

/* Symbol Matrix */
/*+ The column block structure. +*/

typedef struct SymbolCblk_ {
  dague_int_t fcolnum;              /*+ First column index               +*/
  dague_int_t lcolnum;              /*+ Last column index (inclusive)    +*/
  dague_int_t bloknum;              /*+ First block in column (diagonal) +*/
} SymbolCblk;

/*+ The column block structure. +*/

typedef struct SymbolBlok_ {
  dague_int_t frownum;              /*+ First row index            +*/
  dague_int_t lrownum;              /*+ Last row index (inclusive) +*/
  dague_int_t cblknum;              /*+ Facing column block        +*/
  dague_int_t levfval;              /*+ Level-of-fill value        +*/
} SymbolBlok;

/*+ The symbolic block matrix. +*/

typedef struct SymbolMatrix_ {
  dague_int_t           baseval;              /*+ Base value for numberings         +*/
  dague_int_t           cblknbr;              /*+ Number of column blocks           +*/
  dague_int_t           bloknbr;              /*+ Number of blocks                  +*/
  SymbolCblk * restrict cblktab;              /*+ Array of column blocks [+1,based] +*/
  SymbolBlok * restrict bloktab;              /*+ Array of blocks [based]           +*/
  dague_int_t           nodenbr;              /*+ Number of nodes in matrix         +*/
} SymbolMatrix;

int  symbolInit(SymbolMatrix * const symbptr);
void symbolExit(SymbolMatrix * const symbptr);
void symbolBase(SymbolMatrix * const symbptr, const dague_int_t baseval);
int  symbolLoad(SymbolMatrix * const symbptr, FILE * const stream);

/*
 * Data structure for the internal csc of pastix
 */
/* Section: Macros */
/*
  Macro: CSC_FNBR

  Accessor to the number of column block.

  Parameters:
    a - Pointer to the CSC.

  Returns:
    Number of column block.
 */
#define CSC_FNBR(a)     (a)->cscfnbr /* cblk nbr */
/*
  Macro: CSC_FTAB

  Accessor to the array of column blocks.

  Parameters:
    a - Pointer to the CSC.

  Returns:
    Address of the array of column blocks.
*/
#define CSC_FTAB(a)     (a)->cscftab
/*
  Macro: CSC_COLNBR

  Accessor to the number of column in a block column.

  Parameters:
    a - Pointer to the CSC matrix.
    b - Column block index.

  Returns:
    The number of column in the block column
*/
#define CSC_COLNBR(a,b) (a)->cscftab[b].colnbr
/*
  Macro: CSC_COLTAB

  Accessor to the array of start for each column
  in the rows and values arrays.

  Parameters:
    a - Pointer to the CSC matrix.
    b - Column block index.

  Reurns:
    Address of the array of indexes of start for each column
    in the rows and values arrays.

*/
#define CSC_COLTAB(a,b) (a)->cscftab[b].coltab
/*
  Macro: CSC_COL

  Accessor to the index of first element of a column in rows
  and values.

  Parameters:
    a - Pointer to the CSC matrix.
    b - Column block index.
    c - Column index.
*/
#define CSC_COL(a,b,c)  (a)->cscftab[b].coltab[c]
/*
   Macro: CSC_ROWTAB

   Accessor to the array of rows.

   Parameters:
     a - Pointer to the CSC matrix.
*/
#define CSC_ROWTAB(a)   (a)->rowtab
/*
   Macro: CSC_ROW

   Accessor to a row in the CSC.

   Parameters:
     a - Pointer to the CSC matrix.
     b - Index of the row.
*/
#define CSC_ROW(a,b)    (a)->rowtab[b]
/*
   Macro: CSC_VALTAB

   Accessor to the array of values.

   Parameters:
     a - Pointer to the CSC matrix.
*/
#define CSC_VALTAB(a)   (a)->valtab
/*
   Macro: CSC_VAL

   Accessor to a value in the CSC.

   Parameters:
     a - Pointer to the CSC matrix.
     b - Index of the value.
*/
#define CSC_VAL(a,b)    (a)->valtab[b]
/*
   Macro: CSC_FROW

   Accessor to the first row of the column $c$ in
   the column block $b$.

   Parameters:
     a - Pointer to the CSC matrix.
     b - Column block index.
     c - Column index.
*/
#define CSC_FROW(a,b,c) (a)->rowtab[(a)->cscftab[b].coltab[c]]
/*
   Macro: CSC_FROW

   Accessor to the first value of the column $c$ in
   the column block $b$.

   Parameters:
     a - Pointer to the CSC matrix.
     b - Column block index.
     c - Column index.
*/
#define CSC_FVAL(a,b,c) (a)->valtab[(a)->cscftab[b].coltab[c]]
/*
  Macro: CSC_VALNBR

  Compute the Number of element on the matrix.

  Parameters:
    a - Pointer to the CSC matrix.
*/
#define CSC_VALNBR(a)   (a)->cscftab[(a)->cscfnbr\
				     -1].coltab[(a)->cscftab[(a)->cscfnbr \
							     -1].colnbr]

#endif /* _DAGUE_PASTIX_INTERNAL_H_ */
