#ifndef _PASTIX_STRUCTS_H_
#define _PASTIX_STRUCTS_H_

/*+ The column block structure. +*/

typedef struct SymbolCblk_ {
    int    fcolnum;              /*+ First column index               +*/
    int    lcolnum;              /*+ Last column index (inclusive)    +*/
    int    bloknum;              /*+ First block in column (diagonal) +*/
    void * cblkptr;
    int    stride;
} SymbolCblk;

/*+ The column block structure. +*/

typedef struct SymbolBlok_ {
    int frownum;              /*+ First row index            +*/
    int lrownum;              /*+ Last row index (inclusive) +*/
    int coefind;
} SymbolBlok;

/*+ The symbolic block matrix. +*/

typedef struct SymbolMatrix_ {
    int                       baseval;              /*+ Base value for numberings         +*/
    int                       cblknbr;              /*+ Number of column blocks           +*/
    int                       bloknbr;              /*+ Number of blocks                  +*/
    SymbolCblk * restrict     cblktab;              /*+ Array of column blocks [+1,based] +*/
    SymbolBlok * restrict     bloktab;              /*+ Array of blocks [based]           +*/
    int                       nodenbr;              /*+ Number of nodes in matrix         +*/
} SymbolMatrix;

#endif
