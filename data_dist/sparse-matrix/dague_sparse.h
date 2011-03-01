typedef int64_t dague_int_t;

typedef struct dsp_context_s {
    int   format;     /* Matrix file format                         */
    char *matrixname; /* Filename to get the matrix                 */
    char *ordername;  /* Filename where the ordering is stored      */
    char *symbname;   /* Filename where the symbol matrix is stored */
    char *type;       /* Type of the matrix                         */
    char *rhstype;    /* Type of the RHS                            */
    dague_int_t n;
    dague_int_t nnz;
    dague_int_t *colptr;
    dague_int_t *rows;
    void        *values;
    void        *rhs;
    dague_int_t *permtab;
    dague_int_t *peritab;
    dague_sparse_input_symbol_matrix_t *symbmtx;
} dsp_context_t;


/*************************************************************************
 *                  PaStiX structures and calls
 */

/* Ordering */
typedef struct Order_ {
  INT                       cblknbr;              /*+ Number of column blocks             +*/
  INT *                     rangtab;              /*+ Column block range array [based,+1] +*/
  INT *                     permtab;              /*+ Permutation array [based]           +*/
  INT *                     peritab;              /*+ Inverse permutation array [based]   +*/
} Order;

int orderLoad (Order * const ordeptr, FILE * const stream);

/* Symbol Matrix */
/*+ The column block structure. +*/

typedef struct SymbolCblk_ {
  INT                       fcolnum;              /*+ First column index               +*/
  INT                       lcolnum;              /*+ Last column index (inclusive)    +*/
  INT                       bloknum;              /*+ First block in column (diagonal) +*/
} SymbolCblk;

/*+ The column block structure. +*/

typedef struct SymbolBlok_ {
  INT                       frownum;              /*+ First row index            +*/
  INT                       lrownum;              /*+ Last row index (inclusive) +*/
  INT                       cblknum;              /*+ Facing column block        +*/
  INT                       levfval;              /*+ Level-of-fill value        +*/
} SymbolBlok;

/*+ The symbolic block matrix. +*/

typedef struct SymbolMatrix_ {
  INT                       baseval;              /*+ Base value for numberings         +*/
  INT                       cblknbr;              /*+ Number of column blocks           +*/
  INT                       bloknbr;              /*+ Number of blocks                  +*/
  SymbolCblk * restrict     cblktab;              /*+ Array of column blocks [+1,based] +*/
  SymbolBlok * restrict     bloktab;              /*+ Array of blocks [based]           +*/
  INT                       nodenbr;              /*+ Number of nodes in matrix         +*/
} SymbolMatrix;

int  symbolInit(SymbolMatrix * const symbptr);
void symbolExit(SymbolMatrix * const symbptr);
void symbolBase(SymbolMatrix * const symbptr, const INT baseval);
int  symbolLoad(SymbolMatrix * const symbptr, FILE * const stream);

