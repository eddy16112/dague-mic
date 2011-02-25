#include <pastix.h>
#include <read_matrix.h>
#include "dague_sparse.h"

#define FOPEN(stream, filename, mode)					\
  {									\
    stream = NULL;							\
    if (NULL == (stream = fopen(filename, mode)))			\
      {									\
	errorPrint("%s: Couldn't open file : %s with mode %s\n",	\
		   __func__, filename, mode);                           \
	exit(0);                                                        \
      }									\
  }


int dague_sparse_zrdmtx( dsp_context_t *dspctxt )
{ 
    dague_sparse_input_symbol_matrix_t *symbptr;
    SymbolMatrix tmpsymbol;
    Order        tmporder;
    int verbosemode = 3;
    FILE *stream;

    /*
     * Read the matrix to get the csc format 
     */
    z_read_matrix(dspctxt->matrixname, 
                  &(dspctxt->n), 
                  &(dspctxt->colptr), 
                  &(dspctxt->rows), 
                  &((Dague_Complex64_t*)(dspctxt->values)), 
                  &((Dague_Complex64_t*)(dspctxt->rhs)), 
                  &(dspctxt->type), 
                  &(dspctxt->rhstype), 
                  dspctxt->format, 
                  NULL);                  /* MPI communicator */

    /*
     *    Check Matrix format because matrix needs :
     *    - to be in fortran numbering
     *    - to have only the lower triangular part in symmetric case
     *    - to have a graph with a symmetric structure in unsymmetric case
     */
    z_pastix_checkMatrix(NULL,                           /* MPI communicator */
                         verbosemode, 
                         (MTX_ISSYM(dspctxt->type) ? API_SYM_YES : API_SYM_NO), 
                         API_YES,                        /* Fix the csc if there is problem */
                         dspctxt->ncol, 
                         &(dspctxt->colptr), 
                         &(dspctxt->rows), 
                         &(dspctxt->values), 
                         NULL,                           /* Pointer for distributed case */
                         1);                             /* Number of degree of freedom  */
    
    /*
     * Load ordering
     */
    memSet (ordeptr, 0, sizeof (Order));
    FOPEN( stream, dspctxt->ordername, "r" );
    orderLoad( &tmporder, stream );
    fclose(stream);
    free(tmporder.rangtab); /* We don't need rangtab */
    dspctxt->permtab = tmporder.permtab;
    dspctxt->peritab = tmporder.peritab;

    /*
     * Load symbmtx
     */
    FOPEN( stream, dspctxt->symbname, "r" );
    symbolInit(&tmpsymbol);           /* Initialize structure */
    symbolLoad(&tmpsymbol, stream);   /* Load data from file  */
    symbolBase(&tmpsymbol, 0);        /* Base everything to 0 if needed */
    fclose(stream);

    /* Convert to the local data structure */
    symbptr = (dague_sparse_input_symbol_matrix_t *) malloc( sizeof(dague_sparse_input_symbol_matrix_t) );

    symbptr->baseval = tmpsymbol.baseval;
    symbptr->cblknbr = tmpsymbol.cblknbr;
    symbptr->bloknbr = tmpsymbol.bloknbr;
    symbptr->nodenbr = tmpsymbol.nodenbr;

    cblknbr = tmpsymbol.cblknbr;
    bloknbr = tmpsymbol.bloknbr;

    /* Convert the cblktab */
    if (((symbptr->cblktab = (dague_sparse_input_symbol_cblk_t *) memAlloc ((cblknbr+1) * sizeof(dague_sparse_input_symbol_cblk_t))) == NULL) ) {
        fprintf(stderr, "%s: malloc failed for cblktab\n", __func__ );
        exit(0);
    }
    
    for (cblknum = 0; cblknum < cblknbr; cblknum ++) {
        symbptr->cblktab[cblknum].fcolnum = tmpsymbol.cblktab[cblknum].fcolnum;
        symbptr->cblktab[cblknum].lcolnum = tmpsymbol.cblktab[cblknum].lcolnum;
        symbptr->cblktab[cblknum].bloknum = tmpsymbol.cblktab[cblknum].bloknum;
        symbptr->cblktab[cblknum].cblkptr = NULL;
        symbptr->cblktab[cblknum].stride  = 0;
    }
    /* Set last column block */
    symbptr->cblktab[cblknbr].fcolnum = tmpsymbol.nodenbr;
    symbptr->cblktab[cblknbr].lcolnum = tmpsymbol.nodenbr;
    symbptr->cblktab[cblknbr].bloknum = bloknbr;
    symbptr->cblktab[cblknbr].cblkptr = NULL;
    symbptr->cblktab[cblknbr].stride  = 0;
    
    /* Start to free memory to avoid to have both full structure at the same time */
    free(tmpsymbol.cblktab);

    /* Convert the bloktab */
    if (((symbptr->bloktab = (dague_sparse_input_symbol_blok_t *) memAlloc ((bloknbr) * sizeof(dague_sparse_input_symbol_blok_t))) == NULL) ) {
        fprintf(stderr, "%s: malloc failed for bloktab\n", __func__ );
        exit(0);
    }

    cblknum = 0;
    for (bloknum = 0; bloknum < bloknbr; bloknum ++) {
        symbptr->bloktab[bloknum].frownum = tmpsymbol.bloktab[bloknum].frownum;
        symbptr->bloktab[bloknum].lrownum = tmpsymbol.bloktab[bloknum].lrownum;
        symbptr->bloktab[bloknum].coefind = 0;        
        
        while ( ! (bloknum < symbptr->cblktab[cblknum+1].bloknum) )
            cblknum++;
        
        symbptr->cblktab[cblknum].stride += symbptr->bloktab[bloknum].lrownum 
            - symbptr->bloktab[bloknum].frownum + 1;
    }
    free(tmpsymbol.bloktab);

    dspctxt->symbmtx = symbptr;

    return 0;
}
