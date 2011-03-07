/**
 *
 * @file zread.c
 *
 * @author Mathieu Faverge
 * @date 2011-03-01
 * @precisions normal z -> c d s
 *
 **/
#include "dague_config.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "data_dist/matrix/precision.h"
#include <pastix.h>
#include <read_matrix.h>
/*#include "dague_sparse.h"*/
#include "sparse-input.h"
#include "pastix_internal.h"

#define FOPEN(stream, filename, mode)					\
  {									\
    stream = NULL;							\
    if (NULL == (stream = fopen(filename, mode)))			\
      {									\
	fprintf(stderr, "%s: Couldn't open file : %s with mode %s\n",   \
		   __func__, filename, mode);                           \
	exit(0);                                                        \
      }									\
  }

/* Section: Structures */
typedef struct CscFormat_ {
    INT   colnbr;
    INT * coltab;
} CscFormat;

typedef struct CscMatrix_ {
  INT                cscfnbr;
  CscFormat         *cscftab;
  INT               *rowtab;
  Dague_Complex64_t *valtab;
} CscMatrix;

void dague_sparse_zcsc2pack(dsp_context_t     *dspctxt, 
                            const CscMatrix   *cscmtx, 
                            Dague_Complex64_t *transcsc);
void dague_sparse_zcsc2cblk(dsp_context_t     *dspctxt,
                            const CscMatrix   *cscmtx,   
                            Dague_Complex64_t *transcsc, 
                            dague_int_t        itercblk);

void Z_CscOrdistrib(CscMatrix          *thecsc,
                    char               *Type,
                    Dague_Complex64_t **transcsc,
                    const Order        *ord,
                    dague_int_t         Nrow,
                    dague_int_t         Ncol,
                    dague_int_t         Nnzero,
                    dague_int_t        *colptr,
                    dague_int_t        *rowind,
                    Dague_Complex64_t  *val,
                    dague_int_t         forcetrans,
                    const SymbolMatrix *symbmtx,
                    dague_int_t         procnum,
                    dague_int_t         dof);

int dague_sparse_zrdmtx( dsp_context_t *dspctxt )
{ 
    dague_sparse_input_symbol_matrix_t *symbptr;
    SymbolMatrix tmpsymbol;
    Order        tmporder;
    dague_int_t        forcetr  = 0;
    Dague_Complex64_t *transcsc = NULL;
    CscMatrix          cscmtx;
    dague_int_t cblknbr, bloknbr, cblknum, bloknum;
    FILE *stream;
    int verbosemode = 3;

    /*
     * Read the matrix to get the csc format 
     */
    z_read_matrix(dspctxt->matrixname, 
                  &(dspctxt->n), 
                  &(dspctxt->colptr), 
                  &(dspctxt->rows), 
                  (Dague_Complex64_t**)&(dspctxt->values), 
                  (Dague_Complex64_t**)&(dspctxt->rhs), 
                  &(dspctxt->type), 
                  &(dspctxt->rhstype), 
                  dspctxt->format, 
                  0);                  /* MPI communicator */

    /*
     *    Check Matrix format because matrix needs :
     *    - to be in fortran numbering
     *    - to have only the lower triangular part in symmetric case
     *    - to have a graph with a symmetric structure in unsymmetric case
     */
    z_pastix_checkMatrix(0,                              /* MPI communicator */
                         verbosemode, 
                         (MTX_ISSYM(dspctxt->type) ? API_SYM_YES : API_SYM_NO), 
                         API_YES,                        /* Fix the csc if there is problem */
                         dspctxt->n, 
                         &(dspctxt->colptr), 
                         &(dspctxt->rows), 
                         (Dague_Complex64_t**)&(dspctxt->values), 
                         NULL,                           /* Pointer for distributed case */
                         1);                             /* Number of degree of freedom  */

    /*
     * Load ordering
     */
    memset(&tmporder, 0, sizeof (Order));
    FOPEN( stream, dspctxt->ordername, "r" );
    orderLoad( &tmporder, stream );
    fclose(stream);

    /*
     * Load symbmtx
     */
    FOPEN( stream, dspctxt->symbname, "r" );
    symbolInit(&tmpsymbol);           /* Initialize structure */
    symbolLoad(&tmpsymbol, stream);   /* Load data from file  */
    symbolBase(&tmpsymbol, 0);        /* Base everything to 0 if needed */
    fclose(stream);

    /* 
     * Create the reordered csc with the permutation 
     * that will be used for the initialization of the coeftab
     */
    {
        if ( (dspctxt->type[1] == 'S') 
             && (dspctxt->factotype == SPARSE_LU) ) /* LU */
        {
            forcetr = 1;
        }

        Z_CscOrdistrib(&cscmtx, 
                       dspctxt->type,
		       &transcsc, 
                       &tmporder,
		       dspctxt->n, 
                       dspctxt->n, 
                       dspctxt->colptr[dspctxt->n]-1, 
                       dspctxt->colptr,
		       dspctxt->rows, 
                       dspctxt->values, 
                       forcetr,
		       &tmpsymbol, 
                       0,                    /* procnum */
                       1);                   /* dof     */
    }

    /* Clean ordering */
    free(tmporder.rangtab); /* We don't need rangtab */
    dspctxt->permtab = tmporder.permtab;
    dspctxt->peritab = tmporder.peritab;

    /* Convert to the local data structure */
    symbptr = (dague_sparse_input_symbol_matrix_t *) malloc( sizeof(dague_sparse_input_symbol_matrix_t) );

    symbptr->cblknbr = tmpsymbol.cblknbr;
    symbptr->bloknbr = tmpsymbol.bloknbr;
    symbptr->nodenbr = tmpsymbol.nodenbr;

    cblknbr = tmpsymbol.cblknbr;
    bloknbr = tmpsymbol.bloknbr;

    /* Convert the cblktab */
    if (((symbptr->cblktab = (dague_sparse_input_symbol_cblk_t *) malloc ((cblknbr+1) * sizeof(dague_sparse_input_symbol_cblk_t))) == NULL) ) {
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
    if (((symbptr->bloktab = (dague_sparse_input_symbol_blok_t *) malloc ((bloknbr) * sizeof(dague_sparse_input_symbol_blok_t))) == NULL) ) {
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

    /* Allocate and fill-in the coeftab */
    dague_sparse_zcsc2pack(dspctxt, &cscmtx, transcsc);

    return 0;
}

#if 0
void dague_sparse_zcsc_reorder( dsp_context_t *dspctxt )
{
    dague_int_t       *oldcolptr = dspctxt->colptr;
    dague_int_t       *oldrows   = dspctxt->rows;
    Dague_Complex64_t *oldvalues = (Dague_Complex64_t*)(dspctxt->values);
    dague_int_t       *newcolptr;
    dague_int_t       *newrows;
    Dague_Complex64_t *newvalues;

    dague_int_t i, j, tmp, newj;
    
    INT   index, itercol, newcol, iter, rowp1,colidx;
    INT   itercblk;
    INT   itercblk2;
    INT  *globcoltab = NULL;
    INT   strdcol    = 0;
    INT **trscltb    = NULL;
    INT  *trowtab    = NULL;
    INT  *cachetab   = NULL;
    INT   therow;
    INT   iterdofrow;
    INT   iterdofcol;
    INT   nodeidx;
    INT   colsize;
    /* To use common macro with CscdOrdistrib */
    INT  *g2l        = NULL;
    
    /* Global coltab */
    newcolptr = (dague_int_t *)malloc( (dspctxt->n + 1)*sizeof(dague_int_t) );
    memset( newcolptr, 0, (dspctxt->n + 1)*sizeof(dague_int_t) );
    
    /* Generate the permuted csc (symetric) */
    for (j=0; j< dspctxt->n; j++) {
        newj = dspctxt->permtab[j];
        newcolptr[newj] += oldcolptr[j+1] - oldcolptr[j];
        
        if (Type[1] == 'S') { /* If matrix is symmetric */
            for (i=oldcolptr[j]; i<oldcolptr[j+1]; i++) {
                if ((oldrows[i-1]-1) != j) {
                    newj = ord->permtab[oldrows[i-1]-1];
                    (newcolptr[newj])++;
                }
            }
        }
    }
    
    /* Sum values to generate indices in new perm */
    j=0;
    for (i=0; i<(dspctxt->n + 1); i++)
    {
            tmp = newcolptr[i];
            newcolptr[i] = j;
            j += tmp;
    }
  
    /* Allocate new rows and values */
    //CSC_ALLOC;

  /* AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA : j'en suis la **/


  /* Tab: contain the column block number or -1 if not local */
  MALLOC_INTERN(cachetab, (Ncol+1)*dof, INT);
  for (itercol=0; itercol<(Ncol+1)*dof; itercol++)
    cachetab[itercol] = -1;
  for (itercblk=0; itercblk<symbmtx->cblknbr; itercblk++)
    {
      for (itercol=symbmtx->cblktab[itercblk].fcolnum;
	   itercol<symbmtx->cblktab[itercblk].lcolnum+1;
	   itercol++)
	{
	  cachetab[itercol] = itercblk;
	}
    }

  /* Filling in thecsc with values and rows*/
  for (itercol=0; itercol<Ncol; itercol++)
    {
      itercblk = cachetab[ord->permtab[itercol]*dof];

      /* ok put the value */
      for (iter=colptr[itercol]; iter<colptr[itercol+1]; iter++)
	{
	  for (iterdofcol = 0; iterdofcol < dof; iterdofcol++)
	    {
	      for (iterdofrow = 0; iterdofrow < dof; iterdofrow++)
		{

		  rowp1 = rowind[iter-1]-1;
		  therow = ord->permtab[rowp1]*dof + iterdofrow;
		  newcol = ord->permtab[itercol]*dof+iterdofcol;

		  if (itercblk != -1)
		    {
		      SET_CSC_ROW_VAL(itercblk, therow, newcol, val);
		    }

		  itercblk2 = cachetab[therow];

		  if (itercblk2 != -1)
		    {
		      if (Type[1] == 'S')
			{
			  if (rowp1 != itercol)
			    {
			      /* newcol <-> therow */
			      SET_CSC_ROW_VAL(itercblk2, newcol, therow,
					      val);

			    }
			}
		      else
			{
			  if (transcsc != NULL)
			    {
			      SET_TRANS_ROW_VAL(itercblk2, therow, newcol,
						val);
			    }
			}
		    }
		}
	    }
	}
    }
  
  
  memFree_null(cachetab);

  /*
    memFree_null(colptr);
    memFree_null(rowind);
    memFree_null(val);
  */
  if (trscltb != NULL)
    {
      for (index=0; index<symbmtx->cblknbr; index++)
	{
	  memFree_null(trscltb[index]);
	}
      memFree_null(trscltb);
    }

  /* 2nd membre */
  /* restore good coltab */
  colidx = 0;
  for (index=0; index<symbmtx->cblknbr; index++)
    {
      for(iter=0;iter<(CSC_COLNBR(thecsc,index)+1); iter++)
	{
	  newcol = CSC_COL(thecsc,index,iter);
	  CSC_COL(thecsc,index,iter) = colidx;
	  colidx = newcol;
	}
    }
  CSC_SORT;
}
#endif

void dague_sparse_zcsc2pack(dsp_context_t     *dspctxt, 
                            const CscMatrix   *cscmtx, 
                            Dague_Complex64_t *transcsc)
{   
    dague_sparse_input_symbol_matrix_t *symbptr = dspctxt->symbmtx;
    dague_int_t icblk, coefnbr;

    /* Allocate array of values in packed format  */
    for (icblk=0; icblk < symbptr->cblknbr; icblk++)
    {
        coefnbr  = symbptr->cblktab[icblk].lcolnum - symbptr->cblktab[icblk].fcolnum +1;
        coefnbr *= symbptr->cblktab[icblk].stride;
        
        /* Could be done in parallel */
      /* What about LU ? see with Anthony for the symmetric structure */
        symbptr->cblktab[icblk].cblkptr = (void *) malloc (coefnbr * sizeof(Dague_Complex64_t));
        memset( symbptr->cblktab[icblk].cblkptr, 0, coefnbr * sizeof(Dague_Complex64_t));
        dague_sparse_zcsc2cblk(dspctxt, cscmtx, transcsc, icblk);
    }
}

void dague_sparse_zcsc2cblk(dsp_context_t     *dspctxt,
                            const CscMatrix   *cscmtx, 
                            Dague_Complex64_t *transcsc, 
                            dague_int_t        itercblk)
{
    dague_sparse_input_symbol_cblk_t * cblktab;
    dague_sparse_input_symbol_blok_t * bloktab;
    Dague_Complex64_t *coeftab;
    dague_int_t itercoltab;
    dague_int_t iterbloc;
    dague_int_t coefindx;
    dague_int_t iterval;
    
    cblktab = dspctxt->symbmtx->cblktab;
    bloktab = dspctxt->symbmtx->bloktab;
    
    if (itercblk < CSC_FNBR(cscmtx)){
        coeftab = (Dague_Complex64_t*)(dspctxt->symbmtx->cblktab[itercblk].cblkptr);

        for (itercoltab=0;
             itercoltab < CSC_COLNBR(cscmtx,itercblk);
             itercoltab++)
        {
            for (iterval = CSC_COL(cscmtx,itercblk,itercoltab);
                 iterval < CSC_COL(cscmtx,itercblk,itercoltab+1);
                 iterval++)
            {
                /* We skip upper part of the csc, 
                 * the csc has normally at this point a symmetric structure,
                 * so there is no need to go through it twice
                 */
                if (CSC_ROW(cscmtx,iterval) >= cblktab[itercblk].fcolnum)
                {
                    iterbloc = cblktab[itercblk].bloknum;
                    
                    /* in which block are we ? */
                    while ( (iterbloc < cblktab[itercblk+1].bloknum) &&
                            (( bloktab[iterbloc].lrownum < CSC_ROW(cscmtx,iterval)) ||
                             ( bloktab[iterbloc].frownum > CSC_ROW(cscmtx,iterval)) ) )
                    {
                        iterbloc++;
                    }
                    
                    /* Let's check that we are still in the same cblk */
                    if ( iterbloc < cblktab[itercblk+1].bloknum )
                    {
                        /* Starting point of the block */
                        coefindx  = bloktab[iterbloc].coefind;
                        /* Row of the value */
                        coefindx += CSC_ROW(cscmtx,iterval) - bloktab[iterbloc].frownum;
                        /* displacement for the column of the value */
                        coefindx += cblktab[itercblk].stride * itercoltab;
                        
                        coeftab[coefindx] = CSC_VAL(cscmtx,iterval);
                        if (transcsc != NULL) 
                        {
                        /*     SOLV_UCOEFTAB(itercblk)[coefindx] = trandcsc[iterval]; */
                        }
                    }
                    else {
                        fprintf(stderr, "One coefficient is out of the structure\n" );
                    }
                }
            }
        }
    }
}
