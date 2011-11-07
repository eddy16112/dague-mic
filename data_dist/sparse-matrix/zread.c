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

#if defined(PRECISION_z) || defined(PRECISION_c)
#define FORCE_COMPLEX
#define TYPE_COMPLEX
#endif

#if defined(PRECISION_z) || defined(PRECISION_d)
#define FORCE_DOUBLE 
#define PREC_DOUBLE
#endif

#include "pastix_internal.h"
#include <read_matrix.h>
#include "sparse-matrix.h"

#if 0
#define DAGUE_FOPEN(stream, filename, mode)					\
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
#endif

void sparse_matrix_zcsc2pack(sparse_context_t  *dspctxt, 
                            const CscMatrix   *cscmtx, 
                            Dague_Complex64_t *transcsc);
void sparse_matrix_zcsc2cblk(sparse_context_t  *dspctxt,
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


DagDouble_t sparse_matrix_zrdmtx( sparse_context_t *dspctxt )
{
    int verbosemode = 3;
    dague_int_t iparm[IPARM_SIZE];
    DagDouble_t dparm[DPARM_SIZE];
    pastix_data_t *pastix_data = NULL; /* Pointer to a storage structure needed by pastix */

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
                  (driver_type_t)(dspctxt->format), 
                  0);                  /* MPI communicator */

    dspctxt->nnz = dspctxt->colptr[dspctxt->n]-1;

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

    /*******************************************/
    /* Initialize parameters to default values */
    /*******************************************/
    iparm[IPARM_MODIFY_PARAMETER] = API_NO;
    z_pastix(&pastix_data, MPI_COMM_WORLD, 
	     dspctxt->n, 
	     dspctxt->colptr, 
	     dspctxt->rows, 
	     dspctxt->values,
	     NULL,
	     NULL,
	     dspctxt->rhs,
	     1, 
	     iparm, 
	     dparm);

    /*******************************************/
    /*       Customize some parameters         */
    /*******************************************/

    iparm[IPARM_THREAD_NBR] = 1; /* WARNING : update with nbr thread for BLEND splitting !!! */
    iparm[IPARM_SYM] = (MTX_ISSYM(dspctxt->type) ? API_SYM_YES : API_SYM_NO);
    iparm[IPARM_FACTORIZATION] = dspctxt->factotype;
    iparm[IPARM_MATRIX_VERIFICATION] = API_NO;
    iparm[IPARM_VERBOSE]             = 4;         /* UPDATE !!! */
    iparm[IPARM_RHS_MAKING]          = API_RHS_1; /* UPDATE !!! */
    iparm[IPARM_START_TASK]          = API_TASK_ORDERING;
    iparm[IPARM_END_TASK]            = API_TASK_ANALYSE;

    /*******************************************/
    /*           Call pastix                   */
    /*******************************************/
    dspctxt->permtab = malloc(dspctxt->n*sizeof(dague_int_t));
    dspctxt->peritab = malloc(dspctxt->n*sizeof(dague_int_t));

    z_pastix(&pastix_data, MPI_COMM_WORLD, 
	     dspctxt->n, 
	     dspctxt->colptr, 
	     dspctxt->rows, 
	     dspctxt->values,
	     dspctxt->permtab,
	     dspctxt->peritab,
	     dspctxt->rhs,
	     1, 
	     iparm, 
	     dparm);

    pastix_data->solvmatr.coeftab = (Dague_Complex64_t **)malloc( pastix_data->solvmatr.symbmtx.cblknbr * sizeof(Dague_Complex64_t*));
    memset( pastix_data->solvmatr.coeftab, 0, pastix_data->solvmatr.symbmtx.cblknbr * sizeof(Dague_Complex64_t*) );


    {
      SymbolMatrix *symbptr = &(pastix_data->solvmatr.symbmtx);
      dague_int_t icblk;
    
      /* Allocate array of values in packed format  */
      for (icblk=0; icblk < symbptr->cblknbr; icblk++)
        {
          /* Could be done in parallel */
          pastix_data->solvmatr.coeftab[icblk] = (void *) malloc (1 * sizeof(Dague_Complex64_t));
          memset( pastix_data->solvmatr.coeftab[icblk], 0, 1 * sizeof(Dague_Complex64_t));
        }
    }

    dspctxt->desc->pastix_data = pastix_data;
    
    return dparm[DPARM_FACT_FLOPS];
}

#if 0
int sparse_matrix_zrdmtx( sparse_context_t *dspctxt )
{ 
    dague_symbol_matrix_t *symbptr;
    SymbolMatrix       tmpsymbol;
    Order              tmporder;
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
                  (driver_type_t)(dspctxt->format), 
                  0);                  /* MPI communicator */

    dspctxt->nnz = dspctxt->colptr[dspctxt->n]-1;

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
    DAGUE_FOPEN( stream, dspctxt->ordername, "r" );
    orderLoad( &tmporder, stream );
    fclose(stream);

    /*
     * Load symbmtx
     */
    DAGUE_FOPEN( stream, dspctxt->symbname, "r" );
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
             && (dspctxt->factotype == DSPARSE_LU) ) /* LU */
        {
            forcetr = 1;
        }

        Z_CscOrdistrib(&cscmtx, 
                       dspctxt->type,
		       &transcsc, 
                       &tmporder,
		       dspctxt->n, 
                       dspctxt->n, 
                       dspctxt->nnz, 
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
    symbptr = &(dspctxt->desc->symbmtx);

    symbptr->cblknbr = tmpsymbol.cblknbr;
    symbptr->bloknbr = tmpsymbol.bloknbr;
    symbptr->nodenbr = tmpsymbol.nodenbr;

    cblknbr = tmpsymbol.cblknbr;
    bloknbr = tmpsymbol.bloknbr;

    /* Convert the cblktab */
    if (((symbptr->cblktab = (dague_symbol_cblk_t *) malloc ((cblknbr+1) * sizeof(dague_symbol_cblk_t))) == NULL) ) {
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
    if (((symbptr->bloktab = (dague_symbol_blok_t *) malloc ((bloknbr) * sizeof(dague_symbol_blok_t))) == NULL) ) {
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

        symbptr->bloktab[bloknum].coefind = symbptr->cblktab[cblknum].stride;        

        symbptr->cblktab[cblknum].stride += symbptr->bloktab[bloknum].lrownum 
            - symbptr->bloktab[bloknum].frownum + 1;
    }
    free(tmpsymbol.bloktab);

    /* Allocate and fill-in the coeftab */
    sparse_matrix_zcsc2pack(dspctxt, &cscmtx, transcsc);

    return 0;
}

void sparse_matrix_zcsc2pack(sparse_context_t     *dspctxt, 
                            const CscMatrix   *cscmtx, 
                            Dague_Complex64_t *transcsc)
{   
    dague_symbol_matrix_t *symbptr = &(dspctxt->desc->symbmtx);
    dague_int_t icblk, coefnbr;
    
    /* Allocate array of values in packed format  */
    for (icblk=0; icblk < symbptr->cblknbr; icblk++)
    {
        coefnbr  = symbptr->cblktab[icblk].lcolnum - symbptr->cblktab[icblk].fcolnum +1;
        coefnbr *= symbptr->cblktab[icblk].stride;
        
        /* Could be done in parallel */
        symbptr->cblktab[icblk].cblkptr = (void *) malloc (coefnbr * sizeof(Dague_Complex64_t));
        memset( symbptr->cblktab[icblk].cblkptr, 0, coefnbr * sizeof(Dague_Complex64_t));

        if (transcsc != NULL) 
        {
            symbptr->cblktab[icblk].ucblkptr = (void *) malloc (coefnbr * sizeof(Dague_Complex64_t));
            memset( symbptr->cblktab[icblk].ucblkptr, 0, coefnbr * sizeof(Dague_Complex64_t));
        }

        sparse_matrix_zcsc2cblk(dspctxt, cscmtx, transcsc, icblk);
    }
}

void sparse_matrix_zcsc2cblk(sparse_context_t  *dspctxt,
                            const CscMatrix   *cscmtx, 
                            Dague_Complex64_t *transcsc, 
                            dague_int_t        itercblk)
{
    dague_symbol_cblk_t * cblktab;
    dague_symbol_blok_t * bloktab;
    Dague_Complex64_t *coeftab;
    Dague_Complex64_t *ucoeftab;
    dague_int_t itercoltab;
    dague_int_t iterbloc;
    dague_int_t coefindx;
    dague_int_t iterval;
    
    cblktab = dspctxt->desc->symbmtx.cblktab;
    bloktab = dspctxt->desc->symbmtx.bloktab;
    
    if (itercblk < CSC_FNBR(cscmtx)){
        coeftab  = (Dague_Complex64_t*)(dspctxt->desc->symbmtx.cblktab[itercblk].cblkptr);
        ucoeftab = (Dague_Complex64_t*)(dspctxt->desc->symbmtx.cblktab[itercblk].ucblkptr);

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
                          ucoeftab[coefindx] = transcsc[iterval];
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
#endif
