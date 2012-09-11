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


double Z_CscNorm1(const CscMatrix *cscmtx,
                  MPI_Comm         comm);

void Z_CscRhsUpdown(const UpDownVector *updovct, 
                    const SymbolMatrix *symbmtx, 
                    Dague_Complex64_t  *rhs, 
                    const dague_int_t   ncol,
                    const dague_int_t  *invp,
                    const int           dof, 
                    const int           rhsmaking,
                    MPI_Comm            comm);

int Z_buildUpdoVect(pastix_data_t     *pastix_data,
                    dague_int_t       *loc2glob,
                    Dague_Complex64_t *b,
                    MPI_Comm           pastix_comm);

int Z_pastix_fillin_csc( pastix_data_t     *pastix_data,
                         MPI_Comm           pastix_comm,
                         dague_int_t        n,
                         dague_int_t       *colptr,
                         dague_int_t       *row,
                         Dague_Complex64_t *avals,
                         Dague_Complex64_t *b,
                         dague_int_t       *loc2glob);
  
void z_pastix(pastix_data_t **pastix_data, 
              MPI_Comm pastix_comm, 
              dague_int_t n, 
              dague_int_t *colptr, 
              dague_int_t *row, 
              Dague_Complex64_t *avals, 
              dague_int_t *perm, 
              dague_int_t *invp, 
              Dague_Complex64_t *b, 
              dague_int_t rhs, 
              dague_int_t *iparm, 
              DagDouble_t *dparm);

int z_pastix_checkMatrix(MPI_Comm pastix_comm, 
                         dague_int_t verb, 
                         dague_int_t flagsym, 
                         dague_int_t flagcor,
                         dague_int_t n, 
                         dague_int_t **colptr, 
                         dague_int_t **row, 
                         Dague_Complex64_t **avals, 
                         dague_int_t **loc2glob, 
                         dague_int_t dof);

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
    dague_int_t *iparm = dspctxt->iparm;
    DagDouble_t *dparm = dspctxt->dparm;
    pastix_data_t *pastix_data = NULL; /* Pointer to a storage structure needed by pastix */
    DagDouble_t criteria;

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

    iparm[IPARM_THREAD_NBR]    = dspctxt->coresnbr;
    iparm[IPARM_FACTORIZATION] = (dspctxt->factotype == DSPARSE_LDLTH) ? API_FACT_LDLT : dspctxt->factotype;
    iparm[IPARM_VERBOSE]       = dspctxt->verbose;
    iparm[IPARM_RHS_MAKING]    = API_RHS_B; /* RHS initialize to rhs[i] = i by read_matrix */
    iparm[IPARM_START_TASK]    = API_TASK_ORDERING;
    iparm[IPARM_END_TASK]      = API_TASK_ANALYSE;
    iparm[IPARM_ABS]           = 1;

    iparm[IPARM_SYM] = (MTX_ISSYM(dspctxt->type) ? API_SYM_YES : API_SYM_NO);
    iparm[IPARM_MATRIX_VERIFICATION] = API_NO;

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
    
    /* Part of pastix_task_sopalin required by dague */
    /* Initialize PaStiX sopar structure */
    pastix_data->sopar.itermax     = iparm[IPARM_ITERMAX];
    pastix_data->sopar.diagchange  = 0;
    pastix_data->sopar.epsilonraff = dparm[DPARM_EPSILON_REFINEMENT];
    pastix_data->sopar.rberror     = 0;
    pastix_data->sopar.espilondiag = dparm[DPARM_EPSILON_MAGN_CTRL];
    pastix_data->sopar.fakefact    = (iparm[IPARM_FILL_MATRIX] == API_YES) ? API_YES : API_NO;
    pastix_data->sopar.usenocsc    = 0;
    pastix_data->sopar.factotype   = iparm[IPARM_FACTORIZATION];
    pastix_data->sopar.symmetric   = iparm[IPARM_SYM];
    pastix_data->sopar.pastix_comm = pastix_data->pastix_comm;
    pastix_data->sopar.iparm       = iparm;
    pastix_data->sopar.dparm       = dparm;
    pastix_data->sopar.schur       = iparm[IPARM_SCHUR];
    pastix_data->sopar.n           = dspctxt->n;
    pastix_data->sopar.gN          = dspctxt->n;

    if (pastix_data->sopar.b != NULL)
      free(pastix_data->sopar.b);
    pastix_data->sopar.bindtab     = pastix_data->bindtab;

    pastix_data->sopar.type_comm   = iparm[IPARM_THREAD_COMM_MODE];
    pastix_data->sopar.nbthrdcomm  = iparm[IPARM_NB_THREAD_COMM];

    if (pastix_data->cscInternFilled == API_NO)
      {
        Z_pastix_fillin_csc(pastix_data, 
                            pastix_data->pastix_comm, 
                            dspctxt->n,
                            dspctxt->colptr, 
                            dspctxt->rows, 
                            dspctxt->values, 
                            dspctxt->rhs, 
                            NULL);
        pastix_data->cscInternFilled = API_YES;
      }

    // cblksize is used for the graph generated for BigDAT/Stream proposal
    //dspctxt->desc->cblksize = (size_t*)malloc( (pastix_data->solvmatr.symbmtx.cblknbr+1) * sizeof(size_t));

    pastix_data->solvmatr.coeftab = (Dague_Complex64_t **)malloc( pastix_data->solvmatr.symbmtx.cblknbr * sizeof(Dague_Complex64_t*));
    memset( pastix_data->solvmatr.coeftab, 0, pastix_data->solvmatr.symbmtx.cblknbr * sizeof(Dague_Complex64_t*) );

    if ( pastix_data->sopar.transcsc != NULL ) {
      pastix_data->solvmatr.ucoeftab = (Dague_Complex64_t **)malloc( pastix_data->solvmatr.symbmtx.cblknbr * sizeof(Dague_Complex64_t*));
      memset( pastix_data->solvmatr.ucoeftab, 0, pastix_data->solvmatr.symbmtx.cblknbr * sizeof(Dague_Complex64_t*) );
    }

    {
      SymbolMatrix *symbptr = &(pastix_data->solvmatr.symbmtx);
      dague_int_t icblk;
      /*      Dague_Complex64_t value = (Dague_Complex64_t)dspctxt->n * dspctxt->n;*/

      /* Allocate array of values in packed format  */
      //dspctxt->desc->cblksize[ 0 ] = 0;
      for (icblk=0; icblk < symbptr->cblknbr; icblk++)
      {
          dague_int_t fcolnum = pastix_data->solvmatr.symbmtx.cblktab[icblk].fcolnum;
          dague_int_t lcolnum = pastix_data->solvmatr.symbmtx.cblktab[icblk].lcolnum;
          dague_int_t stride  = pastix_data->solvmatr.cblktab[icblk].stride;
          dague_int_t width   = lcolnum - fcolnum + 1;
          dague_int_t size    = stride * width;

          /* Could be done in parallel */
          pastix_data->solvmatr.coeftab[icblk] = (void *) malloc (size * sizeof(Dague_Complex64_t));
          if ( pastix_data->sopar.transcsc != NULL ) {
              pastix_data->solvmatr.ucoeftab[icblk] = (void *) malloc (size * sizeof(Dague_Complex64_t));
          }
/*           pastix_data->solvmatr.coeftab[icblk] = (void *) malloc (1 * sizeof(Dague_Complex64_t)); */
/*           if ( pastix_data->sopar.transcsc != NULL ) { */
/*               pastix_data->solvmatr.ucoeftab[icblk] = (void *) malloc (1 * sizeof(Dague_Complex64_t)); */
/*           } */

/*           dspctxt->desc->cblksize[ icblk+1 ] = dspctxt->desc->cblksize[ icblk ] + size * sizeof(Dague_Complex64_t); */
      }
    }

    /* Tell PaStiX that the coeftab are allocated */
    pastix_data->malcof = 1;

    /* Compute criteria for static pivoting */
    criteria = pastix_data->sopar.espilondiag;
    if ( criteria < 0.0 )
    {
      /* Absolute criteria */
      criteria = -criteria;
    }
    else
      {
        if (pastix_data->sopar.usenocsc != 1)
          {
            if (pastix_data->sopar.fakefact == 1)
              {
                printf("WARNING: Fake factorization means absolute criteria: %e\n", criteria );
              }
            else
              {
                criteria = Z_CscNorm1(&(pastix_data->solvmatr.cscmtx), pastix_data->pastix_comm)
                  *        sqrt( criteria );
              }
          }
      }
    if (verbosemode > 3)
      printf(" - criteria = %g\n", criteria);  
    pastix_data->sopar.espilondiag = criteria;
    
    dspctxt->desc->pastix_data = pastix_data;
    dspctxt->rhsdesc->pastix_data = pastix_data;

    /*D_Ddump_all(&(pastix_data->solvmatr), DUMP_CSC);*/

    return dparm[DPARM_FACT_FLOPS];
}

void sparse_matrix_zcsc2cblk(const SolverMatrix *solvmatr,
                             Dague_Complex64_t  *transcsc, 
                             dague_int_t         itercblk)
{
    const CscMatrix *cscmtx;
    SolverBlok *solvbloktab;
    SymbolBlok *symbbloktab;
    Dague_Complex64_t *coeftab  = NULL;
    Dague_Complex64_t *ucoeftab = NULL;
    dague_int_t itercoltab;
    dague_int_t iterbloc;
    dague_int_t coefindx;
    dague_int_t iterval, stride;
    dague_int_t fcolnum, lcolnum, fbloknum, lbloknum;

    cscmtx      = &(solvmatr->cscmtx);
    solvbloktab = solvmatr->bloktab;
    symbbloktab = solvmatr->symbmtx.bloktab;

    if (itercblk < CSC_FNBR(cscmtx)){
        stride  = solvmatr->cblktab[itercblk].stride;
        fcolnum = solvmatr->symbmtx.cblktab[itercblk].fcolnum;
        lcolnum = solvmatr->symbmtx.cblktab[itercblk].lcolnum;
        fbloknum= solvmatr->symbmtx.cblktab[itercblk].bloknum;
        lbloknum= solvmatr->symbmtx.cblktab[itercblk+1].bloknum;

        coeftab  = (Dague_Complex64_t*)(solvmatr->coeftab[itercblk]);
        memset(coeftab, 0, stride * (lcolnum - fcolnum + 1) * sizeof(Dague_Complex64_t));
        if ( transcsc != NULL ){
          ucoeftab = (Dague_Complex64_t*)(solvmatr->ucoeftab[itercblk]);
          memset(ucoeftab, 0, stride * (lcolnum - fcolnum + 1) * sizeof(Dague_Complex64_t));
        }

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
                if ( CSC_ROW(cscmtx,iterval) >= fcolnum )
                {
                    iterbloc = fbloknum;
                    
                    /* in which block are we ? */
                    while ( (iterbloc < lbloknum) &&
                            (( symbbloktab[iterbloc].lrownum < CSC_ROW(cscmtx,iterval)) ||
                             ( symbbloktab[iterbloc].frownum > CSC_ROW(cscmtx,iterval)) ) )
                    {
                        iterbloc++;
                    }
                    
                    /* Let's check that we are still in the same cblk */
                    if ( iterbloc < lbloknum )
                    {
                        /* Starting point of the block */
                        coefindx  = solvbloktab[iterbloc].coefind;
                        /* Row of the value */
                        coefindx += CSC_ROW(cscmtx,iterval) - symbbloktab[iterbloc].frownum;
                        /* displacement for the column of the value */
                        coefindx += stride * itercoltab;
                        
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



void sparse_matrix_zcheck( sparse_context_t *dspctxt )
{
    dague_int_t *iparm = dspctxt->iparm;
    DagDouble_t *dparm = dspctxt->dparm;
    pastix_data_t *pastix_data = dspctxt->desc->pastix_data;

#if (defined DSPARSE_WITH_SOLVE)
    if (dspctxt->factotype == DSPARSE_LLT )
        iparm[IPARM_START_TASK] = API_TASK_REFINE;
    else
#endif
        iparm[IPARM_START_TASK] = API_TASK_SOLVE;
    iparm[IPARM_END_TASK]   = API_TASK_REFINE;

    /*******************************************/
    /*           Call pastix                   */
    /*******************************************/
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

    return;
}

void sparse_matrix_zclean( sparse_context_t *dspctxt )
{
    dague_int_t *iparm = dspctxt->iparm;
    DagDouble_t *dparm = dspctxt->dparm;
    pastix_data_t *pastix_data = dspctxt->desc->pastix_data;

    iparm[IPARM_START_TASK] = API_TASK_CLEAN;
    iparm[IPARM_END_TASK]   = API_TASK_CLEAN;

    /* Dsparse free the coeftab itself */
    {
        dague_int_t itercblk;
        dague_int_t cblknbr = pastix_data->solvmatr.symbmtx.cblknbr;

        for(itercblk=0; itercblk<cblknbr; itercblk++) {
            free( pastix_data->solvmatr.coeftab[itercblk] );
            pastix_data->solvmatr.coeftab[itercblk] = NULL;
        }
        free( pastix_data->solvmatr.coeftab );
        pastix_data->solvmatr.coeftab = NULL;

        if ( pastix_data->sopar.transcsc != NULL ) {
            for(itercblk=0; itercblk<cblknbr; itercblk++) {
                free( pastix_data->solvmatr.ucoeftab[itercblk] );
                pastix_data->solvmatr.ucoeftab[itercblk] = NULL;
            }
            free( pastix_data->solvmatr.ucoeftab );
            pastix_data->solvmatr.ucoeftab = NULL;
        }
    }
    pastix_data->malcof = 0;

    /*******************************************/
    /*           Call pastix                   */
    /*******************************************/
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

    free( dspctxt->colptr  );
    free( dspctxt->rows    );
    free( dspctxt->values  );
    free( dspctxt->permtab );
    free( dspctxt->peritab );
    free( dspctxt->rhs     );

    return;
}

void sparse_vector_zinit( sparse_context_t *dspctxt )
{
    pastix_data_t *pastix_data = NULL;

    pastix_data = dspctxt->desc->pastix_data;

    if (pastix_data->malsmx)
        {
            free(pastix_data->solvmatr.updovct.sm2xtab);
            pastix_data->malsmx = 0;
        }

    pastix_data->solvmatr.updovct.sm2xnbr = 1;
    pastix_data->solvmatr.updovct.sm2xtab = malloc( pastix_data->solvmatr.updovct.sm2xnbr *
                                                    pastix_data->solvmatr.updovct.sm2xsze *
                                                    dspctxt->rhsdesc->typesze );
    
    pastix_data->malsmx = 1;
    
    Z_buildUpdoVect(pastix_data,
                    NULL,
                    dspctxt->rhs,
                    0);    

    /* Save B for reffinment */
    pastix_data->sopar.b = malloc( pastix_data->solvmatr.updovct.sm2xnbr *
                                   pastix_data->solvmatr.updovct.sm2xsze *
                                   dspctxt->rhsdesc->typesze );
    memcpy(pastix_data->sopar.b, pastix_data->solvmatr.updovct.sm2xtab,
           pastix_data->solvmatr.updovct.sm2xsze*sizeof(FLOAT));

    return;
}

void sparse_vector_zfinalize( sparse_context_t *dspctxt )
{
    dague_int_t   *iparm = dspctxt->iparm;
    //    DagDouble_t   *dparm = dspctxt->dparm;
    pastix_data_t *pastix_data = NULL; 

    pastix_data = dspctxt->desc->pastix_data;

    Z_CscRhsUpdown(&(pastix_data->solvmatr.updovct),
                   &(pastix_data->solvmatr.symbmtx),
                   dspctxt->rhs, 
                   dspctxt->n, 
                   dspctxt->peritab,
                   iparm[IPARM_DOF_NBR],
                   iparm[IPARM_RHS_MAKING],
                   0);
    return;
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

void sparse_matrix_zcsc2pack(sparse_context_t *dspctxt, 
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

#endif
