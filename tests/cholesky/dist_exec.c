/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */


#ifdef USE_MPI
#include "mpi.h"
#endif  /* defined(USE_MPI) */

#include <getopt.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <cblas.h>
#include <math.h>
#include "plasma.h"
#include <../src/common.h>
#include <../src/lapack.h>
#include <../src/context.h>
#include <../src/allocate.h>
#include <sys/time.h>

#include "dplasma.h"
#include "scheduling.h"
#include "profiling.h"
#include "data_management.h"

//#ifdef VTRACE
//#include "vt_user.h"
//#endif

static void runtime_init(int argc, char **argv);
static void runtime_fini(void);

static dplasma_context_t *setup_dplasma(int* pargc, char** pargv[]);
static void cleanup_dplasma(dplasma_context_t* context);
static void warmup_dplasma(dplasma_context_t* dplasma);

static void create_matrix(int N, PLASMA_enum* uplo, double** pA1, double** pA2, 
                          double** pB1, double** pB2, double** pWORK, double** pD, 
                          int LDA, int NRHS, int LDB, PLASMA_desc* local);
static void scatter_matrix(PLASMA_desc* local, DPLASMA_desc* dist);
static void gather_matrix(PLASMA_desc* local, DPLASMA_desc* dist);
static void check_matrix(int N, PLASMA_enum* uplo, double* A1, double* A2, 
                         double* B1, double* B2, double* WORK, double* D, 
                         int LDA, int NRHS, int LDB, PLASMA_desc* local, 
                         double gflops);

static int check_factorization(int, double*, double*, int, int , double);
static int check_solution(int, int, double*, int, double*, double*, int, double);


/* timing profiling etc */
double time_elapsed;

static inline double get_cur_time(){
    double t;
    struct timeval tv;
    gettimeofday(&tv,NULL);
    t=tv.tv_sec+tv.tv_usec/1e6;
    return t;
}
#define TIME_START() do { time_elapsed = get_cur_time(); } while(0)
#define TIME_STOP() do { time_elapsed = get_cur_time() - time_elapsed; } while(0)
#define TIME_PRINT(print) do { \
  time_elapsed = get_cur_time() - time_elapsed; \
  printf("[%d] TIMED %f doing\t", rank, time_elapsed); \
  printf print; \
} while(0)


/* overload exit in MPI mode */
#ifdef USE_MPI
#   define exit(ret) MPI_Abort(MPI_COMM_WORLD, ret)
#endif

typedef enum {
    DO_PLASMA,
    DO_DPLASMA
} backend_argv_t;

/* globals and argv set values */
PLASMA_desc descA;
DPLASMA_desc ddescA;
int do_warmup = 0;
int do_nasty_validations = 0;
int cores = 1;
int nodes = 1;
#define N (ddescA.n)
#define NB (ddescA.nb)
#define rank (ddescA.mpi_rank)
int LDA = 0;
int NRHS = 1;
int LDB = 0;
PLASMA_enum uplo = PlasmaLower;
backend_argv_t backend = DO_DPLASMA;

int main(int argc, char ** argv)
{
    double flops, gflops;
    double *A1;
    double *A2;
    double *B1;
    double *B2;
    double *WORK;
    double *D;
    dplasma_context_t* dplasma;

    //#ifdef VTRACE
      // VT_OFF();
    //#endif

    runtime_init(argc, argv);

    if(0 == rank)
        create_matrix(N, &uplo, &A1, &A2, &B1, &B2, &WORK, &D, LDA, NRHS, LDB, &descA);

    switch(backend)
    {
        case DO_PLASMA: {
            plasma_context_t* plasma = plasma_context_self();

            if(do_warmup)
            {
                TIME_START();
                plasma_parallel_call_2(plasma_pdpotrf, 
                                       PLASMA_enum, uplo, 
                                       PLASMA_desc, descA);
                TIME_PRINT(("_plasma warmup:\t\t%d %d %f Gflops\n", N, PLASMA_NB,
                            (N/1e3*N/1e3*N/1e3/3.0)/(time_elapsed)));
            }
            TIME_START();
            plasma_parallel_call_2(plasma_pdpotrf,
                                   PLASMA_enum, uplo,
                                   PLASMA_desc, descA);
            TIME_PRINT(("_plasma computation:\t%d %d %f Gflops\n", N, PLASMA_NB, 
                        gflops = flops = (N/1e3*N/1e3*N/1e3/3.0)/(time_elapsed)));
            break;
        }
        case DO_DPLASMA: {
            scatter_matrix(&descA, &ddescA);

            //#ifdef VTRACE 
            //    VT_ON();
            //#endif
    
            /*** THIS IS THE DPLASMA COMPUTATION ***/
            TIME_START();
            dplasma = setup_dplasma(&argc, &argv);
            if(0 == rank)
            {
                dplasma_execution_context_t exec_context;

                /* I know what I'm doing ;) */
                exec_context.function = (dplasma_t*)dplasma_find("POTRF");
                dplasma_set_initial_execution_context(&exec_context);
                dplasma_schedule(dplasma, &exec_context);
            }
            TIME_PRINT(("Dplasma initialization:\t%d %d\n", N, NB));

            if(do_warmup)
                warmup_dplasma(dplasma);
    
            /* lets rock! */
            TIME_START();
            dplasma_progress(dplasma);
            TIME_PRINT(("Dplasma computation:\t%d %d %f Gflops\n", N, NB, 
                        gflops = flops = (N/1e3*N/1e3*N/1e3/3.0)/(time_elapsed * nodes)));

            cleanup_dplasma(dplasma);
            /*** END OF DPLASMA COMPUTATION ***/

#ifdef USE_MPI
            MPI_Reduce(&flops, &gflops, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#endif
            gather_matrix(&descA, &ddescA);
            break;
        }
    }

    if(0 == rank)
        check_matrix(N, &uplo, A1, A2, B1, B2, WORK, D, LDA, NRHS, LDB, &descA, gflops);

    runtime_fini();
    return 0;
}

static void print_usage(void)
{
    fprintf(stderr,
            "Mandatory argument:\n"
            "   number           : the size of the matrix\n"
            "Optional arguments:\n"
            "   -c --nb-cores    : number of computing threads to use\n"
            "   -d --dplasma     : use DPLASMA backend (default)\n"
            "   -p --plasma      : use PLASMA backend\n"
            "   -g --grid-rows   : number of processes row in the process grid (must divide the total number of processes (default: 1)\n"
            "   -s --stile-size  : number of tile per row (col) in a super tile (default: 1)\n"
            "   -a --lda         : leading dimension of the matrix A (equal matrix size by default)\n"
            "   -b --ldb         : leading dimension of the RHS B (equal matrix size by default)\n"
            "   -r --nrhs        : Number of Right Hand Side (default: 1)\n"
            "   -x --xcheck      : do extra nasty result validations\n"
            "   -w --warmup      : do some warmup, if > 1 also preload cache\n");
}

static void runtime_init(int argc, char **argv)
{
    struct option long_options[] =
    {
        {"nb-cores",    required_argument,  0, 'c'},
        {"matrix-size", required_argument,  0, 'n'},
        {"lda",         required_argument,  0, 'a'},
        {"nrhs",        required_argument,  0, 'r'},
        {"ldb",         required_argument,  0, 'b'},
        {"grid-rows",   required_argument,  0, 'g'},
        {"stile-size",  required_argument,  0, 's'},
        {"xcheck",      no_argument,        0, 'x'},
        {"warmup",      optional_argument,  0, 'w'},
        {"dplasma",     no_argument,        0, 'd'},
        {"plasma",      no_argument,        0, 'p'},
        {"help",        no_argument,        0, 'h'},
        {0, 0, 0, 0}
    };

#ifdef USE_MPI
    /* mpi init */
    MPI_Init(&argc, &argv);
    
    MPI_Comm_size(MPI_COMM_WORLD, &nodes); 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
#else
    nodes = 1;
    rank = 0;
#endif
    
    /* parse arguments */
    ddescA.GRIDrows = 1;
    ddescA.nrst = ddescA.ncst = 1;
    do
    {
        int c;
        int option_index = 0;
        
        c = getopt_long (argc, argv, "dpxc:n:a:r:b:g:s:w::h",
                         long_options, &option_index);
        
        /* Detect the end of the options. */
        if (c == -1)
            break;
        
        switch (c)
        {
            case 'p': 
                backend = DO_PLASMA;
                break; 
            case 'd':
                backend = DO_DPLASMA;
                break;

            case 'c':
                cores = atoi(optarg);
                if(cores<= 0)
                    cores=1;
                ddescA.cores = cores;
                //printf("Number of cores (computing threads) set to %d\n", cores);
                break;

            case 'n':
                N = atoi(optarg);
                //printf("matrix size set to %d\n", N);
                break;

            case 'g':
                ddescA.GRIDrows = atoi(optarg);
                break;
            case 's':
                ddescA.ncst = ddescA.nrst = atoi(optarg);
                if(ddescA.ncst <= 0)
                {
                    if(0 == rank)
                    {
                        fprintf(stderr, "select a positive value for super tile size\n");
                        exit(2);
                    }
                }                
                //printf("processes receives tiles by blocks of %dx%d\n", ddescA.nrst, ddescA.ncst);
                break;
                
            case 'r':
                NRHS  = atoi(optarg);
                printf("number of RHS set to %d\n", NRHS);
                break;
            case 'a':
                LDA = atoi(optarg);
                printf("LDA set to %d\n", LDA);
                break;                
            case 'b':
                LDB  = atoi(optarg);
                printf("LDB set to %d\n", LDB);
                break;
                
            case 'x':
                do_nasty_validations = 1;
                break; 
                
            case 'w':
                if(optarg)
                    do_warmup = atoi(optarg);
                else
                    do_warmup = 1;
                break;
                
            case 'h':
                if(0 == rank)
                {
                    print_usage();
                    exit(0);
                }
            case '?': /* getopt_long already printed an error message. */
            default:
                break; /* Assume anything else is dplasma/mpi stuff */
        }
    } while(1);
    
    if((DO_PLASMA == backend) && (nodes > 1))
    {
        if(0 == rank)
        {
            fprintf(stderr, "using the PLASMA backend for distributed runs is meaningless. Either use DPLASMA (-d, --dplasma), or run in single node mode.\n");
            exit(2);
        }
    }
    
    while(N == 0)
    {
        if(optind < argc)
        {
            N = atoi(argv[optind++]);
            continue;
        }
        if(0 == rank)
        {
            print_usage(); 
            exit(2);
        }
    } 

    ddescA.GRIDcols = nodes / ddescA.GRIDrows ;
    if((nodes % ddescA.GRIDrows) != 0)
    {
        if(0 == rank)
        {
            fprintf(stderr, "GRIDrows %d does not divide the total number of nodes %d\n", ddescA.GRIDrows, nodes);
            exit(2);
        }
    }
    //printf("Grid is %dx%d\n", ddescA.GRIDrows, ddescA.GRIDcols);

    if(LDA <= 0) 
    {
        LDA = N;
    }
    if(LDB <= 0) 
    {
        LDB = N;        
    }
    
    switch(backend)
    {
        case DO_PLASMA:
            PLASMA_Init(cores);
            break;
        case DO_DPLASMA:
            PLASMA_Init(1);
            break;
    }
}

static void runtime_fini(void)
{
    PLASMA_Finalize();
#ifdef USE_MPI
    MPI_Finalize();
#endif    
}



static dplasma_context_t *setup_dplasma(int* pargc, char** pargv[])
{
    dplasma_context_t *dplasma;
    
    dplasma = dplasma_init(cores, pargc, pargv);
    load_dplasma_objects(dplasma);
    {
        expr_t* constant;
        
        constant = expr_new_int( ddescA.nb );
        dplasma_assign_global_symbol( "NB", constant );
        constant = expr_new_int( ddescA.nt );
        dplasma_assign_global_symbol( "SIZE", constant );
        constant = expr_new_int( ddescA.GRIDrows );
        dplasma_assign_global_symbol( "GRIDrows", constant );
        constant = expr_new_int( ddescA.GRIDcols );
        dplasma_assign_global_symbol( "GRIDcols", constant );
        constant = expr_new_int( ddescA.rowRANK );
        dplasma_assign_global_symbol( "rowRANK", constant );
        constant = expr_new_int( ddescA.colRANK );
        dplasma_assign_global_symbol( "colRANK", constant );
        constant = expr_new_int( ddescA.nrst );
        dplasma_assign_global_symbol( "stileSIZE", constant );
    }
    load_dplasma_hooks(dplasma);
    enumerate_dplasma_tasks(dplasma);
    
    return dplasma;
}

static void cleanup_dplasma(dplasma_context_t* dplasma)
{
#ifdef DPLASMA_PROFILING
    char* filename = NULL;
    
    asprintf( &filename, "%s.%d.profile", "dposv", rank );
    dplasma_profiling_dump_xml(filename);
    free(filename);
#endif  /* DPLASMA_PROFILING */
    
    dplasma_fini(&dplasma);
}

static void warmup_dplasma(dplasma_context_t* dplasma)
{
    TIME_START();
    dplasma_progress(dplasma);
    TIME_PRINT(("Warmup on rank %d:\t%d %d %f Gflops\n", rank, N, NB, (N/1e3*N/1e3*N/1e3/3.0)/(time_elapsed * nodes)));
    
    enumerate_dplasma_tasks(dplasma);
    
    if(0 == rank)    
    {
        /* warm the cache for the first tile */
        dplasma_execution_context_t exec_context;
        if(do_warmup > 1)
        {
            int i, j;
            double useless = 0.0;
            for( i = 0; i < ddescA.nb; i++ ) {
                for( j = 0; j < ddescA.nb; j++ ) {
                    useless += ((double*)ddescA.mat)[i*ddescA.nb+j];
                }
            }
        }

        /* Ok, now get ready for the same thing again. */
        exec_context.function = (dplasma_t*)dplasma_find("POTRF");
        dplasma_set_initial_execution_context(&exec_context);
        dplasma_schedule(dplasma, &exec_context);
    }
# ifdef USE_MPI
    /* Make sure everybody is done with warmup before proceeding */
    MPI_Barrier(MPI_COMM_WORLD);
# endif    
}

#undef N
#undef NB



static void create_matrix(int N, PLASMA_enum* uplo, double** pA1, double** pA2, 
                          double** pB1, double** pB2, double** pWORK, double** pD, 
                          int LDA, int NRHS, int LDB, PLASMA_desc* local)
{
#define A1      (*pA1)
#define A2      (*pA2)
#define B1      (*pB1)
#define B2      (*pB2)
#define WORK    (*pWORK)
#define D       (*pD)
    
    if(do_nasty_validations)
    {
        int LDBxNRHS = LDB*NRHS;
        A1   = (double *)malloc(LDA*N*sizeof(double));
        A2   = (double *)malloc(LDA*N*sizeof(double));
        B1   = (double *)malloc(LDBxNRHS*sizeof(double));
        B2   = (double *)malloc(LDBxNRHS*sizeof(double));
        WORK = (double *)malloc(2*LDA*sizeof(double));
        D    = (double *)malloc(LDA*sizeof(double));
        /* Check if unable to allocate memory */
        if ((!pA1)||(!pA2)||(!pB1)||(!pB2)){
            printf("Out of Memory \n ");
            exit(1);
        }

        /* generating a random matrix */
        generate_matrix(N, A1, A2,  B1, B2,  WORK, D, LDA, NRHS, LDB);
    }
    else
    {
        int i, j;
        
        /* Only need A2 */
        A1 = B1 = B2 = WORK = D = NULL;
        A2   = (double *)malloc(LDA*N*sizeof(double));
        /* Check if unable to allocate memory */
        if (!A2){
            printf("Out of Memory \n ");
            exit(1);
        }
    
        /* generating a random matrix */
        for ( i = 0; i < N; i++)
            for ( j = i; j < N; j++) {
                A2[LDA*j+i] = A2[LDA*i+j] = (double)rand() / RAND_MAX;
            }
        for ( i = 0; i < N; i++) {
            A2[LDA*i+i] = A2[LDA*i+i] + 10*N;
        }
    }
    
    tiling(uplo, N, A2, LDA, NRHS, local);
#undef A1
#undef A2 
#undef B1 
#undef B2 
#undef WORK
#undef D
}

static void scatter_matrix(PLASMA_desc* local, DPLASMA_desc* dist)
{
#ifdef USE_MPI
    MPI_Request * requests;
    int req_count;
    
    TIME_START();
    /* prepare data for block reception  */
    dplasma_desc_bcast(local, dist);
    distribute_data(local, dist, &requests, &req_count);
    /* wait for data distribution to finish before continuing */
    is_data_distributed(dist, requests, req_count);
    TIME_PRINT(("data distribution on rank %d\n", dist->mpi_rank));
    
    if(do_nasty_validations)
    {
        data_dist_verif(local, dist);
#       if defined(PRINT_ALL_BLOCKS)
            if(rank == 0)
                plasma_dump(local);
            data_dump(dist);
#       endif
    }

#else /* NO MPI */
    dplasma_desc_init(local, dist);
#endif
}

static void gather_matrix(PLASMA_desc* local, DPLASMA_desc* dist)
{
# ifdef USE_MPI
    if(do_nasty_validations)
    {
        TIME_START();
        gather_data(local, dist);
        TIME_PRINT(("data reduction on rank %d (to rank 0)\n", dist->mpi_rank));
    }
# endif
}

static void check_matrix(int N, PLASMA_enum* uplo, double* A1, double* A2, 
                         double* B1, double* B2, double* WORK, double* D, 
                         int LDA, int NRHS, int LDB, PLASMA_desc* local, 
                         double gflops)
{    
    int info_solution, info_factorization;
    double eps = (double) 1.0e-13;  /* dlamch("Epsilon");*/

    printf("\n");
    printf("------ TESTS FOR PLASMA DPOTRF + DPOTRS ROUTINE -------  \n");
    printf("            Size of the Matrix %d by %d\n", N, N);
    printf("\n");
    printf(" The matrix A is randomly generated for each test.\n");
    printf("============\n");
    printf(" The relative machine precision (eps) is to be %e \n", eps);
    printf(" Computational tests pass if scaled residuals are less than 10.\n");        
    if(do_nasty_validations)
    {
        untiling(uplo, N, A2, LDA, &descA);
        PLASMA_dpotrs(*uplo, N, NRHS, A2, LDA, B2, LDB);

        /* Check the factorization and the solution */
        info_factorization = check_factorization(N, A1, A2, LDA, *uplo, eps);
        info_solution = check_solution(N, NRHS, A1, LDA, B1, B2, LDB, eps);

        if((info_solution == 0) && (info_factorization == 0)) 
        {
            printf("****************************************************\n");
            printf(" ---- TESTING DPOTRF + DPOTRS ............ PASSED ! \n");
            printf("****************************************************\n");
            printf(" ---- GFLOPS ............................. %.4f\n", gflops);
            printf("****************************************************\n");
        }
        else 
        {
            printf("*****************************************************\n");
            printf(" ---- TESTING DPOTRF + DPOTRS ............ FAILED !  \n");
            printf("*****************************************************\n");
        }
        free(A1); free(B1); free(B2); free(WORK); free(D);
    }
    else
    {
        printf("****************************************************\n");
        printf(" ---- TESTING DPOTRF + DPOTRS ............ SKIPPED !\n");
        printf("****************************************************\n");
        printf(" ---- n= %d np= %d nc= %d %dx%d\t %.4f GFLOPS\n", N, nodes, cores, ddescA.GRIDrows, ddescA.GRIDcols, gflops);
        printf("****************************************************\n");
    }
    free(A2);
}

#undef rank

/*------------------------------------------------------------------------
 * *  Check the factorization of the matrix A2
 * */
static int check_factorization(int N, double *A1, double *A2, int LDA, int uplo, double eps)
{
    double Anorm, Rnorm;
    double alpha;
    char norm='I';
    int info_factorization;
    int i,j;
    
    double *Residual = (double *)malloc(N*N*sizeof(double));
    double *L1       = (double *)malloc(N*N*sizeof(double));
    double *L2       = (double *)malloc(N*N*sizeof(double));
    double *work     = (double *)malloc(N*sizeof(double));
    
    memset((void*)L1, 0, N*N*sizeof(double));
    memset((void*)L2, 0, N*N*sizeof(double));
    
    alpha= 1.0;
    
    dlacpy("ALL", &N, &N, A1, &LDA, Residual, &N);
    
    /* Dealing with L'L or U'U  */
    if (uplo == PlasmaUpper){
        dlacpy(lapack_const(PlasmaUpper), &N, &N, A2, &LDA, L1, &N);
        dlacpy(lapack_const(PlasmaUpper), &N, &N, A2, &LDA, L2, &N);
        cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, (alpha), L1, N, L2, N);
    }
    else{
        dlacpy(lapack_const(PlasmaLower), &N, &N, A2, &LDA, L1, &N);
        dlacpy(lapack_const(PlasmaLower), &N, &N, A2, &LDA, L2, &N);
        cblas_dtrmm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, N, N, (alpha), L1, N, L2, N);
    }
    
    /* Compute the Residual || A -L'L|| */
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            Residual[j*N+i] = L2[j*N+i] - Residual[j*N+i];
    
    Rnorm = dlange(&norm, &N, &N, Residual, &N, work);
    Anorm = dlange(&norm, &N, &N, A1, &LDA, work);
    
    printf("============\n");
    printf("Checking the Cholesky Factorization \n");
    printf("-- ||L'L-A||_oo/(||A||_oo.N.eps) = %e \n",Rnorm/(Anorm*N*eps));
    
    if ( isnan(Rnorm/(Anorm*N*eps)) || (Rnorm/(Anorm*N*eps) > 10.0) ){
        printf("-- Factorization is suspicious ! \n");
        info_factorization = 1;
    }
    else{
        printf("-- Factorization is CORRECT ! \n");
        info_factorization = 0;
    }
    
    free(Residual); free(L1); free(L2); free(work);
    
    return info_factorization;
}


/*------------------------------------------------------------------------
 * *  Check the accuracy of the solution of the linear system
 * */
static int check_solution(int N, int NRHS, double *A1, int LDA, double *B1, double *B2, int LDB, double eps )
{
    int info_solution;
    double Rnorm, Anorm, Xnorm, Bnorm;
    char norm='I';
    double alpha, beta;
    double *work = (double *)malloc(N*sizeof(double));
    alpha = 1.0;
    beta  = -1.0;
    
    Xnorm = dlange(&norm, &N, &NRHS, B2, &LDB, work);
    Anorm = dlange(&norm, &N, &N, A1, &LDA, work);
    Bnorm = dlange(&norm, &N, &NRHS, B1, &LDB, work);
    
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, NRHS, N, (alpha), A1, LDA, B2, LDB, (beta), B1, LDB);
    Rnorm = dlange(&norm, &N, &NRHS, B1, &LDB, work);
    
    printf("============\n");
    printf("Checking the Residual of the solution \n");
    printf("-- ||Ax-B||_oo/((||A||_oo||x||_oo+||B||_oo).N.eps) = %e \n",Rnorm/((Anorm*Xnorm+Bnorm)*N*eps));
    
    if (Rnorm/((Anorm*Xnorm+Bnorm)*N*eps) > 10.0){
        printf("-- The solution is suspicious ! \n");
        info_solution = 1;
    }
    else{
        printf("-- The solution is CORRECT ! \n");
        info_solution = 0;
    }
    
    free(work);
    
    return info_solution;
}
