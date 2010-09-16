/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */


#include "dplasma.h"
#ifdef USE_MPI
#include "remote_dep.h"
#include <mpi.h>
#endif  /* defined(USE_MPI) */

#if defined(HAVE_GETOPT_H)
#include <getopt.h>
#endif  /* defined(HAVE_GETOPT_H) */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

/* CUDA INCLUDE */
#include <cuda.h>
#include <cublas.h>
#include <cuda_runtime_api.h>

#include <cblas.h>
#include <math.h>
#include <plasma.h>
#include <lapack.h>
#include <control/common.h>
#include <control/context.h>
#include <control/allocate.h>

#include "scheduling.h"
#include "profiling.h"
#include "data_management.h"
#include "remote_dep.h"

//#ifdef VTRACE
//#include "vt_user.h"
//#endif

/*
 *  *  * These are used for CUDA in the jdf.
 *   *   */
volatile uint32_t *gpu_lock;
volatile uint32_t *compute_lock;
int *gpu_counter;
int *set_device;
int cpu_counter;
int use_gpu = 10;
int overlap_counter;

static void runtime_init(int argc, char **argv);
static void runtime_fini(void);

static dplasma_context_t *setup_dplasma(int* pargc, char** pargv[]);
static void cleanup_dplasma(dplasma_context_t* context);
static void warmup_dplasma(dplasma_context_t* dplasma);

static void create_matrix(int N, PLASMA_enum* uplo, 
                          float** pA1, float** pA2, 
                          float** pB1, float** pB2, 
                          int LDA, int NRHS, int LDB, PLASMA_desc* local);
static void scatter_matrix(PLASMA_desc* local, DPLASMA_desc* dist);
static void gather_matrix(PLASMA_desc* local, DPLASMA_desc* dist);
static void check_matrix(int N, PLASMA_enum* uplo, 
                         float* A1, float* A2, 
                         float* B1, float* B2,
                         int LDA, int NRHS, int LDB, PLASMA_desc* local, 
                         double gflops);

static int check_factorization(int N, float *A1, float *A2, int LDA, int uplo, float eps);
static int check_solution(int N, int NRHS, float *A1, int LDA, float *B1, float *B2, int LDB, float eps );

/* timing profiling etc */
double time_elapsed;
double sync_time_elapsed;

int dposv_force_nb = 0;
int pri_change = 0;
static int preallocated_tiles = 1024;

static inline double get_cur_time(){
    double t;
    struct timeval tv;
    gettimeofday(&tv,NULL);
    t=tv.tv_sec+tv.tv_usec/1e6;
    return t;
}

#define TIME_START() do { time_elapsed = get_cur_time(); } while(0)
#define TIME_STOP() do { time_elapsed = get_cur_time() - time_elapsed; } while(0)
#define TIME_PRINT(print) do {                                  \
        TIME_STOP();                                            \
        /*printf("[%d] TIMED %f s :\t", rank, time_elapsed);*/  \
        printf print;                                           \
    } while(0)


#ifdef USE_MPI
# define SYNC_TIME_START() do {                 \
        MPI_Barrier(MPI_COMM_WORLD);            \
        sync_time_elapsed = get_cur_time();     \
    } while(0)
# define SYNC_TIME_STOP() do {                                  \
        MPI_Barrier(MPI_COMM_WORLD);                            \
        sync_time_elapsed = get_cur_time() - sync_time_elapsed; \
    } while(0)
# define SYNC_TIME_PRINT(print) do {                                \
        SYNC_TIME_STOP();                                           \
        if(0 == rank) {                                             \
            printf("### TIMED %f s :\t", sync_time_elapsed);        \
            printf print;                                           \
        }                                                           \
    } while(0)

/* overload exit in MPI mode */
#   define exit(ret) MPI_Abort(MPI_COMM_WORLD, ret)

#else 
# define SYNC_TIME_START() do { sync_time_elapsed = get_cur_time(); } while(0)
# define SYNC_TIME_STOP() do { sync_time_elapsed = get_cur_time() - sync_time_elapsed; } while(0)
# define SYNC_TIME_PRINT(print) do { \
    SYNC_TIME_STOP(); \
    if(0 == rank) { \
      printf("### TIMED %f doing\t", sync_time_elapsed); \
      printf print; \
    } \
  } while(0)
#endif

typedef enum {
    DO_PLASMA,
    DO_DPLASMA
} backend_argv_t;

/* globals and argv set values */
int do_warmup = 0;
int do_nasty_validations = 0;
int do_distributed_generation = 1;
backend_argv_t backend = DO_DPLASMA;
int cores = 1;
int nodes = 1;
int nbtasks = -1;
#define N (ddescA.n)
#define NB (ddescA.nb)
#define rank (ddescA.mpi_rank)
int LDA = 0;
int NRHS = 1;
int LDB = 0;
PLASMA_enum uplo = PlasmaLower;

PLASMA_desc descA;
DPLASMA_desc ddescA;

extern int spotrf_cuda_init( int* use_gpu );
extern int spotrf_cuda_fini( int use_gpu );

int main(int argc, char ** argv)
{
    double gflops;
    float *A1;
    float *A2;
    float *B1;
    float *B2;
    dplasma_context_t* dplasma;

    //#ifdef VTRACE
    // VT_OFF();
    //#endif

    runtime_init(argc, argv);

    if(0 == rank)
        create_matrix(N, &uplo, &A1, &A2, &B1, &B2, LDA, NRHS, LDB, &descA);

    switch(backend) {
    case DO_PLASMA: {
        plasma_context_t* plasma = plasma_context_self();

        if(do_warmup)
            {
                TIME_START();
                PLASMA_spotrf_Tile(uplo, &descA);
                TIME_PRINT(("_plasma warmup:\t\t%d %d %f Gflops\n", N, PLASMA_NB,
                            (N/1e3*N/1e3*N/1e3/3.0+N/1e3*N/1e3/2.0)/(time_elapsed)));
            }
        TIME_START();
        PLASMA_spotrf_Tile(uplo, &descA);
        TIME_PRINT(("_plasma computation:\t%d %d %f Gflops\n", N, PLASMA_NB, 
                    gflops = (N/1e3*N/1e3*N/1e3/3.0)/(time_elapsed)));
        break;
    }
    case DO_DPLASMA: {
        //#ifdef VTRACE 
        //    VT_ON();
        //#endif
    
        /*** THIS IS THE DPLASMA COMPUTATION ***/
        TIME_START();
        dplasma = setup_dplasma(&argc, &argv);

        if( 0 != dplasma_description_init(&ddescA, LDA, LDB, NRHS, uplo) ) {
            printf("Failed to initialize the matrix\n");
            exit(-2);
        }

        if(use_gpu != -1) {
            if( 0 == spotrf_cuda_init( &use_gpu ) ) {
                overlap_counter = 0;
                /* cpu counter for GEMM*/
                cpu_counter = 0;
            }
        }

        dplasma_remote_dep_preallocate_buffers( preallocated_tiles, NB*NB*sizeof(float), use_gpu );

        scatter_matrix(&descA, &ddescA);
        TIME_PRINT(("Dplasma initialization:\t%d %d\n", N, NB));
#ifdef USE_MPI
        /**
         * Redefine the default type after dplasma_init.
         */
        {
            char type_name[MPI_MAX_OBJECT_NAME];
    
            snprintf(type_name, MPI_MAX_OBJECT_NAME, "Default MPI_FLOAT*%d*%d", NB, NB);
    
            MPI_Type_contiguous(NB * NB, MPI_FLOAT, &DPLASMA_DEFAULT_DATA_TYPE);
            MPI_Type_set_name(DPLASMA_DEFAULT_DATA_TYPE, type_name);
            MPI_Type_commit(&DPLASMA_DEFAULT_DATA_TYPE);
        }
#endif  /* USE_MPI */

        /**
         * Now the last step of the DPLASMA initialization.
         */
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
            dplasma_assign_global_symbol( "rtileSIZE", constant );
            constant = expr_new_int( ddescA.ncst );
            dplasma_assign_global_symbol( "ctileSIZE", constant );
            constant = expr_new_int( pri_change );
            dplasma_assign_global_symbol( "PRI_CHANGE", constant );
        }
        load_dplasma_hooks(dplasma);
        nbtasks = enumerate_dplasma_tasks(dplasma);

        if(0 == rank) {
            dplasma_execution_context_t exec_context;

            /* I know what I'm doing ;) */
            exec_context.function = (dplasma_t*)dplasma_find("POTRF");
            dplasma_set_initial_execution_context(&exec_context);
            dplasma_schedule(dplasma, &exec_context);
        }
        if(do_warmup)
            warmup_dplasma(dplasma);
    
        /* lets rock! */
        SYNC_TIME_START();
        TIME_START();
        dplasma_progress(dplasma);
        TIME_PRINT(("Dplasma proc %d:\ttasks: %d\t%f task/s\n", rank, nbtasks, nbtasks/time_elapsed));
        SYNC_TIME_PRINT(("Dplasma computation:\t%d %d %f gflops\n", N, NB,
                         gflops = (N/1e3*N/1e3*N/1e3/3.0)/(sync_time_elapsed)));
        printf("[%d] Dplasma priority change at position \t%d\n", rank, ddescA.nt - pri_change);

        cleanup_dplasma(dplasma);
        /*** END OF DPLASMA COMPUTATION ***/
		
        gather_matrix(&descA, &ddescA);
        /* Cleanup CUDA */
        {
            if (use_gpu > 0) {
                spotrf_cuda_fini( use_gpu );
            }
        }
        break;
    }
    }

    if(0 == rank)
        check_matrix(N, &uplo, A1, A2, B1, B2, LDA, NRHS, LDB, &descA, gflops);

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
            "   -s --stile-row   : number of tile per row in a super tile (default: 1)\n"
            "   -e --stile-col   : number of tile per col in a super tile (default: 1)\n"
            "   -a --lda         : leading dimension of the matrix A (equal matrix size by default)\n"
            "   -b --ldb         : leading dimension of the RHS B (equal matrix size by default)\n"
            "   -r --nrhs        : Number of Right Hand Side (default: 1)\n"
            "   -x --xcheck      : do extra nasty result validations\n"
            "   -w --warmup      : do some warmup, if > 1 also preload cache\n"
            "      --gpu         : number of activable GPUs\n"
            "   -P --pri_change  : the position on the diagonal from the end where we switch the priority (default: 0)\n"
            "   -B --block-size  : change the block size from the size tuned by PLASMA\n"
	    "   -A --allocation  : change the number of preallocated reception tiles. Default 1024. For GPU run, all reception tiles *must* be preallocated\n");
}

static void runtime_init(int argc, char **argv)
{
#if defined(HAVE_GETOPT_LONG)
    struct option long_options[] =
    {
        {"nb-cores",    required_argument,  0, 'c'},
        {"matrix-size", required_argument,  0, 'n'},
        {"lda",         required_argument,  0, 'a'},
        {"nrhs",        required_argument,  0, 'r'},
        {"ldb",         required_argument,  0, 'b'},
        {"grid-rows",   required_argument,  0, 'g'},
        {"stile-row",   required_argument,  0, 's'},
        {"stile-col",   required_argument,  0, 'e'},
        {"xcheck",      no_argument,        0, 'x'},
        {"warmup",      optional_argument,  0, 'w'},
        {"dplasma",     no_argument,        0, 'd'},
        {"plasma",      no_argument,        0, 'p'},
        {"gpu",         required_argument,  0, 'u'},
        {"block-size",  required_argument,  0, 'B'},
        {"allocation",  required_argument,  0, 'A'},
        {"pri_change",  required_argument,  0, 'P'},
        {"help",        no_argument,        0, 'h'},
        {0, 0, 0, 0}
    };
#endif  /* defined(HAVE_GETOPT_LONG) */

#ifdef USE_MPI
    /* mpi init */
    MPI_Init(&argc, &argv);
    
    MPI_Comm_size(MPI_COMM_WORLD, &nodes); 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    /*sleep(20);*/
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
#if defined(HAVE_GETOPT_LONG)
            int option_index = 0;
            c = getopt_long (argc, argv, "dpxc:n:a:r:b:g:e:s:w::B:A:P:h",
                             long_options, &option_index);
#else
            c = getopt (argc, argv, "dpxc:n:a:r:b:g:e:s:w::B:A:P:h");
#endif  /* defined(HAVE_GETOPT_LONG) */
        
        /* Detect the end of the options. */
            if (c == -1)
                break;
        
            switch (c) {
            case 'p': 
                backend = DO_PLASMA;
                do_distributed_generation = 0;
                break;
            case 'd':
                backend = DO_DPLASMA;
                break;

            case 'c':
                cores = atoi(optarg);
                if(cores<= 0)
                    cores=1;
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
                ddescA.nrst = atoi(optarg);
                if(ddescA.nrst <= 0) {
                    fprintf(stderr, "select a positive value for the row super tile size\n");
                    exit(2);
                }                
                /*printf("processes receives tiles by blocks of %dx%d\n", ddescA.nrst, ddescA.ncst);*/
                break;
            case 'e':
                ddescA.ncst = atoi(optarg);
                if(ddescA.ncst <= 0) {
                    fprintf(stderr, "select a positive value for the col super tile size\n");
                    exit(2);
                }                
                /*printf("processes receives tiles by blocks of %dx%d\n", ddescA.nrst, ddescA.ncst);*/
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
                do_distributed_generation = 0;
                fprintf(stderr, "Results are checked on rank 0, distributed matrix generation is disabled.\n");
                if(do_warmup) {
                    fprintf(stderr, "Results cannot be correct with warmup! Validations and warmup are exclusive; please select only one.\n");
                    exit(2);
                }
                break; 
            case 'w':
                if(optarg)
                    do_warmup = atoi(optarg);
                else
                    do_warmup = 1;
                if(do_nasty_validations) {
                    fprintf(stderr, "Results cannot be correct with warmup! Validations and warmup are exclusive; please select only one.\n");
                    exit(2);
                }
                break;
                
            case 'B':
                if(optarg) {
                    dposv_force_nb = atoi(optarg);
                } else {
                    fprintf(stderr, "Argument is mandatory for -B (--block-size) flag.\n");
                    exit(2);
                }
                break;
        case 'A':
            if(optarg) {
                preallocated_tiles = atoi(optarg);
            } else {
                fprintf(stderr, "Argument is mandatory for -A (--allocation) flag.\n");
                exit(2);
            }
            break;            
        case 'u':
            use_gpu = atoi(optarg);
            break;

        case 'P':
                pri_change = atoi(optarg);
                break;
        case 'h':
                print_usage();
                exit(0);
        case '?': /* getopt_long already printed an error message. */
        default:
                break; /* Assume anything else is dplasma/mpi stuff */
            }
        } while(1);
    
    if((DO_PLASMA == backend) && (nodes > 1)) {
        fprintf(stderr, "using the PLASMA backend for distributed runs is meaningless. Either use DPLASMA (-d, --dplasma), or run in single node mode.\n");
        exit(2);
    }
    
    while(N == 0) {
        if(optind < argc) {
            N = atoi(argv[optind++]);
            continue;
        }
        print_usage(); 
        exit(2);
    } 
    ddescA.cores = cores;
    ddescA.GRIDcols = nodes / ddescA.GRIDrows ;
    if((nodes % ddescA.GRIDrows) != 0) {
        fprintf(stderr, "GRIDrows %d does not divide the total number of nodes %d\n", ddescA.GRIDrows, nodes);
        exit(2);
    }
    //printf("Grid is %dx%d\n", ddescA.GRIDrows, ddescA.GRIDcols);

    if(LDA <= 0) 
        LDA = N;
    if(LDB <= 0) 
        LDB = N;        

    switch(backend) {
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
   
    dplasma = dplasma_init(cores, pargc, pargv, ddescA.nb);

    load_dplasma_objects(dplasma);

    return dplasma;
}

static void cleanup_dplasma(dplasma_context_t* dplasma)
{
#ifdef DPLASMA_PROFILING
    char* filename = NULL;
    
    asprintf( &filename, "%s.%d.profile", "sposv", rank );
    dplasma_profiling_dump_xml(filename);
    free(filename);
#endif  /* DPLASMA_PROFILING */
    
    dplasma_fini(&dplasma);
}

static void warmup_dplasma(dplasma_context_t* dplasma)
{
    TIME_START();
    dplasma_progress(dplasma);
    TIME_PRINT(("Warmup on rank %d:\t%d %d\n", rank, N, NB));
    
    enumerate_dplasma_tasks(dplasma);
    
    if(0 == rank)    
    {
        /* warm the cache for the first tile */
        dplasma_execution_context_t exec_context;
        if(do_warmup > 1)
        {
            int i, j;
            float useless = 0.0;
            for( i = 0; i < ddescA.nb; i++ ) {
                for( j = 0; j < ddescA.nb; j++ ) {
                    useless += ((float*)ddescA.mat)[i*ddescA.nb+j];
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


static void create_matrix(int N, PLASMA_enum* uplo, 
                          float** pA1, float** pA2, 
                          float** pB1, float** pB2, 
                          int LDA, int NRHS, int LDB, PLASMA_desc* local)
{
#define A1      (*pA1)
#define A2      (*pA2)
#define B1      (*pB1)
#define B2      (*pB2)
    int i, j;

    if(do_distributed_generation) 
    {
        A1 = A2 = B1 = B2 = NULL;
        return;
    }
    
    if(do_nasty_validations)
    {
        A1   = (float *)malloc(LDA*N*sizeof(float));
        A2   = (float *)malloc(LDA*N*sizeof(float));
        B1   = (float *)malloc(LDB*NRHS*sizeof(float));
        B2   = (float *)malloc(LDB*NRHS*sizeof(float));
        /* Check if unable to allocate memory */
        if((!pA1) || (!pA2) || (!pB1) || (!pB2))
        {
            printf("Out of Memory \n ");
            exit(1);
        }

        /* generating a random matrix */
        for ( i = 0; i < N; i++)
            for ( j = i; j < N; j++) {
                A2[LDA*j+i] = A1[LDA*j+i] = (float)rand() / RAND_MAX;
                A2[LDA*i+j] = A1[LDA*i+j] = A1[LDA*j+i];
            }
        for ( i = 0; i < N; i++) {
            A2[LDA*i+i] = A1[LDA*i+i] += 10*N;
        }
        /* Initialize B1 and B2 */
        for ( i = 0; i < N; i++)
            for ( j = 0; j < NRHS; j++)
                B2[LDB*j+i] = B1[LDB*j+i] = (float)rand() / RAND_MAX;
    }
    else
    {        
        /* Only need A2 */
        A1 = B1 = B2 = NULL;
        A2   = (float *)malloc(LDA*N*sizeof(float));
        /* Check if unable to allocate memory */
        if (!A2){
            printf("Out of Memory \n ");
            exit(1);
        }

        /* generating a random matrix */
        for ( i = 0; i < N; i++)
            for ( j = i; j < N; j++) {
                A2[LDA*j+i] = A2[LDA*i+j] = (float)rand() / RAND_MAX;
            }
        for ( i = 0; i < N; i++) {
            A2[LDA*i+i] = A2[LDA*i+i] + 10 * N;
        }
    }
    
    tiling(uplo, N, A2, LDA, NRHS, local);
#undef A1
#undef A2 
#undef B1 
#undef B2 
}

static void scatter_matrix(PLASMA_desc* local, DPLASMA_desc* dist)
{
    if(do_distributed_generation)
    {
        /* Allocate memory for matrices in block layout */
        dist->mat = dplasma_allocate_matrix( dist->nb_elem_r * dist->nb_elem_c * dist->bsiz * sizeof(float),
                                             use_gpu);
        rand_dist_matrix(dist);
        /*TIME_PRINT(("distributed matrix generation on rank %d\n", dist->mpi_rank));*/
        return;
    }
    
    TIME_START();
    if(0 == rank)
    {
        dplasma_desc_init(local, dist);
    }
#ifdef USE_MPI
    dplasma_desc_bcast(local, dist, use_gpu);
    distribute_data(local, dist);
    /*TIME_PRINT(("data distribution on rank %d\n", dist->mpi_rank));*/
    
#if defined(DATA_VERIFICATIONS)
    if(do_nasty_validations)
    {
        data_dist_verif(local, dist);
#if defined(PRINT_ALL_BLOCKS)
        if(rank == 0)
            plasma_dump(local);
        data_dump(dist);
#endif /* PRINT_ALL_BLOCKS */
    }
#endif /* DATA_VERIFICATIONS */
#endif /* USE_MPI */
}

static void gather_matrix(PLASMA_desc* local, DPLASMA_desc* dist)
{
    if(do_distributed_generation) 
    {
        return;
    }
# ifdef USE_MPI
    if(do_nasty_validations)
    {
        TIME_START();
        gather_data(local, dist);
        TIME_PRINT(("data reduction on rank %d (to rank 0)\n", dist->mpi_rank));
    }
# endif
}

static void check_matrix(int N, PLASMA_enum* uplo, 
                         float* A1, float* A2, 
                         float* B1, float* B2,  
                         int LDA, int NRHS, int LDB, PLASMA_desc* local, 
                         double gflops)
{    
    int info_solution, info_factorization;
    float eps = lapack_slamch(lapack_eps);

    printf("\n");
    printf("------ TESTS FOR PLASMA SPOTRF + SPOTRS ROUTINE -------  \n");
    printf("            Size of the Matrix %d by %d\n", N, N);
    printf("\n");
    printf(" The matrix A is randomly generated for each test.\n");
    printf("============\n");
    printf(" The relative machine precision (eps) is to be %e \n", eps);
    printf(" Computational tests pass if scaled residuals are less than 10.\n");        
    if(do_nasty_validations)
    {
        untiling(uplo, N, A2, LDA, local);
        PLASMA_spotrs(*uplo, N, NRHS, A2, LDA, B2, LDB);

        /* Check the factorization and the solution */
        info_factorization = check_factorization(N, A1, A2, LDA, *uplo, eps);
        info_solution = check_solution(N, NRHS, A1, LDA, B1, B2, LDB, eps);

        if((info_solution == 0) && (info_factorization == 0)) 
        {
            printf("****************************************************\n");
            printf(" ---- TESTING SPOTRF + SPOTRS ............ PASSED ! \n");
            printf("****************************************************\n");
            printf(" ---- GFLOPS ............................. %.4f\n", gflops);
            printf("****************************************************\n");
        }
        else 
        {
            printf("*****************************************************\n");
            printf(" ---- TESTING SPOTRF + SPOTRS ............ FAILED !  \n");
            printf("*****************************************************\n");
        }
        free(A1); free(B1); free(B2);
    }
    else
    {
        printf("****************************************************\n");
        printf(" ---- TESTING SPOTRF + SPOTRS ............ SKIPPED !\n");
        printf("****************************************************\n");
        printf(" ---- n= %d np= %d nc= %d g= %dx%d\t %.4f GFLOPS\n", N, nodes, cores, ddescA.GRIDrows, ddescA.GRIDcols, gflops);
        printf("****************************************************\n");
    }
    free(A2);
}

#undef rank


/*------------------------------------------------------------------------
 * *  Check the factorization of the matrix A2
 * */
static int check_factorization(int N, float *A1, float *A2, int LDA, int uplo, float eps)
{
    float Anorm;
    float alpha;
    int info_factorization;
    int i,j;
    
    float *Residual = (float *)malloc(N*N*sizeof(float));
    float *L1       = (float *)malloc(N*N*sizeof(float));
    float *L2       = (float *)malloc(N*N*sizeof(float));
    float *work     = (float *)malloc(N*sizeof(float));
    float Rnorm;

    /*memset((void*)L1, 0, N*N*sizeof(float));*/
    /*memset((void*)L2, 0, N*N*sizeof(float));*/
    
    alpha = 1.0;
    
    lapack_slacpy(lapack_upper_lower, N, N, A1, LDA, Residual, N);

    /* Dealing with L'L or U'U  */
    if (uplo == PlasmaUpper){
        lapack_slacpy(lapack_upper, N, N, A2, LDA, L1, N);
        lapack_slacpy(lapack_upper, N, N, A2, LDA, L2, N);
        cblas_strmm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, (alpha), L1, N, L2, N);
    }
    else{
        lapack_slacpy(lapack_lower, N, N, A2, LDA, L1, N);
        lapack_slacpy(lapack_lower, N, N, A2, LDA, L2, N);
        cblas_strmm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, N, N, (alpha), L1, N, L2, N);
    }
    
    /* Compute the Residual || A -L'L|| */
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            Residual[j*N+i] = L2[j*N+i] - Residual[j*N+i];
    
    Anorm = lapack_slange(lapack_inf_norm, N, N,       A1, LDA, work);
    Rnorm = lapack_slange(lapack_inf_norm, N, N, Residual,   N, work);

    printf("============\n");
    printf("Checking the Cholesky Factorization \n");
    printf("-- eps = %e\n", eps);
    printf("-- Rnorm = %e\n", Rnorm);
    printf("-- ||L'L-A||_oo/(||A||_oo.N.eps) = %e \n", Rnorm/(Anorm*N*eps));
    
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
static int check_solution(int N, int NRHS, float *A1, int LDA, float *B1, float *B2, int LDB, float eps )
{
    int info_solution;
    float Rnorm, Anorm, Xnorm, Bnorm;
    float alpha, beta;
    float *work = (float *)malloc(N*sizeof(float));
    alpha = 1.0;
    beta  = -1.0;
    
    Xnorm = lapack_slange(lapack_inf_norm, N, NRHS, B2, LDB, work);
    Anorm = lapack_slange(lapack_inf_norm, N, N,    A1, LDA, work);
    Bnorm = lapack_slange(lapack_inf_norm, N, NRHS, B1, LDB, work);
    
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, NRHS, N, (alpha), A1, LDA, B2, LDB, (beta), B1, LDB);
    Rnorm = lapack_slange(lapack_inf_norm, N, NRHS, B1, LDB, work);

    printf("============\n");
    printf("Checking the Residual of the solution \n");
    printf("-- ||Ax-B||_oo/((||A||_oo||x||_oo+||B||_oo).N.eps) = %e \n",Rnorm/((Anorm*Xnorm+Bnorm)*N*eps));
    
    if ( isnan(Rnorm/((Anorm*Xnorm+Bnorm)*N*eps)) || Rnorm/((Anorm*Xnorm+Bnorm)*N*eps) > 10.0) {
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
