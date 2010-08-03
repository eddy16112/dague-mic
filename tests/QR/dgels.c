/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague.h"
#if defined(DISTRIBUTED)
#include "remote_dep.h"
#include <mpi.h>
#endif 

#if defined(HAVE_GETOPT_H)
#include <getopt.h>
#endif  /* defined(HAVE_GETOPT_H) */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

#include <cblas.h>
#include <math.h>
#include <plasma.h>
#include <../src/common.h>
#include <../src/lapack.h>
#include <../src/context.h>
#include <../src/allocate.h>

#include "scheduling.h"
#include "profiling.h"
#include "two_dim_rectangle_cyclic.h"
#include "QR.h"

static void runtime_init(int argc, char **argv);
static void runtime_fini(void);

static dague_context_t *setup_dague(int* pargc, char** pargv[]);
static void cleanup_dague(dague_context_t* context);

static void create_datatypes(void);

#if defined(DEBUG_MATRICES)
static void debug_matrices(void);
#else
#define debug_matrices()
#endif

static dague_object_t *dague_QR = NULL;

/* timing profiling etc */
double time_elapsed;
double sync_time_elapsed;

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
        TIME_STOP();                                    \
        /*printf("[%d] TIMED %f s :\t", rank, time_elapsed);*/  \
        printf print;                                           \
    } while(0)

#if defined(DISTRIBUTED)
# define SYNC_TIME_START() do {                 \
        MPI_Barrier(MPI_COMM_WORLD);            \
        sync_time_elapsed = get_cur_time();     \
    } while(0)
# define SYNC_TIME_STOP() do {                  \
        MPI_Barrier(MPI_COMM_WORLD);                    \
        sync_time_elapsed = get_cur_time() - sync_time_elapsed; \
    } while(0)
# define SYNC_TIME_PRINT(print) do { \
        SYNC_TIME_STOP();                                   \
        if(0 == rank) {                                             \
            printf("### TIMED %f s :\t", sync_time_elapsed);    \
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

/* globals and argv set values */
int cores = 1;
int nodes = 1;
int nbtasks = -1;
int N = 0;
int M = 0;
int NB = 144;
int IB = 48;
int MB = 48;
int rank;
int LDA = 0;
int NRHS = 1;
int LDB = 0;
int nrst = 1;
int ncst = 1;
PLASMA_enum uplo = PlasmaLower;
int GRIDrows = 1;

two_dim_block_cyclic_t ddescA;
two_dim_block_cyclic_t ddescT;
#if defined(DISTRIBUTED)
MPI_Datatype LOWER_TILE, UPPER_TILE, LITTLE_T;
#endif

/* TODO Remove this ugly stuff */
extern int dgels_private_memory_initialization(plasma_context_t *plasma, int MB, int NB, int type);

int main(int argc, char ** argv)
{
    double gflops;
    dague_context_t* dague;
        
#if defined(DISTRIBUTED)
    /* mpi init */
    MPI_Init(&argc, &argv);
    
    MPI_Comm_size(MPI_COMM_WORLD, &nodes); 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
#else
    nodes = 1;
    rank = 0;
#endif

   runtime_init(argc, argv);
   /* initializing matrix structure */
   two_dim_block_cyclic_init(&ddescA, matrix_RealDouble, nodes, cores, rank, 
                             NB, NB, IB, M, N, 0, 0, 
                             M, N, nrst, ncst, GRIDrows);
   /* matrix generation */
   generate_tiled_random_mat((tiled_matrix_desc_t *) &ddescA);
   printf("matrix generated\n");
   
   two_dim_block_cyclic_init(&ddescT, matrix_RealDouble, nodes, cores, rank,
                             IB, NB, IB, IB*ddescA.super.mt, N, 0, 0,
                             IB*ddescA.super.mt, N, nrst, ncst, GRIDrows);
   generate_tiled_zero_mat((tiled_matrix_desc_t *) &ddescT);

   create_datatypes();

   /*** THIS IS THE DAGUE COMPUTATION ***/
   TIME_START();
   dague = setup_dague(&argc, &argv);
   if(0 == rank)
       {
           dague_execution_context_t exec_context;
           
           /* I know what I'm doing ;) */
           exec_context.function = (dague_t*)dague_find(dague_QR, "DGEQRT");
           exec_context.dague_object = dague_QR;
           exec_context.priority = 0;
           exec_context.locals[0].value = 0;
           
           dague_schedule(dague, &exec_context);
       }
   TIME_PRINT(("Dague initialization:\t%d %d\n", N, NB));
   
   /* lets rock! */
   SYNC_TIME_START();
   TIME_START();
   dague_progress(dague);
   TIME_PRINT(("Dague proc %d:\ttasks: %d\t%f task/s\n", rank, nbtasks, nbtasks/time_elapsed));
   SYNC_TIME_PRINT(("Dague computation:\t%d %d %f gflops\n", N, NB,
                    gflops = (2*N/1e3*N/1e3*((double)M - N/3.0)/1e3)/(sync_time_elapsed)));
   
   cleanup_dague(dague);
   /*** END OF DAGUE COMPUTATION ***/
   
   runtime_fini();
   return 0;
}

static void print_usage(void)
{
    fprintf(stderr,
            "Mandatory argument:\n"
            "   number           : number of elements per col\n"
            "Optional arguments:\n"
            "   -c --nb-cores    : number of computing threads to use\n"
            "   -M --rows        : number of elements per row (default to N)\n"
            "   -N --cols        : number of elements per col\n"
            "   -g --grid-rows   : number of processes row in the process grid (must divide the total number of processes (default: 1)\n"
            "   -s --stile-row   : number of tile per row in a super tile (default: 1)\n"
            "   -e --stile-col   : number of tile per col in a super tile (default: 1)\n"
            "   -a --lda         : leading dimension of the matrix A (equal matrix size by default)\n"
            "   -b --ldb         : leading dimension of the RHS B (equal matrix size by default)\n"
            "   -r --nrhs        : Number of Right Hand Side (default: 1)\n"
            "   -B --block-size  : change the block size from the size tuned by PLASMA\n");
}

static void runtime_init(int argc, char **argv)
{
#if defined(HAVE_GETOPT_LONG)
    struct option long_options[] =
    {
        {"nb-cores",    required_argument,  0, 'c'},
        {"cols",        required_argument,  0, 'N'},
        {"rows",        required_argument,  0, 'M'},
        {"lda",         required_argument,  0, 'a'},
        {"nrhs",        required_argument,  0, 'r'},
        {"ldb",         required_argument,  0, 'b'},
        {"grid-rows",   required_argument,  0, 'g'},
        {"stile-col",   required_argument,  0, 'e'},
        {"stile-row",   required_argument,  0, 's'},
        {"block-size",  required_argument,  0, 'B'},
        {"internal-block-size", required_argument, 0, 'I'},
        {"help",        no_argument,        0, 'h'},
        {0, 0, 0, 0}
    };
#endif  /* defined(HAVE_GETOPT_LONG) */

    do
    {
        int c;
#if defined(HAVE_GETOPT_LONG)
        int option_index = 0;
        c = getopt_long (argc, argv, "c:N:M:a:r:b:g:e:s:B:I:h",
                         long_options, &option_index);
#else
        c = getopt (argc, argv, "c:N:M:a:r:b:g:e:s:B:I:h");
#endif  /* defined(HAVE_GETOPT_LONG) */
        
        /* Detect the end of the options. */
        if (c == -1)
            break;
        
        switch (c)
        {
            case 'c':
                cores = atoi(optarg);
                if(cores<= 0)
                    cores=1;
                break;

            case 'N':
                N = atoi(optarg);
                break;

            case 'M':
                M = atoi(optarg);
                //printf("matrix size set to %d\n", M);
                break;

            case 'g':
                GRIDrows = atoi(optarg);
                break;

            case 's':
                nrst = atoi(optarg);
                if(nrst <= 0)
                {
                    fprintf(stderr, "select a positive value for the row super tile size\n");
                    exit(2);
                }                
                /*printf("processes receives tiles by blocks of %dx%d\n", ddescA.nrst, ddescA.ncst);*/
                break;

            case 'e':
                ncst = atoi(optarg);
                if(ncst <= 0)
                {
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
                
            case 'B':
                if(optarg)
                {
                    NB = atoi(optarg);
                }
                else
                {
                    fprintf(stderr, "Argument is mandatory for -B (--block-size) flag.\n");
                    exit(2);
                }
                break;

            case 'I':
                if(optarg)
                {
                    IB = atoi(optarg);
                    MB = IB;
                }
                else
                {
                    fprintf(stderr, "Argument is mandatory for -I (--internal-block-size) flag.\n");
                    exit(2);
                }
                break;

            case 'h':
                print_usage();
                exit(0);
            case '?': /* getopt_long already printed an error message. */
            default:
                break; /* Assume anything else is dague/mpi stuff */
        }
    } while(1);
    
    while(N == 0)
    {
        if(optind < argc)
        {
            N = atoi(argv[optind++]);
            continue;
        }
        print_usage(); 
        exit(2);
    } 
    if( M == 0 ) {
        M = N;
    }
     if((nodes % GRIDrows) != 0)
        {
            fprintf(stderr, "GRIDrows %d does not divide the total number of nodes %d\n", ddescA.GRIDrows, nodes);
            exit(2);
        }
    //printf("Grid is %dx%d\n", ddescA.GRIDrows, ddescA.GRIDcols);

    if(LDA <= 0) 
    {
        LDA = M;
    }
    if(LDB <= 0) 
    {
        LDB = M;        
    }

    PLASMA_Init(1);

    {
        plasma_context_t* plasma = plasma_context_self();
        plasma_tune(PLASMA_FUNC_DGELS, M, N, NRHS);
        
        PLASMA_NB = NB;
        PLASMA_NBNBSIZE = PLASMA_NB * PLASMA_NB;

        if( PLASMA_NB % IB != 0 ) {
            fprintf(stderr, "Invalid IB flag: %d (internal block size) does not divide %d (block size)\n",
                    IB, NB);
            exit(1);
        }
        PLASMA_IB = IB;

        PLASMA_IBNBSIZE = PLASMA_IB * PLASMA_NB;

        plasma->autotuning_enabled = 0;
    } 
}

static void runtime_fini(void)
{
    PLASMA_Finalize();
#if defined(DISTRIBUTED)
    MPI_Finalize();
#endif    
}


static dague_context_t *setup_dague(int* pargc, char** pargv[])
{
    dague_context_t *dague;
    int MT, NT, MINMTNT;
    
    dague = dague_init(cores, pargc, pargv, NB);

    create_datatypes();
    
    /* TODO: this should be computed by the generated code, not me */
    MT = (M + (NB-1))/NB;
    NT = (N + (NB-1))/NB;
    MINMTNT = ((MT < NT)  ? MT : NT);

    dague_QR = (dague_object_t*)dague_QR_new((dague_ddesc_t*)&ddescT, (dague_ddesc_t*)&ddescA, 
                                             MB, NB, M, N, 
                                             MT, NT, MINMTNT);
    dague->taskstodo += dague_QR->nb_local_tasks;
    printf("QR %dx%d has %d tasks to run. Total nb tasks to run: %d\n", 
           ddescA.super.nb, ddescA.super.nt, dague_QR->nb_local_tasks, dague->taskstodo);
    printf("GRIDrows = %d, GRIDcols = %d, rrank = %d, crank = %d\n", 
           ddescA.GRIDrows, ddescA.GRIDcols, ddescA.rowRANK, ddescA.colRANK );
    
    dgels_private_memory_initialization(plasma_context_self(), MB, NB, PlasmaRealDouble);
    
    return dague;
}

static void cleanup_dague(dague_context_t* dague)
{
#ifdef DAGUE_PROFILING
    char* filename = NULL;
    
    asprintf( &filename, "%s.%d.profile", "dgels", rank );
    dague_profiling_dump_xml(filename);
    free(filename);
#endif  /* DAGUE_PROFILING */
    
    dague_fini(&dague);
}

/*
 * These datatype creation function works only when the matrix
 * is COLUMN major. In case the matrix storage is ROW major
 * these functions have to be changed.
 */
static void create_datatypes(void)
{
#if defined(DISTRIBUTED)
    plasma_context_t* plasma = plasma_context_self();
    int *blocklens, *indices, count, i;
    MPI_Datatype tmp;
    MPI_Aint lb, ub;
    int IB = PLASMA_IB;

    count = NB; 
    blocklens = (int*)malloc( count * sizeof(int) );
    indices = (int*)malloc( count * sizeof(int) );

    /* UPPER_TILE with the diagonal */
    for( i = 0; i < count; i++ ) {
        blocklens[i] = i + 1;
        indices[i] = i * NB;
    }

    MPI_Type_indexed(count, blocklens, indices, MPI_DOUBLE, &UPPER_TILE);
    MPI_Type_set_name(UPPER_TILE, "Upper");
    MPI_Type_commit(&UPPER_TILE);
    
    MPI_Type_get_extent(UPPER_TILE, &lb, &ub);
    
    /* LOWER_TILE without the diagonal */
    for( i = 0; i < count-1; i++ ) {
        blocklens[i] = NB - i - 1;
        indices[i] = i * NB + i + 1;
    }

    MPI_Type_indexed(count-1, blocklens, indices, MPI_DOUBLE, &tmp);
    MPI_Type_create_resized(tmp, 0, NB*NB*sizeof(double), &LOWER_TILE);
    MPI_Type_set_name(LOWER_TILE, "Lower");
    MPI_Type_commit(&LOWER_TILE);
    
    /* LITTLE_T is a NB*IB rectangle (containing IB*IB Lower tiles) */
    MPI_Type_contiguous(NB*IB, MPI_DOUBLE, &tmp);
    MPI_Type_create_resized(tmp, 0, NB*NB*sizeof(double), &LITTLE_T);
    MPI_Type_set_name(LITTLE_T, "T");
    MPI_Type_commit(&LITTLE_T);
    
    free(blocklens);
    free(indices);
#endif
}

#undef N
#undef NB


#if defined(DEBUG_MATRICES)
#if defined(DISTRIBUTED)
#define A(m,n) dague_get_local_tile_s(&ddescA, m, n)
#define L(m,n) dague_get_local_tile_s(&ddescL, m, n)
#define descA ddescA
#define descL ddescL
#else
#define A(m,n) &(((double*)descA.mat)[descA.bsiz*(m)+descA.bsiz*descA.lmt*(n)])
#define L(m,n) &(((double*)descL.mat)[descL.bsiz*(m)+descL.bsiz*descL.lmt*(n)])
#endif
#define MAXDBLSTRLEN 16

static void debug_matrices(void)
{
    int tilem, tilen;
    int m, n, len, pos;
    double *a;
    char *line;
#if defined(DISTRIBUTED)
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    int rank = 0;
#endif

    len = 32 + (MAXDBLSTRLEN + 1) * descA.nb;
    line = (char *)malloc( len );

    for(tilem = 0; tilem < descA.mt; tilem++) {
        for(tilen = 0; tilen < descA.nt; tilen++) {
#if defined(DISTRIBUTED)
            if( dague_get_rank_for_tile(&ddescA, tilem, tilen) == rank ) {
#endif
                a = A(tilem, tilen);
                fprintf(stderr, "[%d] A(%d, %d) = \n", rank, tilem, tilen);
                pos = 0;
                for(m = 0; m < descA.mb; m++) {
                    for(n = 0; n < descA.nb; n++) {
                        pos += snprintf(line + pos, len-pos, "%9.5f ", a[m + descA.mb * n]);
                    }
                    fprintf(stderr, "[%d]   %s\n", rank, line);
                    pos = 0;
                }
#if defined(DISTRIBUTED)
            }
            MPI_Barrier(MPI_COMM_WORLD);
#endif
        }
    }

    for(tilem = 0; tilem < descL.mt; tilem++) {
        for(tilen = 0; tilen < descL.nt; tilen++) {
#if defined(DISTRIBUTED)
            if( dague_get_rank_for_tile(&ddescL, tilem, tilen) == rank ) {
#endif
                a = L(tilem, tilen);
                fprintf(stderr, "[%d] dL(%d, %d) = \n", rank, tilem, tilen);
                pos = 0;
                for(m = 0; m < descL.mb; m++) {
                    for(n = 0; n < descL.nb; n++) {
                        pos += snprintf(line + pos, len-pos, "%9.5f ", a[m + descL.mb * n]);
                    }
                    fprintf(stderr, "[%d]   %s\n", rank, line);
                    pos = 0;
                }
#if defined(DISTRIBUTED)            }
            MPI_Barrier(MPI_COMM_WORLD);
#endif
        }
    }

    free(line);
}
#undef descA
#undef descL
#undef A
#undef L
#endif /* defined(DEBUG_MATRICES) */
