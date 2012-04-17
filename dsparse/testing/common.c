/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 */
#include "dague_config.h"
#include "dague.h"

#include "common.h"
#include "common_timing.h"
#include <plasma.h>

#include <stdlib.h>
#include <stdio.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#ifdef HAVE_LIMITS_H
#include <limits.h>
#endif
#if defined(HAVE_GETOPT_H)
#include <getopt.h>
#endif  /* defined(HAVE_GETOPT_H) */
#ifdef HAVE_MPI
#include <mpi.h>
#endif
#if defined(HAVE_CUDA)
#include "gpu_data.h"
#endif

#include "dague_prof_grapher.h"
#include "schedulers.h"

/*******************************
 * globals and argv set values *
 *******************************/
#if defined(HAVE_MPI)
MPI_Datatype SYNCHRO = MPI_BYTE;
#endif  /* HAVE_MPI */

static char *dot_filename = NULL;

double time_elapsed = 0.0;
double sync_time_elapsed = 0.0;

/**********************************
 * Command line arguments 
 **********************************/
void print_usage(void)
{
    fprintf(stderr,
            "Mandatory argument:\n"
            " filename          : Matrix filename (required)\n"
            "Optional arguments:\n"
            " -c --cores        : number of concurent threads (default: number of physical hyper-threads)\n"
            " -g --gpus         : number of GPU (default: 0)\n"
            " -o --scheduler    : select the scheduler (default: LFQ)\n"
            "                     Accepted values:\n"
            "                       LFQ -- Local Flat Queues\n"
            "                       GD  -- Global Dequeue\n"
            "                       LHQ -- Local Hierarchical Queues\n"
            "                       AP  -- Absolute Priorities\n"
            "\n"
            " -k --prio-switch  : activate prioritized DAG k steps before the end (default: 0)\n"
            "                   : with no argument, prioritized DAG from the start\n"
            "\n"
            "Matrix format:\n"
            " -0 --rsa          :  RSA format (use Fortran)\n" 
            " -1 --chb          :  CHB format\n"
            " -2 --ccc          :  CCC format\n"
            " -3 --rcc          :  RCC format\n"
            " -4 --olaf         :  OLAF format\n"
            " -5 --peer         :  PEER format\n"
            " -6 --hb           :  HB (double) format\n"
            " -7 --ijv          :  IJV 3files format\n"
            " -8 --mm           :  Matrix Market format\n"
            " -9 --lap          :  Generate a random Laplacian of specified size\n"
            "\n"
            " -n -N --rhs       : Right Hand side filename (default: ./rhsname)\n"
            " -p -P --order     : Ordering filename (default: ./ordername )\n"
            " -s -S --symbol    : Symbol factorization filename (default: ./symbname )\n"
            "\n"
            "Factorization:\n"
            " -f --facto        :  LL^t = 0, LDL^t = 1 and LU = 2 (default: LU)\n" 
            "\n"
            " -x --check        : verify the results\n"
            "\n"
            "    --dot          : create a dot output file (default: don't)\n"
            "\n"
            " -v --verbose      : extra verbose output\n"
            " -h --help         : this message\n"
           );
}

#define GETOPT_STRING "c:o:g::p:P:q:Q:k::n:N:o:O:xv::h0:1:2:3:4:5:6:7:8:9:f:"

#if defined(HAVE_GETOPT_LONG)
static struct option long_options[] =
{
    {"cores",       required_argument,  0, 'c'},
    {"c",           required_argument,  0, 'c'},
    {"o",           required_argument,  0, 'o'},
    {"scheduler",   required_argument,  0, 'o'},
    {"gpus",        required_argument,  0, 'g'},
    {"g",           required_argument,  0, 'g'},
    {"prio-switch", optional_argument,  0, 'k'},
    {"k",           optional_argument,  0, 'k'},

    {"0",           required_argument,  0, '0'},
    {"rsa",         required_argument,  0, '0'},
    {"1",           required_argument,  0, '1'},
    {"chb",         required_argument,  0, '1'},
    {"2",           required_argument,  0, '2'},
    {"ccc",         required_argument,  0, '2'},
    {"3",           required_argument,  0, '3'},
    {"rcc",         required_argument,  0, '3'},
    {"4",           required_argument,  0, '4'},
    {"olaf",        required_argument,  0, '4'},
    {"5",           required_argument,  0, '5'},
    {"peer",        required_argument,  0, '5'},
    {"6",           required_argument,  0, '6'},
    {"hb",          required_argument,  0, '6'},
    {"7",           required_argument,  0, '7'},
    {"ijv",         required_argument,  0, '7'},
    {"8",           required_argument,  0, '8'},
    {"mm",          required_argument,  0, '8'},
    {"9",           required_argument,  0, '9'},
    {"lap",         required_argument,  0, '9'},

    {"f",           required_argument,  0, 'f'},
    {"facto",       required_argument,  0, 'f'},

    {"n",           required_argument,  0, 'n'},
    {"N",           required_argument,  0, 'n'},
    {"rhs",         required_argument,  0, 'n'},
    {"p",           required_argument,  0, 'p'},
    {"P",           required_argument,  0, 'p'},
    {"order",       required_argument,  0, 'p'},
    {"s",           required_argument,  0, 's'},
    {"S",           required_argument,  0, 's'},
    {"symbol",      required_argument,  0, 's'},
    {"check",       no_argument,        0, 'x'},
    {"x",           no_argument,        0, 'x'},

    {"dot",         required_argument,  0, '.'},

    {"verbose",     optional_argument,  0, 'v'},
    {"v",           optional_argument,  0, 'v'},
    {"help",        no_argument,        0, 'h'},
    {"h",           no_argument,        0, 'h'},
    {0, 0, 0, 0}
};
#endif  /* defined(HAVE_GETOPT_LONG) */

static void parse_arguments(int argc, char** argv, int* iparam, char** sparam) 
{
    int opt = 0;
    int c;

    do
    {
#if defined(HAVE_GETOPT_LONG)
        c = getopt_long_only(argc, argv, "",
                        long_options, &opt);
#else
        c = getopt(argc, argv, GETOPT_STRING);
        (void) opt;
#endif  /* defined(HAVE_GETOPT_LONG) */
    
 //       printf("%c: %s = %s\n", c, long_options[opt].name, optarg);
        switch(c)
        {
            case 'c': iparam[IPARAM_NCORES] = atoi(optarg); break;
            case 'o': 
                if( !strcmp(optarg, "LFQ") )
                    iparam[IPARAM_SCHEDULER] = DAGUE_SCHEDULER_LFQ;
                else if( !strcmp(optarg, "AP") )
                    iparam[IPARAM_SCHEDULER] = DAGUE_SCHEDULER_AP;
                else if( !strcmp(optarg, "LHQ") )
                    iparam[IPARAM_SCHEDULER] = DAGUE_SCHEDULER_LHQ;
                else if( !strcmp(optarg, "GD") )
                    iparam[IPARAM_SCHEDULER] = DAGUE_SCHEDULER_GD;
                else {
                    fprintf(stderr, "malformed scheduler value %s (accepted: LFQ AP LHQ GD). Reverting to default LFQ\n",
                            optarg);
                    iparam[IPARAM_SCHEDULER] = DAGUE_SCHEDULER_LFQ;
                }
                break;

            case 'g':
                if(iparam[IPARAM_NGPUS] == -1)
                {
                    fprintf(stderr, "!!! This test does not have GPU support. GPU disabled.\n");
                    break;
                }
                if(optarg)  iparam[IPARAM_NGPUS] = atoi(optarg);
                else        iparam[IPARAM_NGPUS] = INT_MAX;
                break;
            case 'k':
                if(optarg)  iparam[IPARAM_PRIO] = atoi(optarg);
                else        iparam[IPARAM_PRIO] = INT_MAX;
                break;
            
            case '0': iparam[IPARAM_FORMAT] = RSA;        sparam[SPARAM_MATRIX] = strdup(optarg); break;
            case '1': iparam[IPARAM_FORMAT] = CHB;        sparam[SPARAM_MATRIX] = strdup(optarg); break;
            case '2': iparam[IPARAM_FORMAT] = CCC;        sparam[SPARAM_MATRIX] = strdup(optarg); break;
            case '3': iparam[IPARAM_FORMAT] = RCC;        sparam[SPARAM_MATRIX] = strdup(optarg); break;
            case '4': iparam[IPARAM_FORMAT] = OLAF;       sparam[SPARAM_MATRIX] = strdup(optarg); break;
            case '5': iparam[IPARAM_FORMAT] = PEER;       sparam[SPARAM_MATRIX] = strdup(optarg); break;
            case '6': iparam[IPARAM_FORMAT] = HB;         sparam[SPARAM_MATRIX] = strdup(optarg); break;
            case '7': iparam[IPARAM_FORMAT] = THREEFILES; sparam[SPARAM_MATRIX] = strdup(optarg); break;
            case '8': iparam[IPARAM_FORMAT] = MM;         sparam[SPARAM_MATRIX] = strdup(optarg); break;
            case '9': iparam[IPARAM_FORMAT] = LAPLACIAN;  iparam[IPARAM_M] = atoi(optarg); break;
            case 'n': sparam[SPARAM_RHS]      = strdup(optarg); break;
            case 'p': sparam[SPARAM_ORDERING] = strdup(optarg); break;
            case 's': sparam[SPARAM_SYMBOL]   = strdup(optarg); break;
            case 'x': iparam[IPARAM_CHECK] = 1; iparam[IPARAM_VERBOSE] = max(2, iparam[IPARAM_VERBOSE]); break; 
            case '.': iparam[IPARAM_DOT] = 1; dot_filename = strdup(optarg); break;

            case 'f': iparam[IPARAM_FACTORIZATION] = atoi(optarg); break;

            case 'v': 
                if(optarg)  iparam[IPARAM_VERBOSE] = atoi(optarg);
                else        iparam[IPARAM_VERBOSE] = 2;
                break;
            case 'h': print_usage(); exit(0);
            
            case '?': /* getopt_long already printed an error message. */
                exit(1);
            default:
                break; /* Assume anything else is dague/mpi stuff */
        }
    } while(-1 != c);
    int verbose = iparam[IPARAM_RANK] ? 0 : iparam[IPARAM_VERBOSE];
    
    /* Set some sensible default to the number of cores */
    if(iparam[IPARAM_NCORES] <= 0)
    {
        iparam[IPARAM_NCORES] = sysconf(_SC_NPROCESSORS_ONLN);
        if(iparam[IPARAM_NCORES] == -1)
        {
            perror("sysconf(_SC_NPROCESSORS_ONLN)\n");
            iparam[IPARAM_NCORES] = 1;
        }
        if(verbose) 
            fprintf(stderr, "+++ cores detected      : %d\n", iparam[IPARAM_NCORES]);
    }
    if(iparam[IPARAM_NGPUS] < 0) iparam[IPARAM_NGPUS] = 0;
    
    if(verbose > 1) 
        fprintf(stderr, "+++ nodes x cores + gpu : %d x %d + %d (%d+%d)\n",
                iparam[IPARAM_NNODES],
                iparam[IPARAM_NCORES],
                iparam[IPARAM_NGPUS],
                iparam[IPARAM_NNODES] * iparam[IPARAM_NCORES],
                iparam[IPARAM_NNODES] * iparam[IPARAM_NGPUS]); 

    /* Set matrices dimensions to default values if not provided */
    /* Search for N as a bare number if not provided by -N */
    while(0 == sparam[SPARAM_MATRIX])
    {
        if(optind < argc)
        {
            sparam[SPARAM_MATRIX] = strdup( argv[optind++] );
            continue;
        }
        break;
    }
     
    if(sparam[SPARAM_MATRIX]   == NULL ) sparam[SPARAM_MATRIX]   = strdup("./rsaname");
    if(sparam[SPARAM_RHS]      == NULL ) sparam[SPARAM_RHS]      = strdup("./rhsname");
    if(sparam[SPARAM_ORDERING] == NULL ) sparam[SPARAM_ORDERING] = strdup("./ordername");
    if(sparam[SPARAM_SYMBOL]   == NULL ) sparam[SPARAM_SYMBOL]   = strdup("./symbname");

    if(verbose > 1)
    {
        fprintf(stderr, "+++ Matrix              : %s\n", sparam[SPARAM_MATRIX]);
        if ( sparam[SPARAM_MATRIX] ) {
            fprintf(stderr, "+++ RHS                 : %s\n", sparam[SPARAM_RHS]);
        } else {
            fprintf(stderr, "+++ RHS                 : B= A*x with x=1\n" );
        }
        fprintf(stderr, "+++ Ordering            : %s\n", sparam[SPARAM_ORDERING]);
        fprintf(stderr, "+++ Symbol              : %s\n", sparam[SPARAM_SYMBOL]);
    }
}

void param_default(int* iparam, char **sparam)
{
    /* Just in case someone forget to add the initialization :) */
    memset(iparam, 0, IPARAM_SIZEOF * sizeof(int)); 
    memset(sparam, 0, SPARAM_SIZEOF * sizeof(char*)); 
    iparam[IPARAM_NNODES] = 1;
    iparam[IPARAM_NGPUS]  = 0;
    iparam[IPARAM_FORMAT] = RSA;
    iparam[IPARAM_FACTORIZATION] = DSPARSE_LU;
}

#ifdef DAGUE_PROF_TRACE
static char* argvzero;
#endif

dague_context_t* setup_dague(int argc, char **argv, int *iparam, char **sparam)
{
#ifdef DAGUE_PROF_TRACE
    argvzero = argv[0];
#endif
#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &iparam[IPARAM_NNODES]);
    MPI_Comm_rank(MPI_COMM_WORLD, &iparam[IPARAM_RANK]); 
#else
    iparam[IPARAM_NNODES] = 1;
    iparam[IPARAM_RANK] = 0;
#endif
    parse_arguments(argc, argv, iparam, sparam);
    int verbose = iparam[IPARAM_VERBOSE];
    if(iparam[IPARAM_RANK] > 0 && verbose < 4) verbose = 0;
    
    TIME_START();
    dague_context_t* ctx = dague_init(iparam[IPARAM_NCORES], &argc, &argv);
    /* If the number of cores has not been defined as a parameter earlier
     update it with the default parameter computed in dague_init. */
    if(iparam[IPARAM_NCORES] <= 0)
    {
        iparam[IPARAM_NCORES] = ctx->nb_cores;
    }

#if defined(HAVE_CUDA)
    if(iparam[IPARAM_NGPUS] > 0)
    {
        if(0 != dague_gpu_init(&iparam[IPARAM_NGPUS], 0))
        {
            fprintf(stderr, "xxx DAGuE is unable to initialize the CUDA environment.\n");
            exit(3);
        }
    }
#endif

#if defined(DAGUE_PROF_GRAPHER)
    if(iparam[IPARAM_DOT] != 0) {
        dague_prof_grapher_init(dot_filename, iparam[IPARAM_RANK], iparam[IPARAM_NNODES], iparam[IPARAM_NCORES]);
    }
#else
    (void)dot_filename;
    if(iparam[IPARAM_DOT] != 0) {
        fprintf(stderr, 
                "************************************************************************************************\n"
                "*** Warning: dot generation requested, but DAGUE configured with DAGUE_PROF_GRAPHER disabled ***\n"
                "************************************************************************************************\n");
    }
#endif

    dague_set_scheduler( ctx, dague_schedulers_array[ iparam[IPARAM_SCHEDULER] ] );  

    if(verbose > 2) TIME_PRINT(iparam[IPARAM_RANK], ("DAGuE initialized\n"));
    return ctx;
}

void cleanup_dague(dague_context_t* dague, int *iparam, char **sparam)
{
    int i;
#ifdef DAGUE_PROF_TRACE
    char* filename = NULL;
#if defined(HAVE_MPI)
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    asprintf(&filename, "%s.%d.profile", argvzero, rank);
#else
    asprintf(&filename, "%s.profile", argvzero);
#endif
    dague_profiling_dump_dbp(filename);
    free(filename);
#endif  /* DAGUE_PROF_TRACE */

#if defined(HAVE_CUDA)
    if( iparam[IPARAM_NGPUS] > 0 ) {
        if( 0 != dague_gpu_fini() ) {
            fprintf(stderr, "xxx DAGuE is unable to finalize the CUDA environment.\n");
        }
    }
#endif  /* defined(HAVE_CUDA) */
    dague_fini(&dague);

    for(i=0; i < SPARAM_SIZEOF; i++) {
        if( sparam[i] != NULL )
            free(sparam[i]);
    }

#if defined(DAGUE_PROF_GRAPHER)
    if(iparam[IPARAM_DOT] != 0) {
        dague_prof_grapher_fini();
    }
#else
    (void)iparam;
#endif
#ifdef HAVE_MPI
    MPI_Finalize();
#endif    
}

