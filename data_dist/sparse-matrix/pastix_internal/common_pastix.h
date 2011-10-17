/* 
   File: common_pastix.h

   Part of a parallel direct block solver.
   
   These lines are the common data        
   declarations for all modules.          

   Authors:
     Mathieu Faverge    - faverge@labri.fr
     David    GOUDIN     - .
     Pascal   HENON      - henon@labri.fr
     Xavier   LACOSTE    - lacoste@labri.fr
     Francois PELLEGRINI - .
     Pierre   RAMET      - ramet@labri.fr
     
   Dates:
     Version 0.0 - from 08 may 1998    
                   to   08 jan 2001
     Version 1.0 - from 06 jun 2002
                   to   06 jun 2002
*/
#ifndef COMMON_PASTIX_H
#define COMMON_PASTIX_H
#include "api.h"
#include "debug.h"
#include "errors.h"

#ifdef __INTEL_COMPILER
/* Ignore icc remark : "operands are evaluated in unspecified order"*/
#pragma warning(disable:981)
/* Ignore icc remark : "external function definition with no prior declaration" */
#pragma warning(disable:1418)
/* Ignore icc remark : "external declaration in primary source file" */
#pragma warning(disable:1419)
/* Ignore icc remark : " parameter "arg" was never referenced" */
#pragma warning(disable:869)
/* Ignore icc remark : "variable "size" was set but never used" */
#pragma warning(disable:593)
/* Ignore icc remark : "floating-point equality and inequality comparisons are unreliable" */
#pragma warning(disable:1572)
/* Ignore icc remark : "statement is unreachable" */
#pragma warning(disable:111)
#endif

#ifdef OOC_FTGT
#ifndef OOC_FTGT_RESET
#define OOC_FTGT_RESET
#endif
#ifndef OOC
#define OOC
#endif
#endif

#ifdef OOC
#ifndef MEMORY_USAGE
#define MEMORY_USAGE
#endif
#ifndef NUMA_ALLOC /* OOC needs NUMA_ALLOC to have a vector of pointers on each cblk */
#define NUMA_ALLOC
#endif
#endif


/*
** Machine configuration values.
** The end of the X_ARCH variable is built with parts of the
** `uname -m`, `uname -r`, and `uname -s` commands.
*/

#define X_C_NORESTRICT
#ifndef X_C_NORESTRICT
#define X_C_RESTRICT
#endif /* X_C_NORESTRICT */

#if (defined X_ARCHi686_mac)
#define X_ARCHi686_pc_linux
#endif

#if (defined X_ARCHpower_ibm_aix)
#define X_INCLUDE_ESSL
#undef  X_C_RESTRICT
#endif /* (defined X_ARCHpower_ibm_aix) */

#if (defined X_ARCHalpha_compaq_osf1)
#define restrict
/*#define volatile*/
#endif /* (defined X_ARCHalpha_compaq_osf1) */


/*
** Compiler optimizations.
*/

#ifdef X_C_RESTRICT
#ifdef __GNUC__
#define restrict                    __restrict
#endif /* __GNUC__ */
#else /* X_C_RESTRICT */
#define restrict
#endif /* X_C_RESTRICT */

/*
** The includes.
*/

/* Redefinition de malloc,free,printf,fprintf */
#ifdef MARCEL
#include <pthread.h>
#endif

#include            <ctype.h>
#include            <math.h>
#ifdef X_ARCHi686_mac
#include            <malloc/malloc.h>
#else /* X_ARCHi686_mac */
#include            <malloc.h>
#endif /* X_ARCHi686_mac */
#include            <memory.h>
#include            <stdio.h>
#include            <stdarg.h>
#include            <stdlib.h>
#include            <string.h>
#include            <time.h>         /* For the effective calls to clock () */
#include            <limits.h>
#include            <sys/types.h>
#include            <sys/time.h>
#include            <sys/resource.h>
#include            <unistd.h>
#include            <float.h>
#ifdef X_INCLUDE_ESSL
#include            <essl.h>
#endif /* X_INCLUDE_ESSL */

#ifdef X_ASSERT
#include <assert.h>
#endif /* X_ASSERT */

#define MIN(x,y) (((x)<(y))?(x):(y))
#define MAX(x,y) (((x)<(y))?(y):(x))

/*
**  Handling of generic types.
*/

#define byte unsigned char                        /*+ Byte type +*/

#ifdef TYPE_COMPLEX
#ifndef FORCE_COMPLEX
#define FORCE_COMPLEX
#endif
#endif

#ifdef FORCE_COMPLEX
#ifndef TYPE_COMPLEX
#define TYPE_COMPLEX
#endif
#endif

#ifdef PREC_DOUBLE
#ifndef FORCE_DOUBLE
#define FORCE_DOUBLE
#endif
#endif

#ifdef FORCE_DOUBLE
#ifndef PREC_DOUBLE
#define PREC_DOUBLE
#endif
#endif

#ifdef INTSIZE32
#ifndef FORCE_INT32
#define FORCE_INT32
#endif

#endif

#if (defined INTSIZE64 || defined INTSSIZE64)
#ifndef FORCE_INT64
#define FORCE_INT64
#endif
#endif

#ifdef FORCE_INT32
#ifndef INTSIZE32
#define INTSIZE32
#endif
#endif


#ifdef FORCE_INT64
#if !(defined INTSIZE64) && !(defined INTSSIZE64)
#define INTSIZE64
#endif
#endif


#ifdef FORCE_DOUBLE
#define BLAS_DOUBLE
#define BASE_FLOAT double
#else
#define BASE_FLOAT float
#endif

#ifdef FORCE_COMPLEX
#define CPLX
#endif

#ifdef CPLX
#if (defined X_ARCHalpha_compaq_osf1)

#ifndef USE_CXX

#ifndef   _RWSTD_HEADER_REQUIRES_HPP
#include <complex>
#else  /* _RWSTD_HEADER_REQUIRES_HPP */
#include <complex.hpp>
#endif /* _RWSTD_HEADER_REQUIRES_HPP */

#define FLOAT complex<BASE_FLOAT>

#ifdef    FORCE_DOUBLE
#define COMM_FLOAT MPI_DOUBLE_COMPLEX
#define FLOAT_MAX DBL_MAX
#else  /* FORCE_DOUBLE */
#define COMM_FLOAT MPI_COMPLEX
#define FLOAT_MAX MAXFLOAT
#endif /* FORCE_DOUBLE */

#define COMM_SUM GetMpiSum()
#define ABS_FLOAT(x) abs(x)
#define fabs(x) abs(x)
#define cabs(x) abs(x)
#define csqrt(x) sqrt(x)
#define CONJ_FLOAT(x) conj(x)
#define creal(x) real(x)
#define cimag(x) imag(x)
#endif /*USE_CXX*/

#else /*X_ARCHalpha_compaq_osf1*/
#include <complex.h>
#define COMM_FLOAT GetMpiType()
#define COMM_SUM GetMpiSum()

#ifdef    FORCE_DOUBLE
#define FLOAT double complex
#define ABS_FLOAT(x) cabs(x)
#ifdef    _DCMPLX
#define BLAS_FLOAT dcmplx 
#else  /* _DCMPLX */
#define BLAS_FLOAT double complex
#endif /* _DCMPLX */
#define CONJ_FLOAT(x) conj(x) 
#define FLOAT_MAX DBL_MAX
#else  /* FORCE_DOUBLE */
#define FLOAT float complex
#define ABS_FLOAT(x) cabsf(x)
#ifdef    _CMPLX
#define BLAS_FLOAT cmplx
#else  /* _CMPLX */
#define BLAS_FLOAT float complex
#endif /* _CMPLX */
#define CONJ_FLOAT(x) conjf(x)
#define FLOAT_MAX MAXFLOAT
#endif /* FORCE_DOUBLE */

#endif /* X_ARCHalpha_compaq_osf1 */
#else /* CPLX */
#define COMM_SUM MPI_SUM

#ifdef FORCE_DOUBLE
#define FLOAT double
#define ABS_FLOAT(x) fabs(x)
#define CONJ_FLOAT(x) x
#define COMM_FLOAT MPI_DOUBLE
#define FLOAT_MAX DBL_MAX
#else /* FORCE_DOUBLE */
#define FLOAT float
#define FLOAT_MAX MAXFLOAT
#define COMM_FLOAT MPI_FLOAT
#define ABS_FLOAT(x) fabsf(x)
#define CONJ_FLOAT(x) x
#endif /* FORCE_DOUBLE */

#endif /* CPLX */

/*
 *  Définition de la taille des entiers utilisés
 */
#ifdef FORCE_LONG
#define INT           long          /* Long integer type */
#define UINT          unsigned long
#define COMM_INT      MPI_LONG
#elif (defined FORCE_INT32)
#define INT           int32_t
#define UINT          u_int32_t
#define COMM_INT      MPI_INTEGER4
#elif (defined FORCE_INT64)
#define INT           int64_t
#define UINT          u_int64_t
#define COMM_INT      MPI_INTEGER8
#else
#define INT           int           /* Default integer type     */
#define UINT          unsigned int
#define COMM_INT      MPI_INT       /* Generic MPI integer type */
#endif

#ifndef INTSIZEBITS
#define INTSIZEBITS   (sizeof (INT) << 3)
#endif /* INTSIZEBITS */

#define INTVALMAX     ((INT) (((UINT) 1 << (INTSIZEBITS - 1)) - 1))

/*#include "redefine_functions.h"*/

/*
**  Working definitions.
*/
#define memAlloca(size)                    alloca(size)
#ifndef MEMORY_USAGE                            /* TODO : remove mutex protection in multi-thread mode */
#ifdef X_ARCHpower_ibm_aix
#define memAlloc(size)                     mymalloc(size, __FILE__,__LINE__)
#else
#define memAlloc(size)                     malloc(size)
#endif
#define memAlloca(size)                    alloca(size)
#define memRealloc(ptr,size)               realloc((ptr),(size))
#define memFree(ptr)                       ( free ((char *) (ptr)) , 0)
#else
#define memAlloc(size)                     (memAlloc_func(size,__FILE__,__LINE__))
#endif
#define memFree_null(ptr)                  ( memFree ((char *) (ptr)) , (ptr) = NULL , 0)

#define memFreea(ptr,module)               (0)           /* Freeing function if memAlloca implemented as malloc */
#define memSet(ptr,val,siz)                memset((ptr),(val),(siz));
#define memCpy(dst,src,siz)                memcpy((dst),(src),(siz));
#define memMov(dst,src,siz)                memmove((dst),(src),(siz));

#ifdef WARNINGS_MALLOC
/*
  Macro: MALLOC_EXTERN
  
  Allocate a space of size *size* x sizeof(*type*) 
  at the adress indicated by ptr, using external *malloc*.

  Parameters:
    ptr   - address where to allocate.
    size  - Number of elements to allocate.
    types - Type of the elements to allocate.
*/
#define MALLOC_EXTERN(ptr, size, type)					\
  {									\
    if (ptr != NULL)							\
      errorPrintW("non NULL pointer in allocation (line=%d,file=%s)",	\
		   __LINE__,__FILE__);					\
    if ((size) * sizeof(type))						\
      errorPrintW("Allocation of size 0 (line=%d,file=%s)",		\
		  __LINE__,__FILE__);					\
    if (NULL == (ptr = (type *) malloc((size) * sizeof(type))))		\
      {									\
	MALLOC_ERROR(#ptr);						\
      }									\
  }


/*
  Macro: MALLOC_INTERN
  
  Allocate a space of size *size* x sizeof(*type*) 
  at the adress indicated by ptr, using internal *memAlloc*.

  Parameters:
    ptr   - address where to allocate.
    size  - Number of elements to allocate.
    types - Type of the elements to allocate.
*/
#define MALLOC_INTERN(ptr, size, type)					\
  {									\
    if (ptr != NULL)							\
      errorPrintW("non NULL pointer in allocation (line=%d,file=%s)",	\
		  __LINE__,__FILE__);					\
    if ((size) * sizeof(type))						\
      errorPrintW("Allocation of size 0 (line=%d,file=%s)",		\
		  __LINE__,__FILE__);					\
    if (((size) * sizeof(type) > 0) &&					\
	(NULL == (ptr = (type *) memAlloc((size) * sizeof(type)))))	\
      {									\
	MALLOC_ERROR(#ptr);						\
      }									\
  }
#else /* WARNINGS_MALLOC */
/*
  Macro: MALLOC_EXTERN
  
  Allocate a space of size *size* x sizeof(*type*) 
  at the adress indicated by ptr, using external *malloc*.

  Parameters:
    ptr   - address where to allocate.
    size  - Number of elements to allocate.
    types - Type of the elements to allocate.
*/
#define MALLOC_EXTERN(ptr, size, type)					\
  {									\
    ptr = NULL;								\
    if (((size) * sizeof(type) > 0) &&					\
	(NULL == (ptr = (type *) malloc((size) * sizeof(type)))))	\
      {									\
	MALLOC_ERROR(#ptr);						\
      }									\
  }


/*
  Macro: MALLOC_INTERN
  
  Allocate a space of size *size* x sizeof(*type*) 
  at the adress indicated by ptr, using internal *memAlloc*.

  Parameters:
    ptr   - address where to allocate.
    size  - Number of elements to allocate.
    types - Type of the elements to allocate.
*/
#define MALLOC_INTERN(ptr, size, type)					\
  {									\
    ptr = NULL;								\
    if (((size) * sizeof(type) > 0) &&					\
	(NULL == (ptr = (type *) memAlloc((size) * sizeof(type)))))	\
      {									\
	MALLOC_ERROR(#ptr);						\
      }									\
  }
#endif /* WARNINGS_MALLOC */
/*
  Macro: FOPEN 

  Open a file and handle errors.

  Parameters:
    FILE      - Stream (FILE*) to link to the file.
    filenamne - String containing the path to the file.
    mode      - String containing the opening mode. 
  
*/
#define FOPEN(FILE, filenamne, mode)					\
  {									\
    FILE = NULL;							\
    if (NULL == (FILE = fopen(filenamne, mode)))			\
      {									\
	errorPrint("%s:%d Couldn't open file : %s with mode %s\n",	\
		   __FILE__, __LINE__, filenamne, mode);		\
	EXIT(MOD_UNKNOWN,FILE_ERR);					\
      }									\
  }
/*
  Macro: FREAD

  Calls fread function and test his return value

  Parameters:
    buff   - Memory area where to copy read data.
    size   - Size of an element to read.
    count  - Number of elements to read
    stream - Stream to read from
 */
#define FREAD(buff, size, count, stream)		\
  {							\
    if ( 0 == fread(buff, size, count, stream))		\
      {							\
	errorPrint("%s:%d fread error\n",		\
		   __FILE__, __LINE__);			\
	EXIT(MOD_UNKNOWN,FILE_ERR);			\
      }							\
  }
/*
 * Other working definitions
 */

#define MAX_CHAR_PER_LINE 1000


/*
**  Handling of timers.
*/

/** The clock type. **/

typedef struct Clock_ {
  double                    time[2];              /*+ The start and accumulated times +*/
} Clock;

/*
**  Handling of files.
*/

/** The file structure. **/

typedef struct File_ {
  char *                    name;                 /*+ File name    +*/
  FILE *                    pntr;                 /*+ File pointer +*/
  char *                    mode;                 /*+ Opening mode +*/
} File;

/*
**  The function prototypes.
*/

#ifdef X_ARCHalpha_compaq_osf1
#ifndef USE_CXX
extern "C" {
#endif
#endif

#ifdef MEMORY_USAGE
void *                      memAlloc_func       (size_t,char*,int);
void *                      memRealloc          (void *, size_t);
void                        memFree             (void *);
unsigned long               memAllocGetCurrent  (void);
unsigned long               memAllocGetMax      (void);
void                        memAllocTraceReset  (void);
#else
void *                      mymalloc            (size_t,char*,int);
#endif /* MEMORY_USAGE */
#ifdef MEMORY_TRACE
void                        memAllocTrace       (FILE *, double, int);
void                        memAllocUntrace     ();
#else
#define                     memAllocTrace(a, b, c) {}
#define                     memAllocUntrace()      {}
#endif
void *                      memAllocGroup       (void **, ...);
void *                      memReallocGroup     (void *, ...);
void *                      memOffset           (void *, ...);

void                        usagePrint          (FILE * const, const char ** const);

void                        errorProg           (const char * const);
void                        errorPrint          (const char * const, ...);
void                        errorPrintW         (const char * const, ...);

int                         intLoad             (FILE * const, INT * const);
int                         intSave             (FILE * const, const INT);
void                        intAscn             (INT * restrict const, const INT, const INT);
void                        intPerm             (INT * restrict const, const INT);
void                        intRandInit         (void);
INT                         intRandVal          (INT);
void                        intSort1asc1        (void * const, const INT);
void                        intSort2asc1        (void * const, const INT);
void                        intSort2asc2        (void * const, const INT);

void                        clockInit           (Clock * const);
void                        clockStart          (Clock * const);
void                        clockStop           (Clock * const);
double                      clockVal            (Clock * const);
double                      clockGet            (void);

#ifdef X_ARCHalpha_compaq_osf1
#ifndef USE_CXX
}
#endif
#endif

/*
**  The macro definitions.
*/

#define clockInit(clk)              ((clk)->time[0]  = (clk)->time[1] = 0)
#define clockStart(clk)             ((clk)->time[0]  = clockGet ())
#define clockStop(clk)              ((clk)->time[1]  = clockGet ())
#define clockVal(clk)               ((clk)->time[1] - (clk)->time[0])

#define intRandVal(ival)            ((INT) (((UINT) random ()) % ((UINT) (ival))))

#define FORTRAN(nu,nl,pl,pc)                     \
void nu pl;                                      \
void nl pl                                       \
{ nu pc; }                                       \
void nl##_ pl                                    \
{ nu pc; }                                       \
void nl##__ pl                                   \
{ nu pc; }                                       \
void nu pl

#ifdef MARCEL
#define marcel_printf(...) do {  } while(0)
/* #define marcel_printf(...) marcel_fprintf(stderr, __VA_ARGS__) */
#else
#define printf(...) do {  } while(0)
/* #define printf(...) fprintf(stderr, __VA_ARGS__) */
#endif

void api_dumparm(FILE *stream, INT *iparm, double *dparm);
int  api_dparmreader(char * filename, double *dparmtab);
int  api_iparmreader(char * filename, INT    *iparmtab);
void set_iparm(INT    *iparm, enum IPARM_ACCESS offset, INT    value);
void set_dparm(double *dparm, enum DPARM_ACCESS offset, double value);

/*
   Function: qsortIntFloatAsc

   Sort 2 arrays simultaneously, the first array is an
   array of INT and used as key for sorting.
   The second array is an array of FLOAT.

   Parameters:
     pbase       - Array of pointers to the first element of each array to sort.
     total_elems - Number of element in each array.

   Returns:
     Nothing

*/
void qsortIntFloatAsc(void ** const pbase,
		      const INT     total_elems);

/*
   Function: qsort2IntFloatAsc

   Sort 3 arrays simultaneously, the first array is an
   array of INT and used as primary key for sorting.
   The second array is an other array of INT used
   as secondary key.
   The third array is an array of FLOAT.

   Parameters:
     pbase       - Array of pointers to the first element of each array to sort.
     total_elems - Number of element in each array.

   Returns:
     Nothing

*/
void qsort2IntFloatAsc(void ** const pbase,
		       const INT     total_elems);


/*
   Function: qsort2IntAsc

   Sort 2 arrays simultaneously, the first array is an
   array of INT and used as primary key for sorting.
   The second array is an other array of INT used
   as secondary key.

   Parameters:
     pbase       - Array of pointers to the first element of each array to sort.
     total_elems - Number of element in each array.

   Returns:
     Nothing

*/
void qsort2IntAsc(void ** const pbase,
		  const INT     total_elems);

/*
   Function: qsort2SmallIntAsc

   Sort 2 arrays simultaneously, the first array is an
   array of integers (int) and used as primary key for sorting.
   The second array is an other array of int used
   as secondary key.

   Parameters:
     pbase       - Array of pointers to the first element of each array to sort.
     total_elems - Number of element in each array.

   Returns:
     Nothing

*/
void qsort2SmallIntAsc(void ** const pbase,
		  const INT     total_elems);

#define MEMORY_WRITE(mem) ((mem < 1<<10)?mem:((mem < 1<<20 )?mem/(1<<10):((mem < 1<<30 )?(double)mem/(double)(1<<20):(double)mem/(double)(1<<30))))
#define MEMORY_UNIT_WRITE(mem) ((mem < 1<<10)?"o":((mem < 1<<20 )?"Ko":((mem < 1<<30 )?"Mo":"Go")))


/* 
 * Macro to write in file pastix.pid/fname 
 * Don't forget to close the FILE out after 
 */
#define OUT_OPENFILEINDIR(iparm, file, fname, mode)	\
  {							\
    char  outdirname[255];				\
    char  outfilename[255];				\
    sprintf(outdirname, "./pastix.%d", (int)(iparm)[IPARM_PID]);	\
    mkdir(outdirname, 0755);				\
    sprintf(outfilename, "%s/%s", outdirname, fname);	\
    file = fopen(outfilename, mode);			\
  }
#define OUT_CLOSEFILEINDIR(file) fclose(file);

#define PASTIX_MASK_ISTRUE(var, mask) (var == (var | mask))


#endif /* COMMON_PASTIX_H */
