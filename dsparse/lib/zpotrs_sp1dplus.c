#line 2 "zpotrs_sp1dplus.jdf"
  /**
   * PLASMA include for defined and constants.
   *
   * @precisions normal z -> s d c
   *
   */
#include <plasma.h>
#include <core_blas.h>

#include "dague.h"
#include "data_distribution.h"
#include "memory_pool.h"
#include "dplasma/lib/dplasmajdf.h"

#include "data_dist/sparse-matrix/pastix_internal/pastix_internal.h"
#include "data_dist/sparse-matrix/sparse-matrix.h"
#include "dsparse/lib/core_z.h"

#define PRECISION_z


#line 24 "zpotrs_sp1dplus.c"
#include <dague.h>
#include <scheduling.h>
#include <remote_dep.h>
#if defined(HAVE_PAPI)
#include <papime.h>
#endif
#include "zpotrs_sp1dplus.h"

#define DAGUE_zpotrs_sp1dplus_NB_FUNCTIONS 4
#define DAGUE_zpotrs_sp1dplus_NB_DATA 2
#if defined(DAGUE_PROF_TRACE)
int zpotrs_sp1dplus_profiling_array[2*DAGUE_zpotrs_sp1dplus_NB_FUNCTIONS] = {-1};
#define TAKE_TIME(context, key, eid, refdesc, refid) do {   \
   dague_profile_ddesc_info_t info;                         \
   info.desc = (dague_ddesc_t*)refdesc;                     \
   info.id = refid;                                         \
   dague_profiling_trace(context->eu_profile,               \
                         __dague_object->super.super.profiling_array[(key)],\
                         eid, (void*)&info);                \
  } while(0);
#else
#define TAKE_TIME(context, key, id, refdesc, refid)
#endif
#include "dague_prof_grapher.h"
#include <mempool.h>
typedef struct __dague_zpotrs_sp1dplus_internal_object {
 dague_zpotrs_sp1dplus_object_t super;
  /* The list of data repositories */
  data_repo_t *GEMM_BACKWARD_repository;
  data_repo_t *TRSM_BACKWARD_repository;
  data_repo_t *GEMM_FORWARD_repository;
  data_repo_t *TRSM_FORWARD_repository;
} __dague_zpotrs_sp1dplus_internal_object_t;

/* Globals */
#define descA (__dague_object->super.descA)
#define descB (__dague_object->super.descB)
#define datacode (__dague_object->super.datacode)
#define cblknbr (__dague_object->super.cblknbr)
#define bloknbr (__dague_object->super.bloknbr)
#define browsize (__dague_object->super.browsize)
#define p_work (__dague_object->super.p_work)

/* Data Access Macros */
#define A(A0)  (((dague_ddesc_t*)__dague_object->super.A)->data_of((dague_ddesc_t*)__dague_object->super.A, (A0)))

#define B(B0)  (((dague_ddesc_t*)__dague_object->super.B)->data_of((dague_ddesc_t*)__dague_object->super.B, (B0)))


/* Functions Predicates */
#define GEMM_BACKWARD_pred(k, bloknum, fcblk, cblk, gcblk2list, firstbrow, lastbrow) (((dague_ddesc_t*)(__dague_object->super.B))->myrank == ((dague_ddesc_t*)(__dague_object->super.B))->rank_of((dague_ddesc_t*)__dague_object->super.B, cblk))
#define TRSM_BACKWARD_pred(k, gcblk2list, firstbrow, lastbrow, firstblok) (((dague_ddesc_t*)(__dague_object->super.B))->myrank == ((dague_ddesc_t*)(__dague_object->super.B))->rank_of((dague_ddesc_t*)__dague_object->super.B, k))
#define GEMM_FORWARD_pred(k, fcblk, cblk, phony, prev, next) (((dague_ddesc_t*)(__dague_object->super.B))->myrank == ((dague_ddesc_t*)(__dague_object->super.B))->rank_of((dague_ddesc_t*)__dague_object->super.B, fcblk))
#define TRSM_FORWARD_pred(k, gcblk2list, browk1, lastbrow, firstblok, lastblok, firstgemm) (((dague_ddesc_t*)(__dague_object->super.B))->myrank == ((dague_ddesc_t*)(__dague_object->super.B))->rank_of((dague_ddesc_t*)__dague_object->super.B, k))

/* Data Repositories */
#define GEMM_BACKWARD_repo (__dague_object->GEMM_BACKWARD_repository)
#define TRSM_BACKWARD_repo (__dague_object->TRSM_BACKWARD_repository)
#define GEMM_FORWARD_repo (__dague_object->GEMM_FORWARD_repository)
#define TRSM_FORWARD_repo (__dague_object->TRSM_FORWARD_repository)

/* Dependency Tracking Allocation Macro */
#define ALLOCATE_DEP_TRACKING(DEPS, vMIN, vMAX, vNAME, vSYMBOL, PREVDEP, FLAG)               \
do {                                                                                         \
  int _vmin = (vMIN);                                                                        \
  int _vmax = (vMAX);                                                                        \
  (DEPS) = (dague_dependencies_t*)calloc(1, sizeof(dague_dependencies_t) +                   \
                   (_vmax - _vmin) * sizeof(dague_dependencies_union_t));                    \
  /*DEBUG(("Allocate %d spaces for loop %s (min %d max %d) 0x%p last_dep 0x%p\n", */         \
  /*       (_vmax - _vmin + 1), (vNAME), _vmin, _vmax, (void*)(DEPS), (void*)(PREVDEP))); */ \
  (DEPS)->flags = DAGUE_DEPENDENCIES_FLAG_ALLOCATED | (FLAG);                                \
  DAGUE_STAT_INCREASE(mem_bitarray,  sizeof(dague_dependencies_t) + STAT_MALLOC_OVERHEAD +   \
                   (_vmax - _vmin) * sizeof(dague_dependencies_union_t));                    \
  (DEPS)->symbol = (vSYMBOL);                                                                \
  (DEPS)->min = _vmin;                                                                       \
  (DEPS)->max = _vmax;                                                                       \
  (DEPS)->prev = (PREVDEP); /* chain them backward */                                        \
} while (0)                                                                                  

static inline int dague_imin(int a, int b) { return (a <= b) ? a : b; };                     

static inline int dague_imax(int a, int b) { return (a >= b) ? a : b; };                     

static inline int zpotrs_sp1dplus_inline_c_expr1_line_170(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task GEMM_BACKWARD */
  int k = assignments[0].value;
  int bloknum = assignments[1].value;
  int fcblk = assignments[2].value;
  int cblk = assignments[3].value;
  int gcblk2list = assignments[4].value;
  int firstbrow = assignments[5].value;
  int lastbrow = assignments[6].value;

  (void)k;  (void)bloknum;  (void)fcblk;  (void)cblk;  (void)gcblk2list;  (void)firstbrow;  (void)lastbrow;

#line 170 "zpotrs_sp1dplus.jdf"
 assert( gcblk2list != -1 ); return UPDOWN_LISTPTR( gcblk2list+1 )-1; 
#line 125 "zpotrs_sp1dplus.c"
}

static inline int zpotrs_sp1dplus_inline_c_expr2_line_169(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task GEMM_BACKWARD */
  int k = assignments[0].value;
  int bloknum = assignments[1].value;
  int fcblk = assignments[2].value;
  int cblk = assignments[3].value;
  int gcblk2list = assignments[4].value;
  int firstbrow = assignments[5].value;
  int lastbrow = assignments[6].value;

  (void)k;  (void)bloknum;  (void)fcblk;  (void)cblk;  (void)gcblk2list;  (void)firstbrow;  (void)lastbrow;

#line 169 "zpotrs_sp1dplus.jdf"
 assert( gcblk2list != -1 ); return UPDOWN_LISTPTR( gcblk2list   ); 
#line 145 "zpotrs_sp1dplus.c"
}

static inline int zpotrs_sp1dplus_inline_c_expr3_line_168(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task GEMM_BACKWARD */
  int k = assignments[0].value;
  int bloknum = assignments[1].value;
  int fcblk = assignments[2].value;
  int cblk = assignments[3].value;
  int gcblk2list = assignments[4].value;
  int firstbrow = assignments[5].value;
  int lastbrow = assignments[6].value;

  (void)k;  (void)bloknum;  (void)fcblk;  (void)cblk;  (void)gcblk2list;  (void)firstbrow;  (void)lastbrow;

#line 168 "zpotrs_sp1dplus.jdf"
 return UPDOWN_GCBLK2LIST(UPDOWN_LOC2GLOB( fcblk )); 
#line 165 "zpotrs_sp1dplus.c"
}

static inline int zpotrs_sp1dplus_inline_c_expr4_line_166(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task GEMM_BACKWARD */
  int k = assignments[0].value;
  int bloknum = assignments[1].value;
  int fcblk = assignments[2].value;
  int cblk = assignments[3].value;
  int gcblk2list = assignments[4].value;
  int firstbrow = assignments[5].value;
  int lastbrow = assignments[6].value;

  (void)k;  (void)bloknum;  (void)fcblk;  (void)cblk;  (void)gcblk2list;  (void)firstbrow;  (void)lastbrow;

#line 166 "zpotrs_sp1dplus.jdf"
 return sparse_matrix_get_lcblknum( descA, (dague_int_t)bloknum ); 
#line 185 "zpotrs_sp1dplus.c"
}

static inline int zpotrs_sp1dplus_inline_c_expr5_line_165(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task GEMM_BACKWARD */
  int k = assignments[0].value;
  int bloknum = assignments[1].value;
  int fcblk = assignments[2].value;
  int cblk = assignments[3].value;
  int gcblk2list = assignments[4].value;
  int firstbrow = assignments[5].value;
  int lastbrow = assignments[6].value;

  (void)k;  (void)bloknum;  (void)fcblk;  (void)cblk;  (void)gcblk2list;  (void)firstbrow;  (void)lastbrow;

#line 165 "zpotrs_sp1dplus.jdf"
 return SYMB_CBLKNUM(bloknum); 
#line 205 "zpotrs_sp1dplus.c"
}

static inline int zpotrs_sp1dplus_inline_c_expr6_line_164(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task GEMM_BACKWARD */
  int k = assignments[0].value;
  int bloknum = assignments[1].value;
  int fcblk = assignments[2].value;
  int cblk = assignments[3].value;
  int gcblk2list = assignments[4].value;
  int firstbrow = assignments[5].value;
  int lastbrow = assignments[6].value;

  (void)k;  (void)bloknum;  (void)fcblk;  (void)cblk;  (void)gcblk2list;  (void)firstbrow;  (void)lastbrow;

#line 164 "zpotrs_sp1dplus.jdf"
 return UPDOWN_LISTBLOK(k); 
#line 225 "zpotrs_sp1dplus.c"
}

static inline int zpotrs_sp1dplus_inline_c_expr7_line_134(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task TRSM_BACKWARD */
  int k = assignments[0].value;
  int gcblk2list = assignments[1].value;
  int firstbrow = assignments[2].value;
  int lastbrow = assignments[3].value;
  int firstblok = assignments[4].value;

  (void)k;  (void)gcblk2list;  (void)firstbrow;  (void)lastbrow;  (void)firstblok;

#line 134 "zpotrs_sp1dplus.jdf"
 return SYMB_BLOKNUM(k)+1; 
#line 243 "zpotrs_sp1dplus.c"
}

static inline int zpotrs_sp1dplus_inline_c_expr8_line_133(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task TRSM_BACKWARD */
  int k = assignments[0].value;
  int gcblk2list = assignments[1].value;
  int firstbrow = assignments[2].value;
  int lastbrow = assignments[3].value;
  int firstblok = assignments[4].value;

  (void)k;  (void)gcblk2list;  (void)firstbrow;  (void)lastbrow;  (void)firstblok;

#line 133 "zpotrs_sp1dplus.jdf"
 if ( gcblk2list != -1 ) return UPDOWN_LISTPTR( gcblk2list+1 )-1; else return -1; 
#line 261 "zpotrs_sp1dplus.c"
}

static inline int zpotrs_sp1dplus_inline_c_expr9_line_132(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task TRSM_BACKWARD */
  int k = assignments[0].value;
  int gcblk2list = assignments[1].value;
  int firstbrow = assignments[2].value;
  int lastbrow = assignments[3].value;
  int firstblok = assignments[4].value;

  (void)k;  (void)gcblk2list;  (void)firstbrow;  (void)lastbrow;  (void)firstblok;

#line 132 "zpotrs_sp1dplus.jdf"
 if ( gcblk2list != -1 ) return UPDOWN_LISTPTR( gcblk2list   ); else return -1; 
#line 279 "zpotrs_sp1dplus.c"
}

static inline int zpotrs_sp1dplus_inline_c_expr10_line_131(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task TRSM_BACKWARD */
  int k = assignments[0].value;
  int gcblk2list = assignments[1].value;
  int firstbrow = assignments[2].value;
  int lastbrow = assignments[3].value;
  int firstblok = assignments[4].value;

  (void)k;  (void)gcblk2list;  (void)firstbrow;  (void)lastbrow;  (void)firstblok;

#line 131 "zpotrs_sp1dplus.jdf"
 return UPDOWN_GCBLK2LIST(UPDOWN_LOC2GLOB( k )); 
#line 297 "zpotrs_sp1dplus.c"
}

static inline int zpotrs_sp1dplus_inline_c_expr11_line_89(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task GEMM_FORWARD */
  int k = assignments[0].value;
  int fcblk = assignments[1].value;
  int cblk = assignments[2].value;
  int phony = assignments[3].value;
  int prev = assignments[4].value;
  int next = assignments[5].value;

  (void)k;  (void)fcblk;  (void)cblk;  (void)phony;  (void)prev;  (void)next;

#line 89 "zpotrs_sp1dplus.jdf"
 if (phony) return 0; else return sparse_matrix_get_listptr_next( descA, (dague_int_t)k, (dague_int_t)fcblk ); 
#line 316 "zpotrs_sp1dplus.c"
}

static inline int zpotrs_sp1dplus_inline_c_expr12_line_88(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task GEMM_FORWARD */
  int k = assignments[0].value;
  int fcblk = assignments[1].value;
  int cblk = assignments[2].value;
  int phony = assignments[3].value;
  int prev = assignments[4].value;
  int next = assignments[5].value;

  (void)k;  (void)fcblk;  (void)cblk;  (void)phony;  (void)prev;  (void)next;

#line 88 "zpotrs_sp1dplus.jdf"
 if (phony) return 0; else return sparse_matrix_get_listptr_prev( descA, (dague_int_t)k, (dague_int_t)fcblk ); 
#line 335 "zpotrs_sp1dplus.c"
}

static inline int zpotrs_sp1dplus_inline_c_expr13_line_87(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task GEMM_FORWARD */
  int k = assignments[0].value;
  int fcblk = assignments[1].value;
  int cblk = assignments[2].value;
  int phony = assignments[3].value;
  int prev = assignments[4].value;
  int next = assignments[5].value;

  (void)k;  (void)fcblk;  (void)cblk;  (void)phony;  (void)prev;  (void)next;

#line 87 "zpotrs_sp1dplus.jdf"
 return k == SYMB_BLOKNUM( cblk ); 
#line 354 "zpotrs_sp1dplus.c"
}

static inline int zpotrs_sp1dplus_inline_c_expr14_line_86(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task GEMM_FORWARD */
  int k = assignments[0].value;
  int fcblk = assignments[1].value;
  int cblk = assignments[2].value;
  int phony = assignments[3].value;
  int prev = assignments[4].value;
  int next = assignments[5].value;

  (void)k;  (void)fcblk;  (void)cblk;  (void)phony;  (void)prev;  (void)next;

#line 86 "zpotrs_sp1dplus.jdf"
 return sparse_matrix_get_lcblknum( descA, (dague_int_t)k ); 
#line 373 "zpotrs_sp1dplus.c"
}

static inline int zpotrs_sp1dplus_inline_c_expr15_line_85(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task GEMM_FORWARD */
  int k = assignments[0].value;
  int fcblk = assignments[1].value;
  int cblk = assignments[2].value;
  int phony = assignments[3].value;
  int prev = assignments[4].value;
  int next = assignments[5].value;

  (void)k;  (void)fcblk;  (void)cblk;  (void)phony;  (void)prev;  (void)next;

#line 85 "zpotrs_sp1dplus.jdf"
 return SYMB_CBLKNUM(k); 
#line 392 "zpotrs_sp1dplus.c"
}

static inline int zpotrs_sp1dplus_inline_c_expr16_line_54(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task TRSM_FORWARD */
  int k = assignments[0].value;
  int gcblk2list = assignments[1].value;
  int browk1 = assignments[2].value;
  int lastbrow = assignments[3].value;
  int firstblok = assignments[4].value;
  int lastblok = assignments[5].value;
  int firstgemm = assignments[6].value;

  (void)k;  (void)gcblk2list;  (void)browk1;  (void)lastbrow;  (void)firstblok;  (void)lastblok;  (void)firstgemm;

#line 50 "zpotrs_sp1dplus.jdf"
 
    dague_int_t fcblk = SYMB_CBLKNUM(firstblok+1); /* Facing block column of the first etra-daigonal block */
    dague_int_t listptr = UPDOWN_GCBLK2LIST( UPDOWN_LOC2GLOB( fcblk ) ); /* Index in the global list of the column block receiving contributyion */
    return UPDOWN_LISTBLOK( UPDOWN_LISTPTR( listptr+1 ) -1 );
    
#line 416 "zpotrs_sp1dplus.c"
}

static inline int zpotrs_sp1dplus_inline_c_expr17_line_48(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task TRSM_FORWARD */
  int k = assignments[0].value;
  int gcblk2list = assignments[1].value;
  int browk1 = assignments[2].value;
  int lastbrow = assignments[3].value;
  int firstblok = assignments[4].value;
  int lastblok = assignments[5].value;
  int firstgemm = assignments[6].value;

  (void)k;  (void)gcblk2list;  (void)browk1;  (void)lastbrow;  (void)firstblok;  (void)lastblok;  (void)firstgemm;

#line 48 "zpotrs_sp1dplus.jdf"
 return SYMB_BLOKNUM(k+1); 
#line 436 "zpotrs_sp1dplus.c"
}

static inline int zpotrs_sp1dplus_inline_c_expr18_line_47(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task TRSM_FORWARD */
  int k = assignments[0].value;
  int gcblk2list = assignments[1].value;
  int browk1 = assignments[2].value;
  int lastbrow = assignments[3].value;
  int firstblok = assignments[4].value;
  int lastblok = assignments[5].value;
  int firstgemm = assignments[6].value;

  (void)k;  (void)gcblk2list;  (void)browk1;  (void)lastbrow;  (void)firstblok;  (void)lastblok;  (void)firstgemm;

#line 47 "zpotrs_sp1dplus.jdf"
 return SYMB_BLOKNUM(k); 
#line 456 "zpotrs_sp1dplus.c"
}

static inline int zpotrs_sp1dplus_inline_c_expr19_line_46(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task TRSM_FORWARD */
  int k = assignments[0].value;
  int gcblk2list = assignments[1].value;
  int browk1 = assignments[2].value;
  int lastbrow = assignments[3].value;
  int firstblok = assignments[4].value;
  int lastblok = assignments[5].value;
  int firstgemm = assignments[6].value;

  (void)k;  (void)gcblk2list;  (void)browk1;  (void)lastbrow;  (void)firstblok;  (void)lastblok;  (void)firstgemm;

#line 46 "zpotrs_sp1dplus.jdf"
 if (browk1 > 0) return UPDOWN_LISTBLOK(browk1-1); else return 0; 
#line 476 "zpotrs_sp1dplus.c"
}

static inline int zpotrs_sp1dplus_inline_c_expr20_line_45(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task TRSM_FORWARD */
  int k = assignments[0].value;
  int gcblk2list = assignments[1].value;
  int browk1 = assignments[2].value;
  int lastbrow = assignments[3].value;
  int firstblok = assignments[4].value;
  int lastblok = assignments[5].value;
  int firstgemm = assignments[6].value;

  (void)k;  (void)gcblk2list;  (void)browk1;  (void)lastbrow;  (void)firstblok;  (void)lastblok;  (void)firstgemm;

#line 45 "zpotrs_sp1dplus.jdf"
 if ( gcblk2list != -1 ) return UPDOWN_LISTPTR( gcblk2list+1 ); else return -1; 
#line 496 "zpotrs_sp1dplus.c"
}

static inline int zpotrs_sp1dplus_inline_c_expr21_line_43(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task TRSM_FORWARD */
  int k = assignments[0].value;
  int gcblk2list = assignments[1].value;
  int browk1 = assignments[2].value;
  int lastbrow = assignments[3].value;
  int firstblok = assignments[4].value;
  int lastblok = assignments[5].value;
  int firstgemm = assignments[6].value;

  (void)k;  (void)gcblk2list;  (void)browk1;  (void)lastbrow;  (void)firstblok;  (void)lastblok;  (void)firstgemm;

#line 43 "zpotrs_sp1dplus.jdf"
 return UPDOWN_GCBLK2LIST(UPDOWN_LOC2GLOB( k )); 
#line 516 "zpotrs_sp1dplus.c"
}

static inline uint64_t GEMM_BACKWARD_hash(const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object, const assignment_t *assignments)
{
  uint64_t __h = 0;
  (void)__dague_object;
  int k = assignments[0].value;
  int k_min = 0;
  __h += (k - k_min);
  return __h;
}

static inline uint64_t TRSM_BACKWARD_hash(const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object, const assignment_t *assignments)
{
  uint64_t __h = 0;
  (void)__dague_object;
  int k = assignments[0].value;
  int k_min = 0;
  __h += (k - k_min);
  return __h;
}

static inline uint64_t GEMM_FORWARD_hash(const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object, const assignment_t *assignments)
{
  uint64_t __h = 0;
  (void)__dague_object;
  int k = assignments[0].value;
  int k_min = 0;
  __h += (k - k_min);
  return __h;
}

static inline uint64_t TRSM_FORWARD_hash(const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object, const assignment_t *assignments)
{
  uint64_t __h = 0;
  (void)__dague_object;
  int k = assignments[0].value;
  int k_min = 0;
  __h += (k - k_min);
  return __h;
}

/** Predeclarations of the dague_function_t objects */
static const dague_function_t zpotrs_sp1dplus_GEMM_BACKWARD;
static const dague_function_t zpotrs_sp1dplus_TRSM_BACKWARD;
static const dague_function_t zpotrs_sp1dplus_GEMM_FORWARD;
static const dague_function_t zpotrs_sp1dplus_TRSM_FORWARD;
/** Declarations of the pseudo-dague_function_t objects for data */
static const dague_function_t zpotrs_sp1dplus_A = {
  .name = "A",
  .flags = 0x0,
  .dependencies_goal = 0x0,
  .nb_parameters = 0,
  .nb_definitions = 0,
  .params = { NULL, },
  .locals = { NULL, },
  .pred = NULL,
  .in = { NULL, },
  .out = { NULL, },
  .priority = NULL,
  .deps = -1,
  .hook = NULL,
  .release_deps = NULL,
  .body = NULL,
#if defined(DAGUE_SCHED_CACHE_AWARE)
  .cache_rank_function = NULL,
#endif /* defined(DAGUE_SCHED_CACHE_AWARE) */
};
static const dague_function_t zpotrs_sp1dplus_B = {
  .name = "B",
  .flags = 0x0,
  .dependencies_goal = 0x0,
  .nb_parameters = 0,
  .nb_definitions = 0,
  .params = { NULL, },
  .locals = { NULL, },
  .pred = NULL,
  .in = { NULL, },
  .out = { NULL, },
  .priority = NULL,
  .deps = -1,
  .hook = NULL,
  .release_deps = NULL,
  .body = NULL,
#if defined(DAGUE_SCHED_CACHE_AWARE)
  .cache_rank_function = NULL,
#endif /* defined(DAGUE_SCHED_CACHE_AWARE) */
};

/** Predeclarations of the parameters */
static const dague_flow_t flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B;
static const dague_flow_t flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B;
static const dague_flow_t flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C;
static const dague_flow_t flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B;
static const dague_flow_t flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C;
static const dague_flow_t flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_B;
static const dague_flow_t flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B;
static const dague_flow_t flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C;
static const dague_flow_t flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B;
static const dague_flow_t flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C;
static const dague_flow_t flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B;
static const dague_flow_t flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C;
static const dague_flow_t flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C;
static const dague_flow_t flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_B;
static const dague_flow_t flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B;
static const dague_flow_t flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C;
/**********************************************************************************
 *                                GEMM_BACKWARD                                  *
 **********************************************************************************/

static inline int minexpr_of_symb_zpotrs_sp1dplus_GEMM_BACKWARD_k_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return 0;
}
static const expr_t minexpr_of_symb_zpotrs_sp1dplus_GEMM_BACKWARD_k = {
  .op = EXPR_OP_INLINE,
  .inline_func = minexpr_of_symb_zpotrs_sp1dplus_GEMM_BACKWARD_k_fct
};
static inline int maxexpr_of_symb_zpotrs_sp1dplus_GEMM_BACKWARD_k_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return browsize;
}
static const expr_t maxexpr_of_symb_zpotrs_sp1dplus_GEMM_BACKWARD_k = {
  .op = EXPR_OP_INLINE,
  .inline_func = maxexpr_of_symb_zpotrs_sp1dplus_GEMM_BACKWARD_k_fct
};
static const symbol_t symb_zpotrs_sp1dplus_GEMM_BACKWARD_k = {.min = &minexpr_of_symb_zpotrs_sp1dplus_GEMM_BACKWARD_k, .max = &maxexpr_of_symb_zpotrs_sp1dplus_GEMM_BACKWARD_k,  .flags = DAGUE_SYMBOL_IS_STANDALONE};

static inline int expr_of_symb_zpotrs_sp1dplus_GEMM_BACKWARD_bloknum_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zpotrs_sp1dplus_inline_c_expr6_line_164((const dague_object_t*)__dague_object, assignments);
}
static const expr_t expr_of_symb_zpotrs_sp1dplus_GEMM_BACKWARD_bloknum = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_symb_zpotrs_sp1dplus_GEMM_BACKWARD_bloknum_fct
};
static const symbol_t symb_zpotrs_sp1dplus_GEMM_BACKWARD_bloknum = {.min = &expr_of_symb_zpotrs_sp1dplus_GEMM_BACKWARD_bloknum, .max = &expr_of_symb_zpotrs_sp1dplus_GEMM_BACKWARD_bloknum,  .flags = 0x0};

static inline int expr_of_symb_zpotrs_sp1dplus_GEMM_BACKWARD_fcblk_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zpotrs_sp1dplus_inline_c_expr5_line_165((const dague_object_t*)__dague_object, assignments);
}
static const expr_t expr_of_symb_zpotrs_sp1dplus_GEMM_BACKWARD_fcblk = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_symb_zpotrs_sp1dplus_GEMM_BACKWARD_fcblk_fct
};
static const symbol_t symb_zpotrs_sp1dplus_GEMM_BACKWARD_fcblk = {.min = &expr_of_symb_zpotrs_sp1dplus_GEMM_BACKWARD_fcblk, .max = &expr_of_symb_zpotrs_sp1dplus_GEMM_BACKWARD_fcblk,  .flags = 0x0};

static inline int expr_of_symb_zpotrs_sp1dplus_GEMM_BACKWARD_cblk_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zpotrs_sp1dplus_inline_c_expr4_line_166((const dague_object_t*)__dague_object, assignments);
}
static const expr_t expr_of_symb_zpotrs_sp1dplus_GEMM_BACKWARD_cblk = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_symb_zpotrs_sp1dplus_GEMM_BACKWARD_cblk_fct
};
static const symbol_t symb_zpotrs_sp1dplus_GEMM_BACKWARD_cblk = {.min = &expr_of_symb_zpotrs_sp1dplus_GEMM_BACKWARD_cblk, .max = &expr_of_symb_zpotrs_sp1dplus_GEMM_BACKWARD_cblk,  .flags = 0x0};

static inline int expr_of_symb_zpotrs_sp1dplus_GEMM_BACKWARD_gcblk2list_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zpotrs_sp1dplus_inline_c_expr3_line_168((const dague_object_t*)__dague_object, assignments);
}
static const expr_t expr_of_symb_zpotrs_sp1dplus_GEMM_BACKWARD_gcblk2list = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_symb_zpotrs_sp1dplus_GEMM_BACKWARD_gcblk2list_fct
};
static const symbol_t symb_zpotrs_sp1dplus_GEMM_BACKWARD_gcblk2list = {.min = &expr_of_symb_zpotrs_sp1dplus_GEMM_BACKWARD_gcblk2list, .max = &expr_of_symb_zpotrs_sp1dplus_GEMM_BACKWARD_gcblk2list,  .flags = 0x0};

static inline int expr_of_symb_zpotrs_sp1dplus_GEMM_BACKWARD_firstbrow_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zpotrs_sp1dplus_inline_c_expr2_line_169((const dague_object_t*)__dague_object, assignments);
}
static const expr_t expr_of_symb_zpotrs_sp1dplus_GEMM_BACKWARD_firstbrow = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_symb_zpotrs_sp1dplus_GEMM_BACKWARD_firstbrow_fct
};
static const symbol_t symb_zpotrs_sp1dplus_GEMM_BACKWARD_firstbrow = {.min = &expr_of_symb_zpotrs_sp1dplus_GEMM_BACKWARD_firstbrow, .max = &expr_of_symb_zpotrs_sp1dplus_GEMM_BACKWARD_firstbrow,  .flags = 0x0};

static inline int expr_of_symb_zpotrs_sp1dplus_GEMM_BACKWARD_lastbrow_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zpotrs_sp1dplus_inline_c_expr1_line_170((const dague_object_t*)__dague_object, assignments);
}
static const expr_t expr_of_symb_zpotrs_sp1dplus_GEMM_BACKWARD_lastbrow = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_symb_zpotrs_sp1dplus_GEMM_BACKWARD_lastbrow_fct
};
static const symbol_t symb_zpotrs_sp1dplus_GEMM_BACKWARD_lastbrow = {.min = &expr_of_symb_zpotrs_sp1dplus_GEMM_BACKWARD_lastbrow, .max = &expr_of_symb_zpotrs_sp1dplus_GEMM_BACKWARD_lastbrow,  .flags = 0x0};

static inline int pred_of_zpotrs_sp1dplus_GEMM_BACKWARD_as_expr_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;
  int bloknum = assignments[1].value;
  int fcblk = assignments[2].value;
  int cblk = assignments[3].value;
  int gcblk2list = assignments[4].value;
  int firstbrow = assignments[5].value;
  int lastbrow = assignments[6].value;

  /* Silent Warnings: should look into predicate to know what variables are usefull */
  (void)__dague_object;
  (void)k;
  (void)bloknum;
  (void)fcblk;
  (void)cblk;
  (void)gcblk2list;
  (void)firstbrow;
  (void)lastbrow;
  /* Compute Predicate */
  return GEMM_BACKWARD_pred(k, bloknum, fcblk, cblk, gcblk2list, firstbrow, lastbrow);
}
static const expr_t pred_of_zpotrs_sp1dplus_GEMM_BACKWARD_as_expr = {
  .op = EXPR_OP_INLINE,
  .inline_func = pred_of_zpotrs_sp1dplus_GEMM_BACKWARD_as_expr_fct
};
static inline int expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_A_dep1_atline_177_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int cblk = assignments[3].value;

  (void)__dague_object; (void)assignments;
  return cblk;
}
static const expr_t expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_A_dep1_atline_177 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_A_dep1_atline_177_fct
};
static const dep_t flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_A_dep1_atline_177 = {
  .cond = NULL,
  .dague = &zpotrs_sp1dplus_A,
  .datatype_index = DAGUE_zpotrs_sp1dplus_DEFAULT_ARENA,  .call_params = {
    &expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_A_dep1_atline_177
  }
};
static const dague_flow_t flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_A = {
  .name = "A",
  .sym_type = SYM_IN,
  .access_type = ACCESS_READ,
  .flow_index = 0,
  .dep_in  = { &flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_A_dep1_atline_177 },
  .dep_out = { NULL }
};

static inline int expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_B_dep1_atline_179_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int fcblk = assignments[2].value;

  (void)__dague_object; (void)assignments;
  return fcblk;
}
static const expr_t expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_B_dep1_atline_179 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_B_dep1_atline_179_fct
};
static const dep_t flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_B_dep1_atline_179 = {
  .cond = NULL,
  .dague = &zpotrs_sp1dplus_TRSM_BACKWARD,
  .flow = &flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B,
  .datatype_index = DAGUE_zpotrs_sp1dplus_DEFAULT_ARENA,  .call_params = {
    &expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_B_dep1_atline_179
  }
};
static const dague_flow_t flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_B = {
  .name = "B",
  .sym_type = SYM_IN,
  .access_type = ACCESS_READ,
  .flow_index = 1,
  .dep_in  = { &flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_B_dep1_atline_179 },
  .dep_out = { NULL }
};

static inline int expr_of_cond_for_flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C_dep1_iftrue_atline_180_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;
  int lastbrow = assignments[6].value;

  (void)__dague_object; (void)assignments;
  return (k == lastbrow);
}
static const expr_t expr_of_cond_for_flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C_dep1_iftrue_atline_180 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C_dep1_iftrue_atline_180_fct
};
static inline int expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C_dep1_iftrue_atline_180_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int cblk = assignments[3].value;

  (void)__dague_object; (void)assignments;
  return cblk;
}
static const expr_t expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C_dep1_iftrue_atline_180 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C_dep1_iftrue_atline_180_fct
};
static const dep_t flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C_dep1_iftrue_atline_180 = {
  .cond = &expr_of_cond_for_flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C_dep1_iftrue_atline_180,
  .dague = &zpotrs_sp1dplus_TRSM_FORWARD,
  .flow = &flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B,
  .datatype_index = DAGUE_zpotrs_sp1dplus_DEFAULT_ARENA,  .call_params = {
    &expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C_dep1_iftrue_atline_180
  }
};
static inline int expr_of_cond_for_flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C_dep1_iffalse_atline_180_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;
  int lastbrow = assignments[6].value;

  (void)__dague_object; (void)assignments;
  return !(k == lastbrow);
}
static const expr_t expr_of_cond_for_flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C_dep1_iffalse_atline_180 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C_dep1_iffalse_atline_180_fct
};
static inline int expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C_dep1_iffalse_atline_180_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return (k + 1);
}
static const expr_t expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C_dep1_iffalse_atline_180 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C_dep1_iffalse_atline_180_fct
};
static const dep_t flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C_dep1_iffalse_atline_180 = {
  .cond = &expr_of_cond_for_flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C_dep1_iffalse_atline_180,
  .dague = &zpotrs_sp1dplus_GEMM_BACKWARD,
  .flow = &flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C,
  .datatype_index = DAGUE_zpotrs_sp1dplus_DEFAULT_ARENA,  .call_params = {
    &expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C_dep1_iffalse_atline_180
  }
};
static inline int expr_of_cond_for_flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C_dep2_iftrue_atline_193_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;
  int firstbrow = assignments[5].value;

  (void)__dague_object; (void)assignments;
  return (k == firstbrow);
}
static const expr_t expr_of_cond_for_flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C_dep2_iftrue_atline_193 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C_dep2_iftrue_atline_193_fct
};
static inline int expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C_dep2_iftrue_atline_193_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int cblk = assignments[3].value;

  (void)__dague_object; (void)assignments;
  return cblk;
}
static const expr_t expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C_dep2_iftrue_atline_193 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C_dep2_iftrue_atline_193_fct
};
static const dep_t flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C_dep2_iftrue_atline_193 = {
  .cond = &expr_of_cond_for_flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C_dep2_iftrue_atline_193,
  .dague = &zpotrs_sp1dplus_TRSM_BACKWARD,
  .flow = &flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B,
  .datatype_index = DAGUE_zpotrs_sp1dplus_DEFAULT_ARENA,  .call_params = {
    &expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C_dep2_iftrue_atline_193
  }
};
static inline int expr_of_cond_for_flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C_dep2_iffalse_atline_193_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;
  int firstbrow = assignments[5].value;

  (void)__dague_object; (void)assignments;
  return !(k == firstbrow);
}
static const expr_t expr_of_cond_for_flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C_dep2_iffalse_atline_193 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C_dep2_iffalse_atline_193_fct
};
static inline int expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C_dep2_iffalse_atline_193_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return (k - 1);
}
static const expr_t expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C_dep2_iffalse_atline_193 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C_dep2_iffalse_atline_193_fct
};
static const dep_t flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C_dep2_iffalse_atline_193 = {
  .cond = &expr_of_cond_for_flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C_dep2_iffalse_atline_193,
  .dague = &zpotrs_sp1dplus_GEMM_BACKWARD,
  .flow = &flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C,
  .datatype_index = DAGUE_zpotrs_sp1dplus_DEFAULT_ARENA,  .call_params = {
    &expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C_dep2_iffalse_atline_193
  }
};
static const dague_flow_t flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C = {
  .name = "C",
  .sym_type = SYM_INOUT,
  .access_type = ACCESS_RW,
  .flow_index = 2,
  .dep_in  = { &flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C_dep1_iftrue_atline_180, &flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C_dep1_iffalse_atline_180 },
  .dep_out = { &flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C_dep2_iftrue_atline_193, &flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C_dep2_iffalse_atline_193 }
};

static void
iterate_successors_of_zpotrs_sp1dplus_GEMM_BACKWARD(dague_execution_unit_t *eu, dague_execution_context_t *exec_context,
               uint32_t action_mask, dague_ontask_function_t *ontask, void *ontask_arg)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)exec_context->dague_object;
  dague_execution_context_t nc;
  dague_arena_t* arena = NULL;
  int rank_src = 0, rank_dst = 0;
  int k = exec_context->locals[0].value;
  int bloknum = exec_context->locals[1].value;
  int fcblk = exec_context->locals[2].value;
  int cblk = exec_context->locals[3].value;
  int gcblk2list = exec_context->locals[4].value;
  int firstbrow = exec_context->locals[5].value;
  int lastbrow = exec_context->locals[6].value;
  (void)rank_src; (void)rank_dst; (void)__dague_object;
  (void)k;  (void)bloknum;  (void)fcblk;  (void)cblk;  (void)gcblk2list;  (void)firstbrow;  (void)lastbrow;
  nc.dague_object = exec_context->dague_object;
#if defined(DISTRIBUTED)
  rank_src = ((dague_ddesc_t*)__dague_object->super.B)->rank_of((dague_ddesc_t*)__dague_object->super.B, cblk);
#endif
  /* Flow of data A has only IN dependencies */
  /* Flow of data B has only IN dependencies */
  /* Flow of Data C */
  if( action_mask & (1 << 0) ) {
#if defined(DISTRIBUTED)
    arena = __dague_object->super.arenas[DAGUE_zpotrs_sp1dplus_DEFAULT_ARENA];
#endif  /* defined(DISTRIBUTED) */
    if( (k == firstbrow) ) {
      nc.function = (const dague_function_t*)&zpotrs_sp1dplus_TRSM_BACKWARD;
      {
        const int TRSM_BACKWARD_k = cblk;
        if( (TRSM_BACKWARD_k >= (0)) && (TRSM_BACKWARD_k <= (cblknbr)) ) {
          nc.locals[0].value = TRSM_BACKWARD_k;
          const int TRSM_BACKWARD_gcblk2list = zpotrs_sp1dplus_inline_c_expr10_line_131((const dague_object_t*)__dague_object, nc.locals);
          nc.locals[1].value = TRSM_BACKWARD_gcblk2list;
          const int TRSM_BACKWARD_firstbrow = zpotrs_sp1dplus_inline_c_expr9_line_132((const dague_object_t*)__dague_object, nc.locals);
          nc.locals[2].value = TRSM_BACKWARD_firstbrow;
          const int TRSM_BACKWARD_lastbrow = zpotrs_sp1dplus_inline_c_expr8_line_133((const dague_object_t*)__dague_object, nc.locals);
          nc.locals[3].value = TRSM_BACKWARD_lastbrow;
          const int TRSM_BACKWARD_firstblok = zpotrs_sp1dplus_inline_c_expr7_line_134((const dague_object_t*)__dague_object, nc.locals);
          nc.locals[4].value = TRSM_BACKWARD_firstblok;
#if defined(DISTRIBUTED)
            rank_dst = ((dague_ddesc_t*)__dague_object->super.B)->rank_of((dague_ddesc_t*)__dague_object->super.B, TRSM_BACKWARD_k);
#endif
#if defined(DAGUE_DEBUG)
          if( NULL != eu ) {
            char tmp[128], tmp1[128];
            DEBUG(("thread %d release deps of C:%s to B:%s (from node %d to %d)\n", eu->eu_id,
                   dague_service_to_string(exec_context, tmp, 128),
                   dague_service_to_string(&nc, tmp1, 128), rank_src, rank_dst));
          }
#endif
            nc.priority = 0;
            if( DAGUE_ITERATE_STOP == ontask(eu, &nc, exec_context, 0, 0, rank_src, rank_dst, arena, ontask_arg) )
              return;
      }
        }
    } else {
      nc.function = (const dague_function_t*)&zpotrs_sp1dplus_GEMM_BACKWARD;
      {
        const int GEMM_BACKWARD_k = (k - 1);
        if( (GEMM_BACKWARD_k >= (0)) && (GEMM_BACKWARD_k <= (browsize)) ) {
          nc.locals[0].value = GEMM_BACKWARD_k;
          const int GEMM_BACKWARD_bloknum = zpotrs_sp1dplus_inline_c_expr6_line_164((const dague_object_t*)__dague_object, nc.locals);
          nc.locals[1].value = GEMM_BACKWARD_bloknum;
          const int GEMM_BACKWARD_fcblk = zpotrs_sp1dplus_inline_c_expr5_line_165((const dague_object_t*)__dague_object, nc.locals);
          nc.locals[2].value = GEMM_BACKWARD_fcblk;
          const int GEMM_BACKWARD_cblk = zpotrs_sp1dplus_inline_c_expr4_line_166((const dague_object_t*)__dague_object, nc.locals);
          nc.locals[3].value = GEMM_BACKWARD_cblk;
          const int GEMM_BACKWARD_gcblk2list = zpotrs_sp1dplus_inline_c_expr3_line_168((const dague_object_t*)__dague_object, nc.locals);
          nc.locals[4].value = GEMM_BACKWARD_gcblk2list;
          const int GEMM_BACKWARD_firstbrow = zpotrs_sp1dplus_inline_c_expr2_line_169((const dague_object_t*)__dague_object, nc.locals);
          nc.locals[5].value = GEMM_BACKWARD_firstbrow;
          const int GEMM_BACKWARD_lastbrow = zpotrs_sp1dplus_inline_c_expr1_line_170((const dague_object_t*)__dague_object, nc.locals);
          nc.locals[6].value = GEMM_BACKWARD_lastbrow;
#if defined(DISTRIBUTED)
            rank_dst = ((dague_ddesc_t*)__dague_object->super.B)->rank_of((dague_ddesc_t*)__dague_object->super.B, GEMM_BACKWARD_cblk);
#endif
#if defined(DAGUE_DEBUG)
          if( NULL != eu ) {
            char tmp[128], tmp1[128];
            DEBUG(("thread %d release deps of C:%s to C:%s (from node %d to %d)\n", eu->eu_id,
                   dague_service_to_string(exec_context, tmp, 128),
                   dague_service_to_string(&nc, tmp1, 128), rank_src, rank_dst));
          }
#endif
            nc.priority = 0;
            if( DAGUE_ITERATE_STOP == ontask(eu, &nc, exec_context, 0, 1, rank_src, rank_dst, arena, ontask_arg) )
              return;
      }
        }
    }
  }
  (void)nc;(void)arena;(void)eu;(void)ontask;(void)ontask_arg;(void)rank_dst;(void)action_mask;
}

static int release_deps_of_zpotrs_sp1dplus_GEMM_BACKWARD(dague_execution_unit_t *eu, dague_execution_context_t *context, uint32_t action_mask, dague_remote_deps_t *deps)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t *)context->dague_object;
  dague_release_dep_fct_arg_t arg;
  arg.nb_released = 0;
  arg.output_usage = 0;
  arg.action_mask = action_mask;
  arg.deps = deps;
  arg.ready_list = NULL;
  (void)__dague_object;
  if( action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS ) {
    arg.output_entry = data_repo_lookup_entry_and_create( eu, GEMM_BACKWARD_repo, GEMM_BACKWARD_hash(__dague_object, context->locals) );
  }
#if defined(DAGUE_SIM)
  assert(arg.output_entry->sim_exec_date == 0);
  arg.output_entry->sim_exec_date = context->sim_exec_date;
#endif
#if defined(DISTRIBUTED)
  arg.remote_deps_count = 0;
  arg.remote_deps = NULL;
#endif
  iterate_successors_of_zpotrs_sp1dplus_GEMM_BACKWARD(eu, context, action_mask, dague_release_dep_fct, &arg);

  if(action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS) {
    data_repo_entry_addto_usage_limit(GEMM_BACKWARD_repo, arg.output_entry->key, arg.output_usage);
    if( NULL != arg.ready_list ) {
      __dague_schedule(eu, arg.ready_list);
      arg.ready_list = NULL;
    }
  }
#if defined(DISTRIBUTED)
  if( (action_mask & DAGUE_ACTION_SEND_REMOTE_DEPS) && arg.remote_deps_count ) {
    arg.nb_released += dague_remote_dep_activate(eu, context, arg.remote_deps, arg.remote_deps_count);
  }
#endif
  if( action_mask & DAGUE_ACTION_RELEASE_LOCAL_REFS ) {
    int k = context->locals[0].value;
    int bloknum = context->locals[1].value;
    int fcblk = context->locals[2].value;
    int cblk = context->locals[3].value;
    int gcblk2list = context->locals[4].value;
    int firstbrow = context->locals[5].value;
    int lastbrow = context->locals[6].value;
    (void)k; (void)bloknum; (void)fcblk; (void)cblk; (void)gcblk2list; (void)firstbrow; (void)lastbrow;

    data_repo_entry_used_once( eu, TRSM_BACKWARD_repo, context->data[1].data_repo->key );
    (void)AUNREF(context->data[1].data);
    if( (k == lastbrow) ) {
      data_repo_entry_used_once( eu, TRSM_FORWARD_repo, context->data[2].data_repo->key );
      (void)AUNREF(context->data[2].data);
    } else {
      data_repo_entry_used_once( eu, GEMM_BACKWARD_repo, context->data[2].data_repo->key );
      (void)AUNREF(context->data[2].data);
    }
  }
  assert( NULL == arg.ready_list );
  return arg.nb_released;
}

static int hook_of_zpotrs_sp1dplus_GEMM_BACKWARD(dague_execution_unit_t *context, dague_execution_context_t *exec_context)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (__dague_zpotrs_sp1dplus_internal_object_t *)exec_context->dague_object;
  assignment_t tass[MAX_PARAM_COUNT];
  (void)context; (void)__dague_object; (void)tass;
#if defined(DAGUE_SIM)
  int __dague_simulation_date = 0;
#endif
  int k = exec_context->locals[0].value;
  int bloknum = exec_context->locals[1].value;
  int fcblk = exec_context->locals[2].value;
  int cblk = exec_context->locals[3].value;
  int gcblk2list = exec_context->locals[4].value;
  int firstbrow = exec_context->locals[5].value;
  int lastbrow = exec_context->locals[6].value;

  (void)k;  (void)bloknum;  (void)fcblk;  (void)cblk;  (void)gcblk2list;  (void)firstbrow;  (void)lastbrow;

  /** Declare the variables that will hold the data, and all the accounting for each */
  void *A = NULL; (void)A;
  dague_arena_chunk_t *gA = NULL; (void)gA;
  data_repo_entry_t *eA = NULL; (void)eA;
  void *B = NULL; (void)B;
  dague_arena_chunk_t *gB = NULL; (void)gB;
  data_repo_entry_t *eB = NULL; (void)eB;
  void *C = NULL; (void)C;
  dague_arena_chunk_t *gC = NULL; (void)gC;
  data_repo_entry_t *eC = NULL; (void)eC;

  /** Lookup the input data, and store them in the context */
  eA = exec_context->data[0].data_repo;
  gA = exec_context->data[0].data;
  if( NULL == gA ) {
  gA = (dague_arena_chunk_t*) A(cblk);
    exec_context->data[0].data = gA;
    exec_context->data[0].data_repo = eA;
  }
  A = ADATA(gA);
#if defined(DAGUE_SIM)
  if( (NULL != eA) && (eA->sim_exec_date > __dague_simulation_date) )
    __dague_simulation_date =  eA->sim_exec_date;
#endif
  eB = exec_context->data[1].data_repo;
  gB = exec_context->data[1].data;
  if( NULL == gB ) {
  tass[0].value = fcblk;
  eB = data_repo_lookup_entry( TRSM_BACKWARD_repo, TRSM_BACKWARD_hash( __dague_object, tass ));
  gB = eB->data[0];
    exec_context->data[1].data = gB;
    exec_context->data[1].data_repo = eB;
  }
  B = ADATA(gB);
#if defined(DAGUE_SIM)
  if( (NULL != eB) && (eB->sim_exec_date > __dague_simulation_date) )
    __dague_simulation_date =  eB->sim_exec_date;
#endif
  eC = exec_context->data[2].data_repo;
  gC = exec_context->data[2].data;
  if( NULL == gC ) {
  if( (k == lastbrow) ) {
      tass[0].value = cblk;
    eC = data_repo_lookup_entry( TRSM_FORWARD_repo, TRSM_FORWARD_hash( __dague_object, tass ));
    gC = eC->data[0];
  } else {
      tass[0].value = (k + 1);
    eC = data_repo_lookup_entry( GEMM_BACKWARD_repo, GEMM_BACKWARD_hash( __dague_object, tass ));
    gC = eC->data[0];
  }
    exec_context->data[2].data = gC;
    exec_context->data[2].data_repo = eC;
  }
  C = ADATA(gC);
#if defined(DAGUE_SIM)
  if( (NULL != eC) && (eC->sim_exec_date > __dague_simulation_date) )
    __dague_simulation_date =  eC->sim_exec_date;
#endif
#if defined(DAGUE_SIM)
  if( exec_context->function->sim_cost_fct != NULL ) {
    exec_context->sim_exec_date = __dague_simulation_date + exec_context->function->sim_cost_fct(exec_context);
  } else {
    exec_context->sim_exec_date = __dague_simulation_date;
  }
  if( context->largest_simulation_date < exec_context->sim_exec_date )
    context->largest_simulation_date = exec_context->sim_exec_date;
#endif
  /** PAPI events */
#if defined(HAVE_PAPI)
  papime_start_thread_counters();
#endif
  /** Cache Awareness Accounting */
#if defined(DAGUE_CACHE_AWARENESS)
  cache_buf_referenced(context->closest_cache, A);
  cache_buf_referenced(context->closest_cache, B);
  cache_buf_referenced(context->closest_cache, C);
#endif /* DAGUE_CACHE_AWARENESS */


#if !defined(DAGUE_PROF_DRY_BODY)

/*--------------------------------------------------------------------------------*
 *                              GEMM_BACKWARD BODY                              *
 *--------------------------------------------------------------------------------*/

  TAKE_TIME(context, 2*exec_context->function->function_id, GEMM_BACKWARD_hash( __dague_object, exec_context->locals), __dague_object->super.B, ((dague_ddesc_t*)(__dague_object->super.B))->data_key((dague_ddesc_t*)__dague_object->super.B, cblk) );
#line 182 "zpotrs_sp1dplus.jdf"
                                                         
    DRYRUN(
           Dague_Complex64_t *work = (Dague_Complex64_t *)dague_private_memory_pop( p_work );
           //core_zpotrfsp1d_gemm(cblk, k, fcblk, A, C, work, datacode);
           /* GEMM ( N, N, A, B, C ); */
           dague_private_memory_push( p_work, (void *)work );
           );
    printlog(
             "thread %d compute_1dgemm( k=%d, fcblk=%d, cblk=%d, prev=%d, next=%d )\n",
             context->eu_id, k, fcblk, cblk, prev, next);

#line 1234 "zpotrs_sp1dplus.c"
/*--------------------------------------------------------------------------------*
 *                          END OF GEMM_BACKWARD BODY                            *
 *--------------------------------------------------------------------------------*/



#endif /*!defined(DAGUE_PROF_DRY_BODY)*/

  return 0;
}
static int complete_hook_of_zpotrs_sp1dplus_GEMM_BACKWARD(dague_execution_unit_t *context, dague_execution_context_t *exec_context)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (__dague_zpotrs_sp1dplus_internal_object_t *)exec_context->dague_object;
  (void)context; (void)__dague_object;
  int k = exec_context->locals[0].value;
  int bloknum = exec_context->locals[1].value;
  int fcblk = exec_context->locals[2].value;
  int cblk = exec_context->locals[3].value;
  int gcblk2list = exec_context->locals[4].value;
  int firstbrow = exec_context->locals[5].value;
  int lastbrow = exec_context->locals[6].value;

  (void)k;  (void)bloknum;  (void)fcblk;  (void)cblk;  (void)gcblk2list;  (void)firstbrow;  (void)lastbrow;

  TAKE_TIME(context,2*exec_context->function->function_id+1, GEMM_BACKWARD_hash( __dague_object, exec_context->locals ), NULL, 0);
  /** PAPI events */
#if defined(HAVE_PAPI)
  papime_stop_thread_counters();
#endif
#if defined(DISTRIBUTED)
  /** If not working on distributed, there is no risk that data is not in place */
#endif /* DISTRIBUTED */
  dague_prof_grapher_task(exec_context, context->eu_id, GEMM_BACKWARD_hash(__dague_object, exec_context->locals));
  {
    release_deps_of_zpotrs_sp1dplus_GEMM_BACKWARD(context, exec_context,
        DAGUE_ACTION_RELEASE_REMOTE_DEPS |
        DAGUE_ACTION_RELEASE_LOCAL_DEPS |
        DAGUE_ACTION_RELEASE_LOCAL_REFS |
        DAGUE_ACTION_DEPS_MASK,
        NULL);
  }
  return 0;
}

static int zpotrs_sp1dplus_GEMM_BACKWARD_internal_init(__dague_zpotrs_sp1dplus_internal_object_t *__dague_object)
{
  dague_dependencies_t *dep = NULL;
  assignment_t assignments[MAX_LOCAL_COUNT];(void) assignments;
  int nb_tasks = 0, __foundone = 0;
  int32_t  k, bloknum, fcblk, cblk, gcblk2list, firstbrow, lastbrow;
  int32_t  k_min = 0x7fffffff;
  int32_t  k_max = 0;
  (void)__dague_object; (void)__foundone;
  /* First, find the min and max value for each of the dimensions */
  for(k = 0;
      k <= browsize;
      k++) {
    assignments[0].value = k;
    bloknum = zpotrs_sp1dplus_inline_c_expr6_line_164((const dague_object_t*)__dague_object, assignments);
    assignments[1].value = bloknum;
    fcblk = zpotrs_sp1dplus_inline_c_expr5_line_165((const dague_object_t*)__dague_object, assignments);
    assignments[2].value = fcblk;
    cblk = zpotrs_sp1dplus_inline_c_expr4_line_166((const dague_object_t*)__dague_object, assignments);
    assignments[3].value = cblk;
    gcblk2list = zpotrs_sp1dplus_inline_c_expr3_line_168((const dague_object_t*)__dague_object, assignments);
    assignments[4].value = gcblk2list;
    firstbrow = zpotrs_sp1dplus_inline_c_expr2_line_169((const dague_object_t*)__dague_object, assignments);
    assignments[5].value = firstbrow;
    lastbrow = zpotrs_sp1dplus_inline_c_expr1_line_170((const dague_object_t*)__dague_object, assignments);
    assignments[6].value = lastbrow;
    if( !GEMM_BACKWARD_pred(k, bloknum, fcblk, cblk, gcblk2list, firstbrow, lastbrow) ) continue;
    nb_tasks++;
    k_max = dague_imax(k_max, k);
    k_min = dague_imin(k_min, k);
  }

  /**
   * Now, for each of the dimensions, re-iterate on the space,
   * and if at least one value is defined, allocate arrays to point
   * to it. Array dimensions are defined by the (rough) observation above
   **/
  if( 0 != nb_tasks ) {
    ALLOCATE_DEP_TRACKING(dep, k_min, k_max, "k", &symb_zpotrs_sp1dplus_GEMM_BACKWARD_k, NULL, DAGUE_DEPENDENCIES_FLAG_FINAL);
  }
  __dague_object->super.super.dependencies_array[0] = dep;
  __dague_object->super.super.nb_local_tasks += nb_tasks;
  return nb_tasks;
}

static const dague_function_t zpotrs_sp1dplus_GEMM_BACKWARD = {
  .name = "GEMM_BACKWARD",
  .deps = 0,
  .flags = 0x0 | DAGUE_HAS_IN_IN_DEPENDENCIES,
  .function_id = 0,
  .dependencies_goal = 0x7,
  .nb_parameters = 1,
  .nb_definitions = 7,
  .params = { &symb_zpotrs_sp1dplus_GEMM_BACKWARD_k },
  .locals = { &symb_zpotrs_sp1dplus_GEMM_BACKWARD_k, &symb_zpotrs_sp1dplus_GEMM_BACKWARD_bloknum, &symb_zpotrs_sp1dplus_GEMM_BACKWARD_fcblk, &symb_zpotrs_sp1dplus_GEMM_BACKWARD_cblk, &symb_zpotrs_sp1dplus_GEMM_BACKWARD_gcblk2list, &symb_zpotrs_sp1dplus_GEMM_BACKWARD_firstbrow, &symb_zpotrs_sp1dplus_GEMM_BACKWARD_lastbrow },
  .pred = &pred_of_zpotrs_sp1dplus_GEMM_BACKWARD_as_expr,
  .priority = NULL,
  .in = { &flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_A, &flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_B, &flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C },
  .out = { &flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C },
  .iterate_successors = iterate_successors_of_zpotrs_sp1dplus_GEMM_BACKWARD,
  .release_deps = release_deps_of_zpotrs_sp1dplus_GEMM_BACKWARD,
  .hook = hook_of_zpotrs_sp1dplus_GEMM_BACKWARD,
  .complete_execution = complete_hook_of_zpotrs_sp1dplus_GEMM_BACKWARD,
#if defined(DAGUE_SIM)
  .sim_cost_fct = NULL,
#endif
  .key = (dague_functionkey_fn_t*)GEMM_BACKWARD_hash,
};


/**********************************************************************************
 *                                TRSM_BACKWARD                                  *
 **********************************************************************************/

static inline int minexpr_of_symb_zpotrs_sp1dplus_TRSM_BACKWARD_k_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return 0;
}
static const expr_t minexpr_of_symb_zpotrs_sp1dplus_TRSM_BACKWARD_k = {
  .op = EXPR_OP_INLINE,
  .inline_func = minexpr_of_symb_zpotrs_sp1dplus_TRSM_BACKWARD_k_fct
};
static inline int maxexpr_of_symb_zpotrs_sp1dplus_TRSM_BACKWARD_k_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return cblknbr;
}
static const expr_t maxexpr_of_symb_zpotrs_sp1dplus_TRSM_BACKWARD_k = {
  .op = EXPR_OP_INLINE,
  .inline_func = maxexpr_of_symb_zpotrs_sp1dplus_TRSM_BACKWARD_k_fct
};
static const symbol_t symb_zpotrs_sp1dplus_TRSM_BACKWARD_k = {.min = &minexpr_of_symb_zpotrs_sp1dplus_TRSM_BACKWARD_k, .max = &maxexpr_of_symb_zpotrs_sp1dplus_TRSM_BACKWARD_k,  .flags = DAGUE_SYMBOL_IS_STANDALONE};

static inline int expr_of_symb_zpotrs_sp1dplus_TRSM_BACKWARD_gcblk2list_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zpotrs_sp1dplus_inline_c_expr10_line_131((const dague_object_t*)__dague_object, assignments);
}
static const expr_t expr_of_symb_zpotrs_sp1dplus_TRSM_BACKWARD_gcblk2list = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_symb_zpotrs_sp1dplus_TRSM_BACKWARD_gcblk2list_fct
};
static const symbol_t symb_zpotrs_sp1dplus_TRSM_BACKWARD_gcblk2list = {.min = &expr_of_symb_zpotrs_sp1dplus_TRSM_BACKWARD_gcblk2list, .max = &expr_of_symb_zpotrs_sp1dplus_TRSM_BACKWARD_gcblk2list,  .flags = 0x0};

static inline int expr_of_symb_zpotrs_sp1dplus_TRSM_BACKWARD_firstbrow_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zpotrs_sp1dplus_inline_c_expr9_line_132((const dague_object_t*)__dague_object, assignments);
}
static const expr_t expr_of_symb_zpotrs_sp1dplus_TRSM_BACKWARD_firstbrow = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_symb_zpotrs_sp1dplus_TRSM_BACKWARD_firstbrow_fct
};
static const symbol_t symb_zpotrs_sp1dplus_TRSM_BACKWARD_firstbrow = {.min = &expr_of_symb_zpotrs_sp1dplus_TRSM_BACKWARD_firstbrow, .max = &expr_of_symb_zpotrs_sp1dplus_TRSM_BACKWARD_firstbrow,  .flags = 0x0};

static inline int expr_of_symb_zpotrs_sp1dplus_TRSM_BACKWARD_lastbrow_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zpotrs_sp1dplus_inline_c_expr8_line_133((const dague_object_t*)__dague_object, assignments);
}
static const expr_t expr_of_symb_zpotrs_sp1dplus_TRSM_BACKWARD_lastbrow = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_symb_zpotrs_sp1dplus_TRSM_BACKWARD_lastbrow_fct
};
static const symbol_t symb_zpotrs_sp1dplus_TRSM_BACKWARD_lastbrow = {.min = &expr_of_symb_zpotrs_sp1dplus_TRSM_BACKWARD_lastbrow, .max = &expr_of_symb_zpotrs_sp1dplus_TRSM_BACKWARD_lastbrow,  .flags = 0x0};

static inline int expr_of_symb_zpotrs_sp1dplus_TRSM_BACKWARD_firstblok_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zpotrs_sp1dplus_inline_c_expr7_line_134((const dague_object_t*)__dague_object, assignments);
}
static const expr_t expr_of_symb_zpotrs_sp1dplus_TRSM_BACKWARD_firstblok = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_symb_zpotrs_sp1dplus_TRSM_BACKWARD_firstblok_fct
};
static const symbol_t symb_zpotrs_sp1dplus_TRSM_BACKWARD_firstblok = {.min = &expr_of_symb_zpotrs_sp1dplus_TRSM_BACKWARD_firstblok, .max = &expr_of_symb_zpotrs_sp1dplus_TRSM_BACKWARD_firstblok,  .flags = 0x0};

static inline int pred_of_zpotrs_sp1dplus_TRSM_BACKWARD_as_expr_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;
  int gcblk2list = assignments[1].value;
  int firstbrow = assignments[2].value;
  int lastbrow = assignments[3].value;
  int firstblok = assignments[4].value;

  /* Silent Warnings: should look into predicate to know what variables are usefull */
  (void)__dague_object;
  (void)k;
  (void)gcblk2list;
  (void)firstbrow;
  (void)lastbrow;
  (void)firstblok;
  /* Compute Predicate */
  return TRSM_BACKWARD_pred(k, gcblk2list, firstbrow, lastbrow, firstblok);
}
static const expr_t pred_of_zpotrs_sp1dplus_TRSM_BACKWARD_as_expr = {
  .op = EXPR_OP_INLINE,
  .inline_func = pred_of_zpotrs_sp1dplus_TRSM_BACKWARD_as_expr_fct
};
static inline int expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_A_dep1_atline_142_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_A_dep1_atline_142 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_A_dep1_atline_142_fct
};
static const dep_t flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_A_dep1_atline_142 = {
  .cond = NULL,
  .dague = &zpotrs_sp1dplus_A,
  .datatype_index = DAGUE_zpotrs_sp1dplus_DEFAULT_ARENA,  .call_params = {
    &expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_A_dep1_atline_142
  }
};
static const dague_flow_t flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_A = {
  .name = "A",
  .sym_type = SYM_IN,
  .access_type = ACCESS_READ,
  .flow_index = 0,
  .dep_in  = { &flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_A_dep1_atline_142 },
  .dep_out = { NULL }
};

static inline int expr_of_cond_for_flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B_dep1_iftrue_atline_143_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int gcblk2list = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return (gcblk2list <  0);
}
static const expr_t expr_of_cond_for_flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B_dep1_iftrue_atline_143 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B_dep1_iftrue_atline_143_fct
};
static inline int expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B_dep1_iftrue_atline_143_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B_dep1_iftrue_atline_143 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B_dep1_iftrue_atline_143_fct
};
static const dep_t flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B_dep1_iftrue_atline_143 = {
  .cond = &expr_of_cond_for_flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B_dep1_iftrue_atline_143,
  .dague = &zpotrs_sp1dplus_B,
  .datatype_index = DAGUE_zpotrs_sp1dplus_DEFAULT_ARENA,  .call_params = {
    &expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B_dep1_iftrue_atline_143
  }
};
static inline int expr_of_cond_for_flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B_dep1_iffalse_atline_143_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int gcblk2list = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return !(gcblk2list <  0);
}
static const expr_t expr_of_cond_for_flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B_dep1_iffalse_atline_143 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B_dep1_iffalse_atline_143_fct
};
static inline int rangemin_of_expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B_dep1_iffalse_atline_143_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int firstbrow = assignments[2].value;

  (void)__dague_object; (void)assignments;
  return firstbrow;
}
static const expr_t rangemin_of_expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B_dep1_iffalse_atline_143 = {
  .op = EXPR_OP_INLINE,
  .inline_func = rangemin_of_expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B_dep1_iffalse_atline_143_fct
};
static inline int rangemax_of_expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B_dep1_iffalse_atline_143_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int lastbrow = assignments[3].value;

  (void)__dague_object; (void)assignments;
  return lastbrow;
}
static const expr_t rangemax_of_expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B_dep1_iffalse_atline_143 = {
  .op = EXPR_OP_INLINE,
  .inline_func = rangemax_of_expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B_dep1_iffalse_atline_143_fct
};
static const expr_t expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B_dep1_iffalse_atline_143 = {
  .op = EXPR_OP_BINARY_RANGE,
  .u_expr.binary = {
    .op1 = &rangemin_of_expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B_dep1_iffalse_atline_143,
    .op2 = &rangemax_of_expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B_dep1_iffalse_atline_143
  }
};
static const dep_t flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B_dep1_iffalse_atline_143 = {
  .cond = &expr_of_cond_for_flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B_dep1_iffalse_atline_143,
  .dague = &zpotrs_sp1dplus_GEMM_BACKWARD,
  .flow = &flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_B,
  .datatype_index = DAGUE_zpotrs_sp1dplus_DEFAULT_ARENA,  .call_params = {
    &expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B_dep1_iffalse_atline_143
  }
};
static inline int expr_of_cond_for_flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B_dep2_iftrue_atline_154_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return (k == cblknbr);
}
static const expr_t expr_of_cond_for_flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B_dep2_iftrue_atline_154 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B_dep2_iftrue_atline_154_fct
};
static inline int expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B_dep2_iftrue_atline_154_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B_dep2_iftrue_atline_154 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B_dep2_iftrue_atline_154_fct
};
static const dep_t flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B_dep2_iftrue_atline_154 = {
  .cond = &expr_of_cond_for_flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B_dep2_iftrue_atline_154,
  .dague = &zpotrs_sp1dplus_TRSM_FORWARD,
  .flow = &flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B,
  .datatype_index = DAGUE_zpotrs_sp1dplus_DEFAULT_ARENA,  .call_params = {
    &expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B_dep2_iftrue_atline_154
  }
};
static inline int expr_of_cond_for_flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B_dep2_iffalse_atline_154_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return !(k == cblknbr);
}
static const expr_t expr_of_cond_for_flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B_dep2_iffalse_atline_154 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B_dep2_iffalse_atline_154_fct
};
static inline int expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B_dep2_iffalse_atline_154_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int firstblok = assignments[4].value;

  (void)__dague_object; (void)assignments;
  return firstblok;
}
static const expr_t expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B_dep2_iffalse_atline_154 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B_dep2_iffalse_atline_154_fct
};
static const dep_t flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B_dep2_iffalse_atline_154 = {
  .cond = &expr_of_cond_for_flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B_dep2_iffalse_atline_154,
  .dague = &zpotrs_sp1dplus_GEMM_BACKWARD,
  .flow = &flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C,
  .datatype_index = DAGUE_zpotrs_sp1dplus_DEFAULT_ARENA,  .call_params = {
    &expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B_dep2_iffalse_atline_154
  }
};
static const dague_flow_t flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B = {
  .name = "B",
  .sym_type = SYM_INOUT,
  .access_type = ACCESS_RW,
  .flow_index = 1,
  .dep_in  = { &flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B_dep2_iftrue_atline_154, &flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B_dep2_iffalse_atline_154 },
  .dep_out = { &flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B_dep1_iftrue_atline_143, &flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B_dep1_iffalse_atline_143 }
};

static void
iterate_successors_of_zpotrs_sp1dplus_TRSM_BACKWARD(dague_execution_unit_t *eu, dague_execution_context_t *exec_context,
               uint32_t action_mask, dague_ontask_function_t *ontask, void *ontask_arg)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)exec_context->dague_object;
  dague_execution_context_t nc;
  dague_arena_t* arena = NULL;
  int rank_src = 0, rank_dst = 0;
  int k = exec_context->locals[0].value;
  int gcblk2list = exec_context->locals[1].value;
  int firstbrow = exec_context->locals[2].value;
  int lastbrow = exec_context->locals[3].value;
  int firstblok = exec_context->locals[4].value;
  (void)rank_src; (void)rank_dst; (void)__dague_object;
  (void)k;  (void)gcblk2list;  (void)firstbrow;  (void)lastbrow;  (void)firstblok;
  nc.dague_object = exec_context->dague_object;
#if defined(DISTRIBUTED)
  rank_src = ((dague_ddesc_t*)__dague_object->super.B)->rank_of((dague_ddesc_t*)__dague_object->super.B, k);
#endif
  /* Flow of data A has only IN dependencies */
  /* Flow of Data B */
  if( action_mask & (1 << 0) ) {
#if defined(DISTRIBUTED)
    arena = __dague_object->super.arenas[DAGUE_zpotrs_sp1dplus_DEFAULT_ARENA];
#endif  /* defined(DISTRIBUTED) */
    if( !((gcblk2list <  0)) ) {
      nc.function = (const dague_function_t*)&zpotrs_sp1dplus_GEMM_BACKWARD;
      {
        int GEMM_BACKWARD_k;
        for( GEMM_BACKWARD_k = firstbrow;GEMM_BACKWARD_k <= lastbrow; GEMM_BACKWARD_k++ ) {
          if( (GEMM_BACKWARD_k >= (0)) && (GEMM_BACKWARD_k <= (browsize)) ) {
            nc.locals[0].value = GEMM_BACKWARD_k;
            const int GEMM_BACKWARD_bloknum = zpotrs_sp1dplus_inline_c_expr6_line_164((const dague_object_t*)__dague_object, nc.locals);
            nc.locals[1].value = GEMM_BACKWARD_bloknum;
            const int GEMM_BACKWARD_fcblk = zpotrs_sp1dplus_inline_c_expr5_line_165((const dague_object_t*)__dague_object, nc.locals);
            nc.locals[2].value = GEMM_BACKWARD_fcblk;
            const int GEMM_BACKWARD_cblk = zpotrs_sp1dplus_inline_c_expr4_line_166((const dague_object_t*)__dague_object, nc.locals);
            nc.locals[3].value = GEMM_BACKWARD_cblk;
            const int GEMM_BACKWARD_gcblk2list = zpotrs_sp1dplus_inline_c_expr3_line_168((const dague_object_t*)__dague_object, nc.locals);
            nc.locals[4].value = GEMM_BACKWARD_gcblk2list;
            const int GEMM_BACKWARD_firstbrow = zpotrs_sp1dplus_inline_c_expr2_line_169((const dague_object_t*)__dague_object, nc.locals);
            nc.locals[5].value = GEMM_BACKWARD_firstbrow;
            const int GEMM_BACKWARD_lastbrow = zpotrs_sp1dplus_inline_c_expr1_line_170((const dague_object_t*)__dague_object, nc.locals);
            nc.locals[6].value = GEMM_BACKWARD_lastbrow;
#if defined(DISTRIBUTED)
              rank_dst = ((dague_ddesc_t*)__dague_object->super.B)->rank_of((dague_ddesc_t*)__dague_object->super.B, GEMM_BACKWARD_cblk);
#endif
#if defined(DAGUE_DEBUG)
            if( NULL != eu ) {
              char tmp[128], tmp1[128];
              DEBUG(("thread %d release deps of B:%s to B:%s (from node %d to %d)\n", eu->eu_id,
                     dague_service_to_string(exec_context, tmp, 128),
                     dague_service_to_string(&nc, tmp1, 128), rank_src, rank_dst));
            }
#endif
              nc.priority = 0;
              if( DAGUE_ITERATE_STOP == ontask(eu, &nc, exec_context, 0, 1, rank_src, rank_dst, arena, ontask_arg) )
                return;
      }
        }
          }
    }
  }
  (void)nc;(void)arena;(void)eu;(void)ontask;(void)ontask_arg;(void)rank_dst;(void)action_mask;
}

static int release_deps_of_zpotrs_sp1dplus_TRSM_BACKWARD(dague_execution_unit_t *eu, dague_execution_context_t *context, uint32_t action_mask, dague_remote_deps_t *deps)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t *)context->dague_object;
  dague_release_dep_fct_arg_t arg;
  arg.nb_released = 0;
  arg.output_usage = 0;
  arg.action_mask = action_mask;
  arg.deps = deps;
  arg.ready_list = NULL;
  (void)__dague_object;
  if( action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS ) {
    arg.output_entry = data_repo_lookup_entry_and_create( eu, TRSM_BACKWARD_repo, TRSM_BACKWARD_hash(__dague_object, context->locals) );
  }
#if defined(DAGUE_SIM)
  assert(arg.output_entry->sim_exec_date == 0);
  arg.output_entry->sim_exec_date = context->sim_exec_date;
#endif
#if defined(DISTRIBUTED)
  arg.remote_deps_count = 0;
  arg.remote_deps = NULL;
#endif
  iterate_successors_of_zpotrs_sp1dplus_TRSM_BACKWARD(eu, context, action_mask, dague_release_dep_fct, &arg);

  if(action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS) {
    data_repo_entry_addto_usage_limit(TRSM_BACKWARD_repo, arg.output_entry->key, arg.output_usage);
    if( NULL != arg.ready_list ) {
      __dague_schedule(eu, arg.ready_list);
      arg.ready_list = NULL;
    }
  }
#if defined(DISTRIBUTED)
  if( (action_mask & DAGUE_ACTION_SEND_REMOTE_DEPS) && arg.remote_deps_count ) {
    arg.nb_released += dague_remote_dep_activate(eu, context, arg.remote_deps, arg.remote_deps_count);
  }
#endif
  if( action_mask & DAGUE_ACTION_RELEASE_LOCAL_REFS ) {
    int k = context->locals[0].value;
    int gcblk2list = context->locals[1].value;
    int firstbrow = context->locals[2].value;
    int lastbrow = context->locals[3].value;
    int firstblok = context->locals[4].value;
    (void)k; (void)gcblk2list; (void)firstbrow; (void)lastbrow; (void)firstblok;

    if( (k == cblknbr) ) {
      data_repo_entry_used_once( eu, TRSM_FORWARD_repo, context->data[1].data_repo->key );
      (void)AUNREF(context->data[1].data);
    } else {
      data_repo_entry_used_once( eu, GEMM_BACKWARD_repo, context->data[1].data_repo->key );
      (void)AUNREF(context->data[1].data);
    }
  }
  assert( NULL == arg.ready_list );
  return arg.nb_released;
}

static int hook_of_zpotrs_sp1dplus_TRSM_BACKWARD(dague_execution_unit_t *context, dague_execution_context_t *exec_context)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (__dague_zpotrs_sp1dplus_internal_object_t *)exec_context->dague_object;
  assignment_t tass[MAX_PARAM_COUNT];
  (void)context; (void)__dague_object; (void)tass;
#if defined(DAGUE_SIM)
  int __dague_simulation_date = 0;
#endif
  int k = exec_context->locals[0].value;
  int gcblk2list = exec_context->locals[1].value;
  int firstbrow = exec_context->locals[2].value;
  int lastbrow = exec_context->locals[3].value;
  int firstblok = exec_context->locals[4].value;

  (void)k;  (void)gcblk2list;  (void)firstbrow;  (void)lastbrow;  (void)firstblok;

  /** Declare the variables that will hold the data, and all the accounting for each */
  void *A = NULL; (void)A;
  dague_arena_chunk_t *gA = NULL; (void)gA;
  data_repo_entry_t *eA = NULL; (void)eA;
  void *B = NULL; (void)B;
  dague_arena_chunk_t *gB = NULL; (void)gB;
  data_repo_entry_t *eB = NULL; (void)eB;

  /** Lookup the input data, and store them in the context */
  eA = exec_context->data[0].data_repo;
  gA = exec_context->data[0].data;
  if( NULL == gA ) {
  gA = (dague_arena_chunk_t*) A(k);
    exec_context->data[0].data = gA;
    exec_context->data[0].data_repo = eA;
  }
  A = ADATA(gA);
#if defined(DAGUE_SIM)
  if( (NULL != eA) && (eA->sim_exec_date > __dague_simulation_date) )
    __dague_simulation_date =  eA->sim_exec_date;
#endif
  eB = exec_context->data[1].data_repo;
  gB = exec_context->data[1].data;
  if( NULL == gB ) {
  if( (k == cblknbr) ) {
      tass[0].value = k;
    eB = data_repo_lookup_entry( TRSM_FORWARD_repo, TRSM_FORWARD_hash( __dague_object, tass ));
    gB = eB->data[0];
  } else {
      tass[0].value = firstblok;
    eB = data_repo_lookup_entry( GEMM_BACKWARD_repo, GEMM_BACKWARD_hash( __dague_object, tass ));
    gB = eB->data[0];
  }
    exec_context->data[1].data = gB;
    exec_context->data[1].data_repo = eB;
  }
  B = ADATA(gB);
#if defined(DAGUE_SIM)
  if( (NULL != eB) && (eB->sim_exec_date > __dague_simulation_date) )
    __dague_simulation_date =  eB->sim_exec_date;
#endif
#if defined(DAGUE_SIM)
  if( exec_context->function->sim_cost_fct != NULL ) {
    exec_context->sim_exec_date = __dague_simulation_date + exec_context->function->sim_cost_fct(exec_context);
  } else {
    exec_context->sim_exec_date = __dague_simulation_date;
  }
  if( context->largest_simulation_date < exec_context->sim_exec_date )
    context->largest_simulation_date = exec_context->sim_exec_date;
#endif
  /** PAPI events */
#if defined(HAVE_PAPI)
  papime_start_thread_counters();
#endif
  /** Cache Awareness Accounting */
#if defined(DAGUE_CACHE_AWARENESS)
  cache_buf_referenced(context->closest_cache, A);
  cache_buf_referenced(context->closest_cache, B);
#endif /* DAGUE_CACHE_AWARENESS */


#if !defined(DAGUE_PROF_DRY_BODY)

/*--------------------------------------------------------------------------------*
 *                              TRSM_BACKWARD BODY                              *
 *--------------------------------------------------------------------------------*/

  TAKE_TIME(context, 2*exec_context->function->function_id, TRSM_BACKWARD_hash( __dague_object, exec_context->locals), __dague_object->super.B, ((dague_ddesc_t*)(__dague_object->super.B))->data_key((dague_ddesc_t*)__dague_object->super.B, k) );
#line 145 "zpotrs_sp1dplus.jdf"
      DRYRUN(
             /*core_zpotrfsp1d(A, datacode, k, descA->pastix_data->sopar.espilondiag );*/
             /*TRSM( L, L , N, N, A, B )*/
	     );
      printlog(
               "thread %d solvedown_1dplus( cblknum=%d, browk=%d, browk1=%d, lastbrow=%d, firstblok=%d, lastblok=%d )\n",
               context->eu_id, k, browk, browk1, lastbrow, firstblok, lastblok);


#line 1856 "zpotrs_sp1dplus.c"
/*--------------------------------------------------------------------------------*
 *                          END OF TRSM_BACKWARD BODY                            *
 *--------------------------------------------------------------------------------*/



#endif /*!defined(DAGUE_PROF_DRY_BODY)*/

  return 0;
}
static int complete_hook_of_zpotrs_sp1dplus_TRSM_BACKWARD(dague_execution_unit_t *context, dague_execution_context_t *exec_context)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (__dague_zpotrs_sp1dplus_internal_object_t *)exec_context->dague_object;
  (void)context; (void)__dague_object;
  int k = exec_context->locals[0].value;
  int gcblk2list = exec_context->locals[1].value;
  int firstbrow = exec_context->locals[2].value;
  int lastbrow = exec_context->locals[3].value;
  int firstblok = exec_context->locals[4].value;

  (void)k;  (void)gcblk2list;  (void)firstbrow;  (void)lastbrow;  (void)firstblok;

  TAKE_TIME(context,2*exec_context->function->function_id+1, TRSM_BACKWARD_hash( __dague_object, exec_context->locals ), NULL, 0);
  /** PAPI events */
#if defined(HAVE_PAPI)
  papime_stop_thread_counters();
#endif
#if defined(DISTRIBUTED)
  /** If not working on distributed, there is no risk that data is not in place */
  if( (gcblk2list <  0) ) {
    if( ADATA(exec_context->data[1].data) != B(k) ) {
      dague_remote_dep_memcpy( context, exec_context->dague_object, B(k), exec_context->data[1].data, 
                               __dague_object->super.arenas[DAGUE_zpotrs_sp1dplus_DEFAULT_ARENA]->opaque_dtt );
    }
  }
#endif /* DISTRIBUTED */
  dague_prof_grapher_task(exec_context, context->eu_id, TRSM_BACKWARD_hash(__dague_object, exec_context->locals));
  {
    release_deps_of_zpotrs_sp1dplus_TRSM_BACKWARD(context, exec_context,
        DAGUE_ACTION_RELEASE_REMOTE_DEPS |
        DAGUE_ACTION_RELEASE_LOCAL_DEPS |
        DAGUE_ACTION_RELEASE_LOCAL_REFS |
        DAGUE_ACTION_DEPS_MASK,
        NULL);
  }
  return 0;
}

static int zpotrs_sp1dplus_TRSM_BACKWARD_internal_init(__dague_zpotrs_sp1dplus_internal_object_t *__dague_object)
{
  dague_dependencies_t *dep = NULL;
  assignment_t assignments[MAX_LOCAL_COUNT];(void) assignments;
  int nb_tasks = 0, __foundone = 0;
  int32_t  k, gcblk2list, firstbrow, lastbrow, firstblok;
  int32_t  k_min = 0x7fffffff;
  int32_t  k_max = 0;
  (void)__dague_object; (void)__foundone;
  /* First, find the min and max value for each of the dimensions */
  for(k = 0;
      k <= cblknbr;
      k++) {
    assignments[0].value = k;
    gcblk2list = zpotrs_sp1dplus_inline_c_expr10_line_131((const dague_object_t*)__dague_object, assignments);
    assignments[1].value = gcblk2list;
    firstbrow = zpotrs_sp1dplus_inline_c_expr9_line_132((const dague_object_t*)__dague_object, assignments);
    assignments[2].value = firstbrow;
    lastbrow = zpotrs_sp1dplus_inline_c_expr8_line_133((const dague_object_t*)__dague_object, assignments);
    assignments[3].value = lastbrow;
    firstblok = zpotrs_sp1dplus_inline_c_expr7_line_134((const dague_object_t*)__dague_object, assignments);
    assignments[4].value = firstblok;
    if( !TRSM_BACKWARD_pred(k, gcblk2list, firstbrow, lastbrow, firstblok) ) continue;
    nb_tasks++;
    k_max = dague_imax(k_max, k);
    k_min = dague_imin(k_min, k);
  }

  /**
   * Now, for each of the dimensions, re-iterate on the space,
   * and if at least one value is defined, allocate arrays to point
   * to it. Array dimensions are defined by the (rough) observation above
   **/
  if( 0 != nb_tasks ) {
    ALLOCATE_DEP_TRACKING(dep, k_min, k_max, "k", &symb_zpotrs_sp1dplus_TRSM_BACKWARD_k, NULL, DAGUE_DEPENDENCIES_FLAG_FINAL);
  }
  __dague_object->super.super.dependencies_array[1] = dep;
  __dague_object->super.super.nb_local_tasks += nb_tasks;
  return nb_tasks;
}

static const dague_function_t zpotrs_sp1dplus_TRSM_BACKWARD = {
  .name = "TRSM_BACKWARD",
  .deps = 1,
  .flags = 0x0 | DAGUE_HAS_IN_IN_DEPENDENCIES,
  .function_id = 1,
  .dependencies_goal = 0x3,
  .nb_parameters = 1,
  .nb_definitions = 5,
  .params = { &symb_zpotrs_sp1dplus_TRSM_BACKWARD_k },
  .locals = { &symb_zpotrs_sp1dplus_TRSM_BACKWARD_k, &symb_zpotrs_sp1dplus_TRSM_BACKWARD_gcblk2list, &symb_zpotrs_sp1dplus_TRSM_BACKWARD_firstbrow, &symb_zpotrs_sp1dplus_TRSM_BACKWARD_lastbrow, &symb_zpotrs_sp1dplus_TRSM_BACKWARD_firstblok },
  .pred = &pred_of_zpotrs_sp1dplus_TRSM_BACKWARD_as_expr,
  .priority = NULL,
  .in = { &flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_A, &flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B },
  .out = { &flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B },
  .iterate_successors = iterate_successors_of_zpotrs_sp1dplus_TRSM_BACKWARD,
  .release_deps = release_deps_of_zpotrs_sp1dplus_TRSM_BACKWARD,
  .hook = hook_of_zpotrs_sp1dplus_TRSM_BACKWARD,
  .complete_execution = complete_hook_of_zpotrs_sp1dplus_TRSM_BACKWARD,
#if defined(DAGUE_SIM)
  .sim_cost_fct = NULL,
#endif
  .key = (dague_functionkey_fn_t*)TRSM_BACKWARD_hash,
};


/**********************************************************************************
 *                                  GEMM_FORWARD                                  *
 **********************************************************************************/

static inline int minexpr_of_symb_zpotrs_sp1dplus_GEMM_FORWARD_k_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return 0;
}
static const expr_t minexpr_of_symb_zpotrs_sp1dplus_GEMM_FORWARD_k = {
  .op = EXPR_OP_INLINE,
  .inline_func = minexpr_of_symb_zpotrs_sp1dplus_GEMM_FORWARD_k_fct
};
static inline int maxexpr_of_symb_zpotrs_sp1dplus_GEMM_FORWARD_k_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return bloknbr;
}
static const expr_t maxexpr_of_symb_zpotrs_sp1dplus_GEMM_FORWARD_k = {
  .op = EXPR_OP_INLINE,
  .inline_func = maxexpr_of_symb_zpotrs_sp1dplus_GEMM_FORWARD_k_fct
};
static const symbol_t symb_zpotrs_sp1dplus_GEMM_FORWARD_k = {.min = &minexpr_of_symb_zpotrs_sp1dplus_GEMM_FORWARD_k, .max = &maxexpr_of_symb_zpotrs_sp1dplus_GEMM_FORWARD_k,  .flags = DAGUE_SYMBOL_IS_STANDALONE};

static inline int expr_of_symb_zpotrs_sp1dplus_GEMM_FORWARD_fcblk_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zpotrs_sp1dplus_inline_c_expr15_line_85((const dague_object_t*)__dague_object, assignments);
}
static const expr_t expr_of_symb_zpotrs_sp1dplus_GEMM_FORWARD_fcblk = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_symb_zpotrs_sp1dplus_GEMM_FORWARD_fcblk_fct
};
static const symbol_t symb_zpotrs_sp1dplus_GEMM_FORWARD_fcblk = {.min = &expr_of_symb_zpotrs_sp1dplus_GEMM_FORWARD_fcblk, .max = &expr_of_symb_zpotrs_sp1dplus_GEMM_FORWARD_fcblk,  .flags = 0x0};

static inline int expr_of_symb_zpotrs_sp1dplus_GEMM_FORWARD_cblk_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zpotrs_sp1dplus_inline_c_expr14_line_86((const dague_object_t*)__dague_object, assignments);
}
static const expr_t expr_of_symb_zpotrs_sp1dplus_GEMM_FORWARD_cblk = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_symb_zpotrs_sp1dplus_GEMM_FORWARD_cblk_fct
};
static const symbol_t symb_zpotrs_sp1dplus_GEMM_FORWARD_cblk = {.min = &expr_of_symb_zpotrs_sp1dplus_GEMM_FORWARD_cblk, .max = &expr_of_symb_zpotrs_sp1dplus_GEMM_FORWARD_cblk,  .flags = 0x0};

static inline int expr_of_symb_zpotrs_sp1dplus_GEMM_FORWARD_phony_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zpotrs_sp1dplus_inline_c_expr13_line_87((const dague_object_t*)__dague_object, assignments);
}
static const expr_t expr_of_symb_zpotrs_sp1dplus_GEMM_FORWARD_phony = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_symb_zpotrs_sp1dplus_GEMM_FORWARD_phony_fct
};
static const symbol_t symb_zpotrs_sp1dplus_GEMM_FORWARD_phony = {.min = &expr_of_symb_zpotrs_sp1dplus_GEMM_FORWARD_phony, .max = &expr_of_symb_zpotrs_sp1dplus_GEMM_FORWARD_phony,  .flags = 0x0};

static inline int expr_of_symb_zpotrs_sp1dplus_GEMM_FORWARD_prev_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zpotrs_sp1dplus_inline_c_expr12_line_88((const dague_object_t*)__dague_object, assignments);
}
static const expr_t expr_of_symb_zpotrs_sp1dplus_GEMM_FORWARD_prev = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_symb_zpotrs_sp1dplus_GEMM_FORWARD_prev_fct
};
static const symbol_t symb_zpotrs_sp1dplus_GEMM_FORWARD_prev = {.min = &expr_of_symb_zpotrs_sp1dplus_GEMM_FORWARD_prev, .max = &expr_of_symb_zpotrs_sp1dplus_GEMM_FORWARD_prev,  .flags = 0x0};

static inline int expr_of_symb_zpotrs_sp1dplus_GEMM_FORWARD_next_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zpotrs_sp1dplus_inline_c_expr11_line_89((const dague_object_t*)__dague_object, assignments);
}
static const expr_t expr_of_symb_zpotrs_sp1dplus_GEMM_FORWARD_next = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_symb_zpotrs_sp1dplus_GEMM_FORWARD_next_fct
};
static const symbol_t symb_zpotrs_sp1dplus_GEMM_FORWARD_next = {.min = &expr_of_symb_zpotrs_sp1dplus_GEMM_FORWARD_next, .max = &expr_of_symb_zpotrs_sp1dplus_GEMM_FORWARD_next,  .flags = 0x0};

static inline int pred_of_zpotrs_sp1dplus_GEMM_FORWARD_as_expr_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;
  int fcblk = assignments[1].value;
  int cblk = assignments[2].value;
  int phony = assignments[3].value;
  int prev = assignments[4].value;
  int next = assignments[5].value;

  /* Silent Warnings: should look into predicate to know what variables are usefull */
  (void)__dague_object;
  (void)k;
  (void)fcblk;
  (void)cblk;
  (void)phony;
  (void)prev;
  (void)next;
  /* Compute Predicate */
  return GEMM_FORWARD_pred(k, fcblk, cblk, phony, prev, next);
}
static const expr_t pred_of_zpotrs_sp1dplus_GEMM_FORWARD_as_expr = {
  .op = EXPR_OP_INLINE,
  .inline_func = pred_of_zpotrs_sp1dplus_GEMM_FORWARD_as_expr_fct
};
static inline int expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_A_dep1_atline_96_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int fcblk = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return fcblk;
}
static const expr_t expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_A_dep1_atline_96 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_A_dep1_atline_96_fct
};
static const dep_t flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_A_dep1_atline_96 = {
  .cond = NULL,
  .dague = &zpotrs_sp1dplus_A,
  .datatype_index = DAGUE_zpotrs_sp1dplus_DEFAULT_ARENA,  .call_params = {
    &expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_A_dep1_atline_96
  }
};
static const dague_flow_t flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_A = {
  .name = "A",
  .sym_type = SYM_IN,
  .access_type = ACCESS_READ,
  .flow_index = 0,
  .dep_in  = { &flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_A_dep1_atline_96 },
  .dep_out = { NULL }
};

static inline int expr_of_cond_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_B_dep1_iftrue_atline_97_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int phony = assignments[3].value;

  (void)__dague_object; (void)assignments;
  return phony;
}
static const expr_t expr_of_cond_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_B_dep1_iftrue_atline_97 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_B_dep1_iftrue_atline_97_fct
};
static inline int expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_B_dep1_iftrue_atline_97_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int fcblk = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return fcblk;
}
static const expr_t expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_B_dep1_iftrue_atline_97 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_B_dep1_iftrue_atline_97_fct
};
static const dep_t flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_B_dep1_iftrue_atline_97 = {
  .cond = &expr_of_cond_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_B_dep1_iftrue_atline_97,
  .dague = &zpotrs_sp1dplus_B,
  .datatype_index = DAGUE_zpotrs_sp1dplus_DEFAULT_ARENA,  .call_params = {
    &expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_B_dep1_iftrue_atline_97
  }
};
static inline int expr_of_cond_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_B_dep1_iffalse_atline_97_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int phony = assignments[3].value;

  (void)__dague_object; (void)assignments;
  return !phony;
}
static const expr_t expr_of_cond_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_B_dep1_iffalse_atline_97 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_B_dep1_iffalse_atline_97_fct
};
static inline int expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_B_dep1_iffalse_atline_97_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int cblk = assignments[2].value;

  (void)__dague_object; (void)assignments;
  return cblk;
}
static const expr_t expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_B_dep1_iffalse_atline_97 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_B_dep1_iffalse_atline_97_fct
};
static const dep_t flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_B_dep1_iffalse_atline_97 = {
  .cond = &expr_of_cond_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_B_dep1_iffalse_atline_97,
  .dague = &zpotrs_sp1dplus_TRSM_FORWARD,
  .flow = &flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B,
  .datatype_index = DAGUE_zpotrs_sp1dplus_DEFAULT_ARENA,  .call_params = {
    &expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_B_dep1_iffalse_atline_97
  }
};
static const dague_flow_t flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_B = {
  .name = "B",
  .sym_type = SYM_IN,
  .access_type = ACCESS_READ,
  .flow_index = 1,
  .dep_in  = { &flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_B_dep1_iftrue_atline_97, &flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_B_dep1_iffalse_atline_97 },
  .dep_out = { NULL }
};

static inline int expr_of_cond_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep1_iftrue_atline_98_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int prev = assignments[4].value;

  (void)__dague_object; (void)assignments;
  return (prev == 0);
}
static const expr_t expr_of_cond_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep1_iftrue_atline_98 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep1_iftrue_atline_98_fct
};
static inline int expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep1_iftrue_atline_98_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int fcblk = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return fcblk;
}
static const expr_t expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep1_iftrue_atline_98 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep1_iftrue_atline_98_fct
};
static const dep_t flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep1_iftrue_atline_98 = {
  .cond = &expr_of_cond_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep1_iftrue_atline_98,
  .dague = &zpotrs_sp1dplus_B,
  .datatype_index = DAGUE_zpotrs_sp1dplus_DEFAULT_ARENA,  .call_params = {
    &expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep1_iftrue_atline_98
  }
};
static inline int expr_of_cond_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep1_iffalse_atline_98_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int prev = assignments[4].value;

  (void)__dague_object; (void)assignments;
  return !(prev == 0);
}
static const expr_t expr_of_cond_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep1_iffalse_atline_98 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep1_iffalse_atline_98_fct
};
static inline int expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep1_iffalse_atline_98_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int prev = assignments[4].value;

  (void)__dague_object; (void)assignments;
  return prev;
}
static const expr_t expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep1_iffalse_atline_98 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep1_iffalse_atline_98_fct
};
static const dep_t flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep1_iffalse_atline_98 = {
  .cond = &expr_of_cond_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep1_iffalse_atline_98,
  .dague = &zpotrs_sp1dplus_GEMM_FORWARD,
  .flow = &flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C,
  .datatype_index = DAGUE_zpotrs_sp1dplus_DEFAULT_ARENA,  .call_params = {
    &expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep1_iffalse_atline_98
  }
};
static inline int expr_of_cond_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep2_atline_99_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int phony = assignments[3].value;

  (void)__dague_object; (void)assignments;
  return phony;
}
static const expr_t expr_of_cond_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep2_atline_99 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep2_atline_99_fct
};
static inline int expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep2_atline_99_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int fcblk = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return fcblk;
}
static const expr_t expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep2_atline_99 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep2_atline_99_fct
};
static const dep_t flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep2_atline_99 = {
  .cond = &expr_of_cond_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep2_atline_99,
  .dague = &zpotrs_sp1dplus_B,
  .datatype_index = DAGUE_zpotrs_sp1dplus_DEFAULT_ARENA,  .call_params = {
    &expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep2_atline_99
  }
};
static inline int expr_of_cond_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep3_atline_100_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int phony = assignments[3].value;
  int next = assignments[5].value;

  (void)__dague_object; (void)assignments;
  return (!phony && (next == 0));
}
static const expr_t expr_of_cond_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep3_atline_100 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep3_atline_100_fct
};
static inline int expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep3_atline_100_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int fcblk = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return fcblk;
}
static const expr_t expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep3_atline_100 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep3_atline_100_fct
};
static const dep_t flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep3_atline_100 = {
  .cond = &expr_of_cond_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep3_atline_100,
  .dague = &zpotrs_sp1dplus_TRSM_FORWARD,
  .flow = &flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B,
  .datatype_index = DAGUE_zpotrs_sp1dplus_DEFAULT_ARENA,  .call_params = {
    &expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep3_atline_100
  }
};
static inline int expr_of_cond_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep4_atline_120_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int phony = assignments[3].value;
  int next = assignments[5].value;

  (void)__dague_object; (void)assignments;
  return (!phony && (next != 0));
}
static const expr_t expr_of_cond_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep4_atline_120 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep4_atline_120_fct
};
static inline int expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep4_atline_120_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int next = assignments[5].value;

  (void)__dague_object; (void)assignments;
  return next;
}
static const expr_t expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep4_atline_120 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep4_atline_120_fct
};
static const dep_t flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep4_atline_120 = {
  .cond = &expr_of_cond_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep4_atline_120,
  .dague = &zpotrs_sp1dplus_GEMM_FORWARD,
  .flow = &flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C,
  .datatype_index = DAGUE_zpotrs_sp1dplus_DEFAULT_ARENA,  .call_params = {
    &expr_of_p1_for_flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep4_atline_120
  }
};
static const dague_flow_t flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C = {
  .name = "C",
  .sym_type = SYM_INOUT,
  .access_type = ACCESS_RW,
  .flow_index = 2,
  .dep_in  = { &flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep1_iftrue_atline_98, &flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep1_iffalse_atline_98 },
  .dep_out = { &flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep2_atline_99, &flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep3_atline_100, &flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C_dep4_atline_120 }
};

static void
iterate_successors_of_zpotrs_sp1dplus_GEMM_FORWARD(dague_execution_unit_t *eu, dague_execution_context_t *exec_context,
               uint32_t action_mask, dague_ontask_function_t *ontask, void *ontask_arg)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)exec_context->dague_object;
  dague_execution_context_t nc;
  dague_arena_t* arena = NULL;
  int rank_src = 0, rank_dst = 0;
  int k = exec_context->locals[0].value;
  int fcblk = exec_context->locals[1].value;
  int cblk = exec_context->locals[2].value;
  int phony = exec_context->locals[3].value;
  int prev = exec_context->locals[4].value;
  int next = exec_context->locals[5].value;
  (void)rank_src; (void)rank_dst; (void)__dague_object;
  (void)k;  (void)fcblk;  (void)cblk;  (void)phony;  (void)prev;  (void)next;
  nc.dague_object = exec_context->dague_object;
#if defined(DISTRIBUTED)
  rank_src = ((dague_ddesc_t*)__dague_object->super.B)->rank_of((dague_ddesc_t*)__dague_object->super.B, fcblk);
#endif
  /* Flow of data A has only IN dependencies */
  /* Flow of data B has only IN dependencies */
  /* Flow of Data C */
  if( action_mask & (1 << 0) ) {
#if defined(DISTRIBUTED)
    arena = __dague_object->super.arenas[DAGUE_zpotrs_sp1dplus_DEFAULT_ARENA];
#endif  /* defined(DISTRIBUTED) */
#if defined(DISTRIBUTED)
    arena = __dague_object->super.arenas[DAGUE_zpotrs_sp1dplus_DEFAULT_ARENA];
#endif  /* defined(DISTRIBUTED) */
    if( (!phony && (next == 0)) ) {
      nc.function = (const dague_function_t*)&zpotrs_sp1dplus_TRSM_FORWARD;
      {
        const int TRSM_FORWARD_k = fcblk;
        if( (TRSM_FORWARD_k >= (0)) && (TRSM_FORWARD_k <= (cblknbr)) ) {
          nc.locals[0].value = TRSM_FORWARD_k;
          const int TRSM_FORWARD_gcblk2list = zpotrs_sp1dplus_inline_c_expr21_line_43((const dague_object_t*)__dague_object, nc.locals);
          nc.locals[1].value = TRSM_FORWARD_gcblk2list;
          const int TRSM_FORWARD_browk1 = zpotrs_sp1dplus_inline_c_expr20_line_45((const dague_object_t*)__dague_object, nc.locals);
          nc.locals[2].value = TRSM_FORWARD_browk1;
          const int TRSM_FORWARD_lastbrow = zpotrs_sp1dplus_inline_c_expr19_line_46((const dague_object_t*)__dague_object, nc.locals);
          nc.locals[3].value = TRSM_FORWARD_lastbrow;
          const int TRSM_FORWARD_firstblok = zpotrs_sp1dplus_inline_c_expr18_line_47((const dague_object_t*)__dague_object, nc.locals);
          nc.locals[4].value = TRSM_FORWARD_firstblok;
          const int TRSM_FORWARD_lastblok = zpotrs_sp1dplus_inline_c_expr17_line_48((const dague_object_t*)__dague_object, nc.locals);
          nc.locals[5].value = TRSM_FORWARD_lastblok;
          const int TRSM_FORWARD_firstgemm = zpotrs_sp1dplus_inline_c_expr16_line_54((const dague_object_t*)__dague_object, nc.locals);
          nc.locals[6].value = TRSM_FORWARD_firstgemm;
#if defined(DISTRIBUTED)
            rank_dst = ((dague_ddesc_t*)__dague_object->super.B)->rank_of((dague_ddesc_t*)__dague_object->super.B, TRSM_FORWARD_k);
#endif
#if defined(DAGUE_DEBUG)
          if( NULL != eu ) {
            char tmp[128], tmp1[128];
            DEBUG(("thread %d release deps of C:%s to B:%s (from node %d to %d)\n", eu->eu_id,
                   dague_service_to_string(exec_context, tmp, 128),
                   dague_service_to_string(&nc, tmp1, 128), rank_src, rank_dst));
          }
#endif
            nc.priority = 0;
            if( DAGUE_ITERATE_STOP == ontask(eu, &nc, exec_context, 0, 1, rank_src, rank_dst, arena, ontask_arg) )
              return;
      }
        }
    }
#if defined(DISTRIBUTED)
    arena = __dague_object->super.arenas[DAGUE_zpotrs_sp1dplus_DEFAULT_ARENA];
#endif  /* defined(DISTRIBUTED) */
    if( (!phony && (next != 0)) ) {
      nc.function = (const dague_function_t*)&zpotrs_sp1dplus_GEMM_FORWARD;
      {
        const int GEMM_FORWARD_k = next;
        if( (GEMM_FORWARD_k >= (0)) && (GEMM_FORWARD_k <= (bloknbr)) ) {
          nc.locals[0].value = GEMM_FORWARD_k;
          const int GEMM_FORWARD_fcblk = zpotrs_sp1dplus_inline_c_expr15_line_85((const dague_object_t*)__dague_object, nc.locals);
          nc.locals[1].value = GEMM_FORWARD_fcblk;
          const int GEMM_FORWARD_cblk = zpotrs_sp1dplus_inline_c_expr14_line_86((const dague_object_t*)__dague_object, nc.locals);
          nc.locals[2].value = GEMM_FORWARD_cblk;
          const int GEMM_FORWARD_phony = zpotrs_sp1dplus_inline_c_expr13_line_87((const dague_object_t*)__dague_object, nc.locals);
          nc.locals[3].value = GEMM_FORWARD_phony;
          const int GEMM_FORWARD_prev = zpotrs_sp1dplus_inline_c_expr12_line_88((const dague_object_t*)__dague_object, nc.locals);
          nc.locals[4].value = GEMM_FORWARD_prev;
          const int GEMM_FORWARD_next = zpotrs_sp1dplus_inline_c_expr11_line_89((const dague_object_t*)__dague_object, nc.locals);
          nc.locals[5].value = GEMM_FORWARD_next;
#if defined(DISTRIBUTED)
            rank_dst = ((dague_ddesc_t*)__dague_object->super.B)->rank_of((dague_ddesc_t*)__dague_object->super.B, GEMM_FORWARD_fcblk);
#endif
#if defined(DAGUE_DEBUG)
          if( NULL != eu ) {
            char tmp[128], tmp1[128];
            DEBUG(("thread %d release deps of C:%s to C:%s (from node %d to %d)\n", eu->eu_id,
                   dague_service_to_string(exec_context, tmp, 128),
                   dague_service_to_string(&nc, tmp1, 128), rank_src, rank_dst));
          }
#endif
            nc.priority = 0;
            if( DAGUE_ITERATE_STOP == ontask(eu, &nc, exec_context, 0, 2, rank_src, rank_dst, arena, ontask_arg) )
              return;
      }
        }
    }
  }
  (void)nc;(void)arena;(void)eu;(void)ontask;(void)ontask_arg;(void)rank_dst;(void)action_mask;
}

static int release_deps_of_zpotrs_sp1dplus_GEMM_FORWARD(dague_execution_unit_t *eu, dague_execution_context_t *context, uint32_t action_mask, dague_remote_deps_t *deps)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t *)context->dague_object;
  dague_release_dep_fct_arg_t arg;
  arg.nb_released = 0;
  arg.output_usage = 0;
  arg.action_mask = action_mask;
  arg.deps = deps;
  arg.ready_list = NULL;
  (void)__dague_object;
  if( action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS ) {
    arg.output_entry = data_repo_lookup_entry_and_create( eu, GEMM_FORWARD_repo, GEMM_FORWARD_hash(__dague_object, context->locals) );
  }
#if defined(DAGUE_SIM)
  assert(arg.output_entry->sim_exec_date == 0);
  arg.output_entry->sim_exec_date = context->sim_exec_date;
#endif
#if defined(DISTRIBUTED)
  arg.remote_deps_count = 0;
  arg.remote_deps = NULL;
#endif
  iterate_successors_of_zpotrs_sp1dplus_GEMM_FORWARD(eu, context, action_mask, dague_release_dep_fct, &arg);

  if(action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS) {
    data_repo_entry_addto_usage_limit(GEMM_FORWARD_repo, arg.output_entry->key, arg.output_usage);
    if( NULL != arg.ready_list ) {
      __dague_schedule(eu, arg.ready_list);
      arg.ready_list = NULL;
    }
  }
#if defined(DISTRIBUTED)
  if( (action_mask & DAGUE_ACTION_SEND_REMOTE_DEPS) && arg.remote_deps_count ) {
    arg.nb_released += dague_remote_dep_activate(eu, context, arg.remote_deps, arg.remote_deps_count);
  }
#endif
  if( action_mask & DAGUE_ACTION_RELEASE_LOCAL_REFS ) {
    int k = context->locals[0].value;
    int fcblk = context->locals[1].value;
    int cblk = context->locals[2].value;
    int phony = context->locals[3].value;
    int prev = context->locals[4].value;
    int next = context->locals[5].value;
    (void)k; (void)fcblk; (void)cblk; (void)phony; (void)prev; (void)next;

    if( !(phony) ) {
      data_repo_entry_used_once( eu, TRSM_FORWARD_repo, context->data[1].data_repo->key );
      (void)AUNREF(context->data[1].data);
    }
    if( !((prev == 0)) ) {
      data_repo_entry_used_once( eu, GEMM_FORWARD_repo, context->data[2].data_repo->key );
      (void)AUNREF(context->data[2].data);
    }
  }
  assert( NULL == arg.ready_list );
  return arg.nb_released;
}

static int hook_of_zpotrs_sp1dplus_GEMM_FORWARD(dague_execution_unit_t *context, dague_execution_context_t *exec_context)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (__dague_zpotrs_sp1dplus_internal_object_t *)exec_context->dague_object;
  assignment_t tass[MAX_PARAM_COUNT];
  (void)context; (void)__dague_object; (void)tass;
#if defined(DAGUE_SIM)
  int __dague_simulation_date = 0;
#endif
  int k = exec_context->locals[0].value;
  int fcblk = exec_context->locals[1].value;
  int cblk = exec_context->locals[2].value;
  int phony = exec_context->locals[3].value;
  int prev = exec_context->locals[4].value;
  int next = exec_context->locals[5].value;

  (void)k;  (void)fcblk;  (void)cblk;  (void)phony;  (void)prev;  (void)next;

  /** Declare the variables that will hold the data, and all the accounting for each */
  void *A = NULL; (void)A;
  dague_arena_chunk_t *gA = NULL; (void)gA;
  data_repo_entry_t *eA = NULL; (void)eA;
  void *B = NULL; (void)B;
  dague_arena_chunk_t *gB = NULL; (void)gB;
  data_repo_entry_t *eB = NULL; (void)eB;
  void *C = NULL; (void)C;
  dague_arena_chunk_t *gC = NULL; (void)gC;
  data_repo_entry_t *eC = NULL; (void)eC;

  /** Lookup the input data, and store them in the context */
  eA = exec_context->data[0].data_repo;
  gA = exec_context->data[0].data;
  if( NULL == gA ) {
  gA = (dague_arena_chunk_t*) A(fcblk);
    exec_context->data[0].data = gA;
    exec_context->data[0].data_repo = eA;
  }
  A = ADATA(gA);
#if defined(DAGUE_SIM)
  if( (NULL != eA) && (eA->sim_exec_date > __dague_simulation_date) )
    __dague_simulation_date =  eA->sim_exec_date;
#endif
  eB = exec_context->data[1].data_repo;
  gB = exec_context->data[1].data;
  if( NULL == gB ) {
  if( phony ) {
    gB = (dague_arena_chunk_t*) B(fcblk);
  } else {
      tass[0].value = cblk;
    eB = data_repo_lookup_entry( TRSM_FORWARD_repo, TRSM_FORWARD_hash( __dague_object, tass ));
    gB = eB->data[0];
  }
    exec_context->data[1].data = gB;
    exec_context->data[1].data_repo = eB;
  }
  B = ADATA(gB);
#if defined(DAGUE_SIM)
  if( (NULL != eB) && (eB->sim_exec_date > __dague_simulation_date) )
    __dague_simulation_date =  eB->sim_exec_date;
#endif
  eC = exec_context->data[2].data_repo;
  gC = exec_context->data[2].data;
  if( NULL == gC ) {
  if( (prev == 0) ) {
    gC = (dague_arena_chunk_t*) B(fcblk);
  } else {
      tass[0].value = prev;
    eC = data_repo_lookup_entry( GEMM_FORWARD_repo, GEMM_FORWARD_hash( __dague_object, tass ));
    gC = eC->data[0];
  }
    exec_context->data[2].data = gC;
    exec_context->data[2].data_repo = eC;
  }
  C = ADATA(gC);
#if defined(DAGUE_SIM)
  if( (NULL != eC) && (eC->sim_exec_date > __dague_simulation_date) )
    __dague_simulation_date =  eC->sim_exec_date;
#endif
#if defined(DAGUE_SIM)
  if( exec_context->function->sim_cost_fct != NULL ) {
    exec_context->sim_exec_date = __dague_simulation_date + exec_context->function->sim_cost_fct(exec_context);
  } else {
    exec_context->sim_exec_date = __dague_simulation_date;
  }
  if( context->largest_simulation_date < exec_context->sim_exec_date )
    context->largest_simulation_date = exec_context->sim_exec_date;
#endif
  /** PAPI events */
#if defined(HAVE_PAPI)
  papime_start_thread_counters();
#endif
  /** Cache Awareness Accounting */
#if defined(DAGUE_CACHE_AWARENESS)
  cache_buf_referenced(context->closest_cache, A);
  cache_buf_referenced(context->closest_cache, B);
  cache_buf_referenced(context->closest_cache, C);
#endif /* DAGUE_CACHE_AWARENESS */


#if !defined(DAGUE_PROF_DRY_BODY)

/*--------------------------------------------------------------------------------*
 *                              GEMM_FORWARD BODY                                *
 *--------------------------------------------------------------------------------*/

  TAKE_TIME(context, 2*exec_context->function->function_id, GEMM_FORWARD_hash( __dague_object, exec_context->locals), __dague_object->super.B, ((dague_ddesc_t*)(__dague_object->super.B))->data_key((dague_ddesc_t*)__dague_object->super.B, fcblk) );
#line 102 "zpotrs_sp1dplus.jdf"
      if (!phony) {
          Dague_Complex64_t *work = (Dague_Complex64_t *)dague_private_memory_pop( p_work );

          DRYRUN(
                 //core_zpotrfsp1d_gemm(cblk, k, fcblk, A, C, work, datacode);
                 /* GEMM ( N, N, A, B, C ); */
                 );
          printlog(
                   "thread %d compute_1dgemm( k=%d, fcblk=%d, cblk=%d, prev=%d, next=%d )\n",
                   context->eu_id, k, fcblk, cblk, prev, next);

          dague_private_memory_push( p_work, (void *)work );
      } else {
          printlog(
                   "thread %d phony_gemm( k=%d, fcblk=%d, cblk=%d, prev=%d, next=%d )\n",
                   context->eu_id, k, fcblk, cblk, prev, next);
      }

#line 2651 "zpotrs_sp1dplus.c"
/*--------------------------------------------------------------------------------*
 *                            END OF GEMM_FORWARD BODY                            *
 *--------------------------------------------------------------------------------*/



#endif /*!defined(DAGUE_PROF_DRY_BODY)*/

  return 0;
}
static int complete_hook_of_zpotrs_sp1dplus_GEMM_FORWARD(dague_execution_unit_t *context, dague_execution_context_t *exec_context)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (__dague_zpotrs_sp1dplus_internal_object_t *)exec_context->dague_object;
  (void)context; (void)__dague_object;
  int k = exec_context->locals[0].value;
  int fcblk = exec_context->locals[1].value;
  int cblk = exec_context->locals[2].value;
  int phony = exec_context->locals[3].value;
  int prev = exec_context->locals[4].value;
  int next = exec_context->locals[5].value;

  (void)k;  (void)fcblk;  (void)cblk;  (void)phony;  (void)prev;  (void)next;

  TAKE_TIME(context,2*exec_context->function->function_id+1, GEMM_FORWARD_hash( __dague_object, exec_context->locals ), NULL, 0);
  /** PAPI events */
#if defined(HAVE_PAPI)
  papime_stop_thread_counters();
#endif
#if defined(DISTRIBUTED)
  /** If not working on distributed, there is no risk that data is not in place */
  if( phony ) {
    if( ADATA(exec_context->data[2].data) != B(fcblk) ) {
      dague_remote_dep_memcpy( context, exec_context->dague_object, B(fcblk), exec_context->data[2].data, 
                               __dague_object->super.arenas[DAGUE_zpotrs_sp1dplus_DEFAULT_ARENA]->opaque_dtt );
    }
  }
#endif /* DISTRIBUTED */
  dague_prof_grapher_task(exec_context, context->eu_id, GEMM_FORWARD_hash(__dague_object, exec_context->locals));
  {
    release_deps_of_zpotrs_sp1dplus_GEMM_FORWARD(context, exec_context,
        DAGUE_ACTION_RELEASE_REMOTE_DEPS |
        DAGUE_ACTION_RELEASE_LOCAL_DEPS |
        DAGUE_ACTION_RELEASE_LOCAL_REFS |
        DAGUE_ACTION_DEPS_MASK,
        NULL);
  }
  return 0;
}

static int zpotrs_sp1dplus_GEMM_FORWARD_internal_init(__dague_zpotrs_sp1dplus_internal_object_t *__dague_object)
{
  dague_dependencies_t *dep = NULL;
  assignment_t assignments[MAX_LOCAL_COUNT];(void) assignments;
  int nb_tasks = 0, __foundone = 0;
  int32_t  k, fcblk, cblk, phony, prev, next;
  int32_t  k_min = 0x7fffffff;
  int32_t  k_max = 0;
  (void)__dague_object; (void)__foundone;
  /* First, find the min and max value for each of the dimensions */
  for(k = 0;
      k <= bloknbr;
      k++) {
    assignments[0].value = k;
    fcblk = zpotrs_sp1dplus_inline_c_expr15_line_85((const dague_object_t*)__dague_object, assignments);
    assignments[1].value = fcblk;
    cblk = zpotrs_sp1dplus_inline_c_expr14_line_86((const dague_object_t*)__dague_object, assignments);
    assignments[2].value = cblk;
    phony = zpotrs_sp1dplus_inline_c_expr13_line_87((const dague_object_t*)__dague_object, assignments);
    assignments[3].value = phony;
    prev = zpotrs_sp1dplus_inline_c_expr12_line_88((const dague_object_t*)__dague_object, assignments);
    assignments[4].value = prev;
    next = zpotrs_sp1dplus_inline_c_expr11_line_89((const dague_object_t*)__dague_object, assignments);
    assignments[5].value = next;
    if( !GEMM_FORWARD_pred(k, fcblk, cblk, phony, prev, next) ) continue;
    nb_tasks++;
    k_max = dague_imax(k_max, k);
    k_min = dague_imin(k_min, k);
  }

  /**
   * Now, for each of the dimensions, re-iterate on the space,
   * and if at least one value is defined, allocate arrays to point
   * to it. Array dimensions are defined by the (rough) observation above
   **/
  if( 0 != nb_tasks ) {
    ALLOCATE_DEP_TRACKING(dep, k_min, k_max, "k", &symb_zpotrs_sp1dplus_GEMM_FORWARD_k, NULL, DAGUE_DEPENDENCIES_FLAG_FINAL);
  }
  __dague_object->super.super.dependencies_array[2] = dep;
  __dague_object->super.super.nb_local_tasks += nb_tasks;
  return nb_tasks;
}

static int zpotrs_sp1dplus_GEMM_FORWARD_startup_tasks(dague_context_t *context, const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object, dague_execution_context_t** pready_list)
{
  dague_execution_context_t* new_context;
  assignment_t *assignments = NULL;
  int32_t  k = -1, fcblk = -1, cblk = -1, phony = -1, prev = -1, next = -1;
  (void)k; (void)fcblk; (void)cblk; (void)phony; (void)prev; (void)next;
  new_context = (dague_execution_context_t*)dague_thread_mempool_allocate( context->execution_units[0]->context_mempool );
  assignments = new_context->locals;
  /* Parse all the inputs and generate the ready execution tasks */
  for(k = 0;
      k <= bloknbr;
      k++) {
    assignments[0].value = k;
    assignments[1].value = fcblk = zpotrs_sp1dplus_inline_c_expr15_line_85((const dague_object_t*)__dague_object, assignments);
    assignments[2].value = cblk = zpotrs_sp1dplus_inline_c_expr14_line_86((const dague_object_t*)__dague_object, assignments);
    assignments[3].value = phony = zpotrs_sp1dplus_inline_c_expr13_line_87((const dague_object_t*)__dague_object, assignments);
    assignments[4].value = prev = zpotrs_sp1dplus_inline_c_expr12_line_88((const dague_object_t*)__dague_object, assignments);
    assignments[5].value = next = zpotrs_sp1dplus_inline_c_expr11_line_89((const dague_object_t*)__dague_object, assignments);
    if( !GEMM_FORWARD_pred(k, fcblk, cblk, phony, prev, next) ) continue;
    if( !((phony) && ((prev == 0))) ) continue;
    DAGUE_STAT_INCREASE(mem_contexts, sizeof(dague_execution_context_t) + STAT_MALLOC_OVERHEAD);
    DAGUE_LIST_ITEM_SINGLETON( new_context );
    new_context->dague_object = (dague_object_t*)__dague_object;
    new_context->function = (const dague_function_t*)&zpotrs_sp1dplus_GEMM_FORWARD;
    new_context->priority = 0;
  new_context->data[0].data_repo = NULL;
  new_context->data[0].data      = NULL;
  new_context->data[1].data_repo = NULL;
  new_context->data[1].data      = NULL;
  new_context->data[2].data_repo = NULL;
  new_context->data[2].data      = NULL;
#if defined(DAGUE_DEBUG)
    {
      char tmp[128];
      printf("Add startup task %s\n",
             dague_service_to_string(new_context, tmp, 128));
    }
#endif
    dague_list_add_single_elem_by_priority( pready_list, new_context );
    new_context = (dague_execution_context_t*)dague_thread_mempool_allocate( context->execution_units[0]->context_mempool );
    assignments = new_context->locals;
    assignments[0].value = k;
    assignments[1].value = fcblk;
    assignments[2].value = cblk;
    assignments[3].value = phony;
    assignments[4].value = prev;
  }
  dague_thread_mempool_free( context->execution_units[0]->context_mempool, new_context );
  return 0;
}

static const dague_function_t zpotrs_sp1dplus_GEMM_FORWARD = {
  .name = "GEMM_FORWARD",
  .deps = 2,
  .flags = 0x0 | DAGUE_HAS_IN_IN_DEPENDENCIES,
  .function_id = 2,
  .dependencies_goal = 0x7,
  .nb_parameters = 1,
  .nb_definitions = 6,
  .params = { &symb_zpotrs_sp1dplus_GEMM_FORWARD_k },
  .locals = { &symb_zpotrs_sp1dplus_GEMM_FORWARD_k, &symb_zpotrs_sp1dplus_GEMM_FORWARD_fcblk, &symb_zpotrs_sp1dplus_GEMM_FORWARD_cblk, &symb_zpotrs_sp1dplus_GEMM_FORWARD_phony, &symb_zpotrs_sp1dplus_GEMM_FORWARD_prev, &symb_zpotrs_sp1dplus_GEMM_FORWARD_next },
  .pred = &pred_of_zpotrs_sp1dplus_GEMM_FORWARD_as_expr,
  .priority = NULL,
  .in = { &flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_A, &flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_B, &flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C },
  .out = { &flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C },
  .iterate_successors = iterate_successors_of_zpotrs_sp1dplus_GEMM_FORWARD,
  .release_deps = release_deps_of_zpotrs_sp1dplus_GEMM_FORWARD,
  .hook = hook_of_zpotrs_sp1dplus_GEMM_FORWARD,
  .complete_execution = complete_hook_of_zpotrs_sp1dplus_GEMM_FORWARD,
#if defined(DAGUE_SIM)
  .sim_cost_fct = NULL,
#endif
  .key = (dague_functionkey_fn_t*)GEMM_FORWARD_hash,
};


/**********************************************************************************
 *                                  TRSM_FORWARD                                  *
 **********************************************************************************/

static inline int minexpr_of_symb_zpotrs_sp1dplus_TRSM_FORWARD_k_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return 0;
}
static const expr_t minexpr_of_symb_zpotrs_sp1dplus_TRSM_FORWARD_k = {
  .op = EXPR_OP_INLINE,
  .inline_func = minexpr_of_symb_zpotrs_sp1dplus_TRSM_FORWARD_k_fct
};
static inline int maxexpr_of_symb_zpotrs_sp1dplus_TRSM_FORWARD_k_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return cblknbr;
}
static const expr_t maxexpr_of_symb_zpotrs_sp1dplus_TRSM_FORWARD_k = {
  .op = EXPR_OP_INLINE,
  .inline_func = maxexpr_of_symb_zpotrs_sp1dplus_TRSM_FORWARD_k_fct
};
static const symbol_t symb_zpotrs_sp1dplus_TRSM_FORWARD_k = {.min = &minexpr_of_symb_zpotrs_sp1dplus_TRSM_FORWARD_k, .max = &maxexpr_of_symb_zpotrs_sp1dplus_TRSM_FORWARD_k,  .flags = DAGUE_SYMBOL_IS_STANDALONE};

static inline int expr_of_symb_zpotrs_sp1dplus_TRSM_FORWARD_gcblk2list_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zpotrs_sp1dplus_inline_c_expr21_line_43((const dague_object_t*)__dague_object, assignments);
}
static const expr_t expr_of_symb_zpotrs_sp1dplus_TRSM_FORWARD_gcblk2list = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_symb_zpotrs_sp1dplus_TRSM_FORWARD_gcblk2list_fct
};
static const symbol_t symb_zpotrs_sp1dplus_TRSM_FORWARD_gcblk2list = {.min = &expr_of_symb_zpotrs_sp1dplus_TRSM_FORWARD_gcblk2list, .max = &expr_of_symb_zpotrs_sp1dplus_TRSM_FORWARD_gcblk2list,  .flags = 0x0};

static inline int expr_of_symb_zpotrs_sp1dplus_TRSM_FORWARD_browk1_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zpotrs_sp1dplus_inline_c_expr20_line_45((const dague_object_t*)__dague_object, assignments);
}
static const expr_t expr_of_symb_zpotrs_sp1dplus_TRSM_FORWARD_browk1 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_symb_zpotrs_sp1dplus_TRSM_FORWARD_browk1_fct
};
static const symbol_t symb_zpotrs_sp1dplus_TRSM_FORWARD_browk1 = {.min = &expr_of_symb_zpotrs_sp1dplus_TRSM_FORWARD_browk1, .max = &expr_of_symb_zpotrs_sp1dplus_TRSM_FORWARD_browk1,  .flags = 0x0};

static inline int expr_of_symb_zpotrs_sp1dplus_TRSM_FORWARD_lastbrow_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zpotrs_sp1dplus_inline_c_expr19_line_46((const dague_object_t*)__dague_object, assignments);
}
static const expr_t expr_of_symb_zpotrs_sp1dplus_TRSM_FORWARD_lastbrow = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_symb_zpotrs_sp1dplus_TRSM_FORWARD_lastbrow_fct
};
static const symbol_t symb_zpotrs_sp1dplus_TRSM_FORWARD_lastbrow = {.min = &expr_of_symb_zpotrs_sp1dplus_TRSM_FORWARD_lastbrow, .max = &expr_of_symb_zpotrs_sp1dplus_TRSM_FORWARD_lastbrow,  .flags = 0x0};

static inline int expr_of_symb_zpotrs_sp1dplus_TRSM_FORWARD_firstblok_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zpotrs_sp1dplus_inline_c_expr18_line_47((const dague_object_t*)__dague_object, assignments);
}
static const expr_t expr_of_symb_zpotrs_sp1dplus_TRSM_FORWARD_firstblok = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_symb_zpotrs_sp1dplus_TRSM_FORWARD_firstblok_fct
};
static const symbol_t symb_zpotrs_sp1dplus_TRSM_FORWARD_firstblok = {.min = &expr_of_symb_zpotrs_sp1dplus_TRSM_FORWARD_firstblok, .max = &expr_of_symb_zpotrs_sp1dplus_TRSM_FORWARD_firstblok,  .flags = 0x0};

static inline int expr_of_symb_zpotrs_sp1dplus_TRSM_FORWARD_lastblok_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zpotrs_sp1dplus_inline_c_expr17_line_48((const dague_object_t*)__dague_object, assignments);
}
static const expr_t expr_of_symb_zpotrs_sp1dplus_TRSM_FORWARD_lastblok = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_symb_zpotrs_sp1dplus_TRSM_FORWARD_lastblok_fct
};
static const symbol_t symb_zpotrs_sp1dplus_TRSM_FORWARD_lastblok = {.min = &expr_of_symb_zpotrs_sp1dplus_TRSM_FORWARD_lastblok, .max = &expr_of_symb_zpotrs_sp1dplus_TRSM_FORWARD_lastblok,  .flags = 0x0};

static inline int expr_of_symb_zpotrs_sp1dplus_TRSM_FORWARD_firstgemm_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zpotrs_sp1dplus_inline_c_expr16_line_54((const dague_object_t*)__dague_object, assignments);
}
static const expr_t expr_of_symb_zpotrs_sp1dplus_TRSM_FORWARD_firstgemm = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_symb_zpotrs_sp1dplus_TRSM_FORWARD_firstgemm_fct
};
static const symbol_t symb_zpotrs_sp1dplus_TRSM_FORWARD_firstgemm = {.min = &expr_of_symb_zpotrs_sp1dplus_TRSM_FORWARD_firstgemm, .max = &expr_of_symb_zpotrs_sp1dplus_TRSM_FORWARD_firstgemm,  .flags = 0x0};

static inline int pred_of_zpotrs_sp1dplus_TRSM_FORWARD_as_expr_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;
  int gcblk2list = assignments[1].value;
  int browk1 = assignments[2].value;
  int lastbrow = assignments[3].value;
  int firstblok = assignments[4].value;
  int lastblok = assignments[5].value;
  int firstgemm = assignments[6].value;

  /* Silent Warnings: should look into predicate to know what variables are usefull */
  (void)__dague_object;
  (void)k;
  (void)gcblk2list;
  (void)browk1;
  (void)lastbrow;
  (void)firstblok;
  (void)lastblok;
  (void)firstgemm;
  /* Compute Predicate */
  return TRSM_FORWARD_pred(k, gcblk2list, browk1, lastbrow, firstblok, lastblok, firstgemm);
}
static const expr_t pred_of_zpotrs_sp1dplus_TRSM_FORWARD_as_expr = {
  .op = EXPR_OP_INLINE,
  .inline_func = pred_of_zpotrs_sp1dplus_TRSM_FORWARD_as_expr_fct
};
static inline int expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_A_dep1_atline_62_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_A_dep1_atline_62 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_A_dep1_atline_62_fct
};
static const dep_t flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_A_dep1_atline_62 = {
  .cond = NULL,
  .dague = &zpotrs_sp1dplus_A,
  .datatype_index = DAGUE_zpotrs_sp1dplus_DEFAULT_ARENA,  .call_params = {
    &expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_A_dep1_atline_62
  }
};
static const dague_flow_t flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_A = {
  .name = "A",
  .sym_type = SYM_IN,
  .access_type = ACCESS_READ,
  .flow_index = 0,
  .dep_in  = { &flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_A_dep1_atline_62 },
  .dep_out = { NULL }
};

static inline int expr_of_cond_for_flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep1_iftrue_atline_63_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int gcblk2list = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return (gcblk2list <  0);
}
static const expr_t expr_of_cond_for_flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep1_iftrue_atline_63 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep1_iftrue_atline_63_fct
};
static inline int expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep1_iftrue_atline_63_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep1_iftrue_atline_63 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep1_iftrue_atline_63_fct
};
static const dep_t flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep1_iftrue_atline_63 = {
  .cond = &expr_of_cond_for_flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep1_iftrue_atline_63,
  .dague = &zpotrs_sp1dplus_B,
  .datatype_index = DAGUE_zpotrs_sp1dplus_DEFAULT_ARENA,  .call_params = {
    &expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep1_iftrue_atline_63
  }
};
static inline int expr_of_cond_for_flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep1_iffalse_atline_63_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int gcblk2list = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return !(gcblk2list <  0);
}
static const expr_t expr_of_cond_for_flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep1_iffalse_atline_63 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep1_iffalse_atline_63_fct
};
static inline int expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep1_iffalse_atline_63_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int lastbrow = assignments[3].value;

  (void)__dague_object; (void)assignments;
  return lastbrow;
}
static const expr_t expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep1_iffalse_atline_63 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep1_iffalse_atline_63_fct
};
static const dep_t flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep1_iffalse_atline_63 = {
  .cond = &expr_of_cond_for_flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep1_iffalse_atline_63,
  .dague = &zpotrs_sp1dplus_GEMM_FORWARD,
  .flow = &flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_C,
  .datatype_index = DAGUE_zpotrs_sp1dplus_DEFAULT_ARENA,  .call_params = {
    &expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep1_iffalse_atline_63
  }
};
static inline int rangemin_of_expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep2_atline_64_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int firstblok = assignments[4].value;

  (void)__dague_object; (void)assignments;
  return (firstblok + 1);
}
static const expr_t rangemin_of_expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep2_atline_64 = {
  .op = EXPR_OP_INLINE,
  .inline_func = rangemin_of_expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep2_atline_64_fct
};
static inline int rangemax_of_expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep2_atline_64_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int lastblok = assignments[5].value;

  (void)__dague_object; (void)assignments;
  return (lastblok - 1);
}
static const expr_t rangemax_of_expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep2_atline_64 = {
  .op = EXPR_OP_INLINE,
  .inline_func = rangemax_of_expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep2_atline_64_fct
};
static const expr_t expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep2_atline_64 = {
  .op = EXPR_OP_BINARY_RANGE,
  .u_expr.binary = {
    .op1 = &rangemin_of_expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep2_atline_64,
    .op2 = &rangemax_of_expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep2_atline_64
  }
};
static const dep_t flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep2_atline_64 = {
  .cond = NULL,
  .dague = &zpotrs_sp1dplus_GEMM_FORWARD,
  .flow = &flow_of_zpotrs_sp1dplus_GEMM_FORWARD_for_B,
  .datatype_index = DAGUE_zpotrs_sp1dplus_DEFAULT_ARENA,  .call_params = {
    &expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep2_atline_64
  }
};
static inline int expr_of_cond_for_flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep3_iftrue_atline_75_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return (k == cblknbr);
}
static const expr_t expr_of_cond_for_flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep3_iftrue_atline_75 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep3_iftrue_atline_75_fct
};
static inline int expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep3_iftrue_atline_75_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep3_iftrue_atline_75 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep3_iftrue_atline_75_fct
};
static const dep_t flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep3_iftrue_atline_75 = {
  .cond = &expr_of_cond_for_flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep3_iftrue_atline_75,
  .dague = &zpotrs_sp1dplus_TRSM_BACKWARD,
  .flow = &flow_of_zpotrs_sp1dplus_TRSM_BACKWARD_for_B,
  .datatype_index = DAGUE_zpotrs_sp1dplus_DEFAULT_ARENA,  .call_params = {
    &expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep3_iftrue_atline_75
  }
};
static inline int expr_of_cond_for_flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep3_iffalse_atline_75_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return !(k == cblknbr);
}
static const expr_t expr_of_cond_for_flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep3_iffalse_atline_75 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep3_iffalse_atline_75_fct
};
static inline int expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep3_iffalse_atline_75_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)__dague_object_parent;
  int firstgemm = assignments[6].value;

  (void)__dague_object; (void)assignments;
  return firstgemm;
}
static const expr_t expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep3_iffalse_atline_75 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep3_iffalse_atline_75_fct
};
static const dep_t flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep3_iffalse_atline_75 = {
  .cond = &expr_of_cond_for_flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep3_iffalse_atline_75,
  .dague = &zpotrs_sp1dplus_GEMM_BACKWARD,
  .flow = &flow_of_zpotrs_sp1dplus_GEMM_BACKWARD_for_C,
  .datatype_index = DAGUE_zpotrs_sp1dplus_DEFAULT_ARENA,  .call_params = {
    &expr_of_p1_for_flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep3_iffalse_atline_75
  }
};
static const dague_flow_t flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B = {
  .name = "B",
  .sym_type = SYM_INOUT,
  .access_type = ACCESS_RW,
  .flow_index = 1,
  .dep_in  = { &flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep1_iftrue_atline_63, &flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep1_iffalse_atline_63 },
  .dep_out = { &flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep2_atline_64, &flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep3_iftrue_atline_75, &flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B_dep3_iffalse_atline_75 }
};

static void
iterate_successors_of_zpotrs_sp1dplus_TRSM_FORWARD(dague_execution_unit_t *eu, dague_execution_context_t *exec_context,
               uint32_t action_mask, dague_ontask_function_t *ontask, void *ontask_arg)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t*)exec_context->dague_object;
  dague_execution_context_t nc;
  dague_arena_t* arena = NULL;
  int rank_src = 0, rank_dst = 0;
  int k = exec_context->locals[0].value;
  int gcblk2list = exec_context->locals[1].value;
  int browk1 = exec_context->locals[2].value;
  int lastbrow = exec_context->locals[3].value;
  int firstblok = exec_context->locals[4].value;
  int lastblok = exec_context->locals[5].value;
  int firstgemm = exec_context->locals[6].value;
  (void)rank_src; (void)rank_dst; (void)__dague_object;
  (void)k;  (void)gcblk2list;  (void)browk1;  (void)lastbrow;  (void)firstblok;  (void)lastblok;  (void)firstgemm;
  nc.dague_object = exec_context->dague_object;
#if defined(DISTRIBUTED)
  rank_src = ((dague_ddesc_t*)__dague_object->super.B)->rank_of((dague_ddesc_t*)__dague_object->super.B, k);
#endif
  /* Flow of data A has only IN dependencies */
  /* Flow of Data B */
  if( action_mask & (1 << 0) ) {
#if defined(DISTRIBUTED)
    arena = __dague_object->super.arenas[DAGUE_zpotrs_sp1dplus_DEFAULT_ARENA];
#endif  /* defined(DISTRIBUTED) */
    nc.function = (const dague_function_t*)&zpotrs_sp1dplus_GEMM_FORWARD;
    {
      int GEMM_FORWARD_k;
      for( GEMM_FORWARD_k = (firstblok + 1);GEMM_FORWARD_k <= (lastblok - 1); GEMM_FORWARD_k++ ) {
        if( (GEMM_FORWARD_k >= (0)) && (GEMM_FORWARD_k <= (bloknbr)) ) {
          nc.locals[0].value = GEMM_FORWARD_k;
          const int GEMM_FORWARD_fcblk = zpotrs_sp1dplus_inline_c_expr15_line_85((const dague_object_t*)__dague_object, nc.locals);
          nc.locals[1].value = GEMM_FORWARD_fcblk;
          const int GEMM_FORWARD_cblk = zpotrs_sp1dplus_inline_c_expr14_line_86((const dague_object_t*)__dague_object, nc.locals);
          nc.locals[2].value = GEMM_FORWARD_cblk;
          const int GEMM_FORWARD_phony = zpotrs_sp1dplus_inline_c_expr13_line_87((const dague_object_t*)__dague_object, nc.locals);
          nc.locals[3].value = GEMM_FORWARD_phony;
          const int GEMM_FORWARD_prev = zpotrs_sp1dplus_inline_c_expr12_line_88((const dague_object_t*)__dague_object, nc.locals);
          nc.locals[4].value = GEMM_FORWARD_prev;
          const int GEMM_FORWARD_next = zpotrs_sp1dplus_inline_c_expr11_line_89((const dague_object_t*)__dague_object, nc.locals);
          nc.locals[5].value = GEMM_FORWARD_next;
#if defined(DISTRIBUTED)
            rank_dst = ((dague_ddesc_t*)__dague_object->super.B)->rank_of((dague_ddesc_t*)__dague_object->super.B, GEMM_FORWARD_fcblk);
#endif
#if defined(DAGUE_DEBUG)
          if( NULL != eu ) {
            char tmp[128], tmp1[128];
            DEBUG(("thread %d release deps of B:%s to B:%s (from node %d to %d)\n", eu->eu_id,
                   dague_service_to_string(exec_context, tmp, 128),
                   dague_service_to_string(&nc, tmp1, 128), rank_src, rank_dst));
          }
#endif
            nc.priority = 0;
            if( DAGUE_ITERATE_STOP == ontask(eu, &nc, exec_context, 0, 0, rank_src, rank_dst, arena, ontask_arg) )
              return;
    }
      }
        }
#if defined(DISTRIBUTED)
    arena = __dague_object->super.arenas[DAGUE_zpotrs_sp1dplus_DEFAULT_ARENA];
#endif  /* defined(DISTRIBUTED) */
    if( (k == cblknbr) ) {
      nc.function = (const dague_function_t*)&zpotrs_sp1dplus_TRSM_BACKWARD;
      {
        const int TRSM_BACKWARD_k = k;
        if( (TRSM_BACKWARD_k >= (0)) && (TRSM_BACKWARD_k <= (cblknbr)) ) {
          nc.locals[0].value = TRSM_BACKWARD_k;
          const int TRSM_BACKWARD_gcblk2list = zpotrs_sp1dplus_inline_c_expr10_line_131((const dague_object_t*)__dague_object, nc.locals);
          nc.locals[1].value = TRSM_BACKWARD_gcblk2list;
          const int TRSM_BACKWARD_firstbrow = zpotrs_sp1dplus_inline_c_expr9_line_132((const dague_object_t*)__dague_object, nc.locals);
          nc.locals[2].value = TRSM_BACKWARD_firstbrow;
          const int TRSM_BACKWARD_lastbrow = zpotrs_sp1dplus_inline_c_expr8_line_133((const dague_object_t*)__dague_object, nc.locals);
          nc.locals[3].value = TRSM_BACKWARD_lastbrow;
          const int TRSM_BACKWARD_firstblok = zpotrs_sp1dplus_inline_c_expr7_line_134((const dague_object_t*)__dague_object, nc.locals);
          nc.locals[4].value = TRSM_BACKWARD_firstblok;
#if defined(DISTRIBUTED)
            rank_dst = ((dague_ddesc_t*)__dague_object->super.B)->rank_of((dague_ddesc_t*)__dague_object->super.B, TRSM_BACKWARD_k);
#endif
#if defined(DAGUE_DEBUG)
          if( NULL != eu ) {
            char tmp[128], tmp1[128];
            DEBUG(("thread %d release deps of B:%s to B:%s (from node %d to %d)\n", eu->eu_id,
                   dague_service_to_string(exec_context, tmp, 128),
                   dague_service_to_string(&nc, tmp1, 128), rank_src, rank_dst));
          }
#endif
            nc.priority = 0;
            if( DAGUE_ITERATE_STOP == ontask(eu, &nc, exec_context, 0, 1, rank_src, rank_dst, arena, ontask_arg) )
              return;
      }
        }
    } else {
      nc.function = (const dague_function_t*)&zpotrs_sp1dplus_GEMM_BACKWARD;
      {
        const int GEMM_BACKWARD_k = firstgemm;
        if( (GEMM_BACKWARD_k >= (0)) && (GEMM_BACKWARD_k <= (browsize)) ) {
          nc.locals[0].value = GEMM_BACKWARD_k;
          const int GEMM_BACKWARD_bloknum = zpotrs_sp1dplus_inline_c_expr6_line_164((const dague_object_t*)__dague_object, nc.locals);
          nc.locals[1].value = GEMM_BACKWARD_bloknum;
          const int GEMM_BACKWARD_fcblk = zpotrs_sp1dplus_inline_c_expr5_line_165((const dague_object_t*)__dague_object, nc.locals);
          nc.locals[2].value = GEMM_BACKWARD_fcblk;
          const int GEMM_BACKWARD_cblk = zpotrs_sp1dplus_inline_c_expr4_line_166((const dague_object_t*)__dague_object, nc.locals);
          nc.locals[3].value = GEMM_BACKWARD_cblk;
          const int GEMM_BACKWARD_gcblk2list = zpotrs_sp1dplus_inline_c_expr3_line_168((const dague_object_t*)__dague_object, nc.locals);
          nc.locals[4].value = GEMM_BACKWARD_gcblk2list;
          const int GEMM_BACKWARD_firstbrow = zpotrs_sp1dplus_inline_c_expr2_line_169((const dague_object_t*)__dague_object, nc.locals);
          nc.locals[5].value = GEMM_BACKWARD_firstbrow;
          const int GEMM_BACKWARD_lastbrow = zpotrs_sp1dplus_inline_c_expr1_line_170((const dague_object_t*)__dague_object, nc.locals);
          nc.locals[6].value = GEMM_BACKWARD_lastbrow;
#if defined(DISTRIBUTED)
            rank_dst = ((dague_ddesc_t*)__dague_object->super.B)->rank_of((dague_ddesc_t*)__dague_object->super.B, GEMM_BACKWARD_cblk);
#endif
#if defined(DAGUE_DEBUG)
          if( NULL != eu ) {
            char tmp[128], tmp1[128];
            DEBUG(("thread %d release deps of B:%s to C:%s (from node %d to %d)\n", eu->eu_id,
                   dague_service_to_string(exec_context, tmp, 128),
                   dague_service_to_string(&nc, tmp1, 128), rank_src, rank_dst));
          }
#endif
            nc.priority = 0;
            if( DAGUE_ITERATE_STOP == ontask(eu, &nc, exec_context, 0, 2, rank_src, rank_dst, arena, ontask_arg) )
              return;
      }
        }
    }
  }
  (void)nc;(void)arena;(void)eu;(void)ontask;(void)ontask_arg;(void)rank_dst;(void)action_mask;
}

static int release_deps_of_zpotrs_sp1dplus_TRSM_FORWARD(dague_execution_unit_t *eu, dague_execution_context_t *context, uint32_t action_mask, dague_remote_deps_t *deps)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (const __dague_zpotrs_sp1dplus_internal_object_t *)context->dague_object;
  dague_release_dep_fct_arg_t arg;
  arg.nb_released = 0;
  arg.output_usage = 0;
  arg.action_mask = action_mask;
  arg.deps = deps;
  arg.ready_list = NULL;
  (void)__dague_object;
  if( action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS ) {
    arg.output_entry = data_repo_lookup_entry_and_create( eu, TRSM_FORWARD_repo, TRSM_FORWARD_hash(__dague_object, context->locals) );
  }
#if defined(DAGUE_SIM)
  assert(arg.output_entry->sim_exec_date == 0);
  arg.output_entry->sim_exec_date = context->sim_exec_date;
#endif
#if defined(DISTRIBUTED)
  arg.remote_deps_count = 0;
  arg.remote_deps = NULL;
#endif
  iterate_successors_of_zpotrs_sp1dplus_TRSM_FORWARD(eu, context, action_mask, dague_release_dep_fct, &arg);

  if(action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS) {
    data_repo_entry_addto_usage_limit(TRSM_FORWARD_repo, arg.output_entry->key, arg.output_usage);
    if( NULL != arg.ready_list ) {
      __dague_schedule(eu, arg.ready_list);
      arg.ready_list = NULL;
    }
  }
#if defined(DISTRIBUTED)
  if( (action_mask & DAGUE_ACTION_SEND_REMOTE_DEPS) && arg.remote_deps_count ) {
    arg.nb_released += dague_remote_dep_activate(eu, context, arg.remote_deps, arg.remote_deps_count);
  }
#endif
  if( action_mask & DAGUE_ACTION_RELEASE_LOCAL_REFS ) {
    int k = context->locals[0].value;
    int gcblk2list = context->locals[1].value;
    int browk1 = context->locals[2].value;
    int lastbrow = context->locals[3].value;
    int firstblok = context->locals[4].value;
    int lastblok = context->locals[5].value;
    int firstgemm = context->locals[6].value;
    (void)k; (void)gcblk2list; (void)browk1; (void)lastbrow; (void)firstblok; (void)lastblok; (void)firstgemm;

    if( !((gcblk2list <  0)) ) {
      data_repo_entry_used_once( eu, GEMM_FORWARD_repo, context->data[1].data_repo->key );
      (void)AUNREF(context->data[1].data);
    }
  }
  assert( NULL == arg.ready_list );
  return arg.nb_released;
}

static int hook_of_zpotrs_sp1dplus_TRSM_FORWARD(dague_execution_unit_t *context, dague_execution_context_t *exec_context)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (__dague_zpotrs_sp1dplus_internal_object_t *)exec_context->dague_object;
  assignment_t tass[MAX_PARAM_COUNT];
  (void)context; (void)__dague_object; (void)tass;
#if defined(DAGUE_SIM)
  int __dague_simulation_date = 0;
#endif
  int k = exec_context->locals[0].value;
  int gcblk2list = exec_context->locals[1].value;
  int browk1 = exec_context->locals[2].value;
  int lastbrow = exec_context->locals[3].value;
  int firstblok = exec_context->locals[4].value;
  int lastblok = exec_context->locals[5].value;
  int firstgemm = exec_context->locals[6].value;

  (void)k;  (void)gcblk2list;  (void)browk1;  (void)lastbrow;  (void)firstblok;  (void)lastblok;  (void)firstgemm;

  /** Declare the variables that will hold the data, and all the accounting for each */
  void *A = NULL; (void)A;
  dague_arena_chunk_t *gA = NULL; (void)gA;
  data_repo_entry_t *eA = NULL; (void)eA;
  void *B = NULL; (void)B;
  dague_arena_chunk_t *gB = NULL; (void)gB;
  data_repo_entry_t *eB = NULL; (void)eB;

  /** Lookup the input data, and store them in the context */
  eA = exec_context->data[0].data_repo;
  gA = exec_context->data[0].data;
  if( NULL == gA ) {
  gA = (dague_arena_chunk_t*) A(k);
    exec_context->data[0].data = gA;
    exec_context->data[0].data_repo = eA;
  }
  A = ADATA(gA);
#if defined(DAGUE_SIM)
  if( (NULL != eA) && (eA->sim_exec_date > __dague_simulation_date) )
    __dague_simulation_date =  eA->sim_exec_date;
#endif
  eB = exec_context->data[1].data_repo;
  gB = exec_context->data[1].data;
  if( NULL == gB ) {
  if( (gcblk2list <  0) ) {
    gB = (dague_arena_chunk_t*) B(k);
  } else {
      tass[0].value = lastbrow;
    eB = data_repo_lookup_entry( GEMM_FORWARD_repo, GEMM_FORWARD_hash( __dague_object, tass ));
    gB = eB->data[0];
  }
    exec_context->data[1].data = gB;
    exec_context->data[1].data_repo = eB;
  }
  B = ADATA(gB);
#if defined(DAGUE_SIM)
  if( (NULL != eB) && (eB->sim_exec_date > __dague_simulation_date) )
    __dague_simulation_date =  eB->sim_exec_date;
#endif
#if defined(DAGUE_SIM)
  if( exec_context->function->sim_cost_fct != NULL ) {
    exec_context->sim_exec_date = __dague_simulation_date + exec_context->function->sim_cost_fct(exec_context);
  } else {
    exec_context->sim_exec_date = __dague_simulation_date;
  }
  if( context->largest_simulation_date < exec_context->sim_exec_date )
    context->largest_simulation_date = exec_context->sim_exec_date;
#endif
  /** PAPI events */
#if defined(HAVE_PAPI)
  papime_start_thread_counters();
#endif
  /** Cache Awareness Accounting */
#if defined(DAGUE_CACHE_AWARENESS)
  cache_buf_referenced(context->closest_cache, A);
  cache_buf_referenced(context->closest_cache, B);
#endif /* DAGUE_CACHE_AWARENESS */


#if !defined(DAGUE_PROF_DRY_BODY)

/*--------------------------------------------------------------------------------*
 *                              TRSM_FORWARD BODY                                *
 *--------------------------------------------------------------------------------*/

  TAKE_TIME(context, 2*exec_context->function->function_id, TRSM_FORWARD_hash( __dague_object, exec_context->locals), __dague_object->super.B, ((dague_ddesc_t*)(__dague_object->super.B))->data_key((dague_ddesc_t*)__dague_object->super.B, k) );
#line 66 "zpotrs_sp1dplus.jdf"
      DRYRUN(
             /*core_zpotrfsp1d(A, datacode, k, descA->pastix_data->sopar.espilondiag );*/
             /*TRSM( L, L , N, N, A, B )*/
	     );
      printlog(
               "thread %d solvedown_1dplus( cblknum=%d, browk1=%d, lastbrow=%d, firstblok=%d, lastblok=%d )\n",
               context->eu_id, k, /*browk, */ browk1, lastbrow, firstblok, lastblok);


#line 3444 "zpotrs_sp1dplus.c"
/*--------------------------------------------------------------------------------*
 *                            END OF TRSM_FORWARD BODY                            *
 *--------------------------------------------------------------------------------*/



#endif /*!defined(DAGUE_PROF_DRY_BODY)*/

  return 0;
}
static int complete_hook_of_zpotrs_sp1dplus_TRSM_FORWARD(dague_execution_unit_t *context, dague_execution_context_t *exec_context)
{
  const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object = (__dague_zpotrs_sp1dplus_internal_object_t *)exec_context->dague_object;
  (void)context; (void)__dague_object;
  int k = exec_context->locals[0].value;
  int gcblk2list = exec_context->locals[1].value;
  int browk1 = exec_context->locals[2].value;
  int lastbrow = exec_context->locals[3].value;
  int firstblok = exec_context->locals[4].value;
  int lastblok = exec_context->locals[5].value;
  int firstgemm = exec_context->locals[6].value;

  (void)k;  (void)gcblk2list;  (void)browk1;  (void)lastbrow;  (void)firstblok;  (void)lastblok;  (void)firstgemm;

  TAKE_TIME(context,2*exec_context->function->function_id+1, TRSM_FORWARD_hash( __dague_object, exec_context->locals ), NULL, 0);
  /** PAPI events */
#if defined(HAVE_PAPI)
  papime_stop_thread_counters();
#endif
#if defined(DISTRIBUTED)
  /** If not working on distributed, there is no risk that data is not in place */
#endif /* DISTRIBUTED */
  dague_prof_grapher_task(exec_context, context->eu_id, TRSM_FORWARD_hash(__dague_object, exec_context->locals));
  {
    release_deps_of_zpotrs_sp1dplus_TRSM_FORWARD(context, exec_context,
        DAGUE_ACTION_RELEASE_REMOTE_DEPS |
        DAGUE_ACTION_RELEASE_LOCAL_DEPS |
        DAGUE_ACTION_RELEASE_LOCAL_REFS |
        DAGUE_ACTION_DEPS_MASK,
        NULL);
  }
  return 0;
}

static int zpotrs_sp1dplus_TRSM_FORWARD_internal_init(__dague_zpotrs_sp1dplus_internal_object_t *__dague_object)
{
  dague_dependencies_t *dep = NULL;
  assignment_t assignments[MAX_LOCAL_COUNT];(void) assignments;
  int nb_tasks = 0, __foundone = 0;
  int32_t  k, gcblk2list, browk1, lastbrow, firstblok, lastblok, firstgemm;
  int32_t  k_min = 0x7fffffff;
  int32_t  k_max = 0;
  (void)__dague_object; (void)__foundone;
  /* First, find the min and max value for each of the dimensions */
  for(k = 0;
      k <= cblknbr;
      k++) {
    assignments[0].value = k;
    gcblk2list = zpotrs_sp1dplus_inline_c_expr21_line_43((const dague_object_t*)__dague_object, assignments);
    assignments[1].value = gcblk2list;
    browk1 = zpotrs_sp1dplus_inline_c_expr20_line_45((const dague_object_t*)__dague_object, assignments);
    assignments[2].value = browk1;
    lastbrow = zpotrs_sp1dplus_inline_c_expr19_line_46((const dague_object_t*)__dague_object, assignments);
    assignments[3].value = lastbrow;
    firstblok = zpotrs_sp1dplus_inline_c_expr18_line_47((const dague_object_t*)__dague_object, assignments);
    assignments[4].value = firstblok;
    lastblok = zpotrs_sp1dplus_inline_c_expr17_line_48((const dague_object_t*)__dague_object, assignments);
    assignments[5].value = lastblok;
    firstgemm = zpotrs_sp1dplus_inline_c_expr16_line_54((const dague_object_t*)__dague_object, assignments);
    assignments[6].value = firstgemm;
    if( !TRSM_FORWARD_pred(k, gcblk2list, browk1, lastbrow, firstblok, lastblok, firstgemm) ) continue;
    nb_tasks++;
    k_max = dague_imax(k_max, k);
    k_min = dague_imin(k_min, k);
  }

  /**
   * Now, for each of the dimensions, re-iterate on the space,
   * and if at least one value is defined, allocate arrays to point
   * to it. Array dimensions are defined by the (rough) observation above
   **/
  if( 0 != nb_tasks ) {
    ALLOCATE_DEP_TRACKING(dep, k_min, k_max, "k", &symb_zpotrs_sp1dplus_TRSM_FORWARD_k, NULL, DAGUE_DEPENDENCIES_FLAG_FINAL);
  }
  __dague_object->super.super.dependencies_array[3] = dep;
  __dague_object->super.super.nb_local_tasks += nb_tasks;
  return nb_tasks;
}

static int zpotrs_sp1dplus_TRSM_FORWARD_startup_tasks(dague_context_t *context, const __dague_zpotrs_sp1dplus_internal_object_t *__dague_object, dague_execution_context_t** pready_list)
{
  dague_execution_context_t* new_context;
  assignment_t *assignments = NULL;
  int32_t  k = -1, gcblk2list = -1, browk1 = -1, lastbrow = -1, firstblok = -1, lastblok = -1, firstgemm = -1;
  (void)k; (void)gcblk2list; (void)browk1; (void)lastbrow; (void)firstblok; (void)lastblok; (void)firstgemm;
  new_context = (dague_execution_context_t*)dague_thread_mempool_allocate( context->execution_units[0]->context_mempool );
  assignments = new_context->locals;
  /* Parse all the inputs and generate the ready execution tasks */
  for(k = 0;
      k <= cblknbr;
      k++) {
    assignments[0].value = k;
    assignments[1].value = gcblk2list = zpotrs_sp1dplus_inline_c_expr21_line_43((const dague_object_t*)__dague_object, assignments);
    assignments[2].value = browk1 = zpotrs_sp1dplus_inline_c_expr20_line_45((const dague_object_t*)__dague_object, assignments);
    assignments[3].value = lastbrow = zpotrs_sp1dplus_inline_c_expr19_line_46((const dague_object_t*)__dague_object, assignments);
    assignments[4].value = firstblok = zpotrs_sp1dplus_inline_c_expr18_line_47((const dague_object_t*)__dague_object, assignments);
    assignments[5].value = lastblok = zpotrs_sp1dplus_inline_c_expr17_line_48((const dague_object_t*)__dague_object, assignments);
    assignments[6].value = firstgemm = zpotrs_sp1dplus_inline_c_expr16_line_54((const dague_object_t*)__dague_object, assignments);
    if( !TRSM_FORWARD_pred(k, gcblk2list, browk1, lastbrow, firstblok, lastblok, firstgemm) ) continue;
    if( !(((gcblk2list <  0))) ) continue;
    DAGUE_STAT_INCREASE(mem_contexts, sizeof(dague_execution_context_t) + STAT_MALLOC_OVERHEAD);
    DAGUE_LIST_ITEM_SINGLETON( new_context );
    new_context->dague_object = (dague_object_t*)__dague_object;
    new_context->function = (const dague_function_t*)&zpotrs_sp1dplus_TRSM_FORWARD;
    new_context->priority = 0;
  new_context->data[0].data_repo = NULL;
  new_context->data[0].data      = NULL;
  new_context->data[1].data_repo = NULL;
  new_context->data[1].data      = NULL;
#if defined(DAGUE_DEBUG)
    {
      char tmp[128];
      printf("Add startup task %s\n",
             dague_service_to_string(new_context, tmp, 128));
    }
#endif
    dague_list_add_single_elem_by_priority( pready_list, new_context );
    new_context = (dague_execution_context_t*)dague_thread_mempool_allocate( context->execution_units[0]->context_mempool );
    assignments = new_context->locals;
    assignments[0].value = k;
    assignments[1].value = gcblk2list;
    assignments[2].value = browk1;
    assignments[3].value = lastbrow;
    assignments[4].value = firstblok;
    assignments[5].value = lastblok;
  }
  dague_thread_mempool_free( context->execution_units[0]->context_mempool, new_context );
  return 0;
}

static const dague_function_t zpotrs_sp1dplus_TRSM_FORWARD = {
  .name = "TRSM_FORWARD",
  .deps = 3,
  .flags = 0x0 | DAGUE_HAS_IN_IN_DEPENDENCIES,
  .function_id = 3,
  .dependencies_goal = 0x3,
  .nb_parameters = 1,
  .nb_definitions = 7,
  .params = { &symb_zpotrs_sp1dplus_TRSM_FORWARD_k },
  .locals = { &symb_zpotrs_sp1dplus_TRSM_FORWARD_k, &symb_zpotrs_sp1dplus_TRSM_FORWARD_gcblk2list, &symb_zpotrs_sp1dplus_TRSM_FORWARD_browk1, &symb_zpotrs_sp1dplus_TRSM_FORWARD_lastbrow, &symb_zpotrs_sp1dplus_TRSM_FORWARD_firstblok, &symb_zpotrs_sp1dplus_TRSM_FORWARD_lastblok, &symb_zpotrs_sp1dplus_TRSM_FORWARD_firstgemm },
  .pred = &pred_of_zpotrs_sp1dplus_TRSM_FORWARD_as_expr,
  .priority = NULL,
  .in = { &flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_A, &flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B },
  .out = { &flow_of_zpotrs_sp1dplus_TRSM_FORWARD_for_B },
  .iterate_successors = iterate_successors_of_zpotrs_sp1dplus_TRSM_FORWARD,
  .release_deps = release_deps_of_zpotrs_sp1dplus_TRSM_FORWARD,
  .hook = hook_of_zpotrs_sp1dplus_TRSM_FORWARD,
  .complete_execution = complete_hook_of_zpotrs_sp1dplus_TRSM_FORWARD,
#if defined(DAGUE_SIM)
  .sim_cost_fct = NULL,
#endif
  .key = (dague_functionkey_fn_t*)TRSM_FORWARD_hash,
};


static const dague_function_t *zpotrs_sp1dplus_functions[] = {
  &zpotrs_sp1dplus_GEMM_BACKWARD,
  &zpotrs_sp1dplus_TRSM_BACKWARD,
  &zpotrs_sp1dplus_GEMM_FORWARD,
  &zpotrs_sp1dplus_TRSM_FORWARD
};

static void zpotrs_sp1dplus_startup(dague_context_t *context, dague_object_t *dague_object, dague_execution_context_t** pready_list)
{
  zpotrs_sp1dplus_GEMM_FORWARD_startup_tasks(context, (__dague_zpotrs_sp1dplus_internal_object_t*)dague_object, pready_list);
  zpotrs_sp1dplus_TRSM_FORWARD_startup_tasks(context, (__dague_zpotrs_sp1dplus_internal_object_t*)dague_object, pready_list);
}
#undef descA
#undef A
#undef descB
#undef B
#undef datacode
#undef cblknbr
#undef bloknbr
#undef browsize
#undef p_work

dague_zpotrs_sp1dplus_object_t *dague_zpotrs_sp1dplus_new(sparse_matrix_desc_t * descA, dague_ddesc_t * A /* data A */, sparse_vector_desc_t * descB, dague_ddesc_t * B /* data B */, dague_memory_pool_t * p_work)
{
  __dague_zpotrs_sp1dplus_internal_object_t *_res = (__dague_zpotrs_sp1dplus_internal_object_t *)calloc(1, sizeof(__dague_zpotrs_sp1dplus_internal_object_t));
  int GEMM_BACKWARD_nblocal_tasks;
  int TRSM_BACKWARD_nblocal_tasks;
  int GEMM_FORWARD_nblocal_tasks;
  int TRSM_FORWARD_nblocal_tasks;

  _res->super.super.nb_functions    = DAGUE_zpotrs_sp1dplus_NB_FUNCTIONS;
  _res->super.super.functions_array = (const dague_function_t**)malloc(DAGUE_zpotrs_sp1dplus_NB_FUNCTIONS * sizeof(dague_function_t*));
  _res->super.super.dependencies_array = (dague_dependencies_t **)
              calloc(DAGUE_zpotrs_sp1dplus_NB_FUNCTIONS, sizeof(dague_dependencies_t *));
  memcpy(_res->super.super.functions_array, zpotrs_sp1dplus_functions, DAGUE_zpotrs_sp1dplus_NB_FUNCTIONS * sizeof(dague_function_t*));
  _res->super.arenas[DAGUE_zpotrs_sp1dplus_DEFAULT_ARENA] = (dague_arena_t*)calloc(1, sizeof(dague_arena_t));
  /* Now the Parameter-dependent structures: */
  _res->super.descA = descA;
  _res->super.A = A;
  _res->super.descB = descB;
  _res->super.B = B;
  _res->super.datacode = &(descA->pastix_data->solvmatr);
  _res->super.cblknbr = descA->pastix_data->solvmatr.symbmtx.cblknbr - 1;
  _res->super.bloknbr = descA->pastix_data->solvmatr.symbmtx.bloknbr - 2;
  _res->super.browsize = _res->super.bloknbr - _res->super.cblknbr - 1;
  _res->super.p_work = p_work;
  /* If profiling is enabled, the keys for profiling */
#  if defined(DAGUE_PROF_TRACE)
  _res->super.super.profiling_array = zpotrs_sp1dplus_profiling_array;
  if( -1 == zpotrs_sp1dplus_profiling_array[0] ) {
    dague_profiling_add_dictionary_keyword("GEMM_BACKWARD", "fill:CC2828",
                                       sizeof(dague_profile_ddesc_info_t), dague_profile_ddesc_key_to_string,
                                       (int*)&_res->super.super.profiling_array[0 + 2 * zpotrs_sp1dplus_GEMM_BACKWARD.function_id /* GEMM_BACKWARD start key */],
                                       (int*)&_res->super.super.profiling_array[1 + 2 * zpotrs_sp1dplus_GEMM_BACKWARD.function_id /* GEMM_BACKWARD end key */]);
    dague_profiling_add_dictionary_keyword("TRSM_BACKWARD", "fill:7ACC28",
                                       sizeof(dague_profile_ddesc_info_t), dague_profile_ddesc_key_to_string,
                                       (int*)&_res->super.super.profiling_array[0 + 2 * zpotrs_sp1dplus_TRSM_BACKWARD.function_id /* TRSM_BACKWARD start key */],
                                       (int*)&_res->super.super.profiling_array[1 + 2 * zpotrs_sp1dplus_TRSM_BACKWARD.function_id /* TRSM_BACKWARD end key */]);
    dague_profiling_add_dictionary_keyword("GEMM_FORWARD", "fill:28CCCC",
                                       sizeof(dague_profile_ddesc_info_t), dague_profile_ddesc_key_to_string,
                                       (int*)&_res->super.super.profiling_array[0 + 2 * zpotrs_sp1dplus_GEMM_FORWARD.function_id /* GEMM_FORWARD start key */],
                                       (int*)&_res->super.super.profiling_array[1 + 2 * zpotrs_sp1dplus_GEMM_FORWARD.function_id /* GEMM_FORWARD end key */]);
    dague_profiling_add_dictionary_keyword("TRSM_FORWARD", "fill:7A28CC",
                                       sizeof(dague_profile_ddesc_info_t), dague_profile_ddesc_key_to_string,
                                       (int*)&_res->super.super.profiling_array[0 + 2 * zpotrs_sp1dplus_TRSM_FORWARD.function_id /* TRSM_FORWARD start key */],
                                       (int*)&_res->super.super.profiling_array[1 + 2 * zpotrs_sp1dplus_TRSM_FORWARD.function_id /* TRSM_FORWARD end key */]);
  }
#  endif /* defined(DAGUE_PROF_TRACE) */
  /* Create the data repositories for this object */
  GEMM_BACKWARD_nblocal_tasks = zpotrs_sp1dplus_GEMM_BACKWARD_internal_init(_res);
  if( 0 == GEMM_BACKWARD_nblocal_tasks ) GEMM_BACKWARD_nblocal_tasks = 10;
  _res->GEMM_BACKWARD_repository = data_repo_create_nothreadsafe(
          ((unsigned int)(GEMM_BACKWARD_nblocal_tasks * 1.5)) > MAX_DATAREPO_HASH ?
          MAX_DATAREPO_HASH :
          ((unsigned int)(GEMM_BACKWARD_nblocal_tasks * 1.5)), 3);

  TRSM_BACKWARD_nblocal_tasks = zpotrs_sp1dplus_TRSM_BACKWARD_internal_init(_res);
  if( 0 == TRSM_BACKWARD_nblocal_tasks ) TRSM_BACKWARD_nblocal_tasks = 10;
  _res->TRSM_BACKWARD_repository = data_repo_create_nothreadsafe(
          ((unsigned int)(TRSM_BACKWARD_nblocal_tasks * 1.5)) > MAX_DATAREPO_HASH ?
          MAX_DATAREPO_HASH :
          ((unsigned int)(TRSM_BACKWARD_nblocal_tasks * 1.5)), 2);

  GEMM_FORWARD_nblocal_tasks = zpotrs_sp1dplus_GEMM_FORWARD_internal_init(_res);
  if( 0 == GEMM_FORWARD_nblocal_tasks ) GEMM_FORWARD_nblocal_tasks = 10;
  _res->GEMM_FORWARD_repository = data_repo_create_nothreadsafe(
          ((unsigned int)(GEMM_FORWARD_nblocal_tasks * 1.5)) > MAX_DATAREPO_HASH ?
          MAX_DATAREPO_HASH :
          ((unsigned int)(GEMM_FORWARD_nblocal_tasks * 1.5)), 3);

  TRSM_FORWARD_nblocal_tasks = zpotrs_sp1dplus_TRSM_FORWARD_internal_init(_res);
  if( 0 == TRSM_FORWARD_nblocal_tasks ) TRSM_FORWARD_nblocal_tasks = 10;
  _res->TRSM_FORWARD_repository = data_repo_create_nothreadsafe(
          ((unsigned int)(TRSM_FORWARD_nblocal_tasks * 1.5)) > MAX_DATAREPO_HASH ?
          MAX_DATAREPO_HASH :
          ((unsigned int)(TRSM_FORWARD_nblocal_tasks * 1.5)), 2);

  _res->super.super.startup_hook = zpotrs_sp1dplus_startup;
  (void)dague_object_register((dague_object_t*)_res);
  return (dague_zpotrs_sp1dplus_object_t*)_res;
}

void dague_zpotrs_sp1dplus_destroy( dague_zpotrs_sp1dplus_object_t *o )
{
  dague_object_t *d = (dague_object_t *)o;
  __dague_zpotrs_sp1dplus_internal_object_t *io = (__dague_zpotrs_sp1dplus_internal_object_t*)o;
  int i;
  free(d->functions_array);
  d->functions_array = NULL;
  d->nb_functions = 0;
  if( o->arenas[DAGUE_zpotrs_sp1dplus_DEFAULT_ARENA] != NULL ) {
    dague_arena_destruct(o->arenas[DAGUE_zpotrs_sp1dplus_DEFAULT_ARENA]);
    free(o->arenas[DAGUE_zpotrs_sp1dplus_DEFAULT_ARENA]);
  }
  /* Destroy the data repositories for this object */
   data_repo_destroy_nothreadsafe(io->GEMM_BACKWARD_repository);
   data_repo_destroy_nothreadsafe(io->TRSM_BACKWARD_repository);
   data_repo_destroy_nothreadsafe(io->GEMM_FORWARD_repository);
   data_repo_destroy_nothreadsafe(io->TRSM_FORWARD_repository);
  for(i = 0; i < DAGUE_zpotrs_sp1dplus_NB_FUNCTIONS; i++)
    dague_destruct_dependencies( d->dependencies_array[i] );
  free( d->dependencies_array );
  free(o);
}

