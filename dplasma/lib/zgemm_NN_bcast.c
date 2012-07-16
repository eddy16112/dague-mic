#line 2 "zgemm_NN_bcast.jdf"
/*
 *  Copyright (c) 2010
 *
 *  The University of Tennessee and The University
 *  of Tennessee Research Foundation.  All rights
 *  reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#define PRECISION_z

#include <plasma.h>
#include <core_blas.h>

#include "dague.h"
#include "data_distribution.h"
#include "data_dist/matrix/precision.h"
#include "data_dist/matrix/matrix.h"
#include "dplasma/lib/dplasmajdf.h"


#line 25 "zgemm_NN_bcast.c"
#include <dague.h>
#include <scheduling.h>
#include <remote_dep.h>
#include <datarepo.h>
#if defined(HAVE_PAPI)
#include <papime.h>
#endif
#include "zgemm_NN_bcast.h"

#define DAGUE_zgemm_NN_bcast_NB_FUNCTIONS 3
#define DAGUE_zgemm_NN_bcast_NB_DATA 3
#if defined(DAGUE_PROF_TRACE)
int zgemm_NN_bcast_profiling_array[2*DAGUE_zgemm_NN_bcast_NB_FUNCTIONS] = {-1};
#define TAKE_TIME(context, key, eid, refdesc, refid) do {   \
   dague_profile_ddesc_info_t info;                         \
   info.desc = (dague_ddesc_t*)refdesc;                     \
   info.id = refid;                                         \
   dague_profiling_trace(context->eu_profile,               \
                         __dague_object->super.super.profiling_array[(key)],\
                         eid, __dague_object->super.super.object_id, (void*)&info);\
  } while(0);
#else
#define TAKE_TIME(context, key, id, refdesc, refid)
#endif
#include "dague_prof_grapher.h"
#include <mempool.h>
typedef struct __dague_zgemm_NN_bcast_internal_object {
 dague_zgemm_NN_bcast_object_t super;
  /* The list of data repositories */
  data_repo_t *GEMM_repository;
  data_repo_t *READ_B_repository;
  data_repo_t *READ_A_repository;
} __dague_zgemm_NN_bcast_internal_object_t;

/* Globals */
#define transA (__dague_object->super.transA)
#define transB (__dague_object->super.transB)
#define alpha (__dague_object->super.alpha)
#define beta (__dague_object->super.beta)
#define descA (__dague_object->super.descA)
#define descB (__dague_object->super.descB)
#define descC (__dague_object->super.descC)

/* Data Access Macros */
#define C(C0,C1)  (((dague_ddesc_t*)__dague_object->super.C)->data_of((dague_ddesc_t*)__dague_object->super.C, (C0), (C1)))

#define B(B0,B1)  (((dague_ddesc_t*)__dague_object->super.B)->data_of((dague_ddesc_t*)__dague_object->super.B, (B0), (B1)))

#define A(A0,A1)  (((dague_ddesc_t*)__dague_object->super.A)->data_of((dague_ddesc_t*)__dague_object->super.A, (A0), (A1)))


/* Functions Predicates */
#define GEMM_pred(m, n, k) (((dague_ddesc_t*)(__dague_object->super.C))->myrank == ((dague_ddesc_t*)(__dague_object->super.C))->rank_of((dague_ddesc_t*)__dague_object->super.C, m, n))
#define READ_B_pred(k, n) (((dague_ddesc_t*)(__dague_object->super.B))->myrank == ((dague_ddesc_t*)(__dague_object->super.B))->rank_of((dague_ddesc_t*)__dague_object->super.B, k, n))
#define READ_A_pred(m, k) (((dague_ddesc_t*)(__dague_object->super.A))->myrank == ((dague_ddesc_t*)(__dague_object->super.A))->rank_of((dague_ddesc_t*)__dague_object->super.A, m, k))

/* Data Repositories */
#define GEMM_repo (__dague_object->GEMM_repository)
#define READ_B_repo (__dague_object->READ_B_repository)
#define READ_A_repo (__dague_object->READ_A_repository)
/* Dependency Tracking Allocation Macro */
#define ALLOCATE_DEP_TRACKING(DEPS, vMIN, vMAX, vNAME, vSYMBOL, PREVDEP, FLAG)               \
do {                                                                                         \
  int _vmin = (vMIN);                                                                        \
  int _vmax = (vMAX);                                                                        \
  (DEPS) = (dague_dependencies_t*)calloc(1, sizeof(dague_dependencies_t) +                   \
                   (_vmax - _vmin) * sizeof(dague_dependencies_union_t));                    \
  DEBUG3(("Allocate %d spaces for loop %s (min %d max %d) 0x%p last_dep 0x%p\n",    \
           (_vmax - _vmin + 1), (vNAME), _vmin, _vmax, (void*)(DEPS), (void*)(PREVDEP)));    \
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

static inline uint64_t GEMM_hash(const __dague_zgemm_NN_bcast_internal_object_t *__dague_object, const assignment_t *assignments)
{
  uint64_t __h = 0;
  (void)__dague_object;
  int m = assignments[0].value;
  int m_min = 0;
  int m_inc = 1;
  int m_range = ((descC.mt - 1) - m_min + 1 + (m_inc-1))/m_inc;
  int n = assignments[1].value;
  int n_min = 0;
  int n_inc = 1;
  int n_range = ((descC.nt - 1) - n_min + 1 + (n_inc-1))/n_inc;
  int k = assignments[2].value;
  int k_min = 0;
  __h += (m - m_min);
  __h += (n - n_min) * m_range;
  __h += (k - k_min) * m_range * n_range;
  return __h;
}

static inline uint64_t READ_B_hash(const __dague_zgemm_NN_bcast_internal_object_t *__dague_object, const assignment_t *assignments)
{
  uint64_t __h = 0;
  (void)__dague_object;
  int k = assignments[0].value;
  int k_min = 0;
  int k_inc = 1;
  int k_range = ((descB.mt - 1) - k_min + 1 + (k_inc-1))/k_inc;
  int n = assignments[1].value;
  int n_min = 0;
  __h += (k - k_min);
  __h += (n - n_min) * k_range;
  return __h;
}

static inline uint64_t READ_A_hash(const __dague_zgemm_NN_bcast_internal_object_t *__dague_object, const assignment_t *assignments)
{
  uint64_t __h = 0;
  (void)__dague_object;
  int m = assignments[0].value;
  int m_min = 0;
  int m_inc = 1;
  int m_range = ((descA.mt - 1) - m_min + 1 + (m_inc-1))/m_inc;
  int k = assignments[1].value;
  int k_min = 0;
  __h += (m - m_min);
  __h += (k - k_min) * m_range;
  return __h;
}

/** Predeclarations of the dague_function_t objects */
static const dague_function_t zgemm_NN_bcast_GEMM;
static const dague_function_t zgemm_NN_bcast_READ_B;
static const dague_function_t zgemm_NN_bcast_READ_A;
/** Declarations of the pseudo-dague_function_t objects for data */
static const dague_function_t zgemm_NN_bcast_C = {
  .name = "C",
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
};
static const dague_function_t zgemm_NN_bcast_B = {
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
};
static const dague_function_t zgemm_NN_bcast_A = {
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
};

/** Predeclarations of the parameters */
static const dague_flow_t flow_of_zgemm_NN_bcast_GEMM_for_A;
static const dague_flow_t flow_of_zgemm_NN_bcast_GEMM_for_B;
static const dague_flow_t flow_of_zgemm_NN_bcast_GEMM_for_C;
static const dague_flow_t flow_of_zgemm_NN_bcast_READ_B_for_B;
static const dague_flow_t flow_of_zgemm_NN_bcast_READ_A_for_A;
/**********************************************************************************
 *                                      GEMM                                      *
 **********************************************************************************/

static inline int minexpr_of_symb_zgemm_NN_bcast_GEMM_m_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;


  (void)__dague_object; (void)assignments;
  return 0;
}
static const expr_t minexpr_of_symb_zgemm_NN_bcast_GEMM_m = {
  .op = EXPR_OP_INLINE,
  .inline_func = minexpr_of_symb_zgemm_NN_bcast_GEMM_m_fct
};
static inline int maxexpr_of_symb_zgemm_NN_bcast_GEMM_m_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;


  (void)__dague_object; (void)assignments;
  return (descC.mt - 1);
}
static const expr_t maxexpr_of_symb_zgemm_NN_bcast_GEMM_m = {
  .op = EXPR_OP_INLINE,
  .inline_func = maxexpr_of_symb_zgemm_NN_bcast_GEMM_m_fct
};
static const symbol_t symb_zgemm_NN_bcast_GEMM_m = { .name = "m", .context_index = 0, .min = &minexpr_of_symb_zgemm_NN_bcast_GEMM_m, .max = &maxexpr_of_symb_zgemm_NN_bcast_GEMM_m, .cst_inc = 1, .expr_inc = NULL,  .flags = 0x0};

static inline int minexpr_of_symb_zgemm_NN_bcast_GEMM_n_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;


  (void)__dague_object; (void)assignments;
  return 0;
}
static const expr_t minexpr_of_symb_zgemm_NN_bcast_GEMM_n = {
  .op = EXPR_OP_INLINE,
  .inline_func = minexpr_of_symb_zgemm_NN_bcast_GEMM_n_fct
};
static inline int maxexpr_of_symb_zgemm_NN_bcast_GEMM_n_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;


  (void)__dague_object; (void)assignments;
  return (descC.nt - 1);
}
static const expr_t maxexpr_of_symb_zgemm_NN_bcast_GEMM_n = {
  .op = EXPR_OP_INLINE,
  .inline_func = maxexpr_of_symb_zgemm_NN_bcast_GEMM_n_fct
};
static const symbol_t symb_zgemm_NN_bcast_GEMM_n = { .name = "n", .context_index = 1, .min = &minexpr_of_symb_zgemm_NN_bcast_GEMM_n, .max = &maxexpr_of_symb_zgemm_NN_bcast_GEMM_n, .cst_inc = 1, .expr_inc = NULL,  .flags = 0x0};

static inline int minexpr_of_symb_zgemm_NN_bcast_GEMM_k_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;


  (void)__dague_object; (void)assignments;
  return 0;
}
static const expr_t minexpr_of_symb_zgemm_NN_bcast_GEMM_k = {
  .op = EXPR_OP_INLINE,
  .inline_func = minexpr_of_symb_zgemm_NN_bcast_GEMM_k_fct
};
static inline int maxexpr_of_symb_zgemm_NN_bcast_GEMM_k_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;


  (void)__dague_object; (void)assignments;
  return (descA.nt - 1);
}
static const expr_t maxexpr_of_symb_zgemm_NN_bcast_GEMM_k = {
  .op = EXPR_OP_INLINE,
  .inline_func = maxexpr_of_symb_zgemm_NN_bcast_GEMM_k_fct
};
static const symbol_t symb_zgemm_NN_bcast_GEMM_k = { .name = "k", .context_index = 2, .min = &minexpr_of_symb_zgemm_NN_bcast_GEMM_k, .max = &maxexpr_of_symb_zgemm_NN_bcast_GEMM_k, .cst_inc = 1, .expr_inc = NULL,  .flags = 0x0};

static inline int pred_of_zgemm_NN_bcast_GEMM_as_expr_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;
  int m = assignments[0].value;
  int n = assignments[1].value;
  int k = assignments[2].value;

  /* Silent Warnings: should look into predicate to know what variables are usefull */
  (void)__dague_object;
  (void)m;
  (void)n;
  (void)k;
  /* Compute Predicate */
  return GEMM_pred(m, n, k);
}
static const expr_t pred_of_zgemm_NN_bcast_GEMM_as_expr = {
  .op = EXPR_OP_INLINE,
  .inline_func = pred_of_zgemm_NN_bcast_GEMM_as_expr_fct
};
static inline int expr_of_cond_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep1_iftrue_atline_88_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;
  int n = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return (n == 0);
}
static const expr_t expr_of_cond_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep1_iftrue_atline_88 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep1_iftrue_atline_88_fct
};
static inline int expr_of_p1_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep1_iftrue_atline_88_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;
  int m = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return m;
}
static const expr_t expr_of_p1_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep1_iftrue_atline_88 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep1_iftrue_atline_88_fct
};
static inline int expr_of_p2_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep1_iftrue_atline_88_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;
  int k = assignments[2].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p2_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep1_iftrue_atline_88 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep1_iftrue_atline_88_fct
};
static const dep_t flow_of_zgemm_NN_bcast_GEMM_for_A_dep1_iftrue_atline_88 = {
  .cond = &expr_of_cond_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep1_iftrue_atline_88,
  .dague = &zgemm_NN_bcast_READ_A,
  .flow = &flow_of_zgemm_NN_bcast_READ_A_for_A,
  .datatype = { .index = DAGUE_zgemm_NN_bcast_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep1_iftrue_atline_88,
    &expr_of_p2_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep1_iftrue_atline_88
  }
};
static inline int expr_of_cond_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep1_iffalse_atline_88_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;
  int n = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return !(n == 0);
}
static const expr_t expr_of_cond_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep1_iffalse_atline_88 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep1_iffalse_atline_88_fct
};
static inline int expr_of_p1_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep1_iffalse_atline_88_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;
  int m = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return m;
}
static const expr_t expr_of_p1_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep1_iffalse_atline_88 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep1_iffalse_atline_88_fct
};
static inline int expr_of_p2_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep1_iffalse_atline_88_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;


  (void)__dague_object; (void)assignments;
  return 0;
}
static const expr_t expr_of_p2_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep1_iffalse_atline_88 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep1_iffalse_atline_88_fct
};
static inline int expr_of_p3_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep1_iffalse_atline_88_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;
  int k = assignments[2].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p3_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep1_iffalse_atline_88 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p3_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep1_iffalse_atline_88_fct
};
static const dep_t flow_of_zgemm_NN_bcast_GEMM_for_A_dep1_iffalse_atline_88 = {
  .cond = &expr_of_cond_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep1_iffalse_atline_88,
  .dague = &zgemm_NN_bcast_GEMM,
  .flow = &flow_of_zgemm_NN_bcast_GEMM_for_A,
  .datatype = { .index = DAGUE_zgemm_NN_bcast_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep1_iffalse_atline_88,
    &expr_of_p2_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep1_iffalse_atline_88,
    &expr_of_p3_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep1_iffalse_atline_88
  }
};
static inline int expr_of_cond_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep2_atline_89_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;
  int n = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return (n == 0);
}
static const expr_t expr_of_cond_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep2_atline_89 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep2_atline_89_fct
};
static inline int expr_of_p1_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep2_atline_89_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;
  int m = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return m;
}
static const expr_t expr_of_p1_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep2_atline_89 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep2_atline_89_fct
};
static inline int rangemin_of_expr_of_p2_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep2_atline_89_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;


  (void)__dague_object; (void)assignments;
  return 1;
}
static const expr_t rangemin_of_expr_of_p2_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep2_atline_89 = {
  .op = EXPR_OP_INLINE,
  .inline_func = rangemin_of_expr_of_p2_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep2_atline_89_fct
};
static inline int rangemax_of_expr_of_p2_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep2_atline_89_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;


  (void)__dague_object; (void)assignments;
  return (descC.nt - 1);
}
static const expr_t rangemax_of_expr_of_p2_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep2_atline_89 = {
  .op = EXPR_OP_INLINE,
  .inline_func = rangemax_of_expr_of_p2_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep2_atline_89_fct
};
static const expr_t expr_of_p2_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep2_atline_89 = {
  .op = EXPR_OP_RANGE_CST_INCREMENT,
  .u_expr.range = {
    .op1 = &rangemin_of_expr_of_p2_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep2_atline_89,
    .op2 = &rangemax_of_expr_of_p2_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep2_atline_89,
    .increment.cst = 1
  }
};
static inline int expr_of_p3_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep2_atline_89_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;
  int k = assignments[2].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p3_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep2_atline_89 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p3_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep2_atline_89_fct
};
static const dep_t flow_of_zgemm_NN_bcast_GEMM_for_A_dep2_atline_89 = {
  .cond = &expr_of_cond_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep2_atline_89,
  .dague = &zgemm_NN_bcast_GEMM,
  .flow = &flow_of_zgemm_NN_bcast_GEMM_for_A,
  .datatype = { .index = DAGUE_zgemm_NN_bcast_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep2_atline_89,
    &expr_of_p2_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep2_atline_89,
    &expr_of_p3_for_flow_of_zgemm_NN_bcast_GEMM_for_A_dep2_atline_89
  }
};
static const dague_flow_t flow_of_zgemm_NN_bcast_GEMM_for_A = {
  .name = "A",
  .sym_type = SYM_INOUT,
  .access_type = ACCESS_READ,
  .flow_index = 0,
  .dep_in  = { &flow_of_zgemm_NN_bcast_GEMM_for_A_dep1_iftrue_atline_88, &flow_of_zgemm_NN_bcast_GEMM_for_A_dep1_iffalse_atline_88 },
  .dep_out = { &flow_of_zgemm_NN_bcast_GEMM_for_A_dep2_atline_89 }
};

static inline int expr_of_cond_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep1_iftrue_atline_90_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;
  int m = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return (m == 0);
}
static const expr_t expr_of_cond_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep1_iftrue_atline_90 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep1_iftrue_atline_90_fct
};
static inline int expr_of_p1_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep1_iftrue_atline_90_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;
  int k = assignments[2].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p1_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep1_iftrue_atline_90 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep1_iftrue_atline_90_fct
};
static inline int expr_of_p2_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep1_iftrue_atline_90_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;
  int n = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return n;
}
static const expr_t expr_of_p2_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep1_iftrue_atline_90 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep1_iftrue_atline_90_fct
};
static const dep_t flow_of_zgemm_NN_bcast_GEMM_for_B_dep1_iftrue_atline_90 = {
  .cond = &expr_of_cond_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep1_iftrue_atline_90,
  .dague = &zgemm_NN_bcast_READ_B,
  .flow = &flow_of_zgemm_NN_bcast_READ_B_for_B,
  .datatype = { .index = DAGUE_zgemm_NN_bcast_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep1_iftrue_atline_90,
    &expr_of_p2_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep1_iftrue_atline_90
  }
};
static inline int expr_of_cond_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep1_iffalse_atline_90_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;
  int m = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return !(m == 0);
}
static const expr_t expr_of_cond_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep1_iffalse_atline_90 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep1_iffalse_atline_90_fct
};
static inline int expr_of_p1_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep1_iffalse_atline_90_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;


  (void)__dague_object; (void)assignments;
  return 0;
}
static const expr_t expr_of_p1_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep1_iffalse_atline_90 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep1_iffalse_atline_90_fct
};
static inline int expr_of_p2_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep1_iffalse_atline_90_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;
  int n = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return n;
}
static const expr_t expr_of_p2_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep1_iffalse_atline_90 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep1_iffalse_atline_90_fct
};
static inline int expr_of_p3_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep1_iffalse_atline_90_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;
  int k = assignments[2].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p3_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep1_iffalse_atline_90 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p3_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep1_iffalse_atline_90_fct
};
static const dep_t flow_of_zgemm_NN_bcast_GEMM_for_B_dep1_iffalse_atline_90 = {
  .cond = &expr_of_cond_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep1_iffalse_atline_90,
  .dague = &zgemm_NN_bcast_GEMM,
  .flow = &flow_of_zgemm_NN_bcast_GEMM_for_B,
  .datatype = { .index = DAGUE_zgemm_NN_bcast_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep1_iffalse_atline_90,
    &expr_of_p2_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep1_iffalse_atline_90,
    &expr_of_p3_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep1_iffalse_atline_90
  }
};
static inline int expr_of_cond_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep2_atline_91_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;
  int m = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return (m == 0);
}
static const expr_t expr_of_cond_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep2_atline_91 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep2_atline_91_fct
};
static inline int rangemin_of_expr_of_p1_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep2_atline_91_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;


  (void)__dague_object; (void)assignments;
  return 1;
}
static const expr_t rangemin_of_expr_of_p1_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep2_atline_91 = {
  .op = EXPR_OP_INLINE,
  .inline_func = rangemin_of_expr_of_p1_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep2_atline_91_fct
};
static inline int rangemax_of_expr_of_p1_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep2_atline_91_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;


  (void)__dague_object; (void)assignments;
  return descC.mt;
}
static const expr_t rangemax_of_expr_of_p1_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep2_atline_91 = {
  .op = EXPR_OP_INLINE,
  .inline_func = rangemax_of_expr_of_p1_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep2_atline_91_fct
};
static const expr_t expr_of_p1_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep2_atline_91 = {
  .op = EXPR_OP_RANGE_CST_INCREMENT,
  .u_expr.range = {
    .op1 = &rangemin_of_expr_of_p1_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep2_atline_91,
    .op2 = &rangemax_of_expr_of_p1_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep2_atline_91,
    .increment.cst = 1
  }
};
static inline int expr_of_p2_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep2_atline_91_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;
  int n = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return n;
}
static const expr_t expr_of_p2_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep2_atline_91 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep2_atline_91_fct
};
static inline int expr_of_p3_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep2_atline_91_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;
  int k = assignments[2].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p3_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep2_atline_91 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p3_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep2_atline_91_fct
};
static const dep_t flow_of_zgemm_NN_bcast_GEMM_for_B_dep2_atline_91 = {
  .cond = &expr_of_cond_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep2_atline_91,
  .dague = &zgemm_NN_bcast_GEMM,
  .flow = &flow_of_zgemm_NN_bcast_GEMM_for_B,
  .datatype = { .index = DAGUE_zgemm_NN_bcast_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep2_atline_91,
    &expr_of_p2_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep2_atline_91,
    &expr_of_p3_for_flow_of_zgemm_NN_bcast_GEMM_for_B_dep2_atline_91
  }
};
static const dague_flow_t flow_of_zgemm_NN_bcast_GEMM_for_B = {
  .name = "B",
  .sym_type = SYM_INOUT,
  .access_type = ACCESS_READ,
  .flow_index = 1,
  .dep_in  = { &flow_of_zgemm_NN_bcast_GEMM_for_B_dep1_iftrue_atline_90, &flow_of_zgemm_NN_bcast_GEMM_for_B_dep1_iffalse_atline_90 },
  .dep_out = { &flow_of_zgemm_NN_bcast_GEMM_for_B_dep2_atline_91 }
};

static inline int expr_of_cond_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep1_iftrue_atline_92_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;
  int k = assignments[2].value;

  (void)__dague_object; (void)assignments;
  return (k == 0);
}
static const expr_t expr_of_cond_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep1_iftrue_atline_92 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep1_iftrue_atline_92_fct
};
static inline int expr_of_p1_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep1_iftrue_atline_92_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;
  int m = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return m;
}
static const expr_t expr_of_p1_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep1_iftrue_atline_92 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep1_iftrue_atline_92_fct
};
static inline int expr_of_p2_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep1_iftrue_atline_92_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;
  int n = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return n;
}
static const expr_t expr_of_p2_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep1_iftrue_atline_92 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep1_iftrue_atline_92_fct
};
static const dep_t flow_of_zgemm_NN_bcast_GEMM_for_C_dep1_iftrue_atline_92 = {
  .cond = &expr_of_cond_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep1_iftrue_atline_92,
  .dague = &zgemm_NN_bcast_C,
  .datatype = { .index = DAGUE_zgemm_NN_bcast_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep1_iftrue_atline_92,
    &expr_of_p2_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep1_iftrue_atline_92
  }
};
static inline int expr_of_cond_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep1_iffalse_atline_92_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;
  int k = assignments[2].value;

  (void)__dague_object; (void)assignments;
  return !(k == 0);
}
static const expr_t expr_of_cond_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep1_iffalse_atline_92 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep1_iffalse_atline_92_fct
};
static inline int expr_of_p1_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep1_iffalse_atline_92_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;
  int m = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return m;
}
static const expr_t expr_of_p1_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep1_iffalse_atline_92 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep1_iffalse_atline_92_fct
};
static inline int expr_of_p2_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep1_iffalse_atline_92_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;
  int n = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return n;
}
static const expr_t expr_of_p2_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep1_iffalse_atline_92 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep1_iffalse_atline_92_fct
};
static inline int expr_of_p3_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep1_iffalse_atline_92_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;


  (void)__dague_object; (void)assignments;
  return 0;
}
static const expr_t expr_of_p3_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep1_iffalse_atline_92 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p3_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep1_iffalse_atline_92_fct
};
static const dep_t flow_of_zgemm_NN_bcast_GEMM_for_C_dep1_iffalse_atline_92 = {
  .cond = &expr_of_cond_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep1_iffalse_atline_92,
  .dague = &zgemm_NN_bcast_GEMM,
  .flow = &flow_of_zgemm_NN_bcast_GEMM_for_C,
  .datatype = { .index = DAGUE_zgemm_NN_bcast_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep1_iffalse_atline_92,
    &expr_of_p2_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep1_iffalse_atline_92,
    &expr_of_p3_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep1_iffalse_atline_92
  }
};
static inline int expr_of_cond_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep2_iftrue_atline_121_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;
  int k = assignments[2].value;

  (void)__dague_object; (void)assignments;
  return (k == (descA.nt - 1));
}
static const expr_t expr_of_cond_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep2_iftrue_atline_121 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep2_iftrue_atline_121_fct
};
static inline int expr_of_p1_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep2_iftrue_atline_121_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;
  int m = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return m;
}
static const expr_t expr_of_p1_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep2_iftrue_atline_121 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep2_iftrue_atline_121_fct
};
static inline int expr_of_p2_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep2_iftrue_atline_121_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;
  int n = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return n;
}
static const expr_t expr_of_p2_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep2_iftrue_atline_121 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep2_iftrue_atline_121_fct
};
static const dep_t flow_of_zgemm_NN_bcast_GEMM_for_C_dep2_iftrue_atline_121 = {
  .cond = &expr_of_cond_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep2_iftrue_atline_121,
  .dague = &zgemm_NN_bcast_C,
  .datatype = { .index = DAGUE_zgemm_NN_bcast_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep2_iftrue_atline_121,
    &expr_of_p2_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep2_iftrue_atline_121
  }
};
static inline int expr_of_cond_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep2_iffalse_atline_121_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;
  int k = assignments[2].value;

  (void)__dague_object; (void)assignments;
  return !(k == (descA.nt - 1));
}
static const expr_t expr_of_cond_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep2_iffalse_atline_121 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep2_iffalse_atline_121_fct
};
static inline int expr_of_p1_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep2_iffalse_atline_121_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;
  int m = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return m;
}
static const expr_t expr_of_p1_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep2_iffalse_atline_121 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep2_iffalse_atline_121_fct
};
static inline int expr_of_p2_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep2_iffalse_atline_121_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;
  int n = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return n;
}
static const expr_t expr_of_p2_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep2_iffalse_atline_121 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep2_iffalse_atline_121_fct
};
static inline int expr_of_p3_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep2_iffalse_atline_121_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;
  int k = assignments[2].value;

  (void)__dague_object; (void)assignments;
  return (k + 1);
}
static const expr_t expr_of_p3_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep2_iffalse_atline_121 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p3_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep2_iffalse_atline_121_fct
};
static const dep_t flow_of_zgemm_NN_bcast_GEMM_for_C_dep2_iffalse_atline_121 = {
  .cond = &expr_of_cond_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep2_iffalse_atline_121,
  .dague = &zgemm_NN_bcast_GEMM,
  .flow = &flow_of_zgemm_NN_bcast_GEMM_for_C,
  .datatype = { .index = DAGUE_zgemm_NN_bcast_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep2_iffalse_atline_121,
    &expr_of_p2_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep2_iffalse_atline_121,
    &expr_of_p3_for_flow_of_zgemm_NN_bcast_GEMM_for_C_dep2_iffalse_atline_121
  }
};
static const dague_flow_t flow_of_zgemm_NN_bcast_GEMM_for_C = {
  .name = "C",
  .sym_type = SYM_INOUT,
  .access_type = ACCESS_RW,
  .flow_index = 2,
  .dep_in  = { &flow_of_zgemm_NN_bcast_GEMM_for_C_dep1_iftrue_atline_92, &flow_of_zgemm_NN_bcast_GEMM_for_C_dep1_iffalse_atline_92 },
  .dep_out = { &flow_of_zgemm_NN_bcast_GEMM_for_C_dep2_iftrue_atline_121, &flow_of_zgemm_NN_bcast_GEMM_for_C_dep2_iffalse_atline_121 }
};

static void
iterate_successors_of_zgemm_NN_bcast_GEMM(dague_execution_unit_t *eu, dague_execution_context_t *this_task,
               uint32_t action_mask, dague_ontask_function_t *ontask, void *ontask_arg)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)this_task->dague_object;
  dague_execution_context_t nc;
  dague_arena_t* arena = NULL;
  int __nb_elt = -1, vpid_dst = -1;
  int rank_src = 0, rank_dst = 0;
  int m = this_task->locals[0].value;
  int n = this_task->locals[1].value;
  int k = this_task->locals[2].value;
  (void)rank_src; (void)rank_dst; (void)__dague_object; (void)vpid_dst; (void)__nb_elt;
  (void)m;  (void)n;  (void)k;
  nc.dague_object = this_task->dague_object;
#if defined(DISTRIBUTED)
  rank_src = ((dague_ddesc_t*)__dague_object->super.C)->rank_of((dague_ddesc_t*)__dague_object->super.C, m, n);
#endif
  /* Flow of Data A */
  if( action_mask & (1 << 0) ) {
#if defined(DISTRIBUTED)
    arena = __dague_object->super.arenas[DAGUE_zgemm_NN_bcast_DEFAULT_ARENA];
    __nb_elt = 1;
#endif
    if( (n == 0) ) {
      nc.function = (const dague_function_t*)&zgemm_NN_bcast_GEMM;
      {
        const int GEMM_m = m;
          if( (GEMM_m >= (0)) && (GEMM_m <= ((descC.mt - 1))) ) {
            nc.locals[0].value = GEMM_m;
          {
            int GEMM_n;
            for( GEMM_n = 1;GEMM_n <= (descC.nt - 1); GEMM_n+=1) {
                if( (GEMM_n >= (0)) && (GEMM_n <= ((descC.nt - 1))) ) {
                  nc.locals[1].value = GEMM_n;
                {
                  const int GEMM_k = k;
                    if( (GEMM_k >= (0)) && (GEMM_k <= ((descA.nt - 1))) ) {
                      nc.locals[2].value = GEMM_k;
#if defined(DISTRIBUTED)
                      rank_dst = ((dague_ddesc_t*)__dague_object->super.C)->rank_of((dague_ddesc_t*)__dague_object->super.C, GEMM_m, GEMM_n);
                      if( eu != NULL && rank_dst == eu->virtual_process->dague_context->my_rank ) vpid_dst = ((dague_ddesc_t*)__dague_object->super.C)->vpid_of((dague_ddesc_t*)__dague_object->super.C, GEMM_m, GEMM_n);
#else /* !DISTRIBUTED */
                      vpid_dst = ((dague_ddesc_t*)__dague_object->super.C)->vpid_of((dague_ddesc_t*)__dague_object->super.C, GEMM_m, GEMM_n);
#endif /* DISTRIBUTED */
#if defined(DAGUE_DEBUG_VERBOSE1)
                    if( NULL != eu ) {
                      char tmp[128], tmp1[128];
                      DEBUG(("thread %d VP %d release deps of A:%s to A:%s (from node %d to %d)\n",
                             eu->th_id, eu->virtual_process->vp_id,
                             dague_snprintf_execution_context(tmp, 128, this_task),
                             dague_snprintf_execution_context(tmp1, 128, &nc), rank_src, rank_dst));
                    }
#endif
                      nc.flowname = "A";
                      nc.priority = __dague_object->super.super.object_priority;
                      if( DAGUE_ITERATE_STOP == ontask(eu, &nc, this_task, 0, 0, rank_src, rank_dst, vpid_dst, arena, __nb_elt, ontask_arg) )
                        return;
        }
          }
            }
              }
                }
                  }
                    }
    }
  }
  /* Flow of Data B */
  if( action_mask & (1 << 1) ) {
#if defined(DISTRIBUTED)
    arena = __dague_object->super.arenas[DAGUE_zgemm_NN_bcast_DEFAULT_ARENA];
    __nb_elt = 1;
#endif
    if( (m == 0) ) {
      nc.function = (const dague_function_t*)&zgemm_NN_bcast_GEMM;
      {
        int GEMM_m;
        for( GEMM_m = 1;GEMM_m <= descC.mt; GEMM_m+=1) {
            if( (GEMM_m >= (0)) && (GEMM_m <= ((descC.mt - 1))) ) {
              nc.locals[0].value = GEMM_m;
            {
              const int GEMM_n = n;
                if( (GEMM_n >= (0)) && (GEMM_n <= ((descC.nt - 1))) ) {
                  nc.locals[1].value = GEMM_n;
                {
                  const int GEMM_k = k;
                    if( (GEMM_k >= (0)) && (GEMM_k <= ((descA.nt - 1))) ) {
                      nc.locals[2].value = GEMM_k;
#if defined(DISTRIBUTED)
                      rank_dst = ((dague_ddesc_t*)__dague_object->super.C)->rank_of((dague_ddesc_t*)__dague_object->super.C, GEMM_m, GEMM_n);
                      if( eu != NULL && rank_dst == eu->virtual_process->dague_context->my_rank ) vpid_dst = ((dague_ddesc_t*)__dague_object->super.C)->vpid_of((dague_ddesc_t*)__dague_object->super.C, GEMM_m, GEMM_n);
#else /* !DISTRIBUTED */
                      vpid_dst = ((dague_ddesc_t*)__dague_object->super.C)->vpid_of((dague_ddesc_t*)__dague_object->super.C, GEMM_m, GEMM_n);
#endif /* DISTRIBUTED */
#if defined(DAGUE_DEBUG_VERBOSE1)
                    if( NULL != eu ) {
                      char tmp[128], tmp1[128];
                      DEBUG(("thread %d VP %d release deps of B:%s to B:%s (from node %d to %d)\n",
                             eu->th_id, eu->virtual_process->vp_id,
                             dague_snprintf_execution_context(tmp, 128, this_task),
                             dague_snprintf_execution_context(tmp1, 128, &nc), rank_src, rank_dst));
                    }
#endif
                      nc.flowname = "B";
                      nc.priority = __dague_object->super.super.object_priority;
                      if( DAGUE_ITERATE_STOP == ontask(eu, &nc, this_task, 1, 0, rank_src, rank_dst, vpid_dst, arena, __nb_elt, ontask_arg) )
                        return;
        }
          }
            }
              }
                }
                  }
                    }
    }
  }
  /* Flow of Data C */
  if( action_mask & (1 << 2) ) {
#if defined(DISTRIBUTED)
    arena = __dague_object->super.arenas[DAGUE_zgemm_NN_bcast_DEFAULT_ARENA];
    __nb_elt = 1;
#endif
    if( !((k == (descA.nt - 1))) ) {
      nc.function = (const dague_function_t*)&zgemm_NN_bcast_GEMM;
      {
        const int GEMM_m = m;
          if( (GEMM_m >= (0)) && (GEMM_m <= ((descC.mt - 1))) ) {
            nc.locals[0].value = GEMM_m;
          {
            const int GEMM_n = n;
              if( (GEMM_n >= (0)) && (GEMM_n <= ((descC.nt - 1))) ) {
                nc.locals[1].value = GEMM_n;
              {
                const int GEMM_k = (k + 1);
                  if( (GEMM_k >= (0)) && (GEMM_k <= ((descA.nt - 1))) ) {
                    nc.locals[2].value = GEMM_k;
#if defined(DISTRIBUTED)
                    rank_dst = ((dague_ddesc_t*)__dague_object->super.C)->rank_of((dague_ddesc_t*)__dague_object->super.C, GEMM_m, GEMM_n);
                    if( eu != NULL && rank_dst == eu->virtual_process->dague_context->my_rank ) vpid_dst = ((dague_ddesc_t*)__dague_object->super.C)->vpid_of((dague_ddesc_t*)__dague_object->super.C, GEMM_m, GEMM_n);
#else /* !DISTRIBUTED */
                    vpid_dst = ((dague_ddesc_t*)__dague_object->super.C)->vpid_of((dague_ddesc_t*)__dague_object->super.C, GEMM_m, GEMM_n);
#endif /* DISTRIBUTED */
#if defined(DAGUE_DEBUG_VERBOSE1)
                  if( NULL != eu ) {
                    char tmp[128], tmp1[128];
                    DEBUG(("thread %d VP %d release deps of C:%s to C:%s (from node %d to %d)\n",
                           eu->th_id, eu->virtual_process->vp_id,
                           dague_snprintf_execution_context(tmp, 128, this_task),
                           dague_snprintf_execution_context(tmp1, 128, &nc), rank_src, rank_dst));
                  }
#endif
                    nc.flowname = "C";
                    nc.priority = __dague_object->super.super.object_priority;
                    if( DAGUE_ITERATE_STOP == ontask(eu, &nc, this_task, 2, 1, rank_src, rank_dst, vpid_dst, arena, __nb_elt, ontask_arg) )
                      return;
        }
          }
            }
              }
                }
                  }
    }
  }
  (void)nc;(void)arena;(void)eu;(void)ontask;(void)ontask_arg;(void)rank_dst;(void)action_mask;
}

static int release_deps_of_zgemm_NN_bcast_GEMM(dague_execution_unit_t *eu, dague_execution_context_t *context, uint32_t action_mask, dague_remote_deps_t *deps)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t *)context->dague_object;
  dague_release_dep_fct_arg_t arg;
  int __vp_id;
  arg.nb_released = 0;
  arg.output_usage = 0;
  arg.action_mask = action_mask;
  arg.deps = deps;
  arg.ready_lists = (eu != NULL) ? calloc(sizeof(dague_execution_context_t *), eu->virtual_process->dague_context->nb_vp) : NULL;
  (void)__dague_object;
  if( action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS ) {
    arg.output_entry = data_repo_lookup_entry_and_create( eu, GEMM_repo, GEMM_hash(__dague_object, context->locals) );
#if defined(DAGUE_SIM)
    assert(arg.output_entry->sim_exec_date == 0);
    arg.output_entry->sim_exec_date = context->sim_exec_date;
#endif
  }
#if defined(DISTRIBUTED)
  arg.remote_deps_count = 0;
  arg.remote_deps = NULL;
#endif
  iterate_successors_of_zgemm_NN_bcast_GEMM(eu, context, action_mask, dague_release_dep_fct, &arg);

  if(action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS) {
    struct dague_vp** vps = eu->virtual_process->dague_context->virtual_processes;
    data_repo_entry_addto_usage_limit(GEMM_repo, arg.output_entry->key, arg.output_usage);
    for(__vp_id = 0; __vp_id < eu->virtual_process->dague_context->nb_vp; __vp_id++) {
      if( NULL == arg.ready_lists[__vp_id] ) continue;
      __dague_schedule(vps[__vp_id]->execution_units[0], arg.ready_lists[__vp_id]);
      arg.ready_lists[__vp_id] = NULL;
    }
    free(arg.ready_lists);
  }
#if defined(DISTRIBUTED)
  if( 0 == arg.remote_deps_count ) {
    if( NULL != arg.remote_deps ) {
      remote_deps_free(arg.remote_deps);
      arg.remote_deps = NULL;
    }
  }
  else if( (action_mask & DAGUE_ACTION_SEND_REMOTE_DEPS) ) {
    arg.nb_released += dague_remote_dep_activate(eu, context, arg.remote_deps, arg.remote_deps_count);
  }
#endif
  if( action_mask & DAGUE_ACTION_RELEASE_LOCAL_REFS ) {
    int m = context->locals[0].value;
    int n = context->locals[1].value;
    int k = context->locals[2].value;
    (void)m; (void)n; (void)k;

    if( (n == 0) ) {
      data_repo_entry_used_once( eu, READ_A_repo, context->data[0].data_repo->key );
      (void)AUNREF(context->data[0].data);
    } else {
      data_repo_entry_used_once( eu, GEMM_repo, context->data[0].data_repo->key );
      (void)AUNREF(context->data[0].data);
    }
    if( (m == 0) ) {
      data_repo_entry_used_once( eu, READ_B_repo, context->data[1].data_repo->key );
      (void)AUNREF(context->data[1].data);
    } else {
      data_repo_entry_used_once( eu, GEMM_repo, context->data[1].data_repo->key );
      (void)AUNREF(context->data[1].data);
    }
    if( !((k == 0)) ) {
      data_repo_entry_used_once( eu, GEMM_repo, context->data[2].data_repo->key );
      (void)AUNREF(context->data[2].data);
    }
  }
  return arg.nb_released;
}

static int hook_of_zgemm_NN_bcast_GEMM(dague_execution_unit_t *context, dague_execution_context_t *this_task)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (__dague_zgemm_NN_bcast_internal_object_t *)this_task->dague_object;
  assignment_t tass[MAX_PARAM_COUNT];
  (void)context; (void)__dague_object; (void)tass;
#if defined(DAGUE_SIM)
  int __dague_simulation_date = 0;
#endif
  int m = this_task->locals[0].value;
  int n = this_task->locals[1].value;
  int k = this_task->locals[2].value;
  (void)m;  (void)n;  (void)k;

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

  /** Lookup the input data, and store them in the context if any */
  eA = this_task->data[0].data_repo;
  gA = this_task->data[0].data;
  if( NULL == gA ) {
    if( (n == 0) ) {
        tass[0].value = m;
        tass[1].value = k;
      eA = data_repo_lookup_entry( READ_A_repo, READ_A_hash( __dague_object, tass ));
      gA = eA->data[0];
    } else {
        tass[0].value = m;
        tass[1].value = 0;
        tass[2].value = k;
      eA = data_repo_lookup_entry( GEMM_repo, GEMM_hash( __dague_object, tass ));
      gA = eA->data[0];
    }
    this_task->data[0].data = gA;
    this_task->data[0].data_repo = eA;
  }
  A = ADATA(gA);
#if defined(DAGUE_SIM)
  if( (NULL != eA) && (eA->sim_exec_date > __dague_simulation_date) )
    __dague_simulation_date =  eA->sim_exec_date;
#endif
  eB = this_task->data[1].data_repo;
  gB = this_task->data[1].data;
  if( NULL == gB ) {
    if( (m == 0) ) {
        tass[0].value = k;
        tass[1].value = n;
      eB = data_repo_lookup_entry( READ_B_repo, READ_B_hash( __dague_object, tass ));
      gB = eB->data[0];
    } else {
        tass[0].value = 0;
        tass[1].value = n;
        tass[2].value = k;
      eB = data_repo_lookup_entry( GEMM_repo, GEMM_hash( __dague_object, tass ));
      gB = eB->data[1];
    }
    this_task->data[1].data = gB;
    this_task->data[1].data_repo = eB;
  }
  B = ADATA(gB);
#if defined(DAGUE_SIM)
  if( (NULL != eB) && (eB->sim_exec_date > __dague_simulation_date) )
    __dague_simulation_date =  eB->sim_exec_date;
#endif
  eC = this_task->data[2].data_repo;
  gC = this_task->data[2].data;
  if( NULL == gC ) {
    if( (k == 0) ) {
      gC = (dague_arena_chunk_t*) C(m, n);
    } else {
        tass[0].value = m;
        tass[1].value = n;
        tass[2].value = 0;
      eC = data_repo_lookup_entry( GEMM_repo, GEMM_hash( __dague_object, tass ));
      gC = eC->data[2];
    }
    this_task->data[2].data = gC;
    this_task->data[2].data_repo = eC;
  }
  C = ADATA(gC);
#if defined(DAGUE_SIM)
  if( (NULL != eC) && (eC->sim_exec_date > __dague_simulation_date) )
    __dague_simulation_date =  eC->sim_exec_date;
#endif
#if defined(DAGUE_SIM)
  if( this_task->function->sim_cost_fct != NULL ) {
    this_task->sim_exec_date = __dague_simulation_date + this_task->function->sim_cost_fct(this_task);
  } else {
    this_task->sim_exec_date = __dague_simulation_date;
  }
  if( context->largest_simulation_date < this_task->sim_exec_date )
    context->largest_simulation_date = this_task->sim_exec_date;
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
 *                                  GEMM BODY                                    *
 *--------------------------------------------------------------------------------*/

  TAKE_TIME(context, 2*this_task->function->function_id, GEMM_hash( __dague_object, this_task->locals), __dague_object->super.C, ((dague_ddesc_t*)(__dague_object->super.C))->data_key((dague_ddesc_t*)__dague_object->super.C, m, n) );
#line 93 "zgemm_NN_bcast.jdf"
        Dague_Complex64_t lbeta = (k == 0) ? beta : (Dague_Complex64_t)1.0;
        int tempmm = m == descC.mt-1 ? descC.m - m * descC.mb : descC.mb;
        int tempnn = n == descC.nt-1 ? descC.n - n * descC.nb : descC.nb;
        int tempkk = k == descA.nt-1 ? descA.n - k * descA.nb : descA.nb;
        int ldam = BLKLDD(descA, m);
        int ldbk = BLKLDD(descB, k);
        int ldcm = BLKLDD(descC, m);
		
		printf("I am in bcast\n");		

        DRYRUN(
            CORE_zgemm(
                transA, transB,
                tempmm, tempnn, tempkk,
                alpha, A /*A(m, k)*/, ldam,
                       B /*B(k, n)*/, ldbk,
                lbeta, C /*C(m, n)*/, ldcm);
            );

        printlog("thread %d gemm( %d, %d, %d )\n"
                 "    ( %s, %s, %d, %d, %d, %f, A(%d,%d), %d, B(%d,%d), %d, %f, C(%d,%d), %d)\n",
                 context->eu_id, m, n, k,
                 plasma_const( transA ), plasma_const( transB ),
                 tempmm, tempnn, tempkk,
                 creal(alpha), m, k, ldam,
                               k, n, ldbk,
                 creal(lbeta), m, n, ldcm );

#line 1312 "zgemm_NN_bcast.c"
/*--------------------------------------------------------------------------------*
 *                                END OF GEMM BODY                                *
 *--------------------------------------------------------------------------------*/



#endif /*!defined(DAGUE_PROF_DRY_BODY)*/

  return 0;
}
static int complete_hook_of_zgemm_NN_bcast_GEMM(dague_execution_unit_t *context, dague_execution_context_t *this_task)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (__dague_zgemm_NN_bcast_internal_object_t *)this_task->dague_object;
  (void)context; (void)__dague_object;
  int m = this_task->locals[0].value;
  int n = this_task->locals[1].value;
  int k = this_task->locals[2].value;
  (void)m;  (void)n;  (void)k;

  TAKE_TIME(context,2*this_task->function->function_id+1, GEMM_hash( __dague_object, this_task->locals ), NULL, 0);
  /** PAPI events */
#if defined(HAVE_PAPI)
  papime_stop_thread_counters();
#endif
#if defined(DISTRIBUTED)
  /** If not working on distributed, there is no risk that data is not in place */
  if( (k == (descA.nt - 1)) ) {
    if( ADATA(this_task->data[2].data) != C(m, n) ) {
      int __arena_index = DAGUE_zgemm_NN_bcast_DEFAULT_ARENA;
      int __dtt_nb = 1;
      assert( (__arena_index>=0) && (__arena_index < __dague_object->super.arenas_size) );
      assert( __dtt_nb >= 0 );
      dague_remote_dep_memcpy( context, this_task->dague_object, C(m, n), this_task->data[2].data, 
                               __dague_object->super.arenas[__arena_index]->opaque_dtt,
                               __dtt_nb );
    }
  }
#endif /* DISTRIBUTED */
  dague_prof_grapher_task(this_task, context->th_id, context->virtual_process->vp_id, GEMM_hash(__dague_object, this_task->locals));
  release_deps_of_zgemm_NN_bcast_GEMM(context, this_task,
      DAGUE_ACTION_RELEASE_REMOTE_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_REFS |
      DAGUE_ACTION_DEPS_MASK,
      NULL);
  return 0;
}

static int zgemm_NN_bcast_GEMM_internal_init(__dague_zgemm_NN_bcast_internal_object_t *__dague_object)
{
  dague_dependencies_t *dep = NULL;
  assignment_t assignments[MAX_LOCAL_COUNT];(void) assignments;
  int nb_tasks = 0;
  int32_t  m, n, k;
  int32_t  m_min = 0x7fffffff, n_min = 0x7fffffff, k_min = 0x7fffffff;
  int32_t  m_max = 0, n_max = 0, k_max = 0;
  (void)__dague_object;
  int32_t m_start, m_end, m_inc;
  int32_t n_start, n_end, n_inc;
  int32_t k_start, k_end, k_inc;
  /* First, find the min and max value for each of the dimensions */
  for(m = 0;
      m <= (descC.mt - 1);
      m += 1) {
    assignments[0].value = m;
    for(n = 0;
        n <= (descC.nt - 1);
        n += 1) {
      assignments[1].value = n;
      for(k = 0;
          k <= (descA.nt - 1);
          k += 1) {
        assignments[2].value = k;
        if( !GEMM_pred(m, n, k) ) continue;
        nb_tasks++;
        m_max = dague_imax(m_max, m);
        m_min = dague_imin(m_min, m);
        n_max = dague_imax(n_max, n);
        n_min = dague_imin(n_min, n);
        k_max = dague_imax(k_max, k);
        k_min = dague_imin(k_min, k);
      }
    }
  }

  /**
   * Now, for each of the dimensions, re-iterate on the space,
   * and if at least one value is defined, allocate arrays to point
   * to it. Array dimensions are defined by the (rough) observation above
   **/
  DEBUG2(("Allocating dependencies array for zgemm_NN_bcast_GEMM_internal_init\n"));
  dep = NULL;
  m_start = 0;
  m_end = (descC.mt - 1);
  m_inc = 1;
  for(m = dague_imax(m_start, m_min); m <= dague_imin(m_end, m_max); m+=m_inc) {
    assignments[0].value = m;
    n_start = 0;
    n_end = (descC.nt - 1);
    n_inc = 1;
    for(n = dague_imax(n_start, n_min); n <= dague_imin(n_end, n_max); n+=n_inc) {
      assignments[1].value = n;
      k_start = 0;
      k_end = (descA.nt - 1);
      k_inc = 1;
      for(k = dague_imax(k_start, k_min); k <= dague_imin(k_end, k_max); k+=k_inc) {
        assignments[2].value = k;
        if( GEMM_pred(m, n, k) ) {
          /* We did find one! Allocate the dependencies array. */
        if( dep == NULL ) {
          ALLOCATE_DEP_TRACKING(dep, m_min, m_max, "m", &symb_zgemm_NN_bcast_GEMM_m, NULL, DAGUE_DEPENDENCIES_FLAG_NEXT);
        }
        if( dep->u.next[m-m_min] == NULL ) {
          ALLOCATE_DEP_TRACKING(dep->u.next[m-m_min], n_min, n_max, "n", &symb_zgemm_NN_bcast_GEMM_n, dep, DAGUE_DEPENDENCIES_FLAG_NEXT);
        }
        if( dep->u.next[m-m_min]->u.next[n-n_min] == NULL ) {
          ALLOCATE_DEP_TRACKING(dep->u.next[m-m_min]->u.next[n-n_min], k_min, k_max, "k", &symb_zgemm_NN_bcast_GEMM_k, dep->u.next[m-m_min], DAGUE_DEPENDENCIES_FLAG_FINAL);
        }
          break;
        }
      }
    }
  }
  (void)m_start; (void)m_end; (void)m_inc;  (void)n_start; (void)n_end; (void)n_inc;  (void)k_start; (void)k_end; (void)k_inc;
  __dague_object->super.super.dependencies_array[0] = dep;
  __dague_object->super.super.nb_local_tasks += nb_tasks;
  return nb_tasks;
}

static const dague_function_t zgemm_NN_bcast_GEMM = {
  .name = "GEMM",
  .deps = 0,
  .flags = 0x0 | DAGUE_HAS_IN_IN_DEPENDENCIES,
  .function_id = 0,
  .dependencies_goal = 0x7,
  .nb_parameters = 3,
  .nb_definitions = 3,
  .params = { &symb_zgemm_NN_bcast_GEMM_m, &symb_zgemm_NN_bcast_GEMM_n, &symb_zgemm_NN_bcast_GEMM_k },
  .locals = { &symb_zgemm_NN_bcast_GEMM_m, &symb_zgemm_NN_bcast_GEMM_n, &symb_zgemm_NN_bcast_GEMM_k },
  .pred = &pred_of_zgemm_NN_bcast_GEMM_as_expr,
  .priority = NULL,
  .in = { &flow_of_zgemm_NN_bcast_GEMM_for_A, &flow_of_zgemm_NN_bcast_GEMM_for_B, &flow_of_zgemm_NN_bcast_GEMM_for_C },
  .out = { &flow_of_zgemm_NN_bcast_GEMM_for_A, &flow_of_zgemm_NN_bcast_GEMM_for_B, &flow_of_zgemm_NN_bcast_GEMM_for_C },
  .iterate_successors = iterate_successors_of_zgemm_NN_bcast_GEMM,
  .release_deps = release_deps_of_zgemm_NN_bcast_GEMM,
  .hook = hook_of_zgemm_NN_bcast_GEMM,
  .complete_execution = complete_hook_of_zgemm_NN_bcast_GEMM,
#if defined(DAGUE_SIM)
  .sim_cost_fct = NULL,
#endif
  .key = (dague_functionkey_fn_t*)GEMM_hash,
};


/**********************************************************************************
 *                                    READ_B                                    *
 **********************************************************************************/

static inline int minexpr_of_symb_zgemm_NN_bcast_READ_B_k_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;


  (void)__dague_object; (void)assignments;
  return 0;
}
static const expr_t minexpr_of_symb_zgemm_NN_bcast_READ_B_k = {
  .op = EXPR_OP_INLINE,
  .inline_func = minexpr_of_symb_zgemm_NN_bcast_READ_B_k_fct
};
static inline int maxexpr_of_symb_zgemm_NN_bcast_READ_B_k_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;


  (void)__dague_object; (void)assignments;
  return (descB.mt - 1);
}
static const expr_t maxexpr_of_symb_zgemm_NN_bcast_READ_B_k = {
  .op = EXPR_OP_INLINE,
  .inline_func = maxexpr_of_symb_zgemm_NN_bcast_READ_B_k_fct
};
static const symbol_t symb_zgemm_NN_bcast_READ_B_k = { .name = "k", .context_index = 0, .min = &minexpr_of_symb_zgemm_NN_bcast_READ_B_k, .max = &maxexpr_of_symb_zgemm_NN_bcast_READ_B_k, .cst_inc = 1, .expr_inc = NULL,  .flags = 0x0};

static inline int minexpr_of_symb_zgemm_NN_bcast_READ_B_n_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;


  (void)__dague_object; (void)assignments;
  return 0;
}
static const expr_t minexpr_of_symb_zgemm_NN_bcast_READ_B_n = {
  .op = EXPR_OP_INLINE,
  .inline_func = minexpr_of_symb_zgemm_NN_bcast_READ_B_n_fct
};
static inline int maxexpr_of_symb_zgemm_NN_bcast_READ_B_n_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;


  (void)__dague_object; (void)assignments;
  return (descB.nt - 1);
}
static const expr_t maxexpr_of_symb_zgemm_NN_bcast_READ_B_n = {
  .op = EXPR_OP_INLINE,
  .inline_func = maxexpr_of_symb_zgemm_NN_bcast_READ_B_n_fct
};
static const symbol_t symb_zgemm_NN_bcast_READ_B_n = { .name = "n", .context_index = 1, .min = &minexpr_of_symb_zgemm_NN_bcast_READ_B_n, .max = &maxexpr_of_symb_zgemm_NN_bcast_READ_B_n, .cst_inc = 1, .expr_inc = NULL,  .flags = 0x0};

static inline int pred_of_zgemm_NN_bcast_READ_B_as_expr_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;
  int n = assignments[1].value;

  /* Silent Warnings: should look into predicate to know what variables are usefull */
  (void)__dague_object;
  (void)k;
  (void)n;
  /* Compute Predicate */
  return READ_B_pred(k, n);
}
static const expr_t pred_of_zgemm_NN_bcast_READ_B_as_expr = {
  .op = EXPR_OP_INLINE,
  .inline_func = pred_of_zgemm_NN_bcast_READ_B_as_expr_fct
};
static inline int expr_of_p1_for_flow_of_zgemm_NN_bcast_READ_B_for_B_dep1_atline_68_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p1_for_flow_of_zgemm_NN_bcast_READ_B_for_B_dep1_atline_68 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgemm_NN_bcast_READ_B_for_B_dep1_atline_68_fct
};
static inline int expr_of_p2_for_flow_of_zgemm_NN_bcast_READ_B_for_B_dep1_atline_68_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;
  int n = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return n;
}
static const expr_t expr_of_p2_for_flow_of_zgemm_NN_bcast_READ_B_for_B_dep1_atline_68 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgemm_NN_bcast_READ_B_for_B_dep1_atline_68_fct
};
static const dep_t flow_of_zgemm_NN_bcast_READ_B_for_B_dep1_atline_68 = {
  .cond = NULL,
  .dague = &zgemm_NN_bcast_B,
  .datatype = { .index = DAGUE_zgemm_NN_bcast_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgemm_NN_bcast_READ_B_for_B_dep1_atline_68,
    &expr_of_p2_for_flow_of_zgemm_NN_bcast_READ_B_for_B_dep1_atline_68
  }
};
static inline int expr_of_p1_for_flow_of_zgemm_NN_bcast_READ_B_for_B_dep2_atline_71_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;


  (void)__dague_object; (void)assignments;
  return 0;
}
static const expr_t expr_of_p1_for_flow_of_zgemm_NN_bcast_READ_B_for_B_dep2_atline_71 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgemm_NN_bcast_READ_B_for_B_dep2_atline_71_fct
};
static inline int expr_of_p2_for_flow_of_zgemm_NN_bcast_READ_B_for_B_dep2_atline_71_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;
  int n = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return n;
}
static const expr_t expr_of_p2_for_flow_of_zgemm_NN_bcast_READ_B_for_B_dep2_atline_71 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgemm_NN_bcast_READ_B_for_B_dep2_atline_71_fct
};
static inline int rangemin_of_expr_of_p3_for_flow_of_zgemm_NN_bcast_READ_B_for_B_dep2_atline_71_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;


  (void)__dague_object; (void)assignments;
  return 0;
}
static const expr_t rangemin_of_expr_of_p3_for_flow_of_zgemm_NN_bcast_READ_B_for_B_dep2_atline_71 = {
  .op = EXPR_OP_INLINE,
  .inline_func = rangemin_of_expr_of_p3_for_flow_of_zgemm_NN_bcast_READ_B_for_B_dep2_atline_71_fct
};
static inline int rangemax_of_expr_of_p3_for_flow_of_zgemm_NN_bcast_READ_B_for_B_dep2_atline_71_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;


  (void)__dague_object; (void)assignments;
  return (descA.mt - 1);
}
static const expr_t rangemax_of_expr_of_p3_for_flow_of_zgemm_NN_bcast_READ_B_for_B_dep2_atline_71 = {
  .op = EXPR_OP_INLINE,
  .inline_func = rangemax_of_expr_of_p3_for_flow_of_zgemm_NN_bcast_READ_B_for_B_dep2_atline_71_fct
};
static const expr_t expr_of_p3_for_flow_of_zgemm_NN_bcast_READ_B_for_B_dep2_atline_71 = {
  .op = EXPR_OP_RANGE_CST_INCREMENT,
  .u_expr.range = {
    .op1 = &rangemin_of_expr_of_p3_for_flow_of_zgemm_NN_bcast_READ_B_for_B_dep2_atline_71,
    .op2 = &rangemax_of_expr_of_p3_for_flow_of_zgemm_NN_bcast_READ_B_for_B_dep2_atline_71,
    .increment.cst = 1
  }
};
static const dep_t flow_of_zgemm_NN_bcast_READ_B_for_B_dep2_atline_71 = {
  .cond = NULL,
  .dague = &zgemm_NN_bcast_GEMM,
  .flow = &flow_of_zgemm_NN_bcast_GEMM_for_B,
  .datatype = { .index = DAGUE_zgemm_NN_bcast_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgemm_NN_bcast_READ_B_for_B_dep2_atline_71,
    &expr_of_p2_for_flow_of_zgemm_NN_bcast_READ_B_for_B_dep2_atline_71,
    &expr_of_p3_for_flow_of_zgemm_NN_bcast_READ_B_for_B_dep2_atline_71
  }
};
static const dague_flow_t flow_of_zgemm_NN_bcast_READ_B_for_B = {
  .name = "B",
  .sym_type = SYM_INOUT,
  .access_type = ACCESS_RW,
  .flow_index = 0,
  .dep_in  = { &flow_of_zgemm_NN_bcast_READ_B_for_B_dep1_atline_68 },
  .dep_out = { &flow_of_zgemm_NN_bcast_READ_B_for_B_dep2_atline_71 }
};

static void
iterate_successors_of_zgemm_NN_bcast_READ_B(dague_execution_unit_t *eu, dague_execution_context_t *this_task,
               uint32_t action_mask, dague_ontask_function_t *ontask, void *ontask_arg)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)this_task->dague_object;
  dague_execution_context_t nc;
  dague_arena_t* arena = NULL;
  int __nb_elt = -1, vpid_dst = -1;
  int rank_src = 0, rank_dst = 0;
  int k = this_task->locals[0].value;
  int n = this_task->locals[1].value;
  (void)rank_src; (void)rank_dst; (void)__dague_object; (void)vpid_dst; (void)__nb_elt;
  (void)k;  (void)n;
  nc.dague_object = this_task->dague_object;
#if defined(DISTRIBUTED)
  rank_src = ((dague_ddesc_t*)__dague_object->super.B)->rank_of((dague_ddesc_t*)__dague_object->super.B, k, n);
#endif
  /* Flow of Data B */
  if( action_mask & (1 << 0) ) {
#if defined(DISTRIBUTED)
    arena = __dague_object->super.arenas[DAGUE_zgemm_NN_bcast_DEFAULT_ARENA];
    __nb_elt = 1;
#endif
    nc.function = (const dague_function_t*)&zgemm_NN_bcast_GEMM;
    {
      const int GEMM_m = 0;
        if( (GEMM_m >= (0)) && (GEMM_m <= ((descC.mt - 1))) ) {
          nc.locals[0].value = GEMM_m;
        {
          const int GEMM_n = n;
            if( (GEMM_n >= (0)) && (GEMM_n <= ((descC.nt - 1))) ) {
              nc.locals[1].value = GEMM_n;
            {
              int GEMM_k;
              for( GEMM_k = 0;GEMM_k <= (descA.mt - 1); GEMM_k+=1) {
                  if( (GEMM_k >= (0)) && (GEMM_k <= ((descA.nt - 1))) ) {
                    nc.locals[2].value = GEMM_k;
#if defined(DISTRIBUTED)
                    rank_dst = ((dague_ddesc_t*)__dague_object->super.C)->rank_of((dague_ddesc_t*)__dague_object->super.C, GEMM_m, GEMM_n);
                    if( eu != NULL && rank_dst == eu->virtual_process->dague_context->my_rank ) vpid_dst = ((dague_ddesc_t*)__dague_object->super.C)->vpid_of((dague_ddesc_t*)__dague_object->super.C, GEMM_m, GEMM_n);
#else /* !DISTRIBUTED */
                    vpid_dst = ((dague_ddesc_t*)__dague_object->super.C)->vpid_of((dague_ddesc_t*)__dague_object->super.C, GEMM_m, GEMM_n);
#endif /* DISTRIBUTED */
#if defined(DAGUE_DEBUG_VERBOSE1)
                  if( NULL != eu ) {
                    char tmp[128], tmp1[128];
                    DEBUG(("thread %d VP %d release deps of B:%s to B:%s (from node %d to %d)\n",
                           eu->th_id, eu->virtual_process->vp_id,
                           dague_snprintf_execution_context(tmp, 128, this_task),
                           dague_snprintf_execution_context(tmp1, 128, &nc), rank_src, rank_dst));
                  }
#endif
                    nc.flowname = "B";
                    nc.priority = __dague_object->super.super.object_priority;
                    if( DAGUE_ITERATE_STOP == ontask(eu, &nc, this_task, 0, 0, rank_src, rank_dst, vpid_dst, arena, __nb_elt, ontask_arg) )
                      return;
      }
        }
          }
            }
              }
                }
                  }
  }
  (void)nc;(void)arena;(void)eu;(void)ontask;(void)ontask_arg;(void)rank_dst;(void)action_mask;
}

static int release_deps_of_zgemm_NN_bcast_READ_B(dague_execution_unit_t *eu, dague_execution_context_t *context, uint32_t action_mask, dague_remote_deps_t *deps)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t *)context->dague_object;
  dague_release_dep_fct_arg_t arg;
  int __vp_id;
  arg.nb_released = 0;
  arg.output_usage = 0;
  arg.action_mask = action_mask;
  arg.deps = deps;
  arg.ready_lists = (eu != NULL) ? calloc(sizeof(dague_execution_context_t *), eu->virtual_process->dague_context->nb_vp) : NULL;
  (void)__dague_object;
  if( action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS ) {
    arg.output_entry = data_repo_lookup_entry_and_create( eu, READ_B_repo, READ_B_hash(__dague_object, context->locals) );
#if defined(DAGUE_SIM)
    assert(arg.output_entry->sim_exec_date == 0);
    arg.output_entry->sim_exec_date = context->sim_exec_date;
#endif
  }
#if defined(DISTRIBUTED)
  arg.remote_deps_count = 0;
  arg.remote_deps = NULL;
#endif
  iterate_successors_of_zgemm_NN_bcast_READ_B(eu, context, action_mask, dague_release_dep_fct, &arg);

  if(action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS) {
    struct dague_vp** vps = eu->virtual_process->dague_context->virtual_processes;
    data_repo_entry_addto_usage_limit(READ_B_repo, arg.output_entry->key, arg.output_usage);
    for(__vp_id = 0; __vp_id < eu->virtual_process->dague_context->nb_vp; __vp_id++) {
      if( NULL == arg.ready_lists[__vp_id] ) continue;
      __dague_schedule(vps[__vp_id]->execution_units[0], arg.ready_lists[__vp_id]);
      arg.ready_lists[__vp_id] = NULL;
    }
    free(arg.ready_lists);
  }
#if defined(DISTRIBUTED)
  if( 0 == arg.remote_deps_count ) {
    if( NULL != arg.remote_deps ) {
      remote_deps_free(arg.remote_deps);
      arg.remote_deps = NULL;
    }
  }
  else if( (action_mask & DAGUE_ACTION_SEND_REMOTE_DEPS) ) {
    arg.nb_released += dague_remote_dep_activate(eu, context, arg.remote_deps, arg.remote_deps_count);
  }
#endif
  if( action_mask & DAGUE_ACTION_RELEASE_LOCAL_REFS ) {
    int k = context->locals[0].value;
    int n = context->locals[1].value;
    (void)k; (void)n;

  }
  return arg.nb_released;
}

static int hook_of_zgemm_NN_bcast_READ_B(dague_execution_unit_t *context, dague_execution_context_t *this_task)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (__dague_zgemm_NN_bcast_internal_object_t *)this_task->dague_object;
  assignment_t tass[MAX_PARAM_COUNT];
  (void)context; (void)__dague_object; (void)tass;
#if defined(DAGUE_SIM)
  int __dague_simulation_date = 0;
#endif
  int k = this_task->locals[0].value;
  int n = this_task->locals[1].value;
  (void)k;  (void)n;

  /** Declare the variables that will hold the data, and all the accounting for each */
  void *B = NULL; (void)B;
  dague_arena_chunk_t *gB = NULL; (void)gB;
  data_repo_entry_t *eB = NULL; (void)eB;

  /** Lookup the input data, and store them in the context if any */
  eB = this_task->data[0].data_repo;
  gB = this_task->data[0].data;
  if( NULL == gB ) {
    gB = (dague_arena_chunk_t*) B(k, n);
    this_task->data[0].data = gB;
    this_task->data[0].data_repo = eB;
  }
  B = ADATA(gB);
#if defined(DAGUE_SIM)
  if( (NULL != eB) && (eB->sim_exec_date > __dague_simulation_date) )
    __dague_simulation_date =  eB->sim_exec_date;
#endif
#if defined(DAGUE_SIM)
  if( this_task->function->sim_cost_fct != NULL ) {
    this_task->sim_exec_date = __dague_simulation_date + this_task->function->sim_cost_fct(this_task);
  } else {
    this_task->sim_exec_date = __dague_simulation_date;
  }
  if( context->largest_simulation_date < this_task->sim_exec_date )
    context->largest_simulation_date = this_task->sim_exec_date;
#endif
  /** PAPI events */
#if defined(HAVE_PAPI)
  papime_start_thread_counters();
#endif
  /** Cache Awareness Accounting */
#if defined(DAGUE_CACHE_AWARENESS)
  cache_buf_referenced(context->closest_cache, B);
#endif /* DAGUE_CACHE_AWARENESS */


#if !defined(DAGUE_PROF_DRY_BODY)

/*--------------------------------------------------------------------------------*
 *                                  READ_B BODY                                  *
 *--------------------------------------------------------------------------------*/

#line 69 "zgemm_NN_bcast.jdf"
     printlog("rank %u <- B(%d,%d)\n", __dague_object->super.B->myrank, k, n);

#line 1828 "zgemm_NN_bcast.c"
/*--------------------------------------------------------------------------------*
 *                              END OF READ_B BODY                              *
 *--------------------------------------------------------------------------------*/



#endif /*!defined(DAGUE_PROF_DRY_BODY)*/

  return 0;
}
static int complete_hook_of_zgemm_NN_bcast_READ_B(dague_execution_unit_t *context, dague_execution_context_t *this_task)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (__dague_zgemm_NN_bcast_internal_object_t *)this_task->dague_object;
  (void)context; (void)__dague_object;
  int k = this_task->locals[0].value;
  int n = this_task->locals[1].value;
  (void)k;  (void)n;

  /** PAPI events */
#if defined(HAVE_PAPI)
  papime_stop_thread_counters();
#endif
#if defined(DISTRIBUTED)
  /** If not working on distributed, there is no risk that data is not in place */
#endif /* DISTRIBUTED */
  dague_prof_grapher_task(this_task, context->th_id, context->virtual_process->vp_id, READ_B_hash(__dague_object, this_task->locals));
  release_deps_of_zgemm_NN_bcast_READ_B(context, this_task,
      DAGUE_ACTION_RELEASE_REMOTE_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_REFS |
      DAGUE_ACTION_DEPS_MASK,
      NULL);
  return 0;
}

static int zgemm_NN_bcast_READ_B_internal_init(__dague_zgemm_NN_bcast_internal_object_t *__dague_object)
{
  dague_dependencies_t *dep = NULL;
  assignment_t assignments[MAX_LOCAL_COUNT];(void) assignments;
  int nb_tasks = 0;
  int32_t  k, n;
  int32_t  k_min = 0x7fffffff, n_min = 0x7fffffff;
  int32_t  k_max = 0, n_max = 0;
  (void)__dague_object;
  int32_t k_start, k_end, k_inc;
  int32_t n_start, n_end, n_inc;
  /* First, find the min and max value for each of the dimensions */
  for(k = 0;
      k <= (descB.mt - 1);
      k += 1) {
    assignments[0].value = k;
    for(n = 0;
        n <= (descB.nt - 1);
        n += 1) {
      assignments[1].value = n;
      if( !READ_B_pred(k, n) ) continue;
      nb_tasks++;
      k_max = dague_imax(k_max, k);
      k_min = dague_imin(k_min, k);
      n_max = dague_imax(n_max, n);
      n_min = dague_imin(n_min, n);
    }
  }

  /**
   * Now, for each of the dimensions, re-iterate on the space,
   * and if at least one value is defined, allocate arrays to point
   * to it. Array dimensions are defined by the (rough) observation above
   **/
  DEBUG2(("Allocating dependencies array for zgemm_NN_bcast_READ_B_internal_init\n"));
  dep = NULL;
  k_start = 0;
  k_end = (descB.mt - 1);
  k_inc = 1;
  for(k = dague_imax(k_start, k_min); k <= dague_imin(k_end, k_max); k+=k_inc) {
    assignments[0].value = k;
    n_start = 0;
    n_end = (descB.nt - 1);
    n_inc = 1;
    for(n = dague_imax(n_start, n_min); n <= dague_imin(n_end, n_max); n+=n_inc) {
      assignments[1].value = n;
      if( READ_B_pred(k, n) ) {
        /* We did find one! Allocate the dependencies array. */
      if( dep == NULL ) {
        ALLOCATE_DEP_TRACKING(dep, k_min, k_max, "k", &symb_zgemm_NN_bcast_READ_B_k, NULL, DAGUE_DEPENDENCIES_FLAG_NEXT);
      }
      if( dep->u.next[k-k_min] == NULL ) {
        ALLOCATE_DEP_TRACKING(dep->u.next[k-k_min], n_min, n_max, "n", &symb_zgemm_NN_bcast_READ_B_n, dep, DAGUE_DEPENDENCIES_FLAG_FINAL);
      }
        break;
      }
    }
  }
  (void)k_start; (void)k_end; (void)k_inc;  (void)n_start; (void)n_end; (void)n_inc;
  __dague_object->super.super.dependencies_array[1] = dep;
  __dague_object->super.super.nb_local_tasks += nb_tasks;
  return nb_tasks;
}

static int zgemm_NN_bcast_READ_B_startup_tasks(dague_context_t *context, const __dague_zgemm_NN_bcast_internal_object_t *__dague_object, dague_execution_context_t** pready_list)
{
  dague_execution_context_t* new_context, new_context_holder, *new_dynamic_context;
  assignment_t *assignments = NULL;
  int vpid;
  int32_t  k = -1, n = -1;
  (void)k; (void)n;
  new_context = &new_context_holder;
  assignments = new_context->locals;
  new_context->dague_object = (dague_object_t*)__dague_object;
  new_context->function = (const dague_function_t*)&zgemm_NN_bcast_READ_B;
  /* Parse all the inputs and generate the ready execution tasks */
  for(k = 0;
      k <= (descB.mt - 1);
      k+=1) {
    assignments[0].value = k;
    for(n = 0;
        n <= (descB.nt - 1);
        n+=1) {
      assignments[1].value = n;
      if( !READ_B_pred(k, n) ) continue;
      vpid = ((dague_ddesc_t*)__dague_object->super.B)->vpid_of((dague_ddesc_t*)__dague_object->super.B, k, n);
      new_dynamic_context = (dague_execution_context_t*)dague_thread_mempool_allocate( context->virtual_processes[vpid]->execution_units[0]->context_mempool );
      /* Copy only the valid elements from new_context to new_dynamic one */
      new_dynamic_context->dague_object = new_context->dague_object;
      new_dynamic_context->function     = new_context->function;
      memcpy(new_dynamic_context->locals, new_context->locals, 2*sizeof(assignment_t));
      DAGUE_STAT_INCREASE(mem_contexts, sizeof(dague_execution_context_t) + STAT_MALLOC_OVERHEAD);
      DAGUE_LIST_ITEM_SINGLETON( new_dynamic_context );
      new_dynamic_context->priority = __dague_object->super.super.object_priority;
      new_dynamic_context->flowname = "B";
    new_dynamic_context->data[0].data_repo = NULL;
    new_dynamic_context->data[0].data      = NULL;
#if defined(DAGUE_DEBUG_VERBOSE2)
      {
        char tmp[128];
        DEBUG2(("Add startup task %s\n",
               dague_snprintf_execution_context(tmp, 128, new_dynamic_context)));
      }
#endif
      pready_list[vpid] = (dague_execution_context_t*)dague_list_item_ring_push_sorted( (dague_list_item_t*)(pready_list[vpid]), (dague_list_item_t*)new_dynamic_context, dague_execution_context_priority_comparator );
    }
  }
  return 0;
}

static const dague_function_t zgemm_NN_bcast_READ_B = {
  .name = "READ_B",
  .deps = 1,
  .flags = 0x0 | DAGUE_HAS_IN_IN_DEPENDENCIES,
  .function_id = 1,
  .dependencies_goal = 0x1,
  .nb_parameters = 2,
  .nb_definitions = 2,
  .params = { &symb_zgemm_NN_bcast_READ_B_k, &symb_zgemm_NN_bcast_READ_B_n },
  .locals = { &symb_zgemm_NN_bcast_READ_B_k, &symb_zgemm_NN_bcast_READ_B_n },
  .pred = &pred_of_zgemm_NN_bcast_READ_B_as_expr,
  .priority = NULL,
  .in = { &flow_of_zgemm_NN_bcast_READ_B_for_B },
  .out = { &flow_of_zgemm_NN_bcast_READ_B_for_B },
  .iterate_successors = iterate_successors_of_zgemm_NN_bcast_READ_B,
  .release_deps = release_deps_of_zgemm_NN_bcast_READ_B,
  .hook = hook_of_zgemm_NN_bcast_READ_B,
  .complete_execution = complete_hook_of_zgemm_NN_bcast_READ_B,
#if defined(DAGUE_SIM)
  .sim_cost_fct = NULL,
#endif
  .key = (dague_functionkey_fn_t*)READ_B_hash,
};


/**********************************************************************************
 *                                    READ_A                                    *
 **********************************************************************************/

static inline int minexpr_of_symb_zgemm_NN_bcast_READ_A_m_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;


  (void)__dague_object; (void)assignments;
  return 0;
}
static const expr_t minexpr_of_symb_zgemm_NN_bcast_READ_A_m = {
  .op = EXPR_OP_INLINE,
  .inline_func = minexpr_of_symb_zgemm_NN_bcast_READ_A_m_fct
};
static inline int maxexpr_of_symb_zgemm_NN_bcast_READ_A_m_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;


  (void)__dague_object; (void)assignments;
  return (descA.mt - 1);
}
static const expr_t maxexpr_of_symb_zgemm_NN_bcast_READ_A_m = {
  .op = EXPR_OP_INLINE,
  .inline_func = maxexpr_of_symb_zgemm_NN_bcast_READ_A_m_fct
};
static const symbol_t symb_zgemm_NN_bcast_READ_A_m = { .name = "m", .context_index = 0, .min = &minexpr_of_symb_zgemm_NN_bcast_READ_A_m, .max = &maxexpr_of_symb_zgemm_NN_bcast_READ_A_m, .cst_inc = 1, .expr_inc = NULL,  .flags = 0x0};

static inline int minexpr_of_symb_zgemm_NN_bcast_READ_A_k_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;


  (void)__dague_object; (void)assignments;
  return 0;
}
static const expr_t minexpr_of_symb_zgemm_NN_bcast_READ_A_k = {
  .op = EXPR_OP_INLINE,
  .inline_func = minexpr_of_symb_zgemm_NN_bcast_READ_A_k_fct
};
static inline int maxexpr_of_symb_zgemm_NN_bcast_READ_A_k_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;


  (void)__dague_object; (void)assignments;
  return (descA.nt - 1);
}
static const expr_t maxexpr_of_symb_zgemm_NN_bcast_READ_A_k = {
  .op = EXPR_OP_INLINE,
  .inline_func = maxexpr_of_symb_zgemm_NN_bcast_READ_A_k_fct
};
static const symbol_t symb_zgemm_NN_bcast_READ_A_k = { .name = "k", .context_index = 1, .min = &minexpr_of_symb_zgemm_NN_bcast_READ_A_k, .max = &maxexpr_of_symb_zgemm_NN_bcast_READ_A_k, .cst_inc = 1, .expr_inc = NULL,  .flags = 0x0};

static inline int pred_of_zgemm_NN_bcast_READ_A_as_expr_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;
  int m = assignments[0].value;
  int k = assignments[1].value;

  /* Silent Warnings: should look into predicate to know what variables are usefull */
  (void)__dague_object;
  (void)m;
  (void)k;
  /* Compute Predicate */
  return READ_A_pred(m, k);
}
static const expr_t pred_of_zgemm_NN_bcast_READ_A_as_expr = {
  .op = EXPR_OP_INLINE,
  .inline_func = pred_of_zgemm_NN_bcast_READ_A_as_expr_fct
};
static inline int expr_of_p1_for_flow_of_zgemm_NN_bcast_READ_A_for_A_dep1_atline_52_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;
  int m = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return m;
}
static const expr_t expr_of_p1_for_flow_of_zgemm_NN_bcast_READ_A_for_A_dep1_atline_52 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgemm_NN_bcast_READ_A_for_A_dep1_atline_52_fct
};
static inline int expr_of_p2_for_flow_of_zgemm_NN_bcast_READ_A_for_A_dep1_atline_52_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;
  int k = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p2_for_flow_of_zgemm_NN_bcast_READ_A_for_A_dep1_atline_52 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgemm_NN_bcast_READ_A_for_A_dep1_atline_52_fct
};
static const dep_t flow_of_zgemm_NN_bcast_READ_A_for_A_dep1_atline_52 = {
  .cond = NULL,
  .dague = &zgemm_NN_bcast_A,
  .datatype = { .index = DAGUE_zgemm_NN_bcast_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgemm_NN_bcast_READ_A_for_A_dep1_atline_52,
    &expr_of_p2_for_flow_of_zgemm_NN_bcast_READ_A_for_A_dep1_atline_52
  }
};
static inline int rangemin_of_expr_of_p1_for_flow_of_zgemm_NN_bcast_READ_A_for_A_dep2_atline_55_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;


  (void)__dague_object; (void)assignments;
  return 0;
}
static const expr_t rangemin_of_expr_of_p1_for_flow_of_zgemm_NN_bcast_READ_A_for_A_dep2_atline_55 = {
  .op = EXPR_OP_INLINE,
  .inline_func = rangemin_of_expr_of_p1_for_flow_of_zgemm_NN_bcast_READ_A_for_A_dep2_atline_55_fct
};
static inline int rangemax_of_expr_of_p1_for_flow_of_zgemm_NN_bcast_READ_A_for_A_dep2_atline_55_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;


  (void)__dague_object; (void)assignments;
  return (descA.mt - 1);
}
static const expr_t rangemax_of_expr_of_p1_for_flow_of_zgemm_NN_bcast_READ_A_for_A_dep2_atline_55 = {
  .op = EXPR_OP_INLINE,
  .inline_func = rangemax_of_expr_of_p1_for_flow_of_zgemm_NN_bcast_READ_A_for_A_dep2_atline_55_fct
};
static const expr_t expr_of_p1_for_flow_of_zgemm_NN_bcast_READ_A_for_A_dep2_atline_55 = {
  .op = EXPR_OP_RANGE_CST_INCREMENT,
  .u_expr.range = {
    .op1 = &rangemin_of_expr_of_p1_for_flow_of_zgemm_NN_bcast_READ_A_for_A_dep2_atline_55,
    .op2 = &rangemax_of_expr_of_p1_for_flow_of_zgemm_NN_bcast_READ_A_for_A_dep2_atline_55,
    .increment.cst = 1
  }
};
static inline int expr_of_p2_for_flow_of_zgemm_NN_bcast_READ_A_for_A_dep2_atline_55_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;


  (void)__dague_object; (void)assignments;
  return 0;
}
static const expr_t expr_of_p2_for_flow_of_zgemm_NN_bcast_READ_A_for_A_dep2_atline_55 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgemm_NN_bcast_READ_A_for_A_dep2_atline_55_fct
};
static inline int expr_of_p3_for_flow_of_zgemm_NN_bcast_READ_A_for_A_dep2_atline_55_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)__dague_object_parent;
  int k = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p3_for_flow_of_zgemm_NN_bcast_READ_A_for_A_dep2_atline_55 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p3_for_flow_of_zgemm_NN_bcast_READ_A_for_A_dep2_atline_55_fct
};
static const dep_t flow_of_zgemm_NN_bcast_READ_A_for_A_dep2_atline_55 = {
  .cond = NULL,
  .dague = &zgemm_NN_bcast_GEMM,
  .flow = &flow_of_zgemm_NN_bcast_GEMM_for_A,
  .datatype = { .index = DAGUE_zgemm_NN_bcast_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgemm_NN_bcast_READ_A_for_A_dep2_atline_55,
    &expr_of_p2_for_flow_of_zgemm_NN_bcast_READ_A_for_A_dep2_atline_55,
    &expr_of_p3_for_flow_of_zgemm_NN_bcast_READ_A_for_A_dep2_atline_55
  }
};
static const dague_flow_t flow_of_zgemm_NN_bcast_READ_A_for_A = {
  .name = "A",
  .sym_type = SYM_INOUT,
  .access_type = ACCESS_RW,
  .flow_index = 0,
  .dep_in  = { &flow_of_zgemm_NN_bcast_READ_A_for_A_dep1_atline_52 },
  .dep_out = { &flow_of_zgemm_NN_bcast_READ_A_for_A_dep2_atline_55 }
};

static void
iterate_successors_of_zgemm_NN_bcast_READ_A(dague_execution_unit_t *eu, dague_execution_context_t *this_task,
               uint32_t action_mask, dague_ontask_function_t *ontask, void *ontask_arg)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t*)this_task->dague_object;
  dague_execution_context_t nc;
  dague_arena_t* arena = NULL;
  int __nb_elt = -1, vpid_dst = -1;
  int rank_src = 0, rank_dst = 0;
  int m = this_task->locals[0].value;
  int k = this_task->locals[1].value;
  (void)rank_src; (void)rank_dst; (void)__dague_object; (void)vpid_dst; (void)__nb_elt;
  (void)m;  (void)k;
  nc.dague_object = this_task->dague_object;
#if defined(DISTRIBUTED)
  rank_src = ((dague_ddesc_t*)__dague_object->super.A)->rank_of((dague_ddesc_t*)__dague_object->super.A, m, k);
#endif
  /* Flow of Data A */
  if( action_mask & (1 << 0) ) {
#if defined(DISTRIBUTED)
    arena = __dague_object->super.arenas[DAGUE_zgemm_NN_bcast_DEFAULT_ARENA];
    __nb_elt = 1;
#endif
    nc.function = (const dague_function_t*)&zgemm_NN_bcast_GEMM;
    {
      int GEMM_m;
      for( GEMM_m = 0;GEMM_m <= (descA.mt - 1); GEMM_m+=1) {
          if( (GEMM_m >= (0)) && (GEMM_m <= ((descC.mt - 1))) ) {
            nc.locals[0].value = GEMM_m;
          {
            const int GEMM_n = 0;
              if( (GEMM_n >= (0)) && (GEMM_n <= ((descC.nt - 1))) ) {
                nc.locals[1].value = GEMM_n;
              {
                const int GEMM_k = k;
                  if( (GEMM_k >= (0)) && (GEMM_k <= ((descA.nt - 1))) ) {
                    nc.locals[2].value = GEMM_k;
#if defined(DISTRIBUTED)
                    rank_dst = ((dague_ddesc_t*)__dague_object->super.C)->rank_of((dague_ddesc_t*)__dague_object->super.C, GEMM_m, GEMM_n);
                    if( eu != NULL && rank_dst == eu->virtual_process->dague_context->my_rank ) vpid_dst = ((dague_ddesc_t*)__dague_object->super.C)->vpid_of((dague_ddesc_t*)__dague_object->super.C, GEMM_m, GEMM_n);
#else /* !DISTRIBUTED */
                    vpid_dst = ((dague_ddesc_t*)__dague_object->super.C)->vpid_of((dague_ddesc_t*)__dague_object->super.C, GEMM_m, GEMM_n);
#endif /* DISTRIBUTED */
#if defined(DAGUE_DEBUG_VERBOSE1)
                  if( NULL != eu ) {
                    char tmp[128], tmp1[128];
                    DEBUG(("thread %d VP %d release deps of A:%s to A:%s (from node %d to %d)\n",
                           eu->th_id, eu->virtual_process->vp_id,
                           dague_snprintf_execution_context(tmp, 128, this_task),
                           dague_snprintf_execution_context(tmp1, 128, &nc), rank_src, rank_dst));
                  }
#endif
                    nc.flowname = "A";
                    nc.priority = __dague_object->super.super.object_priority;
                    if( DAGUE_ITERATE_STOP == ontask(eu, &nc, this_task, 0, 0, rank_src, rank_dst, vpid_dst, arena, __nb_elt, ontask_arg) )
                      return;
      }
        }
          }
            }
              }
                }
                  }
  }
  (void)nc;(void)arena;(void)eu;(void)ontask;(void)ontask_arg;(void)rank_dst;(void)action_mask;
}

static int release_deps_of_zgemm_NN_bcast_READ_A(dague_execution_unit_t *eu, dague_execution_context_t *context, uint32_t action_mask, dague_remote_deps_t *deps)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (const __dague_zgemm_NN_bcast_internal_object_t *)context->dague_object;
  dague_release_dep_fct_arg_t arg;
  int __vp_id;
  arg.nb_released = 0;
  arg.output_usage = 0;
  arg.action_mask = action_mask;
  arg.deps = deps;
  arg.ready_lists = (eu != NULL) ? calloc(sizeof(dague_execution_context_t *), eu->virtual_process->dague_context->nb_vp) : NULL;
  (void)__dague_object;
  if( action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS ) {
    arg.output_entry = data_repo_lookup_entry_and_create( eu, READ_A_repo, READ_A_hash(__dague_object, context->locals) );
#if defined(DAGUE_SIM)
    assert(arg.output_entry->sim_exec_date == 0);
    arg.output_entry->sim_exec_date = context->sim_exec_date;
#endif
  }
#if defined(DISTRIBUTED)
  arg.remote_deps_count = 0;
  arg.remote_deps = NULL;
#endif
  iterate_successors_of_zgemm_NN_bcast_READ_A(eu, context, action_mask, dague_release_dep_fct, &arg);

  if(action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS) {
    struct dague_vp** vps = eu->virtual_process->dague_context->virtual_processes;
    data_repo_entry_addto_usage_limit(READ_A_repo, arg.output_entry->key, arg.output_usage);
    for(__vp_id = 0; __vp_id < eu->virtual_process->dague_context->nb_vp; __vp_id++) {
      if( NULL == arg.ready_lists[__vp_id] ) continue;
      __dague_schedule(vps[__vp_id]->execution_units[0], arg.ready_lists[__vp_id]);
      arg.ready_lists[__vp_id] = NULL;
    }
    free(arg.ready_lists);
  }
#if defined(DISTRIBUTED)
  if( 0 == arg.remote_deps_count ) {
    if( NULL != arg.remote_deps ) {
      remote_deps_free(arg.remote_deps);
      arg.remote_deps = NULL;
    }
  }
  else if( (action_mask & DAGUE_ACTION_SEND_REMOTE_DEPS) ) {
    arg.nb_released += dague_remote_dep_activate(eu, context, arg.remote_deps, arg.remote_deps_count);
  }
#endif
  if( action_mask & DAGUE_ACTION_RELEASE_LOCAL_REFS ) {
    int m = context->locals[0].value;
    int k = context->locals[1].value;
    (void)m; (void)k;

  }
  return arg.nb_released;
}

static int hook_of_zgemm_NN_bcast_READ_A(dague_execution_unit_t *context, dague_execution_context_t *this_task)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (__dague_zgemm_NN_bcast_internal_object_t *)this_task->dague_object;
  assignment_t tass[MAX_PARAM_COUNT];
  (void)context; (void)__dague_object; (void)tass;
#if defined(DAGUE_SIM)
  int __dague_simulation_date = 0;
#endif
  int m = this_task->locals[0].value;
  int k = this_task->locals[1].value;
  (void)m;  (void)k;

  /** Declare the variables that will hold the data, and all the accounting for each */
  void *A = NULL; (void)A;
  dague_arena_chunk_t *gA = NULL; (void)gA;
  data_repo_entry_t *eA = NULL; (void)eA;

  /** Lookup the input data, and store them in the context if any */
  eA = this_task->data[0].data_repo;
  gA = this_task->data[0].data;
  if( NULL == gA ) {
    gA = (dague_arena_chunk_t*) A(m, k);
    this_task->data[0].data = gA;
    this_task->data[0].data_repo = eA;
  }
  A = ADATA(gA);
#if defined(DAGUE_SIM)
  if( (NULL != eA) && (eA->sim_exec_date > __dague_simulation_date) )
    __dague_simulation_date =  eA->sim_exec_date;
#endif
#if defined(DAGUE_SIM)
  if( this_task->function->sim_cost_fct != NULL ) {
    this_task->sim_exec_date = __dague_simulation_date + this_task->function->sim_cost_fct(this_task);
  } else {
    this_task->sim_exec_date = __dague_simulation_date;
  }
  if( context->largest_simulation_date < this_task->sim_exec_date )
    context->largest_simulation_date = this_task->sim_exec_date;
#endif
  /** PAPI events */
#if defined(HAVE_PAPI)
  papime_start_thread_counters();
#endif
  /** Cache Awareness Accounting */
#if defined(DAGUE_CACHE_AWARENESS)
  cache_buf_referenced(context->closest_cache, A);
#endif /* DAGUE_CACHE_AWARENESS */


#if !defined(DAGUE_PROF_DRY_BODY)

/*--------------------------------------------------------------------------------*
 *                                  READ_A BODY                                  *
 *--------------------------------------------------------------------------------*/

#line 53 "zgemm_NN_bcast.jdf"
    printlog("rank %u <- A(%d,%d)\n", __dague_object->super.A->myrank, m, k);

#line 2360 "zgemm_NN_bcast.c"
/*--------------------------------------------------------------------------------*
 *                              END OF READ_A BODY                              *
 *--------------------------------------------------------------------------------*/



#endif /*!defined(DAGUE_PROF_DRY_BODY)*/

  return 0;
}
static int complete_hook_of_zgemm_NN_bcast_READ_A(dague_execution_unit_t *context, dague_execution_context_t *this_task)
{
  const __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (__dague_zgemm_NN_bcast_internal_object_t *)this_task->dague_object;
  (void)context; (void)__dague_object;
  int m = this_task->locals[0].value;
  int k = this_task->locals[1].value;
  (void)m;  (void)k;

  /** PAPI events */
#if defined(HAVE_PAPI)
  papime_stop_thread_counters();
#endif
#if defined(DISTRIBUTED)
  /** If not working on distributed, there is no risk that data is not in place */
#endif /* DISTRIBUTED */
  dague_prof_grapher_task(this_task, context->th_id, context->virtual_process->vp_id, READ_A_hash(__dague_object, this_task->locals));
  release_deps_of_zgemm_NN_bcast_READ_A(context, this_task,
      DAGUE_ACTION_RELEASE_REMOTE_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_REFS |
      DAGUE_ACTION_DEPS_MASK,
      NULL);
  return 0;
}

static int zgemm_NN_bcast_READ_A_internal_init(__dague_zgemm_NN_bcast_internal_object_t *__dague_object)
{
  dague_dependencies_t *dep = NULL;
  assignment_t assignments[MAX_LOCAL_COUNT];(void) assignments;
  int nb_tasks = 0;
  int32_t  m, k;
  int32_t  m_min = 0x7fffffff, k_min = 0x7fffffff;
  int32_t  m_max = 0, k_max = 0;
  (void)__dague_object;
  int32_t m_start, m_end, m_inc;
  int32_t k_start, k_end, k_inc;
  /* First, find the min and max value for each of the dimensions */
  for(m = 0;
      m <= (descA.mt - 1);
      m += 1) {
    assignments[0].value = m;
    for(k = 0;
        k <= (descA.nt - 1);
        k += 1) {
      assignments[1].value = k;
      if( !READ_A_pred(m, k) ) continue;
      nb_tasks++;
      m_max = dague_imax(m_max, m);
      m_min = dague_imin(m_min, m);
      k_max = dague_imax(k_max, k);
      k_min = dague_imin(k_min, k);
    }
  }

  /**
   * Now, for each of the dimensions, re-iterate on the space,
   * and if at least one value is defined, allocate arrays to point
   * to it. Array dimensions are defined by the (rough) observation above
   **/
  DEBUG2(("Allocating dependencies array for zgemm_NN_bcast_READ_A_internal_init\n"));
  dep = NULL;
  m_start = 0;
  m_end = (descA.mt - 1);
  m_inc = 1;
  for(m = dague_imax(m_start, m_min); m <= dague_imin(m_end, m_max); m+=m_inc) {
    assignments[0].value = m;
    k_start = 0;
    k_end = (descA.nt - 1);
    k_inc = 1;
    for(k = dague_imax(k_start, k_min); k <= dague_imin(k_end, k_max); k+=k_inc) {
      assignments[1].value = k;
      if( READ_A_pred(m, k) ) {
        /* We did find one! Allocate the dependencies array. */
      if( dep == NULL ) {
        ALLOCATE_DEP_TRACKING(dep, m_min, m_max, "m", &symb_zgemm_NN_bcast_READ_A_m, NULL, DAGUE_DEPENDENCIES_FLAG_NEXT);
      }
      if( dep->u.next[m-m_min] == NULL ) {
        ALLOCATE_DEP_TRACKING(dep->u.next[m-m_min], k_min, k_max, "k", &symb_zgemm_NN_bcast_READ_A_k, dep, DAGUE_DEPENDENCIES_FLAG_FINAL);
      }
        break;
      }
    }
  }
  (void)m_start; (void)m_end; (void)m_inc;  (void)k_start; (void)k_end; (void)k_inc;
  __dague_object->super.super.dependencies_array[2] = dep;
  __dague_object->super.super.nb_local_tasks += nb_tasks;
  return nb_tasks;
}

static int zgemm_NN_bcast_READ_A_startup_tasks(dague_context_t *context, const __dague_zgemm_NN_bcast_internal_object_t *__dague_object, dague_execution_context_t** pready_list)
{
  dague_execution_context_t* new_context, new_context_holder, *new_dynamic_context;
  assignment_t *assignments = NULL;
  int vpid;
  int32_t  m = -1, k = -1;
  (void)m; (void)k;
  new_context = &new_context_holder;
  assignments = new_context->locals;
  new_context->dague_object = (dague_object_t*)__dague_object;
  new_context->function = (const dague_function_t*)&zgemm_NN_bcast_READ_A;
  /* Parse all the inputs and generate the ready execution tasks */
  for(m = 0;
      m <= (descA.mt - 1);
      m+=1) {
    assignments[0].value = m;
    for(k = 0;
        k <= (descA.nt - 1);
        k+=1) {
      assignments[1].value = k;
      if( !READ_A_pred(m, k) ) continue;
      vpid = ((dague_ddesc_t*)__dague_object->super.A)->vpid_of((dague_ddesc_t*)__dague_object->super.A, m, k);
      new_dynamic_context = (dague_execution_context_t*)dague_thread_mempool_allocate( context->virtual_processes[vpid]->execution_units[0]->context_mempool );
      /* Copy only the valid elements from new_context to new_dynamic one */
      new_dynamic_context->dague_object = new_context->dague_object;
      new_dynamic_context->function     = new_context->function;
      memcpy(new_dynamic_context->locals, new_context->locals, 2*sizeof(assignment_t));
      DAGUE_STAT_INCREASE(mem_contexts, sizeof(dague_execution_context_t) + STAT_MALLOC_OVERHEAD);
      DAGUE_LIST_ITEM_SINGLETON( new_dynamic_context );
      new_dynamic_context->priority = __dague_object->super.super.object_priority;
      new_dynamic_context->flowname = "A";
    new_dynamic_context->data[0].data_repo = NULL;
    new_dynamic_context->data[0].data      = NULL;
#if defined(DAGUE_DEBUG_VERBOSE2)
      {
        char tmp[128];
        DEBUG2(("Add startup task %s\n",
               dague_snprintf_execution_context(tmp, 128, new_dynamic_context)));
      }
#endif
      pready_list[vpid] = (dague_execution_context_t*)dague_list_item_ring_push_sorted( (dague_list_item_t*)(pready_list[vpid]), (dague_list_item_t*)new_dynamic_context, dague_execution_context_priority_comparator );
    }
  }
  return 0;
}

static const dague_function_t zgemm_NN_bcast_READ_A = {
  .name = "READ_A",
  .deps = 2,
  .flags = 0x0 | DAGUE_HAS_IN_IN_DEPENDENCIES,
  .function_id = 2,
  .dependencies_goal = 0x1,
  .nb_parameters = 2,
  .nb_definitions = 2,
  .params = { &symb_zgemm_NN_bcast_READ_A_m, &symb_zgemm_NN_bcast_READ_A_k },
  .locals = { &symb_zgemm_NN_bcast_READ_A_m, &symb_zgemm_NN_bcast_READ_A_k },
  .pred = &pred_of_zgemm_NN_bcast_READ_A_as_expr,
  .priority = NULL,
  .in = { &flow_of_zgemm_NN_bcast_READ_A_for_A },
  .out = { &flow_of_zgemm_NN_bcast_READ_A_for_A },
  .iterate_successors = iterate_successors_of_zgemm_NN_bcast_READ_A,
  .release_deps = release_deps_of_zgemm_NN_bcast_READ_A,
  .hook = hook_of_zgemm_NN_bcast_READ_A,
  .complete_execution = complete_hook_of_zgemm_NN_bcast_READ_A,
#if defined(DAGUE_SIM)
  .sim_cost_fct = NULL,
#endif
  .key = (dague_functionkey_fn_t*)READ_A_hash,
};


static const dague_function_t *zgemm_NN_bcast_functions[] = {
  &zgemm_NN_bcast_GEMM,
  &zgemm_NN_bcast_READ_B,
  &zgemm_NN_bcast_READ_A
};

static void zgemm_NN_bcast_startup(dague_context_t *context, dague_object_t *dague_object, dague_execution_context_t** pready_list)
{
  zgemm_NN_bcast_READ_B_startup_tasks(context, (__dague_zgemm_NN_bcast_internal_object_t*)dague_object, pready_list);
  zgemm_NN_bcast_READ_A_startup_tasks(context, (__dague_zgemm_NN_bcast_internal_object_t*)dague_object, pready_list);
}
static void zgemm_NN_bcast_destructor( dague_zgemm_NN_bcast_object_t *o )
{
  dague_object_t *d = (dague_object_t *)o;
  __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (__dague_zgemm_NN_bcast_internal_object_t*)o; (void)__dague_object;
  int i;
  free(d->functions_array);
  d->functions_array = NULL;
  d->nb_functions = 0;
  for(i =0; i < o->arenas_size; i++) {
    if( o->arenas[i] != NULL ) {
      dague_arena_destruct(o->arenas[i]);
      free(o->arenas[i]);
      o->arenas[i] = NULL;
    }
  }
  free( o->arenas );
  o->arenas = NULL;
  o->arenas_size = 0;
  /* Destroy the data repositories for this object */
   data_repo_destroy_nothreadsafe(__dague_object->GEMM_repository);
   data_repo_destroy_nothreadsafe(__dague_object->READ_B_repository);
   data_repo_destroy_nothreadsafe(__dague_object->READ_A_repository);
  for(i = 0; i < DAGUE_zgemm_NN_bcast_NB_FUNCTIONS; i++) {
    dague_destruct_dependencies( d->dependencies_array[i] );
    d->dependencies_array[i] = NULL;
  }
  free( d->dependencies_array );
  d->dependencies_array = NULL;
  dague_object_unregister( d );
  free(o);
}

#undef transA
#undef transB
#undef alpha
#undef beta
#undef descA
#undef A
#undef descB
#undef B
#undef descC
#undef C

dague_zgemm_NN_bcast_object_t *dague_zgemm_NN_bcast_new(int transA, int transB, Dague_Complex64_t alpha, Dague_Complex64_t beta, tiled_matrix_desc_t descA, dague_ddesc_t * A /* data A */, tiled_matrix_desc_t descB, dague_ddesc_t * B /* data B */, tiled_matrix_desc_t descC, dague_ddesc_t * C /* data C */)
{
  __dague_zgemm_NN_bcast_internal_object_t *__dague_object = (__dague_zgemm_NN_bcast_internal_object_t *)calloc(1, sizeof(__dague_zgemm_NN_bcast_internal_object_t));
  int i;
  int GEMM_nblocal_tasks;
  int READ_B_nblocal_tasks;
  int READ_A_nblocal_tasks;

  __dague_object->super.super.nb_functions    = DAGUE_zgemm_NN_bcast_NB_FUNCTIONS;
  __dague_object->super.super.functions_array = (const dague_function_t**)malloc(DAGUE_zgemm_NN_bcast_NB_FUNCTIONS * sizeof(dague_function_t*));
  __dague_object->super.super.dependencies_array = (dague_dependencies_t **)
              calloc(DAGUE_zgemm_NN_bcast_NB_FUNCTIONS, sizeof(dague_dependencies_t *));
  memcpy(__dague_object->super.super.functions_array, zgemm_NN_bcast_functions, DAGUE_zgemm_NN_bcast_NB_FUNCTIONS * sizeof(dague_function_t*));
  /* Compute the amount of arenas: */
  /*   DAGUE_zgemm_NN_bcast_DEFAULT_ARENA  ->  0 */
  __dague_object->super.arenas_size = 1;
  __dague_object->super.arenas = (dague_arena_t **)malloc(__dague_object->super.arenas_size * sizeof(dague_arena_t*));
  for(i = 0; i < __dague_object->super.arenas_size; i++) {
    __dague_object->super.arenas[i] = (dague_arena_t*)calloc(1, sizeof(dague_arena_t));
  }
  /* Now the Parameter-dependent structures: */
  __dague_object->super.transA = transA;
  __dague_object->super.transB = transB;
  __dague_object->super.alpha = alpha;
  __dague_object->super.beta = beta;
  __dague_object->super.descA = descA;
  __dague_object->super.A = A;
  __dague_object->super.descB = descB;
  __dague_object->super.B = B;
  __dague_object->super.descC = descC;
  __dague_object->super.C = C;
  /* If profiling is enabled, the keys for profiling */
#  if defined(DAGUE_PROF_TRACE)
  __dague_object->super.super.profiling_array = zgemm_NN_bcast_profiling_array;
  if( -1 == zgemm_NN_bcast_profiling_array[0] ) {
    dague_profiling_add_dictionary_keyword("GEMM", "fill:CC2828",
                                       sizeof(dague_profile_ddesc_info_t), dague_profile_ddesc_key_to_string,
                                       (int*)&__dague_object->super.super.profiling_array[0 + 2 * zgemm_NN_bcast_GEMM.function_id /* GEMM start key */],
                                       (int*)&__dague_object->super.super.profiling_array[1 + 2 * zgemm_NN_bcast_GEMM.function_id /* GEMM end key */]);
  }
#  endif /* defined(DAGUE_PROF_TRACE) */
  /* Create the data repositories for this object */
  GEMM_nblocal_tasks = zgemm_NN_bcast_GEMM_internal_init(__dague_object);
  if( 0 == GEMM_nblocal_tasks ) GEMM_nblocal_tasks = 10;
  __dague_object->GEMM_repository = data_repo_create_nothreadsafe(
          ((unsigned int)(GEMM_nblocal_tasks * 1.5)) > MAX_DATAREPO_HASH ?
          MAX_DATAREPO_HASH :
          ((unsigned int)(GEMM_nblocal_tasks * 1.5)), 3);

  READ_B_nblocal_tasks = zgemm_NN_bcast_READ_B_internal_init(__dague_object);
  if( 0 == READ_B_nblocal_tasks ) READ_B_nblocal_tasks = 10;
  __dague_object->READ_B_repository = data_repo_create_nothreadsafe(
          ((unsigned int)(READ_B_nblocal_tasks * 1.5)) > MAX_DATAREPO_HASH ?
          MAX_DATAREPO_HASH :
          ((unsigned int)(READ_B_nblocal_tasks * 1.5)), 1);

  READ_A_nblocal_tasks = zgemm_NN_bcast_READ_A_internal_init(__dague_object);
  if( 0 == READ_A_nblocal_tasks ) READ_A_nblocal_tasks = 10;
  __dague_object->READ_A_repository = data_repo_create_nothreadsafe(
          ((unsigned int)(READ_A_nblocal_tasks * 1.5)) > MAX_DATAREPO_HASH ?
          MAX_DATAREPO_HASH :
          ((unsigned int)(READ_A_nblocal_tasks * 1.5)), 1);

  __dague_object->super.super.startup_hook      = zgemm_NN_bcast_startup;
  __dague_object->super.super.object_destructor = (dague_destruct_object_fn_t)zgemm_NN_bcast_destructor;
  (void)dague_object_register((dague_object_t*)__dague_object);
  return (dague_zgemm_NN_bcast_object_t*)__dague_object;
}

#line 125 "zgemm_NN_bcast.jdf"


#line 2657 "zgemm_NN_bcast.c"
