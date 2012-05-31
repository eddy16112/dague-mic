/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DAGUE_CONFIG_H_HAS_BEEN_INCLUDED
#error "dague_config_bottom.h should only be included from dague_config.h"
#endif

/*
 * Flex is trying to include the unistd.h file. As there is no configure
 * option or this, the flex generated files will try to include the file
 * even on platforms without unistd.h (such as Windows). Therefore, if we
 * know this file is not available, we can prevent flex from including it.
 */
#ifndef HAVE_UNISTD_H
#define YY_NO_UNISTD_H
#endif

/*
 * BEGIN_C_DECLS should be used at the beginning of your declarations,
 * so that C++ compilers don't mangle their names.  Use END_C_DECLS at
 * the end of C declarations.
 */
#undef BEGIN_C_DECLS
#undef END_C_DECLS
#if defined(c_plusplus) || defined(__cplusplus)
# define BEGIN_C_DECLS extern "C" {
# define END_C_DECLS }
#else
#define BEGIN_C_DECLS          /* empty */
#define END_C_DECLS            /* empty */
#endif

#if defined(HAVE_STDDEF_H)
#include <stddef.h>
#endif  /* HAVE_STDDEF_H */
#include <stdint.h>

#if defined(HAVE_MPI)
# define DISTRIBUTED
#else
# undef DISTRIBUTED
#endif

#if defined(DAGUE_PROF_DRY_RUN)
# define DAGUE_PROF_DRY_BODY
# define DAGUE_PROF_DRY_DEP
#endif

#ifndef DAGUE_DIST_EAGER_LIMIT
#define RDEP_MSG_EAGER_LIMIT    0
#else
#define RDEP_MSG_EAGER_LIMIT    (((size_t)DAGUE_DIST_EAGER_LIMIT)*1024)
#endif

#ifndef DAGUE_DIST_EAGER_LIMIT
#define RDEP_MSG_SHORT_LIMIT    0
#else
#define RDEP_MSG_SHORT_LIMIT    (((size_t)DAGUE_DIST_SHORT_LIMIT)*1024)
#endif

#if (DAGUE_DEBUG_VERBOSE >= 3)
#   define DAGUE_DEBUG_VERBOSE3
#   define DAGUE_DEBUG_VERBOSE2
#   define DAGUE_DEBUG_VERBOSE1
#elif (DAGUE_DEBUG_VERBOSE >= 2)
#   define DAGUE_DEBUG_VERBOSE2
#   define DAGUE_DEBUG_VERBOSE1
#elif (DAGUE_DEBUG_VERBOSE >= 1)
#   define DAGUE_DEBUG_VERBOSE1
#endif

#if defined(DAGUE_SCHED_DEPS_MASK)
typedef uint32_t dague_dependency_t;
#else
/**
 * Should be large enough to support MAX_PARAM_COUNT values.
 */
typedef uint32_t dague_dependency_t;

#endif

/*
 * A set of constants defining the capabilities of the underlying
 * runtime.
 */
#define MAX_LOCAL_COUNT  20
#define MAX_PARAM_COUNT  20

#define MAX_DEP_IN_COUNT  10
#define MAX_DEP_OUT_COUNT 10

#define MAX_TASK_STRLEN 128

