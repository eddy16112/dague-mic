#ifndef _DAGUE_PASTIX_INTERNAL_H_
#define _DAGUE_PASTIX_INTERNAL_H_

/*#include <pastix.h>*/

#define NUMA_ALLOC
#define FORCE_NOMPI
#define MEMORY_USAGE
#define WITH_SCOTCH
#define FORCE_INT64
#define INTSSIZE64

#include "common_pastix.h"
#include "sopalin_define.h"

#include "nompi.h"
#include <scotch.h>

#include "graph.h"
#include "dof.h"
#include "ftgt.h"
#include "symbol.h"
#include "csc.h"
#include "updown.h"
#include "queue.h"
#include "bulles.h"
#include "solver.h"
#include "assembly.h"
#include "param_blend.h"
#include "order.h"
#include "fax.h"
#include "kass.h"
#include "blend.h"
#include "solverRealloc.h"
#include "sopalin_thread.h"
#include "stack.h"
#include "sopalin3d.h"
#include "sopalin_init.h"
#include "sopalin_option.h"
#include "csc_intern_updown.h"
#include "csc_intern_build.h"
#include "coefinit.h"
#include "out.h"
#include "pastixstr.h"

#define pastix_int_t     int64_t
#define pastix_uint_t    u_int64_t

typedef double DagDouble_t;

#endif /* _DAGUE_PASTIX_INTERNAL_H_ */
