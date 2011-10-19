/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 */
#ifndef _DSPARSE_H_
#define _DSPARSE_H_

#include "dague_config.h"

#define dsparse_error(__func, __msg) fprintf(stderr, "%s: %s\n", (__func), (__msg))

#include "data_dist/sparse-matrix/sparse-matrix.h"

#include "dsparse/include/dsparse_s.h"
#include "dsparse/include/dsparse_d.h"
#include "dsparse/include/dsparse_c.h"
#include "dsparse/include/dsparse_z.h"

#endif /* _DSPARSE_H_ */
