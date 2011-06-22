/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _DAGUE_scheduling_h
#define _DAGUE_scheduling_h

#include "dague.h"

/**
 * Add the dague_object_t to the execution queue of the dague_context_t. As
 * a result all ready tasks on the dague_object_t are scheduledfor execution
 * on the dague_context_t, and the total number of tasks on the dague_context_t
 * is increased by the number of local tasks on the dague_object_t.
 *
 * @param [INOUT] The dague context where the tasks generated by the dague_object_t
 *                are to be executed.
 * @param [INOUT] The dague object with pending tasks.
 *
 * @return 0 If the enqueue operation succeeded.
 */
int dague_enqueue( dague_context_t*, dague_object_t* );

/**
 * Start the execution of the dague_context_t. The other threads will
 * asynchronously start the execution of the pending tasks. The execution cannot
 * complete without the submitting thread getting involved, either using
 * dague_wait or dague_test.
 *
 * @param [INOUT] The dague context which is to be started.
 *
 * @return 0 If the execution is ongoing.
 */
int dague_start( dague_context_t* );

/**
 * Check the status of an ongoing execution, started with dague_start
 *
 * @param [INOUT] The dague context where the execution is taking place.
 *
 * @return 0 If the execution is still ongoing.
 * @return 1 If the execution is completed, and the dague_context has no
 *           more pending tasks. All subsequent calls on the same context
 *           will automatically succeed.
 */
int dague_test( dague_context_t* );

/**
 * Wait until all the possible tasks on the dague_context_t are executed
 * before returning.
 *
 * @param [INOUT] The dague context where the execution is taking place.
 *
 * @return 0 If the execution is completed.
 * @return * Any other error raised by the tasks themselves.
 */
int dague_wait( dague_context_t* );

/**
 * Mark a execution context as being ready to be scheduled, i.e. all
 * input dependencies are resolved. The execution context can be
 * executed immediately or delayed until resources become available.
 *
 * @param [IN] The execution context to be executed. This include
 *             calling the attached hook (if any) as well as marking
 *             all dependencies as completed.
 *
 * @return  0 If the execution was succesful and all output dependencies
 *            has been correctly marked.
 * @return -1 If something went wrong.
 */
int dague_schedule( dague_context_t*, const dague_execution_context_t* );
int __dague_schedule( dague_execution_unit_t*, dague_execution_context_t*);

int dague_progress(dague_context_t* context);
void* __dague_progress(dague_execution_unit_t* eu_context);

void dague_register_nb_tasks(dague_context_t* context, int32_t n);



//#ifdef DEPRECATED
/**
 * Signal the termination of the execution context to all dependencies of 
 * its dependencies.  
 * 
 * @param [IN]  The exeuction context of the finished task.
 * @param [IN]  when forward_remote is 0, only local (in the sense of the 
 *              process grid predicates) dependencies are satisfied.
 *
 * @return 0    If the dependencies have successfully been signaled.
 * @return -1   If something went wrong. 
 */
int dague_trigger_dependencies( const struct dague_object *dague_object,
                                dague_execution_unit_t*,
                                const dague_execution_context_t*,
                                int forward_remote );

int dague_complete_execution( dague_execution_unit_t *eu_context,
                              dague_execution_context_t *exec_context );

//#endif /* DEPRECATED */

#endif  /* _DAGUE_scheduling_h */

