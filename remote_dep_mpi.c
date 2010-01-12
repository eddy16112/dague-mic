/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/* /!\  THIS FILE IS NOT INTENDED TO BE COMPILED ON ITS OWN
 *      It should be included from remote_dep.c if USE_MPI is defined
 */

#include <mpi.h>


/* TODO: smart use of dplasma context instead of ugly globals */
static MPI_Comm dep_comm;
static MPI_Request dep_req;
/* TODO: fix heterogeneous restriction by using mpi datatypes */ 
#define dep_dtt MPI_BYTE
#define dep_count sizeof(dplasma_execution_context_t)
static dplasma_execution_context_t dep_buff;

int dplasma_remote_dep_init(dplasma_context_t* context)
{
    MPI_Comm_dup(MPI_COMM_WORLD, &dep_comm);
    MPI_Recv_init(&dep_buff, dep_count, dep_dtt, MPI_ANY_SOURCE, REMOTE_DEP_ACTIVATE_TAG, dep_comm, &dep_req);
    MPI_Start(&dep_req);
    return 0;
}

int dplasma_remote_dep_fini(dplasma_context_t* context)
{
    MPI_Cancel(&dep_req);
    MPI_Request_free(&dep_req);
    MPI_Comm_free(&dep_comm);
    return 0;
}



int dplasma_remote_dep_activate(dplasma_execution_unit_t* eu_context,
                                const dplasma_execution_context_t* origin,
                                const param_t* origin_param,
                                dplasma_execution_context_t* exec_context,
                                const param_t* dest_param )
{
    int rank; 
    
    rank = remote_dep_compute_grid_rank(eu_context, origin, exec_context);
    assert(rank >= 0);
    return MPI_Send((void*) origin, dep_count, dep_dtt, rank, REMOTE_DEP_ACTIVATE_TAG, dep_comm);
}


int dplasma_remote_dep_progress(dplasma_execution_unit_t* eu_context)
{
    MPI_Status status;
    int flag;
#ifdef _DEBUG
    char tmp[128];
#endif
    
    MPI_Test(&dep_req, &flag, &status);
    if(flag)
    {
        DEBUG(("%s -> local\tFROM REMOTE process rank %d\n", dplasma_service_to_string(&dep_buff, tmp, 128), status.MPI_SOURCE));
        dplasma_signal_dependencies(eu_context, &dep_buff, 0);
        
        MPI_Start(&dep_req);
        return 1;
    }
    return 0;
}


