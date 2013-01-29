/************************************************************/
/**                                                        **/
/**   NAME       : blend.h                                 **/
/**                                                        **/
/**   AUTHORS    : Pascal HENON                            **/
/**                                                        **/
/**   FUNCTION   : Part of a parallel direct block solver. **/
/**                Partition and distribute data           **/
/**                for an optimal parallel resolution      **/
/**                                                        **/
/**   DATES      : # Version 0.0  : from : 22 jul 1998     **/
/**                                 to                     **/
/**                                                        **/
/************************************************************/
#ifndef BLEND_H
#define BLEND_H

/*
 * Function: solverBlend
 *
 * Main blend function
 *
 * Build the elimination graph from the symbolic partition.
 *
 * Build the cost matrix from the symbolic partition.
 *
 * Build the elimination tree from the symbolic partition.
 *
 * Distribute each column bloc on candidate processors.
 *
 * Build a new symbol matrix...
 *
 * Parameters:
 *   solvmtx    - Solver matrix structure.
 *   symbmtx    - Symbol matrix
 *   assemb1D   -
 *   assemb2D   -
 *   clustnbr   - Number of MPI processes.
 *   thrdlocnbr - Number of threads.
 *   cudanbr    - Number of cuda devices.
 *   clustnum   - Processor ID number.
 *   option     - Blend parameters.
 *   dofptr     -
 */
void          solverBlend(SolverMatrix * solvmtx,
                          SymbolMatrix * symbmtx,
                          Assembly1D   * assemb1D,
                          Assembly2D   * assemb2D,
                          int            clustnbr,
                          int            thrdlocnbr,
                          int            cudanbr,
                          int            clusnum,
                          BlendParam   * option,
                          const Dof    * dofptr);


#endif /* BLEND_H */
