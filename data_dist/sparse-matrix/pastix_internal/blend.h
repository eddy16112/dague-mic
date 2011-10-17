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

void          solverBlend(SolverMatrix *, Assembly1D *, Assembly2D *, int, int, int, BlendParam *, const Dof *dofptr);


