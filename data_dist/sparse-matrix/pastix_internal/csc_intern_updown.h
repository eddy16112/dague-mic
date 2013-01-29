/*
  File: csc_intern_updown.h

  Build UpDownVector from user right-hand-side and CSCd.
  Retrieve soltion from UpDownVector.
  Construct UpDownVector such as X[i] = 1, or X[i] = i.

*/
#ifndef CSC_INTERN_UPDOWN_H
#define CSC_INTERN_UPDOWN_H

/*
  Function: CscdUpdownRhs

  Fill-in UpDownVector structure from user right-hand-side member.

  Parameters:
    updovct - UpDownVector structure to fill-in.
    symbmtx - Solver matrix.
    rhs     - Right-hand-side member.
    perm    - reverse permutation tabular.
    dof      - Number of degree of freedom.
 */
void CscUpdownRhs(UpDownVector       *updovct,
		  const SolverMatrix *symbmtx, 
		  const FLOAT        *rhs, 
		  const INT          *perm,
		  int                 dof);

/*
  Function: CscdUpdownRhs

  Fill-in UpDownVector structure from user distributed right-hand-side member.

  Parameters:
    updovct - UpDownVector structure to fill-in.
    symbmtx - Solver matrix.
    rhs     - Right-hand-side member.
    invp    - reverse permutation tabular.
    g2l     - local numbers of global nodes, if not local contains -owner
    dof      - Number of degree of freedom.
 */
void CscdUpdownRhs(UpDownVector       *updovct,
		   const SolverMatrix *symbmtx, 
		   const FLOAT        *rhs, 
		   const INT          *invp,
		   const INT          *g2l,
		   const INT           ln,
		   int                 dof);

/*
  Function:CscdRhsUpdown

  Builds solution from UpDownVector structure

  Parameters:
    updovct  - UpDownVector structure containing the solution.
    symbmtx  - Solver matrix structure.
    rhs      - Solution to fill.
    ncol     - Number of columns in local matrix.
    dof      - Number of degree of freedom.
    comm     - MPI communicator.
  
 */
void CscRhsUpdown(const UpDownVector *updovct, 
		  const SolverMatrix *symbmtx, 
		  FLOAT              *rhs, 
		  const INT           ncol,
		  const INT          *invp,
		  const int           dof, 
		  const int           rhsmaking, 
		  MPI_Comm            comm);

/*
  Function:CscdRhsUpdown

  Builds distributed solution from
  UpDownVector structure

  Parameters:
    updovct  - UpDownVector structure containing the solution.
    symbmtx  - Solver matrix structure.
    x        - Solution to fill.
    ncol     - Number of columns in local matrix.
    g2l      - local numbers of global nodes, if not local contains -owner
    ord      - ordering
    dof      - Number of degree of freedom.
    comm     - MPI communicator.
  
 */
void CscdRhsUpdown(const UpDownVector *updovct, 
		   const SolverMatrix *symbmtx, 
		   FLOAT              *x,
		   const INT           ncol, 
		   const INT          *g2l,
		   const INT          *invp,
		   int                 dof,  
		   MPI_Comm            comm);

/*
  Function: Csc2updown

  Fill-in UpDownVector structure such as the solution of
  the system Ax=b is x_i=1 (API_RHS_1) or x_i=i (API_RHS_I). 

  Parameters:
    cscmtx   - internal CSCd matrix.
    updovct  - UpDownVector structure to fill-in.
    symbmtx  - Solver matrix.
    mode     - wanted solution API_RHS_1 or API_RHS_I.
    comm     - MPI communicator.
*/
void Csc2updown(const CscMatrix    *cscmtx, 
		UpDownVector       *updovct,
		const SolverMatrix *symbmtx, 
		int                 mode,
		MPI_Comm            comm);


/*
  Function: Csc2updown_X0

  Fill-in initial X0 for reffinement if we don't want to use
  Solve step.

  (iparm[IPARM_ONLY_RAFF] == API_YES)

  Parameters:
    updovct - UpDownVector structure were to copy B as the first X0 used for raffinement.
    symbmtx - Solver matrix.
    mode    - Rule to construct X0 (API_RHS_1 : X0[i] = 1, API_RHS_I : X0[i] = i).
    comm    - MPI_Communicator.
*/
void Csc2updown_X0(UpDownVector *updovct, 
		   /*const*/ SolverMatrix *symbmtx, 
		   int mode, 
		   MPI_Comm comm);

#endif /* CSC_INTERN_UPDOWN_H */
