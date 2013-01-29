/*
  File: csc_intern_build.h
  
  Functions to build internal CSCd from user CSCd.

  Function to free internal CSCd.

*/
#ifndef CSC_INTERN_BUILD_H
#define CSC_INTERN_BUILD_H

/*
  Function: CscOrdistrib

  Fill in *thecsc* CSC matrix in column block representation.

  Parameters: 
  
  thecsc     - Matrix in block column CSC format to fill in.
  Type       - 3 charactères for matrix type : only Type[1] is used to check if matrix is Symetric(S) or not(U).
  transcsc   - Transpose of the CSC in non symetric mode.
  ord        - ordering
  Nrow       - Number of rows.
  Ncol       - Number of columns.
  Nnzero     - Number of non zeros in the matrix.
  colptr     - Index in *rowind* and *val* of the start of each column.
  rowind     - Index of the elements.
  val        - values of the elements.
  forcetrans - If matrix symetric, transcsc will be the copy of the CSC_VALTAB.
  symbmtx    - Solver matrix
  procnum    - MPI process number.
  dof        - Number of degree of freedom
*/
void CscOrdistrib(CscMatrix          *thecsc, 
		  char               *Type, 
		  FLOAT             **transcsc,
		  const Order        *ord, 
		  INT                 Nrow, 
		  INT                 Ncol,
		  INT                 Nnzero, 
		  INT                *colptr, 
		  INT                *rowind, 
		  FLOAT              *val, 
		  INT                 forcetrans,
		  const SolverMatrix *symbmtx, 
		  INT                 procnum, 
		  INT                 dof);

/*
  Function: CscdOrdistrib

  Fill in *thecsc* CSC matrix in column block representation.

  - Construct cachetab (sizeof(INT)*globalNbCol) which will contain
  the column block wich will own each column (internal numerotation), 
  or -1 if not local 

  - Build newcoltab (sizeof(INT)*globalNbCol) which will contain the 
  coltab corresponding to the local internal CSCd.
  This CSCd correspond to the given CSCd adding upper part in Symmetric matrix.
  Also count number of triples (i,j,v) to send to each other processors.

  - Send the information about how many triples will be sent
  
  - Fill-in the arrays containing triples to send and send them.

  - Receive those arrays and correct the newcoltab arrays with information 
  from others processors.

  - Build CSC_COLNBR from symbolic matrix informations and CSC_COL from newcoltab.

  - Construct transpose matrix, in symmetric mode, transcsc == CSC_VALTAB; in 
  unsymmetric mode, allocate trowtab (number of total local elements) , 
  and build trscltb which contains number of elements, 
  in each column of each column bloc.

  - fill-in internal CSC row and values from local given CSCd, 
  also fill-in trowtab and transcsc in unsymmetric mode.
  CSC_COL and trscltb are incremented for each element added. 

  - fill-in  internal CSC row and values from iniformation received,
  also fill in transposed CSCd in unsymmetric mode.
  CSC_COL and trscltb are incremented for each element added.

  - restore CSC_COL.
  
  - sort internal CSCd.

  - sort intranal transposed CSCd.

  Parameters: 
  
  thecsc     - Matrix in block column CSC format to fill in.
  Type       - 3 charactères for matrix type : only Type[1] is used to check if matrix is Symetric(S) or not(U).
  transcsc   - Transpose of the CSC in non symetric mode.
  ord        - ordering
  Ncol       - Number of columns.
  colptr     - Index in *rowind* and *val* of the start of each column.
  rowind     - Index of the elements.
  val        - values of the elements.
  l2g        - global numbers of local nodes.
  gNcol      - global number of columns.
  g2l        - local numbers of global nodes, if not local contains -owner
  forcetrans - If matrix symetric, transcsc will be the copy of the CSC_VALTAB.
  symbmtx    - Solver matrix
  procnum    - MPI process number.
  dof        - Number of degree of freedom
  comm       - MPI communicator.
*/
void CscdOrdistrib(CscMatrix          *thecsc, 
		   char               *Type, 
		   FLOAT             **transcsc,
		   const Order        *ord, 
		   INT                 Ncol,
		   INT                *colptr, 
		   INT                *rowind, 
		   FLOAT              *val, 
		   INT                *l2g,
		   INT                 gNcol,
		   INT                *g2l,
		   INT                 forcetrans,
		   const SolverMatrix *symbmtx, 
		   INT                 procnum,
		   INT                 dof,
		   MPI_Comm            comm);

/* 
   Function: CscExit
   
   Free the internal CSCd structure.
   
   Parameters:
     thecsc - Internal CSCd to free.
*/
void CscExit(CscMatrix *thecsc);
#endif /* CSC_INTERN_BUILD_H */
