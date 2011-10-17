/************************************************************/
/**                                                        **/
/**   NAME       : kass.h                                  **/
/**                                                        **/
/**   AUTHORS    : Pascal HENON                            **/
/**                                                        **/
/**   FUNCTION   : Compute a block structure of the factor **/
/**                obtained by a ILU(k) factorization      **/
/**                                                        **/
/**                                                        **/
/**   DATES      : # Version 0.0  : from : 30/01/2006      **/
/**                                 to                     **/
/**                                                        **/
/************************************************************/

void kass(int            levelk, 
	  int            rat, 
	  SymbolMatrix * symbptr, 
	  INT            baseval,
	  INT            vertnbr, 
	  INT            edgenbr, 
	  INT          * verttab,
	  INT          * edgetab, 
	  Order        * orderptr, 
	  MPI_Comm       pastix_comm);

/* void kass(int alpha, int rat, SymbolMatrix * symbptr, Graph * graphptr, Order * orderptr, MPI_Comm pastix_comm); */

void ifax(INT n, INT *ia, INT *ja, INT levelk, INT  cblknbr, INT *rangtab, INT *perm, INT *iperm, SymbolMatrix *symbmtx);

