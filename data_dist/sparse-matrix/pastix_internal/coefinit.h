/*
  File: coefinit.h

  Allocation and initialisation of the coeficient of the solver matrix.  
*/
#ifndef COEFINIT_H
#define COEFINIT_H

/* Section: Functions declarations*/

/*
  Function: CoefMatrix_Allocate
  
  Allocate matrix coefficients in coeftab and ucoeftab.

  Should be first called with me = -1 to allocated coeftab.
  Then, in NUMA_ALLOC mode should be called with me set to thread ID 
  to allocate column blocks coefficients arrays.
  
  Parameters
 
     datacode  - solverMatrix 
     factotype - factorization type (LU, LLT ou LDLT)
     me        - thread number. (-1 for first call, 
                 from main thread. >=0 to allocate column blocks 
		 assigned to each thread.)
 
*/
void CoefMatrix_Allocate (SopalinParam    *sopar,
			  SolverMatrix    *datacode,
			  pthread_mutex_t *mutex,
			  INT              factotype, 
			  INT              me);

/*
  Function: CoefMatrix_Init

  Init coeftab and ucoeftab coefficients.
  
  OOC only works with NUMA_ALLOC

  Parameters:
     datacode     - solverMatrix 
     barrier      - Barrier used for thread synchronisation.
     me           - Thread ID 
     iparm        - Integer parameters array.
     transcsc     - vecteur transcsc
     sopalin_data - <Sopalin_Data_t> structure for NUMA version.
*/
void CoefMatrix_Init     (SolverMatrix         *datacode, 
			  sopthread_barrier_t  *barrier, 
			  INT                   me,
			  INT                  *iparm, 
			  FLOAT               **transcsc, 
			  Sopalin_Data_t       *sopalin_data);

/*
  Function: CoefMatrix_Free
  
  Free the solver matrix coefficient tabular : coeftab and ucoeftab.
  
  WARNING: Call it with one unnique thread. 

  Parameters:
    datacode   - solverMatrix 
    factotype  - factorisation type (<API_FACT>)
    
*/  
void CoefMatrix_Free     (SopalinParam *sopar,
			  SolverMatrix *datacode, 
			  INT           factotype);


#endif /* COEFINIT_H */
