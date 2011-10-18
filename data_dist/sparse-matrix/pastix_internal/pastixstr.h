#ifndef PASTIX_STR_H
#define PASTIX_STR_H

/*
   struct: pastix_data_t

   Structure used to store datas for a step by step execution.
*/

typedef struct pastix_data_t {
  SolverMatrix     solvmatr;         /*+ Matrix informations                                                 +*/
  SopalinParam     sopar;            /*+ Sopalin parameters                                                  +*/
  Order            ordemesh;         /*+ Order                                                               +*/
#ifdef WITH_SCOTCH
  SCOTCH_Graph     grafmesh;         /*+ Graph                                                               +*/
  int              malgrf;           /*+ boolean indicating if grafmesh has been allocated                   +*/
#endif /* WITH_SCOTCH */
#ifdef DISTRIBUTED
#ifdef WITH_SCOTCH
  SCOTCH_Dordering ordedat;          /*+ distributed scotch order                                            +*/
  SCOTCH_Dgraph    dgraph;
  INT             *PTS_permtab;
  INT             *PTS_peritab;
#endif /* WITH_SCOTCH */
  INT             *glob2loc;         /*+ local column number of global column, or -(owner+1) is not local    +*/
#endif
#ifdef DISTRIBUTED
  INT              ncol_int;         /*+ Number of local columns in internal CSCD                            +*/
  INT             *l2g_int;          /*+ Local to global column numbers in internal CSCD                     +*/
  int              malrhsd_int;      /*+ Indicates if internal distributed rhs has been allocated            +*/
  int              mal_l2g_int;
  FLOAT           *b_int;            /*+ Local part of the right-hand-side                                   +*/
  INT             *loc2glob2;        /*+ local2global column number                                          +*/
#endif
  INT              gN;               /*+ global column number                                                +*/
  INT              n;                /*+ local column number                                                 +*/
  int              procnbr;          /*+ Number of MPI tasks                                                 +*/
  int              procnum;          /*+ Local MPI rank                                                      +*/
  INT             *iparm;            /*+ Vecteur de parametres entiers                                       +*/
  double          *dparm;            /*+ Vecteur de parametres floattant                                     +*/
  INT              n2;               /*+ Number of local columns                                             +*/
  INT             *col2;             /*+ column tabular for the CSC matrix                                   +*/
				     /*+ (index of first element of each col in row and values tabulars)     +*/
  INT             *row2;             /*+ tabular containing row number of each element of                    +*/
				     /*+  the CSC matrix, ordered by column.                                 +*/
  int              bmalcolrow;       /*+ boolean indicating if col2 ans row2 have been allocated             +*/
  int              malord;           /*+ boolean indicating if ordemesh has been allocated                   +*/
  int              malcsc;           /*+ boolean indicating if solvmatr->cscmtx has beek allocated           +*/
  int              malsmx;           /*+ boolean indicating if solvmatr->updovct.sm2xtab has been allocated  +*/
  int              malslv;           /*+ boolean indicating if solvmatr has been allocated                   +*/
  int              malcof;           /*+ boolean indicating if coeficients tabular(s) has(ve) been allocated +*/
  MPI_Comm         pastix_comm;      /*+ PaStiX MPI communicator                                             +*/
  int             *bindtab;          /*+ Tabular giving for each thread a CPU to bind it too                 +*/
  INT              nschur;           /*+ Number of entries for the Schur complement.                         +*/
  INT             *listschur;        /*+ List of entries for the schur complement.                           +*/
  FLOAT           *schur_tab;
  INT              schur_tab_set;
  int              cscInternFilled;
  int              scaling;          /*+ Indicates if the matrix has been scaled                             +*/
  FLOAT           *scalerowtab;      /*+ Describes how the matrix has been scaled                            +*/
  FLOAT           *iscalerowtab;
  FLOAT           *scalecoltab;
  FLOAT           *iscalecoltab;
} pastix_data_t;

#endif /* PASTIX_STR_H */
