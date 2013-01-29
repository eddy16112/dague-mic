/************************************************************/
/**                                                        **/
/**   NAME       : updown.h                                **/
/**                                                        **/
/**   AUTHORS    : David GOUDIN                            **/
/**                Pascal HENON                            **/
/**                Francois PELLEGRINI                     **/
/**                Pierre RAMET                            **/
/**                                                        **/
/**   FUNCTION   : Part of a parallel direct block solver. **/
/**                These lines are the data declarations   **/
/**                for the UpDown step  .                  **/
/**                                                        **/
/**   DATES      : # Version 0.0  : from : 22 jul 1998     **/
/**                                 to     28 oct 1998     **/
/**                                                        **/
/************************************************************/

#ifndef UPDOWN_H
#define UPDOWN_H

/*+ UpDown block structure. +*/

typedef struct UpDownCblk_  {
  INT                       sm2xind;              /*+ Index in the rhs local vector of the unknowns corresponding to the diag blok +*/
  INT *                     browproctab;          /*+ Brow                               +*/
  INT *                     browcblktab;          /*+ Brow                               +*/
  INT                       browprocnbr;          /*+ Brow size                          +*/
  INT                       msgnbr;               /*+ Number of messages                 +*/
  INT volatile              msgcnt;               /*+ Number of messages                 +*/
  INT                       ctrbnbr;              /*+ Number of contributions            +*/
  INT volatile              ctrbcnt;              /*+ Number of contributions            +*/
} UpDownCblk;


/*+ UpDown vector structure. +*/

typedef struct UpDownVector_ {
  UpDownCblk *              cblktab;              /*+ Array of solver column blocks      +*/
  FLOAT *                   sm2xtab;              /*+ Unknown vector                     +*/
  INT                       sm2xmax;              /*+ Maximum of coefficients per unknown vector +*/
  INT                       sm2xsze;              /*+ Size of sm2xtab                    +*/
  INT                       sm2xnbr;              /*+ Number of sm2x                     +*/
  INT *                     gcblk2list;           /*+ Global cblknum -> index in listptr +*/
  INT                       gcblk2listnbr;        /*+ Size of gcblk2list                 +*/
  INT *                     listptr;              /*+ Index in list                      +*/
  INT                       listptrnbr;           /*+ Size of listptr                    +*/
  INT *                     listcblk;             /*+ List of cblk in a same row         +*/
  INT *                     listblok;             /*+ List of blok in a same row         +*/
  INT                       listnbr;              /*+ Size of list                       +*/
  INT *                     loc2glob;             /*+ Local cblknum -> global cblknum    +*/
  INT                       loc2globnbr;          /*+ Size of loc2glob                   +*/
  INT *                     lblk2gcblk;           /*+ Local blok -> global facing cblk   +*/
  INT                       gcblknbr;             /*+ total number of cblk               +*/
  INT                       gnodenbr;             /*+ total number of nodes              +*/
  INT                       downmsgnbr;           /*+ Nb messages receive during down    +*/
  INT                       upmsgnbr;             /*+ Nb messages receive during up      +*/
} UpDownVector;

#endif /* UPDOWN_H */
