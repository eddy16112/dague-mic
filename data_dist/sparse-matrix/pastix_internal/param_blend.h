/************************************************************/
/**                                                        **/
/**   NAME       : param_blend.h                           **/
/**                                                        **/
/**   AUTHORS    : Pascal HENON                            **/
/**                                                        **/
/**   FUNCTION   : Part of a parallel direct block solver. **/
/**                These lines are the data declarations   **/
/**                for the parameters of blend.            **/
/**                                                        **/
/**   DATES      : # Version 0.0  : from : 22 jul 1998     **/
/**                                 to     08 sep 1998     **/
/**                                                        **/
/************************************************************/

/*
**  The type and structure definitions.
*/

/*+ The parameters structure definition +*/

typedef struct BlendParam_ {
  char * hpf_filename;          /*+ file name for HPF distribution  +*/
  char * trace_filename;        /*+ file name for Paragraph traces  +*/
  char * ps_filename;           /*+ file name for matrix postscript +*/
  
  INT    hpf;             /*+ gener an HPF distribution file                                    +*/
  INT    tracegen ;       /*+ gener a simulated Paragraph execution trace file                  +*/
  INT    ps ;             /*+ gener a post-script of symbol matrix and elimination tree         +*/
  INT    assembly;        /*+ Gener the info structure needed to assemble                       +*/
  char * solvmtx_filename;/*+ name of solver matrix files (in sequential mode                   +*/ 
  INT    sequentiel;      /*+ Exec blend in sequentiel mode -> all solver matrix files generated+*/
  INT    count_ops ;      /*+ print costs in term of number of elementary operations            +*/
  INT    debug ;          /*+ make some check at certains execution points                      +*/
  INT    timer;           /*+ print execution time                                              +*/
  INT    recover;         /*+ take acount of a recover time estimation for ftgt                 +*/  
  INT    blcolmin ;       /*+ minimun number of column for a good use of BLAS primitives        +*/
  INT    blcolmax;
  INT    blblokmin ;      /*+ size of blockage for a good use of BLAS primitives  in 2D distribution +*/
  INT    blblokmax;
  INT    abs;             /*+ adaptative block size: := 0 all block are cut to blcolmin else try to make (ncand*abs) column +*/ 
  INT    leader;          /*+ Processor leader for not parallele task (ex: gener assembly1D     +*/
  INT    allcand;         /*+ All processor are candidat for a splitted cblk                    +*/
  INT    nocrossproc;     /*+ Crossing processor forbiden in the splitting phase                +*/
  INT    forceE2;
  INT    level2D;         /*+ number of level to treat with a 2D distribution                   +*/
  INT    candcorrect;   
  INT    clusterprop;     /*+ Proportionnal mapping with clustering for upper layers            +*/
  INT    costlevel;       /*+ Calcul du cout de chaque sous arbre dans candtab                  +*/
  INT    autolevel;       /*+ Level to shift 1D to 2D is automaticly computed                   +*/
  INT    malt_limit;      /*+ Limit for AUB memory allocations    (in octets)                   +*/
  INT    smpnbr;          /*+ Number of smp node                                                +*/
  INT    procnbr;         /*+ Number of physical processors in a smp node                       +*/
  double ratiolimit;  
  INT    dense_endblock;   /*+ Treat the square right lower part of the matrix as a dense matrix+*/  
  INT    ooc;              /*+ To use the out of core version of Pastix                         +*/
  INT    ricar;            /*+ If set to 1 then use blend dedicated to ricar                    +*/
  double oocmemlimit;      /*+ limit of physical memory for ooc                                 +*/
  INT   *iparm;            /*+ In/Out Integer parameters +*/
  double *dparm;           /*+ In/Out Float parameters   +*/
  INT     n;               /*+ Size of the matrix        +*/  
} BlendParam;




INT      blendParamInit(BlendParam *);
void     blendParamExit(BlendParam *);

