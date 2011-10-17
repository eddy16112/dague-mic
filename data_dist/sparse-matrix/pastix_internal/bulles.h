#ifndef BULLES_H
#define BULLES_H

/*+ The node structure. +*/
typedef struct BubbleTreeNode_ {
  int                       fathnum;      /*+ index of the father node               +*/
  int                       sonsnbr;
  int                       fsonnum;
  int                       fcandnum;     /*+ first bubble proc                      +*/
  int                       lcandnum;     /*+ last bubble proc                       +*/
  double                    costlevel;    /*+ cost of way from the root to this node +*/
  int                       treelevel;    /*+ cost of way from the root to this node +*/
  INT                       priomin;      /*+ Minimal priority of tasks owned by the bubble +*/
  INT                       priomax;      /*+ Maximal priority of tasks owned by the bubble +*/
  Queue *                   taskheap;     /*+ Liste de taches de la bulle            +*/
} BubbleTreeNode;


/*+ The bubble tree. +*/
typedef struct BubbleTree_ {
  int                       leavesnbr;    /*+ Number of leaves in tree  +*/
  int                       nodenbr;      /*+ Number of nodes           +*/
  int                       nodemax;      /*+ Number max of nodes       +*/
  int                      *sonstab;      /*+ Array of node             +*/
  BubbleTreeNode           *nodetab;      /*+ Array of node             +*/
} BubbleTree;


#define BFATHER(btree, r)     (btree)->nodetab[r].fathnum
#define BNBSON(btree, r)      (btree)->nodetab[r].sonsnbr
#define BROOT(btree)          (btree)->nodetab[(btree)->leavesnbr]
#define BSON(btree, n, r)     (btree)->sonstab[(btree)->nodetab[n].fsonnum + r ]

void  Bubble_InitTree  (BubbleTree *, int);
void  Bubble_Free      (BubbleTree *);
int   Bubble_Add       (BubbleTree *, INT, INT, double, INT);
void  Bubble_BuildTree (const BubbleTree *);
void  Bubble_Print     (const BubbleTree *, const double *, FILE*);

#endif /* BULLES_H */






