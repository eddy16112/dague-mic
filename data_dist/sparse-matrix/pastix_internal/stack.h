#ifndef STACK_H
#define STACK_H

typedef struct faststack {
  INT  pos;
  INT *tab;
} faststack_t;

#define FASTSTACK_INIT(s)   {(s).pos = 0; (s).tab[0] = -1; }
#define FASTSTACK_ADD(s, v) {((s).pos) ++; (s).tab[(s).pos] = (v);}
#define FASTSTACK_TOP(s, v) {v = (s).tab[(s).pos]; (s).pos--;}

#endif /* STACK_H */
