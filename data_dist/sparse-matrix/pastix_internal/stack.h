#ifndef STACK_H
#define STACK_H

typedef struct faststack {
  INT  pos;
  INT *tab;
} faststack_t;

#define FASTSTACK_INIT(s)   (s).pos = 0;
#define FASTSTACK_ADD(s, v) (s).pos += 1; (s).tab[(s).pos] = (v);
#define FASTSTACK_TOP(s, v) v = (s).tab[(s).pos]; (s).pos--;

#endif /* STACK_H */
