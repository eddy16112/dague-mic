#include <stdio.h>
#include <stdlib.h>
#include "dep.h"

void dump_dep(const dep_t *d, const char *prefix)
{
    int i;

    printf("%s", prefix);
    if( NULL != d->cond ) {
        printf("if ");
        expr_dump(d->cond);
        printf(" then ");
    }
    printf("%s %s(", d->sym_name, d->dplasma_name);
    for(i = 0; i < MAX_CALL_PARAM_COUNT && NULL != d->call_params[i]; i++) {
        expr_dump(d->call_params[i]);
        if( i+1 < MAX_CALL_PARAM_COUNT && NULL != d->call_params[i+1] ) {
            printf(", ");
        } 
    }
    printf(")\n");
}
