#ifndef SW_CPU_H
#define SW_CPU_H

#define S_LEN 512
#define N 1000
int cpu(char ** query, char ** reference, int ** sc_mat, char ** dir_mat, int * res, char ** simple_rev_cigar);

#endif