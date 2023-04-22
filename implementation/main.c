#include "cpu.h"
#include "gpu_pure.cuh"
#include "gpu_conventional.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#define S_LEN 512
#define N 1000

int main(int argc, char *argv[]) {
    srand(time(NULL));

    char alphabet[5] = {'A', 'C', 'G', 'T', 'N'};

    char **query = (char **) malloc(N * sizeof(char *));
    for (int i = 0; i < N; i++)
        query[i] = (char *) malloc(S_LEN * sizeof(char));

    char **reference = (char **) malloc(N * sizeof(char *));
    for (int i = 0; i < N; i++)
        reference[i] = (char *) malloc(S_LEN * sizeof(char));

    int **sc_mat = (int **) malloc((S_LEN + 1) * sizeof(int *));
    for (int i = 0; i < (S_LEN + 1); i++)
        sc_mat[i] = (int *) malloc((S_LEN + 1) * sizeof(int));

    char **dir_mat = (char **) malloc((S_LEN + 1) * sizeof(char *));
    for (int i = 0; i < (S_LEN + 1); i++)
        dir_mat[i] = (char *) malloc((S_LEN + 1) * sizeof(char));

    int *res = (int *) malloc(N * sizeof(int));
    char **simple_rev_cigar = (char **) malloc(N * sizeof(char *));
    for (int i = 0; i < N; i++)
        simple_rev_cigar[i] = (char *) malloc(S_LEN * 2 * sizeof(char));

    // randomly generate sequences
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < S_LEN; j++) {
            query[i][j] = alphabet[rand() % 5];
            reference[i][j] = alphabet[rand() % 5];
        }
    }

    cpu(query, reference, sc_mat, dir_mat, res, simple_rev_cigar);
    if(!strcmp(argv[1],"pure"))
        gpu_pure(query, reference, sc_mat, dir_mat, res, simple_rev_cigar);
    else if(!strcmp(argv[1],"conventional"))
        gpu_conventional(query, reference, sc_mat, dir_mat, res, simple_rev_cigar);
    else
        printf("Invalid argument\n");


}