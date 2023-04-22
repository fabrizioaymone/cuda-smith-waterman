#include "gpu_conventional.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define S_LEN 512
#define DIM S_LEN+1
#define N 1000

#define CHECK(call)                                                                       \
    {                                                                                     \
        const cudaError_t err = call;                                                     \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }

#define CHECK_KERNELCALL()                                                                \
    {                                                                                     \
        const cudaError_t err = cudaGetLastError();                                       \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }


double get_time_gpu() // function to get the time of day in seconds
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}


__device__ int max4_gpu(int n1, int n2, int n3, int n4) {
    int tmp1, tmp2;
    tmp1 = n1 > n2 ? n1 : n2;
    tmp2 = n3 > n4 ? n3 : n4;
    tmp1 = tmp1 > tmp2 ? tmp1 : tmp2;
    return tmp1;
}


__device__ void backtrace_gpu(char *simple_rev_cigar, char dir_mat[], int i, int j, int max_cigar_len) {
    int k;
    for (k = 0; k < max_cigar_len && dir_mat[i * (S_LEN + 1) + j] != 0; k++) {
        int dir = dir_mat[i * (S_LEN + 1) + j];
        if (dir == 1 || dir == 2) {
            i--;
            j--;
        } else if (dir == 3)
            i--;
        else if (dir == 4)
            j--;

        simple_rev_cigar[k] = dir;
        // if(k==0 && )
    }
}


__global__ void diagonal_kernel(char *query, char *reference, int *res, char *simple_rev_cigar, char *dir_mat_gpu) {
    int threadId = threadIdx.x;
    int n = blockIdx.x;
    char *dir_mat = &dir_mat_gpu[n * (S_LEN + 1) * (S_LEN + 1)];

    __shared__ char shared_query[S_LEN];
    __shared__ char shared_reference[S_LEN];

    if (threadId < S_LEN)
        shared_query[threadId] = query[n * S_LEN + threadId];
    else if (threadId < S_LEN * 2)
        shared_reference[threadId - S_LEN] = reference[n * S_LEN + (threadId - S_LEN)];

   // __syncthreads();

    __shared__ int mat[S_LEN + 1];
    __shared__ int prev[S_LEN + 1];
    __shared__ int prev2[S_LEN + 1];

    //matrix initialisation
    prev[0] = 0;
    prev[1] = 0;
    prev2[0] = 0;
   // __syncthreads();

    int ins = -2, del = -2, match = 1, mismatch = -1; // penalties
    __shared__ int maxi, maxj;
    __shared__ int max;
    max = -1;
    __shared__ int threadMax;

    for (int diag = 2; diag < (2 * S_LEN + 1); diag++) {
        int length = DIM - abs(DIM - (diag + 1));
        int length_1 = DIM - abs(DIM - (diag));
        int length_2 = DIM - abs(DIM - (diag - 1));
        if (DIM > diag) {
            mat[0] = 0;
            mat[length - 1] = 0;
        }
        threadMax = -1;

        __syncthreads();

        int row = (DIM > diag) ? (diag - threadId) : ((DIM - 1) - threadId);
        int col = (DIM > diag) ? threadId : ((diag - (DIM - 1)) + threadId);

        int tmp = -2;
        if (threadId < length) {
            if (row != 0 && col != 0) {
                // compare the sequences characters

                int comparison = (shared_query[row - 1] == shared_reference[col - 1]) ? match : mismatch;

                // compute the cell knowing the comparison result


                tmp = max4_gpu(prev2[threadId + (length_2 - length) / 2] + comparison,
                               prev[(DIM > diag) ? threadId : (threadId + 1)] + del,
                               prev[(DIM > diag) ? (threadId - 1) : threadId] + ins,
                               0);
                //max4(sc_mat[i - 1][j - 1] + comparison, sc_mat[i - 1][j] + del, sc_mat[i][j - 1] + ins, 0);

                char dir;

                if (tmp == (prev2[threadId + (length_2 - length) / 2] + comparison))
                    dir = comparison == match ? 1 : 2;
                else if (tmp == (prev[(DIM > diag) ? threadId : (threadId + 1)] + del))
                    dir = 3;
                else if (tmp == (prev[(DIM > diag) ? (threadId - 1) : threadId] + ins))
                    dir = 4;
                else
                    dir = 0;

                dir_mat[row * (S_LEN + 1) + col] = dir;
                mat[threadId] = tmp;
                atomicMax(&max, tmp);

            }
        }
        __syncthreads(); // se non sincronizzi hai thread che non entrando nell'if sopra ti modificano prev e prev2 prima che i thread nell'if li usino per calcolare la score
        if (threadId < length_1)
            prev2[threadId] = prev[threadId];
        if (threadId < length)
            prev[threadId] = mat[threadId];

        __syncthreads();

         if (tmp == max) {
            atomicMax(&threadMax, threadId);
         }
        __syncthreads();

        if (tmp == max && threadId == threadMax) {
            maxi = row;
            maxj = col;
        }
        __syncthreads();
    }

    __syncthreads();
    res[n] = max;

    if (threadId == 0)
        backtrace_gpu(&simple_rev_cigar[n * S_LEN * 2], dir_mat, maxi, maxj, S_LEN * 2);

}


int gpu_conventional(char **query, char **reference, int **sc_mat, char **dir_mat, int *res, char **simple_rev_cigar) {

    //declare global variables


    char *query_gpu;
    char *reference_gpu;
    int *res_gpu;
    char *simple_rev_cigar_gpu;
    char *dir_mat_gpu;

    int *res_host = (int *) malloc(N * sizeof(int));

    char *simple_rev_cigar_host = (char *) malloc(N * S_LEN * 2 * sizeof(char));

    //allocate space in global memory and copy original ones -> linearize matrices!!

    CHECK(cudaMalloc((void **) &query_gpu, N * S_LEN * sizeof(char)));
    for (int i = 0; i < N; i++) CHECK(
            cudaMemcpy(&query_gpu[i * S_LEN], query[i], S_LEN * sizeof(char), cudaMemcpyHostToDevice));

    CHECK(cudaMalloc((void **) &reference_gpu, N * S_LEN * sizeof(char)));
    for (int i = 0; i < N; i++) CHECK(
            cudaMemcpy(&reference_gpu[i * S_LEN], reference[i], S_LEN * sizeof(char), cudaMemcpyHostToDevice));

    CHECK(cudaMalloc((void **) &res_gpu, N * sizeof(int)));

    CHECK(cudaMalloc((void **) &simple_rev_cigar_gpu, N * S_LEN * 2 * sizeof(char)));

    CHECK(cudaMalloc((void **) &dir_mat_gpu, N * (S_LEN + 1) * (S_LEN + 1) * sizeof(char)));
    CHECK(cudaMemset((void *) dir_mat_gpu, 0, N * (S_LEN + 1) * (S_LEN + 1) * sizeof(char)));

    dim3 gridsize(1000);
    dim3 blocksize(1024);


    double start_gpu = get_time_gpu();

    diagonal_kernel<<<gridsize, blocksize>>>(query_gpu, reference_gpu, res_gpu, simple_rev_cigar_gpu, dir_mat_gpu);
    CHECK_KERNELCALL();


    CHECK(cudaDeviceSynchronize());
    double end_gpu = get_time_gpu();

    printf("SW Time GPU: %.10lf\n", end_gpu - start_gpu);

    CHECK(cudaMemcpy(res_host, res_gpu, N * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(simple_rev_cigar_host, simple_rev_cigar_gpu, N * S_LEN * 2 * sizeof(char),
                     cudaMemcpyDeviceToHost));





    // verify res are correct
    for (int i = 0; i < N; i++)
        if (res_host[i] != res[i]) {
            printf("ERROR! N: %d res: %d res_host: %d\n", i, res[i], res_host[i]);
        }

    // verify cigars are correct
    for (int i = 0; i < N; i++) {
        int flag = 0;
        for (int j = 0; j < S_LEN * 2 && !flag; j++) {

            if (simple_rev_cigar[i][j] != simple_rev_cigar_host[i * S_LEN * 2 + j]) {
                printf("ERROR! N: %d index:%d cpu_cigar: %d gpu_cigar: %d\n", i, j, simple_rev_cigar[i][j],
                       simple_rev_cigar_host[i * S_LEN * 2 + j]);
                flag = 1;
            }
        }
    }

    return 0;
}

