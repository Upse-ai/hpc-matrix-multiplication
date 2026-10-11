#include <stdio.h>
#include <stdlib.h>

#define THREADS_PER_BLOCK 128

/* Grid-stride loop: each thread computes multiple elements */
__global__ void matmul_kernel(double *A, double *B, double *C, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    int total_elements = n * n;

    for (int idx = tid; idx < total_elements; idx += total_threads) {
        int row = idx / n;
        int col = idx % n;
        double sum = 0.0;
        for (int k = 0; k < n; k++)
            sum += A[row * n + k] * B[k * n + col];
        C[idx] = sum;
    }
}

int main(int argc, char *argv[])
{
    int N      = (argc > 1) ? atoi(argv[1]) : 1024;
    int num_sm = (argc > 2) ? atoi(argv[2]) : 66;

    size_t bytes = (size_t)N * N * sizeof(double);

    /* Host memory */
    double *h_A = (double*)malloc(bytes);
    double *h_B = (double*)malloc(bytes);
    double *h_C = (double*)malloc(bytes);

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    for (int i = 0; i < N * N; i++) {
        h_A[i] = (double)i / (N * N);
        h_B[i] = (double)(N * N - i) / (N * N);
    }

    /* Device memory */
    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    /* Copy A and B from host to device */
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    /* GPU timer */
    cudaEvent_t t_start, t_end;
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);

    cudaEventRecord(t_start);
    matmul_kernel<<<num_sm, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, N);
    cudaEventRecord(t_end);
    cudaEventSynchronize(t_end);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t_start, t_end);

    /* Copy C from device back to host */
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    double checksum = 0.0;
    for (int i = 0; i < N * N; i++)
        checksum += h_C[i];

    printf("N        = %d\n", N);
    printf("SMs      = %d\n", num_sm);
    printf("Threads  = %d\n", num_sm * THREADS_PER_BLOCK);
    printf("Time     = %.4f s\n", ms / 1000.0);
    printf("Checksum = %.6f\n", checksum);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
