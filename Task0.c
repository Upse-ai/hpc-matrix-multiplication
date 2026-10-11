#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double get_time(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char *argv[])
{
    int N = (argc > 1) ? atoi(argv[1]) : 1024;

    double *A = malloc(N * N * sizeof(double));
    double *B = malloc(N * N * sizeof(double));
    double *C = malloc(N * N * sizeof(double));

    if (!A || !B || !C) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    for (int i = 0; i < N * N; i++) {
        A[i] = (double)i / (N * N);
        B[i] = (double)(N * N - i) / (N * N);
    }

    double t_start = get_time();

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }

    double t_end = get_time();

    double checksum = 0.0;
    for (int i = 0; i < N * N; i++)
        checksum += C[i];

    printf("N        = %d\n", N);
    printf("Time     = %.4f s\n", t_end - t_start);
    printf("Checksum = %.6f\n", checksum);

    free(A);
    free(B);
    free(C);
    return 0;
}
