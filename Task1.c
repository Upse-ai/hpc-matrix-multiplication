#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

static void init_matrices(double *A, double *B, double *C, int N)
{
    for (int i = 0; i < N * N; i++) {
        A[i] = (double)i / (N * N);
        B[i] = (double)(N * N - i) / (N * N);
        C[i] = 0.0;
    }
}

static double checksum(double *C, int N)
{
    double s = 0.0;
    for (int i = 0; i < N * N; i++)
        s += C[i];
    return s;
}

/* IJK loop order */
static double run_ijk(double *A, double *B, double *C, int schedule_type, int N)
{
    memset(C, 0, N * N * sizeof(double));
    double t = omp_get_wtime();

    if (schedule_type == 0) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                double sum = 0.0;
                for (int k = 0; k < N; k++)
                    sum += A[i * N + k] * B[k * N + j];
                C[i * N + j] = sum;
            }
    } else if (schedule_type == 1) {
        #pragma omp parallel for schedule(dynamic, 16)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                double sum = 0.0;
                for (int k = 0; k < N; k++)
                    sum += A[i * N + k] * B[k * N + j];
                C[i * N + j] = sum;
            }
    } else {
        #pragma omp parallel for schedule(static) collapse(2)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                double sum = 0.0;
                for (int k = 0; k < N; k++)
                    sum += A[i * N + k] * B[k * N + j];
                C[i * N + j] = sum;
            }
    }

    return omp_get_wtime() - t;
}

/* IKJ loop order */
static double run_ikj(double *A, double *B, double *C, int schedule_type, int N)
{
    memset(C, 0, N * N * sizeof(double));
    double t = omp_get_wtime();

    if (schedule_type == 0) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++)
            for (int k = 0; k < N; k++) {
                double a_ik = A[i * N + k];
                for (int j = 0; j < N; j++)
                    C[i * N + j] += a_ik * B[k * N + j];
            }
    } else if (schedule_type == 1) {
        #pragma omp parallel for schedule(dynamic, 16)
        for (int i = 0; i < N; i++)
            for (int k = 0; k < N; k++) {
                double a_ik = A[i * N + k];
                for (int j = 0; j < N; j++)
                    C[i * N + j] += a_ik * B[k * N + j];
            }
    } else {
        #pragma omp parallel for schedule(guided)
        for (int i = 0; i < N; i++)
            for (int k = 0; k < N; k++) {
                double a_ik = A[i * N + k];
                for (int j = 0; j < N; j++)
                    C[i * N + j] += a_ik * B[k * N + j];
            }
    }

    return omp_get_wtime() - t;
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

    printf("Threads = %d, N = %d\n\n", omp_get_max_threads(), N);
    printf("%-35s %10s %15s\n", "Configuration", "Time (s)", "Checksum");
    printf("%-35s %10s %15s\n", "-----------------------------------", "----------", "---------------");

    double t;
    init_matrices(A, B, C, N);
    t = run_ijk(A, B, C, 0, N);
    printf("%-35s %10.4f %15.2f\n", "IJK  schedule(static)",             t, checksum(C, N));

    init_matrices(A, B, C, N);
    t = run_ijk(A, B, C, 1, N);
    printf("%-35s %10.4f %15.2f\n", "IJK  schedule(dynamic,16)",         t, checksum(C, N));

    init_matrices(A, B, C, N);
    t = run_ijk(A, B, C, 2, N);
    printf("%-35s %10.4f %15.2f\n", "IJK  schedule(static)+collapse(2)", t, checksum(C, N));

    init_matrices(A, B, C, N);
    t = run_ikj(A, B, C, 0, N);
    printf("%-35s %10.4f %15.2f\n", "IKJ  schedule(static)",             t, checksum(C, N));

    init_matrices(A, B, C, N);
    t = run_ikj(A, B, C, 1, N);
    printf("%-35s %10.4f %15.2f\n", "IKJ  schedule(dynamic,16)",         t, checksum(C, N));

    init_matrices(A, B, C, N);
    t = run_ikj(A, B, C, 2, N);
    printf("%-35s %10.4f %15.2f\n", "IKJ  schedule(guided)",             t, checksum(C, N));

    free(A);
    free(B);
    free(C);
    return 0;
}
