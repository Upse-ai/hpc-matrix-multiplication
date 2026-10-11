#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = (argc > 1) ? atoi(argv[1]) : 1024;

    /* Each process gets rows_per_proc rows of A and C */
    int rows_per_proc = N / size;

    double *A       = NULL;
    double *B       = malloc(N * N * sizeof(double));
    double *C       = NULL;
    double *A_local = malloc(rows_per_proc * N * sizeof(double));
    double *C_local = malloc(rows_per_proc * N * sizeof(double));

    if (rank == 0) {
        A = malloc(N * N * sizeof(double));
        C = malloc(N * N * sizeof(double));

        for (int i = 0; i < N * N; i++) {
            A[i] = (double)i / (N * N);
            B[i] = (double)(N * N - i) / (N * N);
        }
    }

    /* Broadcast B to all processes */
    double t_bcast_start = MPI_Wtime();
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double t_bcast = MPI_Wtime() - t_bcast_start;

    /* Scatter rows of A to all processes */
    double t_scatter_start = MPI_Wtime();
    MPI_Scatter(A, rows_per_proc * N, MPI_DOUBLE,
                A_local, rows_per_proc * N, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    double t_scatter = MPI_Wtime() - t_scatter_start;

    /* Each process computes its rows of C using IKJ order */
    for (int i = 0; i < rows_per_proc * N; i++)
        C_local[i] = 0.0;

    double t_start = MPI_Wtime();

    for (int i = 0; i < rows_per_proc; i++)
        for (int k = 0; k < N; k++) {
            double a_ik = A_local[i * N + k];
            for (int j = 0; j < N; j++)
                C_local[i * N + j] += a_ik * B[k * N + j];
        }

    double t_compute = MPI_Wtime() - t_start;

    /* Gather C from all processes back to root */
    double t_gather_start = MPI_Wtime();
    MPI_Gather(C_local, rows_per_proc * N, MPI_DOUBLE,
               C, rows_per_proc * N, MPI_DOUBLE,
               0, MPI_COMM_WORLD);
    double t_gather = MPI_Wtime() - t_gather_start;

    if (rank == 0) {
        double checksum = 0.0;
        for (int i = 0; i < N * N; i++)
            checksum += C[i];

        printf("Processes = %d\n", size);
        printf("N         = %d\n", N);
        printf("Bcast     = %.4f s\n", t_bcast);
        printf("Scatter   = %.4f s\n", t_scatter);
        printf("Compute   = %.4f s\n", t_compute);
        printf("Gather    = %.4f s\n", t_gather);
        printf("Total     = %.4f s\n", t_bcast + t_scatter + t_compute + t_gather);
        printf("Checksum  = %.6f\n", checksum);

        free(A);
        free(C);
    }

    free(B);
    free(A_local);
    free(C_local);

    MPI_Finalize();
    return 0;
}
