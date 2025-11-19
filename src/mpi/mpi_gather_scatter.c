#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    const int N = 4;
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int sendbuf[4] = {1, 2, 3, 4};
    int gatherbuf[N];
    int sum;

    MPI_Scatter(sendbuf, 1, MPI_INT, &gatherbuf[rank], 1, MPI_INT, 0, MPI_COMM_WORLD);
    int square = gatherbuf[rank] * gatherbuf[rank];
    // MPI_Gather(&square, 1, MPI_INT, gatherbuf, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Reduce(&square, &sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        // for (int i = 0; i < N; i++)
        //     printf("Square[%d] = %d\n", i, gatherbuf[i]);
        printf("Sum %d\n", sum);
    }
    MPI_Finalize();
}