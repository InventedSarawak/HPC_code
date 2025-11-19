#include <mpi.h>
#include <stdio.h>
#define MASTER_TASK 0 // I take data from all other tasks
#define NUM_DATA_ITEMS 1

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == MASTER_TASK) {
        int i, recvBuffer;
        for (i = 1; i < size; i++) {
            MPI_Recv(&recvBuffer, NUM_DATA_ITEMS, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("I, %d, just heard from task %d.\n", MASTER_TASK, recvBuffer);
        }
    } else {
        MPI_Send(&rank, NUM_DATA_ITEMS, MPI_INT, MASTER_TASK, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return 0;
}
