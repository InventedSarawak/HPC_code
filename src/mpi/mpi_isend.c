#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MASTER_TASK 0 // I take data from all other tasks
#define NUM_DATA_ITEMS 1

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size == 1) {
        MPI_Finalize();
        return 0;
    }

    if (rank == MASTER_TASK) {
        int i;
        int nworkers = size - 1;
        int *recvBuffer = malloc(nworkers * sizeof(int));
        MPI_Request *reqs = malloc(nworkers * sizeof(MPI_Request));
        MPI_Status status;

        /* post non-blocking receives for all workers */
        for (i = 0; i < nworkers; ++i) {
            MPI_Irecv(&recvBuffer[i], NUM_DATA_ITEMS, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &reqs[i]);
        }

        /* wait for incoming messages and process as they arrive */
        for (i = 0; i < nworkers; ++i) {
            int idx;
            MPI_Waitany(nworkers, reqs, &idx, &status); /* idx is request index */
            int source = status.MPI_SOURCE;
            printf("Master: received %d from task %d (buffer idx %d)\n", recvBuffer[idx], source, idx);
        }

        free(recvBuffer);
        free(reqs);
    } else {
        MPI_Request req;
        MPI_Status status;
        /* non-blocking send from worker to master */
        MPI_Isend(&rank, NUM_DATA_ITEMS, MPI_INT, MASTER_TASK, 0, MPI_COMM_WORLD, &req);

        /* do other work here if needed */

        /* ensure send completion before finalizing */
        MPI_Wait(&req, &status);
    }

    MPI_Finalize();
    return 0;
}