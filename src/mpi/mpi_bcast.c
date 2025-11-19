#include <mpi.h>
#include <stdio.h>
#include <string.h>

#define MSGSIZE 64

int main(int argc, char **argv) {
    int rank, size;
    int value;
    char msg[MSGSIZE];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        value = 12345;
        snprintf(msg, MSGSIZE, "Hello from root (rank %d)", rank);
    } else {
        value = 0;
        msg[0] = '\0';
    }

    /* broadcast an int and a fixed-size char buffer from root (0) to all ranks */
    MPI_Bcast(&value, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(msg, MSGSIZE, MPI_CHAR, 0, MPI_COMM_WORLD);

    printf("rank %d/%d: received value=%d msg=\"%s\"\n", rank, size, value, msg);

    MPI_Finalize();
    return 0;
}