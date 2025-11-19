#include <mpi.h>
#include <stdio.h>
int main(int argc, char *argv[]) {
    int rank, size;
    char buf[64];
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Stdout has some issues regarding the ordering of prints from different processes.
    // Since it is an asynchronous system call, the order in which different processes print to
    // stdout is unpredictable.
    puts("Hello, World!");
    MPI_Barrier(MPI_COMM_WORLD);
    snprintf(buf, sizeof buf, "I am %d of %d", rank, size);
    puts(buf);
    MPI_Finalize();
    return 0;
}
