#include <mpi.h>
#include <stdio.h>

#define MASTER 0
#define SLAVE 1

/*
Write MPI program to print hello from each process in the comm world
Write MPI program to do point-to-point communication:

Master process:     send msg with tag 1134
Master process:     wait for msg with tag 4114
Slave process:      receive msg with tag 1134
Slave process:      send back msg with tag 4114

Write MPI program for sending and receiving
Process 1 sends 4 characters to process 0
Process 0 receives an integer (4 bytes)
*/

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int send_a = 69, recv_a, send_b = 67, recv_b; // a means MASTER, b means SLAVE

    if (rank == MASTER) {
        MPI_Send(&send_a, 1, MPI_INT, SLAVE, 1134, MPI_COMM_WORLD);
        MPI_Recv(&recv_a, 1, MPI_INT, SLAVE, 4114, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("MASTER - sent: %d, recvd: %d\n", send_a, recv_a);
    } else {
        MPI_Send(&send_b, 1, MPI_INT, MASTER, 4114, MPI_COMM_WORLD);
        MPI_Recv(&recv_b, 1, MPI_INT, MASTER, 1134, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("SLAVE - sent: %d, recvd: %d\n", send_b, recv_b);
    }
    MPI_Finalize();
    return 0;
}