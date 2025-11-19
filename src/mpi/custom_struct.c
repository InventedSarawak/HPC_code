#include <mpi.h>
#include <stdio.h>

typedef struct {
    int id;
    float value;
} Data;

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // --- 1) Describe block lengths of each field ---
    int block_lengths[2] = {1, 1};

    // --- 2) Describe the types of each field ---
    MPI_Datatype types[2] = {MPI_INT, MPI_FLOAT};

    // --- 3) Describe the *byte offsets* inside the struct ---
    MPI_Aint offsets[2];
    offsets[0] = offsetof(Data, id);    // offset of int id
    offsets[1] = offsetof(Data, value); // offset of float value

    // --- 4) Create the MPI datatype ---
    MPI_Datatype MPI_DATA;
    MPI_Type_create_struct(2, block_lengths, offsets, types, &MPI_DATA);
    MPI_Type_commit(&MPI_DATA);

    // --- Example usage ---
    if (rank == 0) {
        Data d;
        d.id = 42;
        d.value = 3.14f;

        MPI_Send(&d, 1, MPI_DATA, 1, 0, MPI_COMM_WORLD);
        printf("Rank 0 sent struct: {id=%d, value=%f}\n", d.id, d.value);
    }

    if (rank == 1) {
        Data d;
        MPI_Recv(&d, 1, MPI_DATA, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Rank 1 received struct: {id=%d, value=%f}\n", d.id, d.value);
    }

    MPI_Type_free(&MPI_DATA);
    MPI_Finalize();
    return 0;
}
