# MPI Functions and Usage

## Basic MPI Setup

## 1. `MPI_Init()`

-   **Usage:** Initializes the MPI execution environment.
-   **Example:**
    ```c
    #include <mpi.h>
    int main(int argc, char *argv[]) {
        MPI_Init(&argc, &argv);
        // MPI code here
        MPI_Finalize();
        return 0;
    }
    ```

## 2. `MPI_Finalize()`

-   **Usage:** Terminates the MPI execution environment.
-   **Example:**
    ```c
    MPI_Finalize();
    ```

## 3. `MPI_Comm_rank()`

-   **Usage:** Returns the rank (process ID) within a communicator.
-   **Example:**
    ```c
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("I am process %d\n", rank);
    ```

## 4. `MPI_Comm_size()`

-   **Usage:** Returns the total number of processes in a communicator.
-   **Example:**
    ```c
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    printf("Total processes: %d\n", size);
    ```

## 5. `MPI_COMM_WORLD`

-   **Usage:** Default communicator containing all processes.
-   **Example:**
    ```c
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    ```

## Point-to-Point Communication

## 6. `MPI_Send()`

-   **Usage:** Sends data from one process to another (blocking).
-   **Example:**
    ```c
    int data = 42;
    int dest_rank = 1;
    int tag = 100;
    MPI_Send(&data, 1, MPI_INT, dest_rank, tag, MPI_COMM_WORLD);
    ```

## 7. `MPI_Recv()`

-   **Usage:** Receives data from another process (blocking).
-   **Example:**
    ```c
    int data;
    int source_rank = 0;
    int tag = 100;
    MPI_Status status;
    MPI_Recv(&data, 1, MPI_INT, source_rank, tag, MPI_COMM_WORLD, &status);
    ```

## 8. `MPI_STATUS_IGNORE`

-   **Usage:** Used when status information is not needed in receive operations.
-   **Example:**
    ```c
    MPI_Recv(&data, 1, MPI_INT, 0, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    ```

## 9. `MPI_Status`

-   **Usage:** Structure containing information about received messages.
-   **Example:**
    ```c
    MPI_Status status;
    MPI_Recv(&data, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    int source = status.MPI_SOURCE;
    int tag = status.MPI_TAG;
    ```

## 10. `MPI_ANY_SOURCE`

-   **Usage:** Receive from any source process.
-   **Example:**
    ```c
    MPI_Recv(&data, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
    ```

## 11. `MPI_ANY_TAG`

-   **Usage:** Receive messages with any tag.
-   **Example:**
    ```c
    MPI_Recv(&data, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    ```

## Non-blocking Communication

## 12. `MPI_Isend()`

-   **Usage:** Non-blocking send operation.
-   **Example:**
    ```c
    int data = 42;
    MPI_Request request;
    MPI_Isend(&data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &request);
    // Do other work...
    MPI_Wait(&request, MPI_STATUS_IGNORE);
    ```

## 13. `MPI_Irecv()`

-   **Usage:** Non-blocking receive operation.
-   **Example:**
    ```c
    int data;
    MPI_Request request;
    MPI_Irecv(&data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &request);
    // Do other work...
    MPI_Wait(&request, MPI_STATUS_IGNORE);
    ```

## 14. `MPI_Request`

-   **Usage:** Handle for non-blocking operations.
-   **Example:**
    ```c
    MPI_Request request;
    MPI_Isend(&data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &request);
    ```

## 15. `MPI_Wait()`

-   **Usage:** Waits for a non-blocking operation to complete.
-   **Example:**
    ```c
    MPI_Status status;
    MPI_Wait(&request, &status);
    ```

## 16. `MPI_Waitany()`

-   **Usage:** Waits for any one of multiple operations to complete.
-   **Example:**
    ```c
    MPI_Request requests[4];
    int index;
    MPI_Status status;
    MPI_Waitany(4, requests, &index, &status);
    printf("Request %d completed\n", index);
    ```

## Collective Communication

## 17. `MPI_Bcast()`

-   **Usage:** Broadcasts data from one process to all others.
-   **Example:**
    ```c
    int data;
    int root = 0;
    if (rank == root) data = 12345;
    MPI_Bcast(&data, 1, MPI_INT, root, MPI_COMM_WORLD);
    ```

## 18. `MPI_Scatter()`

-   **Usage:** Distributes data from one process to all others.
-   **Example:**
    ```c
    int sendbuf[4] = {1, 2, 3, 4};  // Only meaningful at root
    int recvbuf;
    MPI_Scatter(sendbuf, 1, MPI_INT, &recvbuf, 1, MPI_INT, 0, MPI_COMM_WORLD);
    ```

## 19. `MPI_Gather()`

-   **Usage:** Collects data from all processes to one process.
-   **Example:**
    ```c
    int sendbuf = rank;
    int recvbuf[4];  // Only meaningful at root
    MPI_Gather(&sendbuf, 1, MPI_INT, recvbuf, 1, MPI_INT, 0, MPI_COMM_WORLD);
    ```

## 20. `MPI_Reduce()`

-   **Usage:** Performs reduction operation on data from all processes.
-   **Example:**
    ```c
    int local_sum = rank;
    int global_sum;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    ```

## 21. `MPI_Allreduce()`

-   **Usage:** Performs reduction and broadcasts result to all processes.
-   **Example:**
    ```c
    int local_sum = rank;
    int global_sum;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    ```

## Reduction Operations

## 22. `MPI_SUM`

-   **Usage:** Addition reduction operation.
-   **Example:**
    ```c
    MPI_Reduce(&local_val, &result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    ```

## 23. `MPI_MAX`

-   **Usage:** Maximum value reduction operation.
-   **Example:**
    ```c
    MPI_Reduce(&local_val, &result, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    ```

## 24. `MPI_MIN`

-   **Usage:** Minimum value reduction operation.
-   **Example:**
    ```c
    MPI_Reduce(&local_val, &result, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    ```

## 25. `MPI_PROD`

-   **Usage:** Product reduction operation.
-   **Example:**
    ```c
    MPI_Reduce(&local_val, &result, 1, MPI_INT, MPI_PROD, 0, MPI_COMM_WORLD);
    ```

## Synchronization

## 26. `MPI_Barrier()`

-   **Usage:** Synchronizes all processes in a communicator.
-   **Example:**
    ```c
    printf("Before barrier - Process %d\n", rank);
    MPI_Barrier(MPI_COMM_WORLD);
    printf("After barrier - Process %d\n", rank);
    ```

## MPI Data Types

## 27. `MPI_INT`

-   **Usage:** MPI data type for integers.
-   **Example:**
    ```c
    int data = 42;
    MPI_Send(&data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    ```

## 28. `MPI_FLOAT`

-   **Usage:** MPI data type for single-precision floating point.
-   **Example:**
    ```c
    float data = 3.14f;
    MPI_Send(&data, 1, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
    ```

## 29. `MPI_DOUBLE`

-   **Usage:** MPI data type for double-precision floating point.
-   **Example:**
    ```c
    double data = 3.14159;
    MPI_Send(&data, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
    ```

## 30. `MPI_CHAR`

-   **Usage:** MPI data type for characters.
-   **Example:**
    ```c
    char message[64] = "Hello";
    MPI_Send(message, 64, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
    ```

## Custom Data Types

## 31. `MPI_Datatype`

-   **Usage:** Handle for MPI data types.
-   **Example:**
    ```c
    MPI_Datatype custom_type;
    ```

## 32. `MPI_Type_create_struct()`

-   **Usage:** Creates a structured data type from multiple fields.
-   **Example:**

    ```c
    typedef struct {
        int id;
        float value;
    } Data;

    int block_lengths[2] = {1, 1};
    MPI_Datatype types[2] = {MPI_INT, MPI_FLOAT};
    MPI_Aint offsets[2];
    offsets[0] = offsetof(Data, id);
    offsets[1] = offsetof(Data, value);

    MPI_Datatype MPI_DATA;
    MPI_Type_create_struct(2, block_lengths, offsets, types, &MPI_DATA);
    ```

## 33. `MPI_Type_commit()`

-   **Usage:** Commits a data type for use in communication.
-   **Example:**
    ```c
    MPI_Type_commit(&custom_type);
    ```

## 34. `MPI_Type_free()`

-   **Usage:** Frees a user-defined data type.
-   **Example:**
    ```c
    MPI_Type_free(&custom_type);
    ```

## 35. `MPI_Aint`

-   **Usage:** Address integer type for byte offsets.
-   **Example:**
    ```c
    MPI_Aint offsets[2];
    offsets[0] = offsetof(struct_type, field1);
    offsets[1] = offsetof(struct_type, field2);
    ```

## 36. `offsetof()`

-   **Usage:** C macro to get byte offset of structure member.
-   **Example:**
    ```c
    #include <stddef.h>
    MPI_Aint offset = offsetof(Data, value);
    ```

## Error Handling

## 37. Error checking

-   **Usage:** Most MPI functions return error codes.
-   **Example:**
    ```c
    int error = MPI_Send(&data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    if (error != MPI_SUCCESS) {
        printf("MPI_Send failed\n");
    }
    ```

## Timing

## 38. `MPI_Wtime()`

-   **Usage:** Returns elapsed wall-clock time in seconds.
-   **Example:**
    ```c
    double start_time = MPI_Wtime();
    // ... computation ...
    double end_time = MPI_Wtime();
    printf("Elapsed time: %f seconds\n", end_time - start_time);
    ```

## Process Groups and Communicators

## 39. Communicator concepts

-   **Usage:** Groups of processes that can communicate.
-   **Example:**
    ```c
    MPI_Comm new_comm;
    // Split communicator based on color and key
    MPI_Comm_split(MPI_COMM_WORLD, color, key, &new_comm);
    ```

## Compilation and Execution

## 40. `mpicc` compiler

-   **Usage:** MPI C compiler wrapper.
-   **Example:**
    ```bash
    mpicc -o program program.c
    ```

## 41. `mpirun` or `mpiexec`

-   **Usage:** Execute MPI programs with multiple processes.
-   **Example:**
    ```bash
    mpirun -np 4 ./program          # Run with 4 processes
    mpiexec -n 4 ./program          # Alternative syntax
    ```

## Common Patterns

## 42. Master-Worker pattern

-   **Usage:** One process coordinates work distribution.
-   **Example:**
    ```c
    #define MASTER 0
    if (rank == MASTER) {
        // Distribute work and collect results
        for (int worker = 1; worker < size; worker++) {
            MPI_Send(&work_data, 1, MPI_INT, worker, 0, MPI_COMM_WORLD);
        }
    } else {
        // Worker processes
        MPI_Recv(&work_data, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // Process work_data
        MPI_Send(&result, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
    }
    ```

## 43. Ring communication

-   **Usage:** Each process communicates with its neighbors.
-   **Example:**
    ```c
    int next = (rank + 1) % size;
    int prev = (rank - 1 + size) % size;
    MPI_Send(&data, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
    MPI_Recv(&data, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    ```

---

Refer to the official MPI documentation for advanced features like one-sided communication, parallel I/O, and dynamic process management.
