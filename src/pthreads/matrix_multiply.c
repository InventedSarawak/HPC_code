#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define N 4 // size of NxN matrices (can change)

int A[N][N], B[N][N], C[N][N]; // matrices

typedef struct {
    int row; // row of C to compute
} thread_arg_t;

void *matrix_multiply(void *arg) {
    thread_arg_t *targ = (thread_arg_t *)arg;
    int i = targ->row;

    for (int j = 0; j < N; j++) {
        C[i][j] = 0;
        for (int k = 0; k < N; k++) {
            C[i][j] += A[i][k] * B[k][j];
        }
    }

    pthread_exit(NULL);
}

int main() {
    pthread_t threads[N];
    thread_arg_t args[N];

    // Initialize matrices with some values
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i][j] = rand() % 10;
            B[i][j] = rand() % 10;
        }

    // Create threads to compute each row
    for (int i = 0; i < N; i++) {
        args[i].row = i;
        pthread_create(&threads[i], NULL, matrix_multiply, &args[i]);
    }

    // Wait for threads to finish
    for (int i = 0; i < N; i++) {
        pthread_join(threads[i], NULL);
    }

    // Print matrices
    printf("Matrix A:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            printf("%d ", A[i][j]);
        printf("\n");
    }

    printf("\nMatrix B:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            printf("%d ", B[i][j]);
        printf("\n");
    }

    printf("\nResult Matrix C = A x B:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            printf("%d ", C[i][j]);
        printf("\n");
    }

    return 0;
}
