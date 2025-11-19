#define _POSIX_C_SOURCE 199309L
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct mat2d {
    int n;
    int **mat;
} mat2d;

typedef struct mat1d {
    int n;
    int *mat;
} mat1d;

mat1d *multiply_matrix_2d_1d_sequential(mat2d *A, mat1d *x) {
    mat1d *y = (mat1d *)malloc(sizeof(mat1d));
    y->n = A->n;
    y->mat = (int *)malloc(A->n * sizeof(int));
    for (int i = 0; i < A->n; i++) {
        y->mat[i] = 0;
        for (int j = 0; j < A->n; j++) {
            y->mat[i] += A->mat[i][j] * x->mat[j];
        }
    }
    return y;
}

mat1d *multiply_matrix_2d_1d_row_partition(mat2d *A, mat1d *x) {
    mat1d *y = (mat1d *)malloc(sizeof(mat1d));
    y->n = A->n;
    y->mat = (int *)malloc(A->n * sizeof(int));
#pragma omp parallel for
    for (int i = 0; i < A->n; i++) {
        y->mat[i] = 0;
        for (int j = 0; j < A->n; j++) {
            y->mat[i] += A->mat[i][j] * x->mat[j];
        }
    }
    return y;
}

mat1d *multiply_matrix_2d_1d_2d_partition(mat2d *A, mat1d *x) {
    mat1d *y = (mat1d *)malloc(sizeof(mat1d));
    y->n = A->n;
    y->mat = (int *)malloc(A->n * sizeof(int));
    int len = A->n;
#pragma omp parallel for shared(A, x, y, len)
    for (int i = 0; i < len; i++) {
        int sum = 0;
#pragma omp simd reduction(+ : sum)
        for (int j = 0; j < len; j++) {
            sum += A->mat[i][j] * x->mat[j];
        }
        y->mat[i] = sum;
    }
    return y;
}

mat2d *generate_matrix_2d(int n) {
    mat2d *A = (mat2d *)malloc(sizeof(mat2d));
    A->n = n;
    A->mat = (int **)malloc(n * sizeof(int *));
    for (int i = 0; i < n; i++) {
        A->mat[i] = (int *)malloc(n * sizeof(int));
        for (int j = 0; j < n; j++) {
            A->mat[i][j] = rand() % 100;
        }
    }
    return A;
}

mat1d *generate_matrix_1d(int n) {
    mat1d *x = (mat1d *)malloc(sizeof(mat1d));
    x->n = n;
    x->mat = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        x->mat[i] = rand() % 100;
    }
    return x;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <max test case length> <algorithm>\n", argv[0]);
        printf("Algorithm options: seq | omp | 2d\n");
        return 1;
    }
    int N = atoi(argv[1]);
    if (N <= 0) {
        printf("Invalid test case length: %s\n", argv[1]);
        return 1;
    }
    char *alg = argv[2];
    srand(time(NULL));

    int min_size = 500;
    if (N < min_size) {
        printf("Max test case length must be at least %d\n", min_size);
        return 1;
    }
    long *times_ns = (long *)malloc((N - min_size + 1) * sizeof(long));

    for (int n = min_size; n <= N; n++) {
        mat2d *A = generate_matrix_2d(n);
        mat1d *x = generate_matrix_1d(n);

        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        mat1d *y = NULL;
        if (strcmp(alg, "seq") == 0)
            y = multiply_matrix_2d_1d_sequential(A, x);
        else if (strcmp(alg, "row") == 0)
            y = multiply_matrix_2d_1d_row_partition(A, x);
        else if (strcmp(alg, "2d") == 0)
            y = multiply_matrix_2d_1d_2d_partition(A, x);
        else {
            printf("Unknown algorithm: %s\n", alg);
            free(x->mat);
            free(x);
            for (int i = 0; i < n; i++)
                free(A->mat[i]);
            free(A->mat);
            free(A);
            free(times_ns);
            return 1;
        }

        clock_gettime(CLOCK_MONOTONIC, &end);
        long elapsed_ns = (end.tv_sec - start.tv_sec) * 1000000000L + (end.tv_nsec - start.tv_nsec);
        times_ns[n - min_size] = elapsed_ns;

        for (int i = 0; i < n; i++) {
            free(A->mat[i]);
        }
        free(A->mat);
        free(A);
        free(x->mat);
        free(x);
        free(y->mat);
        free(y);
    }

    for (int i = 0; i <= N - min_size; i++) {
        printf("%ld\n", times_ns[i]);
    }
    free(times_ns);
    return 0;
}