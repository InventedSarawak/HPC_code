#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <iostream>

#define MAX_BLOCK_LENGTH 512

using namespace std;

__global__ void mul2d(int **A, int **B, int **C, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len*len) return;
    int i = idx / len;
    int j = idx % len;
    int sum = 0;
    for (int k = 0; k < len; k++) {
        sum += A[i][k] * B[k][j];
    }
    C[j][i] = sum;
    __syncthreads();
}

int **generate_matrix(int len, bool generateRandomData) {
    if (len <= 0) return nullptr;
    int **arr = (int **)malloc(len * sizeof(int *));
    if (!arr) return nullptr;
    for (int i = 0; i < len; i++) {
        arr[i] = (int *)malloc(len * sizeof(int));
        if (!arr[i]) {
            // free previously allocated rows
            for (int k = 0; k < i; ++k) free(arr[k]);
            free(arr);
            return nullptr;
        }
    }
    if (generateRandomData) {
        for (int i = 0; i < len; i++) {
            for (int j = 0; j < len; j++) {
                arr[i][j] = rand() % 5;
            }
        }
    } else {
        for (int i = 0; i < len; ++i)
            for (int j = 0; j < len; ++j)
                arr[i][j] = 0;
    }
    return arr;
}

void print_matrix(int **arr, int len) {
    if (!arr) { printf("(null matrix)\n"); return; }
    for(int i = 0; i < len; i ++) {
        for(int j = 0; j < len; j ++) {
            printf("%d ", arr[i][j]);
        }
        printf("\n");
    }
}

int **cudaAllocAndCopyMatrix2D(int **h_mat, int len) {
    int **d_mat;
    int **h_rows = (int **)malloc(len * sizeof(int *));
    
    // allocate device array of row pointers
    cudaMalloc(&d_mat, len * sizeof(int*));

    for (int i = 0; i < len; i++) {
        int *d_row;
        cudaMalloc(&d_row, len * sizeof(int));                 // allocate row
        cudaMemcpy(d_row, h_mat[i], len * sizeof(int), cudaMemcpyHostToDevice);
        h_rows[i] = d_row;                                     // store row pointer
    }

    // copy row pointer array to device
    cudaMemcpy(d_mat, h_rows, len * sizeof(int*), cudaMemcpyHostToDevice);

    free(h_rows);
    return d_mat;
}

void cudaFreeMatrix2D(int **d_mat, int len) {
    int **h_rows = (int**)malloc(len * sizeof(int*));
    cudaMemcpy(h_rows, d_mat, len * sizeof(int*), cudaMemcpyDeviceToHost);

    for (int i = 0; i < len; i++)
        cudaFree(h_rows[i]);   // free each row

    cudaFree(d_mat);           // free pointer table
    free(h_rows);
}

bool cudaCopyMatrix2D(int **d_mat, int **h_mat, int len) {
    int **h_rows = (int**)malloc(len * sizeof(int*));
    cudaMemcpy(h_rows, d_mat, len * sizeof(int*), cudaMemcpyDeviceToHost);

    for (int i = 0; i < len; i++) {
        cudaMemcpy(h_mat[i], h_rows[i], len * sizeof(int), cudaMemcpyDeviceToHost);
    }

    free(h_rows);
    return true;
}


void cudaFreeMatrix(int *d_mat) {
    if (d_mat) cudaFree(d_mat);
}

void free_matrix(int **arr, int len) {
    if (!arr) return;
    for (int i = 0; i < len; ++i) free(arr[i]);
    free(arr);
}

void multiply_matrix(int **A, int **B, int **C, int len) {
    if (len <= 0) return;
    size_t total = (size_t)len * len;
    int threads = MAX_BLOCK_LENGTH;
    int blocks = (int)((total + threads - 1) / threads);

    int **dev_A = nullptr, **dev_B = nullptr, **dev_C = nullptr;

    dev_A = cudaAllocAndCopyMatrix2D(A, len);
    if (!dev_A) {
        fprintf(stderr, "Failed to allocate/copy dev_A\n");
        return;
    }
    dev_B = cudaAllocAndCopyMatrix2D(B, len);
    if (!dev_B) {
        fprintf(stderr, "Failed to allocate/copy dev_B\n");
        cudaFreeMatrix2D(dev_A, len);
        return;
    }

    dev_C = cudaAllocAndCopyMatrix2D(C, len);
    if (!dev_C) {
        fprintf(stderr, "Failed to allocate dev_C\n");
        cudaFreeMatrix2D(dev_A, len);
        cudaFreeMatrix2D(dev_B, len);
    return;
}

    mul2d<<<blocks, threads>>>(dev_A, dev_B, dev_C, len);
    cudaDeviceSynchronize();

    if (!cudaCopyMatrix2D(dev_C, C, len)) {
        fprintf(stderr, "Failed to copy dev_C to host\n");
    }

    cudaFreeMatrix2D(dev_A, len);
    cudaFreeMatrix2D(dev_B, len);
    cudaFreeMatrix2D(dev_C, len);
}

int main(int argc, char **argv) {
    srand(time(0));
    int len = 3; // default size
    if (argc > 1) len = atoi(argv[1]);
    if (len <= 0) {
        printf("Invalid matrix size: %d\n", len);
        return 1;
    }

    int **A = generate_matrix(len, true);
    int **B = generate_matrix(len, true);
    int **C = generate_matrix(len, false);

    if (!A || !B || !C) {
        printf("Failed to allocate matrices\n");
        free_matrix(A, len);
        free_matrix(B, len);
        free_matrix(C, len);
        return 1;
    }

    printf("Matrix A:\n"); print_matrix(A, len);
    printf("Matrix B:\n"); print_matrix(B, len);
    multiply_matrix(A, B, C, len);
    printf("Matrix C:\n"); print_matrix(C, len);

    free_matrix(A, len);
    free_matrix(B, len);
    free_matrix(C, len);
    return 0;
}
