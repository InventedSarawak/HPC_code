#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__global__ void shared_print(int* data, size_t len) {
    __shared__ int sharedData[2];
    sharedData[len - threadIdx.x - 1] = data[threadIdx.x];
    __syncthreads();
    printf("Value read by thread %d: %d\n", threadIdx.x, sharedData[threadIdx.x]);
}

int main(int argc, char** argv) {
    int numBlocks = 1;
    int threadsPerBlock = 2;
    int value[] = {10, 20};
    size_t len = sizeof(value) / sizeof(value[0]);
    int *d_value;
    cudaMalloc(&d_value, len * sizeof(int));
    cudaMemcpy(d_value, value, len * sizeof(int), cudaMemcpyHostToDevice);
    shared_print<<<numBlocks, threadsPerBlock>>>(d_value, len);
    cudaDeviceSynchronize();
    return 0;
}
