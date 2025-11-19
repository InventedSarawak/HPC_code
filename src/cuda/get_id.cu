#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__global__ void print_id() {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("thread id: %d, block id: %d, global id: %d\n", tid, bid, gid);
}

int main() {
    int numBlocks = 3;
    int threadsPerBlock = 4;
    print_id<<<numBlocks, threadsPerBlock>>>();
    cudaDeviceSynchronize();
    return 0;
}