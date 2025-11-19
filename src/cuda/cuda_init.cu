#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define SMALL_N 10
#define LARGE_N (1 << 20)
#define THREADS_PER_BLOCK 256

__global__ void testKernel() { printf("Block ID: %d, Thread ID: %d\n", blockIdx.x, threadIdx.x); }

__global__ void printThreads() {
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Hello from thread %d in block %d (Global ID: %d)\n", threadIdx.x, blockIdx.x, threadId);
}

__global__ void vectorAddSmall(int *a, int *b, int *c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < SMALL_N)
        c[tid] = a[tid] + b[tid];
}

__global__ void vectorAddLarge(int *a, int *b, int *c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        c[tid] = a[tid] + b[tid];
}

void demonstrateBasicKernel() {
    printf("\n=== Basic Kernel Demonstration ===\n");
    int numBlocks = 2;
    int threadsPerBlock = 32;

    testKernel<<<numBlocks, threadsPerBlock>>>();
    cudaDeviceSynchronize();
    printf("Basic kernel execution completed.\n");
}

void demonstrateThreadPrinting() {
    printf("\n=== Thread Printing Demonstration ===\n");
    int threadsPerBlock = 8;
    int numBlocks = 2;

    printThreads<<<numBlocks, threadsPerBlock>>>();
    cudaDeviceSynchronize();
    printf("Thread printing demonstration completed.\n");
}

void demonstrateSmallVectorAddition() {
    printf("\n=== Small Vector Addition Demonstration ===\n");

    int a[SMALL_N], b[SMALL_N], c[SMALL_N];
    int *dev_a, *dev_b, *dev_c;

    cudaError_t err;
    err = cudaMalloc((void **)&dev_a, SMALL_N * sizeof(int));
    if (err != cudaSuccess) {
        printf("CUDA malloc failed!\n");
        return;
    }

    err = cudaMalloc((void **)&dev_b, SMALL_N * sizeof(int));
    if (err != cudaSuccess) {
        printf("CUDA malloc failed!\n");
        return;
    }

    err = cudaMalloc((void **)&dev_c, SMALL_N * sizeof(int));
    if (err != cudaSuccess) {
        printf("CUDA malloc failed!\n");
        return;
    }

    // Generating "random" values from here
    for (int i = 0; i < SMALL_N; i++) {
        a[i] = -i;
        b[i] = i * i;
    }

    cudaMemcpy(dev_a, a, SMALL_N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, SMALL_N * sizeof(int), cudaMemcpyHostToDevice);

    vectorAddSmall<<<1, SMALL_N>>>(dev_a, dev_b, dev_c);

    cudaMemcpy(c, dev_c, SMALL_N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Small vector addition results:\n");
    for (int i = 0; i < SMALL_N; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}

void demonstrateLargeVectorAddition() {
    printf("\n=== Large Vector Addition Demonstration ===\n");

    int *a, *b, *c;
    int *dev_a, *dev_b, *dev_c;
    size_t size = LARGE_N * sizeof(int);

    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    for (int i = 0; i < LARGE_N; i++) {
        a[i] = i;
        b[i] = 2 * i;
    }

    cudaMalloc((void **)&dev_a, size);
    cudaMalloc((void **)&dev_b, size);
    cudaMalloc((void **)&dev_c, size);

    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

    // This finds the number of blocks using
    // blocks = ceil(large_n/threads_per_block)
    int blocks = (LARGE_N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    vectorAddLarge<<<blocks, THREADS_PER_BLOCK>>>(dev_a, dev_b, dev_c, LARGE_N);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);

    printf("Vector addition of %d elements completed in %f ms\n", LARGE_N, elapsedTime);
    printf("Printing first 10 additions:\n");

    for (int i = 0; i < 10; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    free(a);
    free(b);
    free(c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void queryDeviceInformation() {
    printf("\n=== Device Information ===\n");

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    printf("Number of CUDA Devices: %d\n\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        printf("  Device %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total Global Memory: %.2f GB\n",
               prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max Threads per Multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Max Grid Size: %d x %d x %d\n", prop.maxGridSize[0], prop.maxGridSize[1],
               prop.maxGridSize[2]);
        printf("  Max Threads per Dimension: %d x %d x %d\n", prop.maxThreadsDim[0],
               prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Shared Memory per Block: %.2f KB\n", prop.sharedMemPerBlock / 1024.0);
        printf("  Warp Size: %d\n", prop.warpSize);
        printf("  Memory Clock Rate: %.2f MHz\n", prop.memoryClockRate / 1000.0);
        printf("  GPU Clock Rate: %.2f MHz\n", prop.clockRate / 1000.0);
        printf("  L2 Cache Size: %d KB\n", prop.l2CacheSize / 1024);
        printf("  Max Memory Pitch: %.2f MB\n", prop.memPitch / (1024.0 * 1024.0));
        printf("  Concurrent Kernels: %s\n", prop.concurrentKernels ? "Yes" : "No");
        printf("  ECC Support: %s\n", prop.ECCEnabled ? "Yes" : "No");
        printf("\n");
    }
}

int main() {
    printf("CUDA Tutorial - Various Functions Demonstration\n");
    printf("==============================================\n");

    queryDeviceInformation();

    demonstrateBasicKernel();

    demonstrateThreadPrinting();

    demonstrateSmallVectorAddition();

    demonstrateLargeVectorAddition();

    printf("\n=== All demonstrations completed successfully! ===\n");

    return 0;
}