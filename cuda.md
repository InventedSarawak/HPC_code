# CUDA Functions and Usage

## Basic Kernel Launch

## 1. `__global__` function qualifier

-   **Usage:** Declares a function as a CUDA kernel that runs on the device (GPU).
-   **Example:**
    ```cuda
    __global__ void myKernel() {
        // kernel code
    }
    ```

## 2. `<<<blocks, threads>>>` execution configuration

-   **Usage:** Launches a kernel with specified grid and block dimensions.
-   **Example:**
    ```cuda
    int numBlocks = 2;
    int threadsPerBlock = 256;
    myKernel<<<numBlocks, threadsPerBlock>>>();
    ```

## 3. `<<<blocks, threads, sharedMemSize>>>` execution configuration

-   **Usage:** Launches a kernel with dynamic shared memory allocation.
-   **Example:**
    ```cuda
    myKernel<<<blocks, threads, 1024>>>();  // 1024 bytes of shared memory
    ```

## Built-in Variables

## 4. `threadIdx`

-   **Usage:** Built-in variable containing the thread index within a block.
-   **Example:**
    ```cuda
    __global__ void kernel() {
        int tid = threadIdx.x;  // x-dimension thread index
        printf("Thread ID: %d\n", tid);
    }
    ```

## 5. `blockIdx`

-   **Usage:** Built-in variable containing the block index within the grid.
-   **Example:**
    ```cuda
    __global__ void kernel() {
        int bid = blockIdx.x;  // x-dimension block index
        printf("Block ID: %d\n", bid);
    }
    ```

## 6. `blockDim`

-   **Usage:** Built-in variable containing the dimensions of the block.
-   **Example:**
    ```cuda
    __global__ void kernel() {
        int blockSize = blockDim.x;  // number of threads in block
        printf("Block size: %d\n", blockSize);
    }
    ```

## 7. `gridDim`

-   **Usage:** Built-in variable containing the dimensions of the grid.
-   **Example:**
    ```cuda
    __global__ void kernel() {
        int numBlocks = gridDim.x;  // number of blocks in grid
        printf("Grid size: %d\n", numBlocks);
    }
    ```

## Global Thread Index Calculation

## 8. Global thread index

-   **Usage:** Calculate unique global thread ID across all blocks.
-   **Example:**
    ```cuda
    __global__ void kernel() {
        int gid = blockIdx.x * blockDim.x + threadIdx.x;
        printf("Global ID: %d\n", gid);
    }
    ```

## Memory Management

## 9. `cudaMalloc()`

-   **Usage:** Allocates memory on the GPU device.
-   **Example:**
    ```cuda
    int *d_array;
    size_t size = 1024 * sizeof(int);
    cudaError_t err = cudaMalloc((void **)&d_array, size);
    ```

## 10. `cudaFree()`

-   **Usage:** Frees memory previously allocated on the GPU device.
-   **Example:**
    ```cuda
    cudaFree(d_array);
    ```

## 11. `cudaMemcpy()`

-   **Usage:** Copies data between host and device memory.
-   **Example:**
    ```cuda
    // Host to Device
    cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice);
    // Device to Host
    cudaMemcpy(h_array, d_array, size, cudaMemcpyDeviceToHost);
    // Device to Device
    cudaMemcpy(d_dest, d_src, size, cudaMemcpyDeviceToDevice);
    ```

## 12. `cudaMemcpy()` direction flags

-   **Usage:** Specifies the direction of memory transfer.
-   **Example:**
    ```cuda
    cudaMemcpyHostToDevice    // Host → Device
    cudaMemcpyDeviceToHost    // Device → Host
    cudaMemcpyDeviceToDevice  // Device → Device
    cudaMemcpyHostToHost      // Host → Host
    ```

## Synchronization

## 13. `cudaDeviceSynchronize()`

-   **Usage:** Blocks host execution until all device operations complete.
-   **Example:**
    ```cuda
    myKernel<<<blocks, threads>>>();
    cudaDeviceSynchronize();  // Wait for kernel to finish
    ```

## 14. `__syncthreads()`

-   **Usage:** Synchronizes all threads within a block (barrier).
-   **Example:**
    ```cuda
    __global__ void kernel() {
        // Phase 1 computation
        __syncthreads();  // Wait for all threads in block
        // Phase 2 computation
    }
    ```

## Shared Memory

## 15. `__shared__` memory qualifier

-   **Usage:** Declares variables in fast shared memory accessible by all threads in a block.
-   **Example:**
    ```cuda
    __global__ void kernel() {
        __shared__ int sharedData[256];
        sharedData[threadIdx.x] = threadIdx.x;
        __syncthreads();
    }
    ```

## 16. Dynamic shared memory

-   **Usage:** Allocate shared memory dynamically at kernel launch.
-   **Example:**
    ```cuda
    __global__ void kernel() {
        extern __shared__ int dynamicShared[];
        dynamicShared[threadIdx.x] = threadIdx.x;
    }
    // Launch with: kernel<<<blocks, threads, sharedMemSize>>>();
    ```

## Error Handling

## 17. `cudaError_t`

-   **Usage:** CUDA error type for checking API call success.
-   **Example:**
    ```cuda
    cudaError_t err = cudaMalloc(&d_ptr, size);
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    ```

## 18. `cudaGetErrorString()`

-   **Usage:** Returns a string description of a CUDA error.
-   **Example:**
    ```cuda
    cudaError_t err = cudaGetLastError();
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    ```

## 19. `cudaGetLastError()`

-   **Usage:** Returns the last error from a CUDA runtime call.
-   **Example:**
    ```cuda
    myKernel<<<blocks, threads>>>();
    cudaError_t err = cudaGetLastError();
    ```

## Device Properties and Information

## 20. `cudaGetDeviceCount()`

-   **Usage:** Returns the number of CUDA-capable devices.
-   **Example:**
    ```cuda
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Number of devices: %d\n", deviceCount);
    ```

## 21. `cudaGetDeviceProperties()`

-   **Usage:** Returns properties of a CUDA device.
-   **Example:**
    ```cuda
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);  // Device 0
    printf("Device name: %s\n", prop.name);
    printf("Total memory: %zu bytes\n", prop.totalGlobalMem);
    ```

## 22. `cudaDeviceProp` structure

-   **Usage:** Structure containing device properties.
-   **Example:**
    ```cuda
    cudaDeviceProp prop;
    // Key properties:
    prop.name                      // Device name
    prop.totalGlobalMem           // Total global memory
    prop.sharedMemPerBlock        // Shared memory per block
    prop.maxThreadsPerBlock       // Max threads per block
    prop.multiProcessorCount      // Number of SMs
    prop.warpSize                 // Warp size
    prop.major, prop.minor        // Compute capability
    ```

## Event-based Timing

## 23. `cudaEvent_t`

-   **Usage:** CUDA event type for timing and synchronization.
-   **Example:**
    ```cuda
    cudaEvent_t start, stop;
    ```

## 24. `cudaEventCreate()`

-   **Usage:** Creates a CUDA event.
-   **Example:**
    ```cuda
    cudaEvent_t event;
    cudaEventCreate(&event);
    ```

## 25. `cudaEventRecord()`

-   **Usage:** Records an event in a CUDA stream.
-   **Example:**
    ```cuda
    cudaEventRecord(start, 0);  // Record in default stream
    myKernel<<<blocks, threads>>>();
    cudaEventRecord(stop, 0);
    ```

## 26. `cudaEventSynchronize()`

-   **Usage:** Waits for an event to complete.
-   **Example:**
    ```cuda
    cudaEventSynchronize(stop);
    ```

## 27. `cudaEventElapsedTime()`

-   **Usage:** Computes elapsed time between two events.
-   **Example:**
    ```cuda
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel time: %f ms\n", milliseconds);
    ```

## 28. `cudaEventDestroy()`

-   **Usage:** Destroys a CUDA event.
-   **Example:**
    ```cuda
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    ```

## Thread and Block Bounds Checking

## 29. Thread bounds checking

-   **Usage:** Ensure thread doesn't access out-of-bounds memory.
-   **Example:**
    ```cuda
    __global__ void kernel(int *data, int n) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < n) {  // Bounds check
            data[tid] = tid;
        }
    }
    ```

## Block Size Calculation

## 30. Grid size calculation

-   **Usage:** Calculate number of blocks needed for a given problem size.
-   **Example:**
    ```cuda
    int n = 1000000;  // Problem size
    int threadsPerBlock = 256;
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;  // Ceiling division
    ```

## Memory Access Patterns

## 31. Coalesced memory access

-   **Usage:** Ensure consecutive threads access consecutive memory locations.
-   **Example:**
    ```cuda
    __global__ void kernel(int *data) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        data[tid] = tid;  // Coalesced access pattern
    }
    ```

## Warp-level Operations

## 32. Warp size considerations

-   **Usage:** Understanding that threads execute in groups of 32 (warp size).
-   **Example:**
    ```cuda
    __global__ void kernel() {
        int laneId = threadIdx.x % 32;  // Lane within warp
        int warpId = threadIdx.x / 32;  // Warp within block
    }
    ```

## Compilation and Linking

## 33. `nvcc` compiler

-   **Usage:** NVIDIA CUDA Compiler for compiling .cu files.
-   **Example:**
    ```bash
    nvcc -o program program.cu
    nvcc -arch=sm_70 -o program program.cu  # Target specific architecture
    ```

## 34. Architecture targeting

-   **Usage:** Specify target GPU architecture for optimization.
-   **Example:**
    ```bash
    nvcc -arch=sm_35 program.cu  # Kepler
    nvcc -arch=sm_50 program.cu  # Maxwell
    nvcc -arch=sm_70 program.cu  # Volta
    nvcc -arch=sm_80 program.cu  # Ampere
    ```

