# CUDA Programming Basics — Exam Answers

## Chapter 2 — Parallel Programming Models & GPUs (CUDA Programming Basics)

### Short Questions

**What is data parallelism? How does CUDA support data parallel computation?**  
Data parallelism is performing the same operation on different pieces of distributed data simultaneously. CUDA supports this by launching many threads, each processing a different data element in parallel.

**Explain the concept of SIMT. How is it different from SIMD?**  
SIMT (Single Instruction, Multiple Threads) is NVIDIA’s execution model where each thread has its own context but executes the same instruction. SIMD (Single Instruction, Multiple Data) has one instruction stream for multiple data, but less flexibility—SIMT allows threads to diverge.

**What is a CUDA kernel? How is it launched from the host?**  
A CUDA kernel is a function executed on the GPU by many threads in parallel. It is launched from the host using the syntax:  
```cuda
kernel<<<numBlocks, threadsPerBlock>>>(args);
```

**Define:**  
- **Grid:** A collection of blocks launched for a kernel.  
- **Block:** A group of threads that can cooperate via shared memory and synchronization.  
- **Thread:** The smallest unit of execution; each runs kernel code independently.  
- **Warp:** A group of 32 threads executed together in lockstep on NVIDIA GPUs.

**What is thread divergence? Why does it degrade performance?**  
Thread divergence occurs when threads in a warp follow different execution paths (e.g., due to `if` statements). This causes serial execution of different paths, reducing parallel efficiency.

**Why do we use `__global__` and `__device__` qualifiers in CUDA?**  
- `__global__`: Marks a function as a kernel callable from host, runs on device.  
- `__device__`: Marks a function callable only from device code, runs on device.

**What is the difference between host and device code?**  
Host code runs on the CPU; device code runs on the GPU. Host launches kernels and manages memory; device code is executed in parallel by GPU threads.

**What is the responsibility of the GPU thread scheduler?**  
It assigns warps to available execution units (SMs), manages context switching, and hides memory latency by switching between warps.

**Explain block dimension (`blockDim`) and thread index (`threadIdx`) with an example.**  
`blockDim` gives the number of threads per block; `threadIdx` gives a thread’s index within its block.  
Example:  
```cuda
int tid = blockIdx.x * blockDim.x + threadIdx.x;
```

**Why is massive multithreading required for GPUs?**  
To hide memory latency and keep execution units busy, GPUs need thousands of threads so that when some threads wait for memory, others can execute.

---

### Long Questions

**Explain the SIMT architecture of NVIDIA GPUs with a neat diagram.**  
SIMT architecture groups threads into warps (32 threads). All threads in a warp execute the same instruction but can have different data and control flow. (Draw: SM → multiple warps → threads.)

**Explain thread hierarchy in CUDA with an example of converting 1D index to 2D block/thread coordinates.**  
CUDA organizes threads as: grid → blocks → threads.  
Example:  
```cuda
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

**Describe how CUDA handles thread scheduling and latency hiding.**  
CUDA schedules warps on SMs. When a warp stalls (e.g., waiting for memory), the scheduler quickly switches to another ready warp, hiding latency.

**Compare CUDA and CPU-based parallel programming models.**  
CUDA: Many lightweight threads, SIMT, explicit memory hierarchy, programmer manages parallelism.  
CPU: Fewer heavyweight threads, SIMD/SIMT less explicit, hardware manages most parallelism.

**Explain memory coalescing. Why is it important?**  
Memory coalescing is when consecutive threads access consecutive memory addresses, allowing efficient memory transactions. It’s important for maximizing memory bandwidth.

**Write and explain the structure of a typical CUDA program.**  
1. Allocate host/device memory  
2. Copy data to device  
3. Launch kernel  
4. Copy results back  
5. Free memory

**What are warps? Explain warp execution and warp divergence with examples.**  
A warp is 32 threads executed together. If threads in a warp diverge (e.g., `if (threadIdx.x % 2)`), the warp serializes the paths, reducing efficiency.

**How does GPU multithreading help hide memory latency?**  
When one warp waits for memory, the scheduler switches to another ready warp, so computation continues without stalling.

**Explain the kernel launch configuration syntax with examples.**  
Syntax:  
```cuda
kernel<<<numBlocks, threadsPerBlock>>>(args);
```
Example:  
```cuda
vectorAdd<<<256, 256>>>(a, b, c);
```

**Discuss the challenges in writing highly scalable data-parallel programs.**  
Challenges: load balancing, memory bandwidth, synchronization, avoiding divergence, maximizing occupancy, and efficient memory access.

---

## Chapter 3 — Thread Organization & Synchronization

### Short Questions

**What is shared memory? How does it differ from global memory?**  
Shared memory is fast, on-chip memory accessible by threads in a block. Global memory is large but slow and accessible by all threads.

**What is bank conflict in shared memory?**  
A bank conflict occurs when multiple threads access different addresses in the same memory bank, causing serialization.

**Explain the need for `__syncthreads()` in CUDA.**  
`__syncthreads()` synchronizes all threads in a block, ensuring all have reached the same point before proceeding (e.g., after writing to shared memory).

**What are memory banks? Why do conflicts reduce performance?**  
Shared memory is divided into banks. If multiple threads access the same bank, accesses are serialized, reducing performance.

**Define:**  
- **Local memory:** Private to each thread, used for register spills or large arrays.  
- **Constant memory:** Read-only memory cached for all threads.  
- **Texture memory:** Read-only memory optimized for 2D spatial locality.

**What is tiling? Why is it used in matrix multiplication?**  
Tiling divides data into small blocks (tiles) that fit in shared memory, reducing global memory accesses and improving performance.

**What are the differences between `__syncthreads()` and warp-level synchronization?**  
`__syncthreads()` synchronizes all threads in a block; warp-level sync (e.g., `__syncwarp()`) synchronizes threads within a warp only.

**What is a reduction operation?**  
A parallel operation that combines elements (e.g., sum, max) into a single result.

**Why is shared memory faster than global memory?**  
Shared memory is on-chip and has much lower latency than off-chip global memory.

**What happens if a thread reaches `__syncthreads()` but others do not?**  
This causes deadlock; the block hangs because all threads must reach the barrier.

---

### Long Questions

**Explain the memory hierarchy of CUDA GPUs with a diagram.**  
(Diagram: Registers → Shared memory (per block) → L1/L2 cache → Global memory → Host memory.)  
Registers (fastest, per thread), shared memory (fast, per block), global memory (large, slow), constant/texture memory (read-only, cached).

**Describe how shared memory can be used to optimize matrix multiplication.**  
By loading tiles of input matrices into shared memory, threads can reuse data, reducing global memory accesses and improving speed.

**Discuss the design and implementation of a parallel reduction algorithm.**  
Parallel reduction uses shared memory and synchronizations to combine values in a tree-like fashion, halving the number of active threads each step.

**Explain shared memory bank conflicts with examples.**  
If threads access addresses that map to the same bank, accesses serialize. Example:  
```cuda
shared[threadIdx.x * 2] // all even indices map to same bank
```

**How does thread synchronization work inside a block? Why can’t we synchronize across blocks?**  
`__syncthreads()` synchronizes threads within a block. There’s no built-in way to sync across blocks because blocks may execute independently and in any order.

**Explain why grid-level synchronization is not allowed in CUDA.**  
Blocks are scheduled independently and may not all be resident at once, so global synchronization is not guaranteed.

**Describe the tiling technique for improving memory locality in CUDA programs.**  
Tiling loads sub-blocks of data into shared memory, allowing threads to reuse data and reduce global memory traffic.

**Explain how memory coalescing is related to thread and data organization.**  
Coalescing requires threads to access consecutive memory addresses; proper data layout and thread assignment are crucial for this.

**Show how shared memory can be used for cooperative thread computation.**  
Threads load data into shared memory, synchronize, then collaboratively process the data (e.g., matrix multiplication, reduction).

**Discuss performance considerations when organizing threads and memory accesses in CUDA.**  
Considerations: maximize occupancy, avoid divergence, use shared memory efficiently, coalesce global memory accesses, minimize bank conflicts, and balance workload.
