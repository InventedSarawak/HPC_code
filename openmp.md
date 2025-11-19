# OpenMP Directives and Usage

## Basic Parallel Constructs

## 1. `#pragma omp parallel`

-   **Usage:** Creates a team of threads to execute the enclosed code block in parallel.
-   **Example:**
    ```c
    #pragma omp parallel
    {
        // code executed by all threads
    }
    ```

## 2. `#pragma omp parallel num_threads(n)`

-   **Usage:** Creates a parallel region with `n` threads.
-   **Example:**
    ```c
    #pragma omp parallel num_threads(8)
    {
        // code executed by 8 threads
    }
    ```

## 3. `num_threads`

-   **Usage:** Specifies the number of threads to use in a parallel region.
-   **Example:**
    ```c
    #pragma omp parallel num_threads(4)
    {
        // code executed by 4 threads
    }
    ```

## Work-Sharing Constructs

## 4. `#pragma omp for`

-   **Usage:** Distributes loop iterations among threads in a parallel region.
-   **Example:**
    ```c
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        // loop body
    }
    ```

## 5. `#pragma omp sections`

-   **Usage:** Divides code into sections, each executed by a different thread.
-   **Example:**
    ```c
    #pragma omp parallel sections
    {
        #pragma omp section
        // section 1
        #pragma omp section
        // section 2
    }
    ```

## 6. `#pragma omp single`

-   **Usage:** Specifies a block of code to be executed by only one thread in a team.
-   **Example:**
    ```c
    #pragma omp single
    {
        // executed by one thread only
    }
    ```

## 7. `#pragma omp master`

-   **Usage:** Restricts code block execution to the master thread.
-   **Example:**
    ```c
    #pragma omp master
    {
        // executed by master thread only
    }
    ```

## Loop Scheduling

## 8. `schedule` clause

-   **Usage:** Specifies how loop iterations are distributed among threads.
-   **Syntax:** `schedule([modifier [modifier]:]kind[,chunk_size])`
-   **Example:**
    ```c
    #pragma omp parallel for schedule(static, 4)
    for (int i = 0; i < n; i++) {
        // loop body
    }
    ```

## 9. `static` scheduling

-   **Usage:** Divides iterations into chunks of specified size and assigns them to threads in round-robin fashion.
-   **Example:**
    ```c
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        // evenly distributed iterations
    }
    ```

## 10. `dynamic` scheduling

-   **Usage:** Assigns chunks of iterations to threads dynamically as they become available.
-   **Example:**
    ```c
    #pragma omp parallel for schedule(dynamic, 10)
    for (int i = 0; i < n; i++) {
        // dynamic load balancing
    }
    ```

## 11. `guided` scheduling

-   **Usage:** Similar to dynamic but chunk size decreases exponentially.
-   **Example:**
    ```c
    #pragma omp parallel for schedule(guided, 5)
    for (int i = 0; i < n; i++) {
        // decreasing chunk sizes
    }
    ```

## 12. `auto` scheduling

-   **Usage:** Compiler/runtime decides the best scheduling.
-   **Example:**
    ```c
    #pragma omp parallel for schedule(auto)
    for (int i = 0; i < n; i++) {
        // implementation-defined scheduling
    }
    ```

## 13. `runtime` scheduling

-   **Usage:** Scheduling determined at runtime via OMP_SCHEDULE environment variable.
-   **Example:**
    ```c
    #pragma omp parallel for schedule(runtime)
    for (int i = 0; i < n; i++) {
        // runtime-determined scheduling
    }
    ```

## Data Sharing Clauses

## 14. `shared` clause

-   **Usage:** Specifies that variables are shared among all threads.
-   **Example:**
    ```c
    int shared_var = 0;
    #pragma omp parallel shared(shared_var)
    {
        // all threads access the same shared_var
    }
    ```

## 15. `private` clause

-   **Usage:** Each thread gets its own private copy of the variable.
-   **Example:**
    ```c
    int private_var;
    #pragma omp parallel private(private_var)
    {
        // each thread has its own private_var
    }
    ```

## 16. `firstprivate` clause

-   **Usage:** Each thread gets a private copy initialized with the original value.
-   **Example:**
    ```c
    int var = 10;
    #pragma omp parallel firstprivate(var)
    {
        // each thread starts with var = 10
    }
    ```

## 17. `lastprivate` clause

-   **Usage:** Private variable's value from the last iteration is copied back to the original variable.
-   **Example:**
    ```c
    int var;
    #pragma omp parallel for lastprivate(var)
    for (int i = 0; i < n; i++) {
        var = i; // value from last iteration preserved
    }
    ```

## 18. `default` clause

-   **Usage:** Sets the default data-sharing attribute for variables.
-   **Example:**
    ```c
    #pragma omp parallel default(none) shared(a) private(b)
    {
        // explicit data sharing required
    }
    ```

## 19. `reduction` clause

-   **Usage:** Performs a reduction operation on variables across threads.
-   **Example:**
    ```c
    int sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        sum += array[i];
    }
    ```

## Synchronization Constructs

## 20. `#pragma omp critical`

-   **Usage:** Ensures only one thread executes the enclosed code at a time (mutual exclusion).
-   **Example:**
    ```c
    #pragma omp critical
    {
        // critical section
    }
    ```

## 21. `#pragma omp atomic`

-   **Usage:** Ensures atomic (indivisible) update of a variable.
-   **Example:**
    ```c
    #pragma omp atomic
    x++;
    ```

## 22. `#pragma omp barrier`

-   **Usage:** Synchronizes all threads in a team; all threads wait until all have reached the barrier.
-   **Example:**
    ```c
    #pragma omp barrier
    ```

## 23. `#pragma omp ordered`

-   **Usage:** Specifies a block of code to be executed in a specific order.
-   **Example:**
    ```c
    #pragma omp parallel for ordered
    for (int i = 0; i < n; i++) {
        #pragma omp ordered
        {
            // executed in order
        }
    }
    ```

## 24. `nowait` clause

-   **Usage:** Removes the implicit barrier at the end of a work-sharing construct.
-   **Example:**
    ```c
    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < n; i++) {
            // no barrier after this loop
        }
        // threads continue here without waiting
    }
    ```

## 25. `lock` functions

-   **Usage:** Manual lock operations for thread synchronization.
-   **Example:**
    ```c
    #include <omp.h>
    omp_lock_t lock;
    omp_init_lock(&lock);
    omp_set_lock(&lock);
    // critical section
    omp_unset_lock(&lock);
    omp_destroy_lock(&lock);
    ```

## Loop Constructs

## 26. `collapse` clause

-   **Usage:** Specifies how many loops to collapse into a single iteration space.
-   **Example:**
    ```c
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            // both loops parallelized
        }
    }
    ```

## Conditional and Control

## 27. `if` clause

-   **Usage:** Conditionally creates a parallel region based on a boolean expression.
-   **Example:**
    ```c
    int condition = 1;
    #pragma omp parallel if(condition)
    {
        // parallel only if condition is true
    }
    ```

## Runtime Functions

## 28. `omp_get_num_threads()`

-   **Usage:** Returns the number of threads in the current parallel region.
-   **Example:**
    ```c
    #include <omp.h>
    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        printf("Number of threads: %d\n", nthreads);
    }
    ```

## 29. `omp_get_thread_num()`

-   **Usage:** Returns the thread number (ID) of the calling thread within the team.
-   **Example:**
    ```c
    #include <omp.h>
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        printf("Thread ID: %d\n", tid);
    }
    ```

## 30. `omp_set_num_threads()`

-   **Usage:** Sets the number of threads to use in subsequent parallel regions.
-   **Example:**
    ```c
    #include <omp.h>
    omp_set_num_threads(8);
    #pragma omp parallel
    {
        // code executed by 8 threads
    }
    ```

## Thread-Private Data

## 31. `#pragma omp threadprivate`

-   **Usage:** Makes a global variable private to each thread.
-   **Example:**
    ```c
    int global_var;
    #pragma omp threadprivate(global_var)
    ```

## 32. `copyin` clause

-   **Usage:** Copies the master thread's value to threadprivate variables.
-   **Example:**
    ```c
    int global_var;
    #pragma omp threadprivate(global_var)
    #pragma omp parallel copyin(global_var)
    {
        // all threads get master's global_var value
    }
    ```

## Memory and System

## 33. `#pragma omp flush`

-   **Usage:** Synchronizes memory between threads.
-   **Example:**
    ```c
    #pragma omp flush
    ```

## 34. `proc_bind` clause

-   **Usage:** Controls thread affinity to processors.
-   **Example:**
    ```c
    #pragma omp parallel proc_bind(close)
    {
        // threads bound to nearby processors
    }
    ```

## Environment Variables

## 35. `OMP_NUM_THREADS`

-   **Usage:** Environment variable that sets the default number of threads.
-   **Example:**
    ```bash
    export OMP_NUM_THREADS=4
    ./program
    ```

---

Refer to the official OpenMP documentation for more advanced directives and clauses.
