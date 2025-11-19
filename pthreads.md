# Pthread Functions and Usage

## 1. `pthread_t`

-   **Usage:** Thread identifier type used to represent a thread.
-   **Example:**
    ```c
    pthread_t thread;
    ```

## 2. `pthread_create()`

-   **Usage:** Creates a new thread and starts executing the specified function.
-   **Example:**
    ```c
    pthread_t thread;
    pthread_create(&thread, NULL, function_name, arg);
    ```

## 3. `pthread_join()`

-   **Usage:** Waits for a thread to terminate and retrieves its return value.
-   **Example:**
    ```c
    pthread_join(thread, NULL);
    ```

## 4. `pthread_exit()`

-   **Usage:** Terminates the calling thread and optionally returns a value.
-   **Example:**
    ```c
    pthread_exit(NULL);
    ```

## 5. `pthread_mutex_t`

-   **Usage:** Mutex (mutual exclusion) variable type for thread synchronization.
-   **Example:**
    ```c
    pthread_mutex_t mutex;
    ```

## 6. `pthread_mutex_init()`

-   **Usage:** Initializes a mutex.
-   **Example:**
    ```c
    pthread_mutex_t mutex;
    pthread_mutex_init(&mutex, NULL);
    ```

## 7. `pthread_mutex_lock()`

-   **Usage:** Locks a mutex, blocking if already locked by another thread.
-   **Example:**
    ```c
    pthread_mutex_lock(&mutex);
    ```

## 8. `pthread_mutex_unlock()`

-   **Usage:** Unlocks a mutex, allowing other threads to acquire it.
-   **Example:**
    ```c
    pthread_mutex_unlock(&mutex);
    ```

## 9. `pthread_mutex_destroy()`

-   **Usage:** Destroys a mutex and frees its resources.
-   **Example:**
    ```c
    pthread_mutex_destroy(&mutex);
    ```

## 10. `pthread_cond_t`

-   **Usage:** Condition variable type for thread synchronization and signaling.
-   **Example:**
    ```c
    pthread_cond_t condition;
    ```

## 11. `pthread_cond_init()`

-   **Usage:** Initializes a condition variable.
-   **Example:**
    ```c
    pthread_cond_t condition;
    pthread_cond_init(&condition, NULL);
    ```

## 12. `pthread_cond_wait()`

-   **Usage:** Blocks the calling thread until the condition variable is signaled.
-   **Example:**
    ```c
    pthread_mutex_lock(&mutex);
    pthread_cond_wait(&condition, &mutex);
    pthread_mutex_unlock(&mutex);
    ```

## 13. `pthread_cond_signal()`

-   **Usage:** Wakes up one thread waiting on the condition variable.
-   **Example:**
    ```c
    pthread_cond_signal(&condition);
    ```

## 14. `pthread_cond_broadcast()`

-   **Usage:** Wakes up all threads waiting on the condition variable.
-   **Example:**
    ```c
    pthread_cond_broadcast(&condition);
    ```

## 15. `pthread_cond_destroy()`

-   **Usage:** Destroys a condition variable and frees its resources.
-   **Example:**
    ```c
    pthread_cond_destroy(&condition);
    ```

## 16. `PTHREAD_MUTEX_INITIALIZER`

-   **Usage:** Static initializer for mutex variables.
-   **Example:**
    ```c
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    ```

## 17. `PTHREAD_COND_INITIALIZER`

-   **Usage:** Static initializer for condition variables.
-   **Example:**
    ```c
    pthread_cond_t condition = PTHREAD_COND_INITIALIZER;
    ```

