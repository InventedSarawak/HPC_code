#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define NUM_THREADS 5

typedef struct {
    pthread_mutex_t count_lock;
    pthread_cond_t ok_to_proceed;
    int count;
} mylib_barrier_t;

void mylib_init_barrier(mylib_barrier_t *b) {
    b->count = 0;
    pthread_mutex_init(&(b->count_lock), NULL);
    pthread_cond_init(&(b->ok_to_proceed), NULL);
}

void mylib_barrier(mylib_barrier_t *b, int num_threads) {
    pthread_mutex_lock(&(b->count_lock));
    b->count++;
    if (b->count == num_threads) {
        b->count = 0;
        pthread_cond_broadcast(&(b->ok_to_proceed));
    } else
        pthread_cond_wait(&(b->ok_to_proceed), &(b->count_lock));
    pthread_mutex_unlock(&(b->count_lock));
}

mylib_barrier_t barrier;

void *worker(void *arg) {
    int id = *(int *)arg;

    printf("Thread %d: starting work\n", id);
    usleep((rand() % 500 + 100) * 1000); // simulate work

    printf("Thread %d: waiting at barrier\n", id);
    mylib_barrier(&barrier, NUM_THREADS);

    printf("Thread %d: passed the barrier\n", id);
    return NULL;
}

int main() {
    pthread_t threads[NUM_THREADS];
    int ids[NUM_THREADS];

    srand(time(NULL));
    mylib_init_barrier(&barrier);

    // Create threads
    for (int i = 0; i < NUM_THREADS; i++) {
        ids[i] = i + 1;
        pthread_create(&threads[i], NULL, worker, &ids[i]);
    }

    // Wait for threads to finish
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("All threads have passed the barrier.\n");
    return 0;
}