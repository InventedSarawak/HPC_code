#include <pthread.h>
#include <stdio.h>

#define NUM_THREADS 20
#define NUM_INCREMENTS 10000

int counter = 0;
pthread_mutex_t counter_mutex;

void *increment() {
    for (int i = 0; i < NUM_INCREMENTS; i++)
        counter++;
    return NULL;
}

void *safe_increment() {
    for (int i = 0; i < NUM_INCREMENTS; i++) {
        pthread_mutex_lock(&counter_mutex);
        counter++;
        pthread_mutex_unlock(&counter_mutex);
    }
    return NULL;
}

int main() {
    pthread_t threads[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++)
        pthread_create(&threads[i], NULL, increment, NULL);
    for (int i = 0; i < NUM_THREADS; i++)
        pthread_join(threads[i], NULL);

    printf("Counter without lock: %d\n", counter);

    counter = 0;
    pthread_mutex_init(&counter_mutex, NULL);
    for (int i = 0; i < NUM_THREADS; i++)
        pthread_create(&threads[i], NULL, safe_increment, NULL);
    for (int i = 0; i < NUM_THREADS; i++)
        pthread_join(threads[i], NULL);
    printf("Counter with lock: %d\n", counter);
    pthread_mutex_destroy(&counter_mutex);

    return 0;
}