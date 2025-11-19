#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define BUFFER_SIZE 5
#define PRODUCERS 3
#define CONSUMERS 4
#define ITEMS_TO_PRODUCE 10

int buffer[BUFFER_SIZE];
int count = 0;
int in = 0;
int out = 0;

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t not_full = PTHREAD_COND_INITIALIZER;
pthread_cond_t not_empty = PTHREAD_COND_INITIALIZER;

bool done = false; // Signal consumers to stop

void *producer(void *arg) {
    int id = *(int *)arg;
    for (int i = 0; i < ITEMS_TO_PRODUCE; i++) {
        pthread_mutex_lock(&mutex);

        while (count == BUFFER_SIZE) {
            pthread_cond_wait(&not_full, &mutex);
        }

        int item = id * 100 + i;
        buffer[in] = item;
        in = (in + 1) % BUFFER_SIZE;
        count++;
        printf("Producer %d produced %d | Buffer count: %d\n", id, item, count);

        pthread_cond_signal(&not_empty);
        pthread_mutex_unlock(&mutex);

        usleep(rand() % 100000);
    }
    return NULL;
}

void *consumer(void *arg) {
    int id = *(int *)arg;
    while (true) {
        pthread_mutex_lock(&mutex);

        // Wait until buffer has items or producers are done
        while (count == 0 && !done) {
            pthread_cond_wait(&not_empty, &mutex);
        }

        if (count == 0 && done) { // No more items coming
            pthread_mutex_unlock(&mutex);
            break;
        }

        int item = buffer[out];
        out = (out + 1) % BUFFER_SIZE;
        count--;
        printf("Consumer %d consumed %d | Buffer count: %d\n", id, item, count);

        pthread_cond_signal(&not_full);
        pthread_mutex_unlock(&mutex);

        usleep(rand() % 150000);
    }
    printf("Consumer %d exiting.\n", id);
    return NULL;
}

int main() {
    pthread_t producers[PRODUCERS], consumers[CONSUMERS];
    int ids[PRODUCERS > CONSUMERS ? PRODUCERS : CONSUMERS];

    // Create producer threads
    for (int i = 0; i < PRODUCERS; i++) {
        ids[i] = i + 1;
        pthread_create(&producers[i], NULL, producer, &ids[i]);
    }

    // Create consumer threads
    for (int i = 0; i < CONSUMERS; i++) {
        ids[i] = i + 1;
        pthread_create(&consumers[i], NULL, consumer, &ids[i]);
    }

    // Wait for producers to finish
    for (int i = 0; i < PRODUCERS; i++) {
        pthread_join(producers[i], NULL);
    }

    // Signal consumers to exit
    pthread_mutex_lock(&mutex);
    done = true;
    pthread_cond_broadcast(&not_empty); // Wake all consumers
    pthread_mutex_unlock(&mutex);

    // Wait for consumers to finish
    for (int i = 0; i < CONSUMERS; i++) {
        pthread_join(consumers[i], NULL);
    }

    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&not_full);
    pthread_cond_destroy(&not_empty);

    printf("All producers and consumers have exited. Program ending.\n");
    return 0;
}
