#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define NUM_READERS 3
#define NUM_WRITERS 2

typedef struct {
    int readers;         // active readers
    int writer;          // 1 if writer active, 0 otherwise
    int pending_writers; // waiting writers
    pthread_cond_t readers_proceed;
    pthread_cond_t writer_proceed;
    pthread_mutex_t read_write_lock;
} mylib_rwlock_t;

// Initialize the lock
void mylib_rwlock_init(mylib_rwlock_t *rw) {
    rw->readers = 0;
    rw->writer = 0;
    rw->pending_writers = 0;
    pthread_mutex_init(&(rw->read_write_lock), NULL);
    pthread_cond_init(&(rw->readers_proceed), NULL);
    pthread_cond_init(&(rw->writer_proceed), NULL);
}

// Acquire read lock
void mylib_rwlock_rdlock(mylib_rwlock_t *rw) {
    pthread_mutex_lock(&rw->read_write_lock);

    // Wait while a writer is active or writers are pending (writer priority)
    while (rw->writer || rw->pending_writers > 0) {
        pthread_cond_wait(&rw->readers_proceed, &rw->read_write_lock);
    }

    rw->readers++; // increment active readers
    pthread_mutex_unlock(&rw->read_write_lock);
}

// Acquire write lock
void mylib_rwlock_wrlock(mylib_rwlock_t *rw) {
    pthread_mutex_lock(&rw->read_write_lock);
    rw->pending_writers++; // signal a writer is waiting

    // Wait while readers or a writer are active
    while (rw->writer || rw->readers > 0) {
        pthread_cond_wait(&rw->writer_proceed, &rw->read_write_lock);
    }

    rw->pending_writers--;
    rw->writer = 1; // mark writer active
    pthread_mutex_unlock(&rw->read_write_lock);
}

// Unlock (called by both readers and writers)
void mylib_rwlock_unlock(mylib_rwlock_t *rw) {
    pthread_mutex_lock(&rw->read_write_lock);

    if (rw->writer) {
        rw->writer = 0;
    } else if (rw->readers > 0) {
        rw->readers--;
    }

    // Wake threads
    if (rw->pending_writers > 0) {
        // Give priority to waiting writers
        pthread_cond_signal(&rw->writer_proceed);
    } else {
        // Wake all readers
        pthread_cond_broadcast(&rw->readers_proceed);
    }

    pthread_mutex_unlock(&rw->read_write_lock);
}

mylib_rwlock_t rwlock;

void *reader(void *arg) {
    int id = *(int *)arg;
    for (int i = 0; i < 5; i++) {
        mylib_rwlock_rdlock(&rwlock);
        printf("Reader %d: reading...\n", id);
        usleep((rand() % 500 + 100) * 1000); // simulate read time
        printf("Reader %d: done reading\n", id);
        mylib_rwlock_unlock(&rwlock);
        usleep((rand() % 500 + 100) * 1000); // simulate thinking time
    }
    return NULL;
}

void *writer(void *arg) {
    int id = *(int *)arg;
    for (int i = 0; i < 5; i++) {
        mylib_rwlock_wrlock(&rwlock);
        printf("Writer %d: writing...\n", id);
        usleep((rand() % 500 + 100) * 1000); // simulate write time
        printf("Writer %d: done writing\n", id);
        mylib_rwlock_unlock(&rwlock);
        usleep((rand() % 500 + 100) * 1000); // simulate thinking time
    }
    return NULL;
}

int main() {
    srand(time(NULL));
    pthread_t rthreads[NUM_READERS], wthreads[NUM_WRITERS];
    int ids[NUM_READERS > NUM_WRITERS ? NUM_READERS : NUM_WRITERS];

    // Initialize lock
    mylib_rwlock_init(&rwlock);

    // Create reader threads
    for (int i = 0; i < NUM_READERS; i++) {
        ids[i] = i + 1;
        pthread_create(&rthreads[i], NULL, reader, &ids[i]);
    }

    // Create writer threads
    for (int i = 0; i < NUM_WRITERS; i++) {
        ids[i] = i + 1;
        pthread_create(&wthreads[i], NULL, writer, &ids[i]);
    }

    // Wait for readers to finish
    for (int i = 0; i < NUM_READERS; i++) {
        pthread_join(rthreads[i], NULL);
    }

    // Wait for writers to finish
    for (int i = 0; i < NUM_WRITERS; i++) {
        pthread_join(wthreads[i], NULL);
    }

    printf("Simulation complete.\n");
    return 0;
}