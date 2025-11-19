#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define N 5

pthread_mutex_t chopsticks[N];

typedef enum { DEADLOCK = 0, LAST_PHILO_RIGHT_FIRST, EVEN_ODD } PhilosopherMode;

/*
In the current code the threads will get stuck in a deadlock.
This is because the threads are trying to lock the same chopstick at the same time.
And so they will keep waiting for each other to unlock the chopstick.
And the program will never end implying that the philosophers will never eat.
*/

// Implementation with deadlock
void *philosopher(void *num) {
    int id = *(int *)num;
    while (1) {
        printf("Philosopher %d is thinking...\n", id);
        sleep(4);

        printf("Philosopher %d is hungry\n", id);

        // Pick left chopstick
        pthread_mutex_lock(&chopsticks[id]);
        printf("Philosopher %d picked up left chopstick\n", id);
        sleep(2);
        // Pick right chopstick
        pthread_mutex_lock(&chopsticks[(id + 1) % N]);
        printf("Philosopher %d picked up right chopstick\n", id);

        printf("Philosopher %d is eating...\n", id);
        sleep(1);

        // Put down chopsticks
        pthread_mutex_unlock(&chopsticks[id]);
        pthread_mutex_unlock(&chopsticks[(id + 1) % N]);

        printf("Philosopher %d finished eating\n", id);
    }
    return NULL;
}

// Implementation involving the last philosopher picking up the right chopstick first
void *philosopher_last_philosopher_right_first(void *num) {
    int id = *(int *)num;
    while (1) {
        printf("Philosopher %d is thinking...\n", id);
        sleep(4);

        printf("Philosopher %d is hungry\n", id);

        if (id == N - 1) {
            // Last Philosopher picks up right chopstick first
            pthread_mutex_lock(&chopsticks[(id + 1) % N]);
            printf("Philosopher %d picked up right chopstick\n", id);
            sleep(2);
            pthread_mutex_lock(&chopsticks[id]);
            printf("Philosopher %d picked up left chopstick\n", id);
        } else {
            // Rest of them pick the left chopstick first
            pthread_mutex_lock(&chopsticks[id]);
            printf("Philosopher %d picked up left chopstick\n", id);
            sleep(2);
            pthread_mutex_lock(&chopsticks[(id + 1) % N]);
            printf("Philosopher %d picked up right chopstick\n", id);
        }

        printf("Philosopher %d is eating...\n", id);
        sleep(1);

        // Put down chopsticks
        pthread_mutex_unlock(&chopsticks[id]);
        pthread_mutex_unlock(&chopsticks[(id + 1) % N]);

        printf("Philosopher %d finished eating\n", id);
    }
    return NULL;
}

// Implementation involving the even philosophers picking up the right chopstick first and the odd picking up the left first
void *philosopher_even_odd(void *num) {
    int id = *(int *)num;
    while (1) {
        printf("Philosopher %d is thinking...\n", id);
        sleep(4);

        printf("Philosopher %d is hungry\n", id);

        if (id % 2 == 0) {
            // Even philosophers picks up right chopstick first
            pthread_mutex_lock(&chopsticks[(id + 1) % N]);
            printf("Philosopher %d picked up right chopstick\n", id);
            sleep(2);
            pthread_mutex_lock(&chopsticks[id]);
            printf("Philosopher %d picked up left chopstick\n", id);
        } else {
            // Left ones of them pick the left chopstick first
            pthread_mutex_lock(&chopsticks[id]);
            printf("Philosopher %d picked up left chopstick\n", id);
            sleep(2);
            pthread_mutex_lock(&chopsticks[(id + 1) % N]);
            printf("Philosopher %d picked up right chopstick\n", id);
        }

        printf("Philosopher %d is eating...\n", id);
        sleep(1);

        // Put down chopsticks
        pthread_mutex_unlock(&chopsticks[id]);
        pthread_mutex_unlock(&chopsticks[(id + 1) % N]);

        printf("Philosopher %d finished eating\n", id);
    }
    return NULL;
}
int main(int argc, char *argv[]) {
    pthread_t philo[N];
    int ids[N];
    PhilosopherMode mode = DEADLOCK;

    if (argc > 1) {
        int opt = atoi(argv[1]);
        if (opt >= 0 && opt <= 2)
            mode = (PhilosopherMode)opt;
        else {
            printf("Usage: %s [0=deadlock|1=last_right_first|2=even_odd]\n", argv[0]);
            return 1;
        }
    } else {
        printf("Usage: %s [0=deadlock|1=last_right_first|2=even_odd]\n", argv[0]);
        printf("Example: %s 1\n", argv[0]);
        return 1;
    }

    for (int i = 0; i < N; i++)
        pthread_mutex_init(&chopsticks[i], NULL);

    void *(*philo_func)(void *);
    switch (mode) {
    case DEADLOCK:
        philo_func = philosopher;
        break;
    case LAST_PHILO_RIGHT_FIRST:
        philo_func = philosopher_last_philosopher_right_first;
        break;
    case EVEN_ODD:
        philo_func = philosopher_even_odd;
        break;
    }

    for (int i = 0; i < N; i++) {
        ids[i] = i;
        pthread_create(&philo[i], NULL, philo_func, &ids[i]);
    }

    for (int i = 0; i < N; i++)
        pthread_join(philo[i], NULL);

    return 0;
}