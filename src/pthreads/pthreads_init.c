#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

void *print_some_message(void *arg) {
    printf("message: %s\n", (char *)arg);
    return NULL;
}

int main(int argc, char **argv) {
    pthread_t test_thread;
    int ret;
    char *msg = "This is a message";
    ret = pthread_create(&test_thread, NULL, &print_some_message, msg);
    if (ret != 0) {
        printf("Error: pthread_create() encountered an error");
        exit(EXIT_FAILURE);
    }
    pthread_join(test_thread, NULL);
    return 0;
}