#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define COUNT 10000000

int main(int argc, char **argv) {
    int num_threads;
#pragma omp parallel for collapse(2) num_threads(128)
    for (int i = 0; i < COUNT; i++) {
        for (int j = 0; j < COUNT; j++) {
            num_threads = omp_get_num_threads();
            if (j % 1000000 == 0 && i % 1000000 == 0)
                printf("* ");
        }
    }
    printf("\n%d\n", num_threads);
    return 0;
}