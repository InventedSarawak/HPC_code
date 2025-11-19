#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define COUNT 1000

int main() {
#pragma omp parallel for num_threads(8) if (COUNT > 500) ordered
    for (int i = 0; i < COUNT; i++) {
        double x = 0.0;
#pragma omp parallel for
        for (int j = 0; j < 1000; j++) {
            x += j * 0.001;
        }
        printf("Thread %d: x = %f\n", omp_get_thread_num(), x);
    }
    return 0;
}
