#include <omp.h>
#include <stdio.h>

#define COUNT 1000

int main() {
#pragma omp parallel num_threads(4) if (COUNT > 500)
    {
        int a;
        int b;
        a = omp_get_num_threads();
        b = omp_get_thread_num();
        printf("Total threads: %d, Thread ID: %d\n", a, b);
    }
    return 0;
}