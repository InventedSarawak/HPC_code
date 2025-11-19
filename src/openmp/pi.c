#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

static long long num_steps[] = {10LL, 100LL, 10000LL, 1000000LL, 10000000LL, 100000000LL, 1000000000LL};
long double step;

int main(int argc, char **argv) {
    omp_set_num_threads(omp_get_max_threads());
    for (int idx = 0; idx < sizeof(num_steps) / sizeof(long long); idx++) {
        long long i;
        long double x, pi, sum = 0.0L;
        long double step = 1.0L / (long double)num_steps[idx];

#pragma omp parallel for reduction(+ : sum)
        for (i = 0; i < num_steps[idx]; i++) {
            x = (i + 0.5L) * step;
            sum += 4.0L / (1.0L + x * x);
        }
        pi = step * sum;
        printf("With %lld steps, our estimate of pi is %.16Lf\n", num_steps[idx], pi);
        pi = 0.0L;
    }

    return 0;
}