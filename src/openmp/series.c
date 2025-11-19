#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int f(int x) { return x * x; }

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <n>\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]);
    int *v = malloc(n * sizeof(int));
    int *indices = malloc(n * sizeof(int));

    for (int i = 0; i < n; i++) {
        indices[i] = 0;
        v[i] = 0;
    }

#pragma omp parallel for default(none) shared(v, indices, n)
    for (int i = 0; i < n; i += 1) {
#pragma omp critical
        v[indices[i]] += f(i);
    }

    printf("%d\n", v[0]);

    free(v);
    free(indices);
    return 0;
}