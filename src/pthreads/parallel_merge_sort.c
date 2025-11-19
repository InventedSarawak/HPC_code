#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_THREADS 4

typedef struct {
    int *arr;
    int left;
    int right;
} ThreadArgs;

void merge(int *arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;
    int *L = malloc(n1 * sizeof(int));
    int *R = malloc(n2 * sizeof(int));
    for (int i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (int i = 0; i < n2; i++)
        R[i] = arr[mid + 1 + i];
    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2)
        arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
    while (i < n1)
        arr[k++] = L[i++];
    while (j < n2)
        arr[k++] = R[j++];
    free(L);
    free(R);
}

void merge_sort(int *arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        merge_sort(arr, left, mid);
        merge_sort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

void *parallel_merge_sort(void *arg) {
    ThreadArgs *args = (ThreadArgs *)arg;
    int left = args->left;
    int right = args->right;
    int *arr = args->arr;

    if (right - left < 1000) {
        merge_sort(arr, left, right);
        return NULL;
    }

    int mid = left + (right - left) / 2;
    pthread_t tid1, tid2;
    ThreadArgs args1 = {arr, left, mid};
    ThreadArgs args2 = {arr, mid + 1, right};

    pthread_create(&tid1, NULL, parallel_merge_sort, &args1);
    pthread_create(&tid2, NULL, parallel_merge_sort, &args2);

    pthread_join(tid1, NULL);
    pthread_join(tid2, NULL);

    merge(arr, left, mid, right);
    return NULL;
}

int main() {
    int n;
    printf("Enter number of elements: ");
    scanf("%d", &n);
    int *arr = malloc(n * sizeof(int));
    printf("Enter elements:\n");
    for (int i = 0; i < n; i++)
        scanf("%d", &arr[i]);

    ThreadArgs args = {arr, 0, n - 1};
    parallel_merge_sort(&args);

    printf("Sorted array:\n");
    for (int i = 0; i < n; i++)
        printf("%d ", arr[i]);
    printf("\n");
    free(arr);
    return 0;
}