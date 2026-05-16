#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

double now_seconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    int n = 1000000;
    
    if (argc > 1) {
        n = atoi(argv[1]);
    }
    
    int *arr = (int *)malloc(n * sizeof(int));
    if (arr == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }
    
    // Initialize array
    double t_init_start = now_seconds();
    #pragma omp parallel for schedule(static)
for (int i = 0; i < n; i++) {
        arr[i] = (i % 100) + 1;
    }
    double t_init_end = now_seconds();
    
    int sum = 0;
    int product = 1;
    int count = 0;
    
    // Sum reduction
    double t_sum_start = now_seconds();
    #pragma omp parallel for schedule(static) reduction(+:sum)
for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    double t_sum_end = now_seconds();
    
    // Product reduction (reset for clean timing)
    product = 1;
    double t_prod_start = now_seconds();
    #pragma omp parallel for schedule(static) reduction(*:product)
for (int i = 0; i < n; i++) {
        product *= (arr[i] % 10 + 1);  // Avoid overflow with modulo
    }
    double t_prod_end = now_seconds();
    
    // Count reduction
    double t_count_start = now_seconds();
    #pragma omp parallel for schedule(static) reduction(+:count)
for (int i = 0; i < n; i++) {
        if (arr[i] % 2 == 0) {
            count += 1;
        }
    }
    double t_count_end = now_seconds();
    
    printf("========== Reduction Test Results ==========\n");
    printf("Array size: %d elements\n\n", n);
    printf("Computed values:\n");
    printf("  Sum:   %d\n", sum);
    printf("  Product: %d (limited to prevent overflow)\n", product);
    printf("  Count:  %d (even numbers)\n\n", count);
    printf("Execution times:\n");
    printf("  Init:  %.6f seconds\n", t_init_end - t_init_start);
    printf("  Sum:   %.6f seconds\n", t_sum_end - t_sum_start);
    printf("  Product: %.6f seconds\n", t_prod_end - t_prod_start);
    printf("  Count: %.6f seconds\n", t_count_end - t_count_start);
    printf("  Total: %.6f seconds\n\n", 
           (t_init_end - t_init_start) + (t_sum_end - t_sum_start) + 
           (t_prod_end - t_prod_start) + (t_count_end - t_count_start));
    printf("==========================================\n");
    
    free(arr);
    return 0;
}
