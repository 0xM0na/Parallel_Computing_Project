#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double now_seconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    int N = 512;

    if (argc > 1) {
        N = atoi(argv[1]);
    }

    long total = (long)N * (long)N;

    double *A = (double *)malloc(total * sizeof(double));
    double *B = (double *)malloc(total * sizeof(double));
    double *C = (double *)malloc(total * sizeof(double));

    if (A == NULL || B == NULL || C == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }

    for (long i = 0; i < total; i++) {
        A[i] = (i % 100) * 0.01;
        B[i] = ((i + 1) % 100) * 0.01;
        C[i] = 0.0;
    }

    double start = now_seconds();

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;

            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }

            C[i * N + j] = sum;
        }
    }

    double end = now_seconds();

    double checksum = 0.0;

    for (long i = 0; i < total; i++) {
        checksum += C[i];
    }

    printf("N=%d time=%f checksum=%f\n", N, end - start, checksum);

    free(A);
    free(B);
    free(C);

    return 0;
}