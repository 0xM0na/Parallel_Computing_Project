/*
 * dgemm_benchmark.c  –  CSCI465/ECEN433 Nile University Spring 2026
 *
 * V0: Sequential baseline           (i,j,k)
 * V1: Naive parallel                (i,j,k  + omp parallel for on i)
 * V2: Loop interchange, sequential  (i,k,j)
 * V3: Full pipeline                 (i,k,j  + cache-aligned chunk + adaptive schedule)
 *
 * Build:  gcc -O2 -fopenmp -march=native -std=c99 -o dgemm_benchmark dgemm_benchmark.c -lm
 * Run:    OMP_NUM_THREADS=N ./dgemm_benchmark <size> <runs>
 */
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static long get_l2_bytes(void) {
    const char *paths[] = {
        "/sys/devices/system/cpu/cpu0/cache/index2/size",
        "/sys/devices/system/cpu/cpu0/cache/index1/size", NULL
    };
    for (int i = 0; paths[i]; i++) {
        FILE *f = fopen(paths[i], "r");
        if (!f) continue;
        long val = 0; char unit = 'K';
        int rc = fscanf(f, "%ld%c", &val, &unit); (void)rc;
        fclose(f);
        if (val <= 0) continue;
        return (unit=='M'||unit=='m') ? val*1024L*1024L : val*1024L;
    }
    return 2L*1024L*1024L;
}

static void init(double *A, double *B, double *C, int N) {
    long tot = (long)N*N;
    for (long i = 0; i < tot; i++) {
        A[i] = (i%100)*0.01; B[i] = ((i+1)%100)*0.01; C[i] = 0.0;
    }
}
static double csum(const double *C, int N) {
    double s=0; for(long i=0;i<(long)N*N;i++) s+=C[i]; return s;
}

/* V0: sequential i,j,k */
static double v0(double *A, double *B, double *C, int N) {
    init(A,B,C,N);
    double t=now_sec();
    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++) {
            double s=0.0;
            for (int k=0;k<N;k++) s+=A[i*N+k]*B[k*N+j];
            C[i*N+j]=s;
        }
    return now_sec()-t;
}

/* V1: naive parallel i,j,k */
static double v1(double *A, double *B, double *C, int N) {
    init(A,B,C,N);
    double t=now_sec();
    #pragma omp parallel for schedule(static)
    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++) {
            double s=0.0;
            for (int k=0;k<N;k++) s+=A[i*N+k]*B[k*N+j];
            C[i*N+j]=s;
        }
    return now_sec()-t;
}

/* V2: loop interchange only, sequential i,k,j */
static double v2(double *A, double *B, double *C, int N) {
    init(A,B,C,N);
    memset(C,0,(long)N*N*sizeof(double));
    double t=now_sec();
    for (int i=0;i<N;i++)
        for (int k=0;k<N;k++)
            for (int j=0;j<N;j++)
                C[i*N+j]+=A[i*N+k]*B[k*N+j];
    return now_sec()-t;
}

/* V3: full pipeline – interchange + cache-line chunk + adaptive schedule */
static double v3(double *A, double *B, double *C, int N) {
    init(A,B,C,N);
    memset(C,0,(long)N*N*sizeof(double));
    /* (1) cache-line-aligned chunk: 64B / 8B = 8 doubles */
    const int chunk = 64 / (int)sizeof(double);
    /* (2) adaptive schedule based on working set vs L2 */
    long ws = 3L*N*N*(long)sizeof(double);
    long l2 = get_l2_bytes();
    double t=now_sec();
    if (ws <= l2) {
        /* working set fits in L2 → static partitioning is optimal */
        #pragma omp parallel for schedule(static)
        for (int i=0;i<N;i++)
            for (int k=0;k<N;k++)
                for (int j=0;j<N;j++)
                    C[i*N+j]+=A[i*N+k]*B[k*N+j];
    } else {
        /* working set exceeds L2 → dynamic with cache-line chunk */
        #pragma omp parallel for schedule(dynamic, 8)
        for (int i=0;i<N;i++)
            for (int k=0;k<N;k++)
                for (int j=0;j<N;j++)
                    C[i*N+j]+=A[i*N+k]*B[k*N+j];
    }
    (void)chunk; /* documents the derivation; value used literally above */
    return now_sec()-t;
}

int main(int argc, char **argv) {
    int N    = argc>1 ? atoi(argv[1]) : 512;
    int runs = argc>2 ? atoi(argv[2]) : 3;
    int nth  = omp_get_max_threads();
    long l2  = get_l2_bytes();
    long ws  = 3L*N*N*(long)sizeof(double);

    printf("# N=%d  threads=%d  L2=%ldKiB  WS=%.1fMiB  sched=%s\n",
           N, nth, l2/1024, ws/1048576.0, ws<=l2?"static":"dynamic");
    printf("%-6s %-12s %-12s %-12s %-12s %-10s %-10s %-10s\n",
           "N","V0_seq","V1_naivepar","V2_ikj_seq","V3_full",
           "Sp_V1","Sp_V2","Sp_V3");

    long tot=(long)N*N;
    double *A=malloc(tot*sizeof(double));
    double *B=malloc(tot*sizeof(double));
    double *C=malloc(tot*sizeof(double));
    if(!A||!B||!C){fprintf(stderr,"OOM\n");return 1;}

    double t0=0,t1=0,t2=0,t3=0;
    for(int r=0;r<runs;r++){
        t0+=v0(A,B,C,N);
        t1+=v1(A,B,C,N);
        t2+=v2(A,B,C,N);
        t3+=v3(A,B,C,N);
    }
    t0/=runs; t1/=runs; t2/=runs; t3/=runs;

    /* correctness */
    double r0,r1,r2,r3;
    v0(A,B,C,N); r0=csum(C,N);
    v1(A,B,C,N); r1=csum(C,N);
    v2(A,B,C,N); r2=csum(C,N);
    v3(A,B,C,N); r3=csum(C,N);

    printf("%-6d %-12.4f %-12.4f %-12.4f %-12.4f %-10.3f %-10.3f %-10.3f\n",
           N,t0,t1,t2,t3, t0/t1, t0/t2, t0/t3);
    printf("\nCorrectness (checksum):\n");
    printf("  V0: %.4f\n  V1: %.4f %s\n  V2: %.4f %s\n  V3: %.4f %s\n",
           r0,
           r1, fabs(r1-r0)<0.5?"OK":"MISMATCH",
           r2, fabs(r2-r0)<0.5?"OK":"MISMATCH",
           r3, fabs(r3-r0)<0.5?"OK":"MISMATCH");
    printf("\nCache analysis (single-thread speedups):\n");
    printf("  V2 vs V0 (interchange only): %.2fx\n", t0/t2);
    printf("  V3 vs V0 (full pipeline):    %.2fx\n", t0/t3);

    free(A);free(B);free(C);
    return 0;
}
