#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

static double now_sec(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec+ts.tv_nsec*1e-9;
}
static long get_l2_bytes(void){
    const char *p[]=
        {"/sys/devices/system/cpu/cpu0/cache/index2/size",
         "/sys/devices/system/cpu/cpu0/cache/index1/size",NULL};
    for(int i=0;p[i];i++){
        FILE*f=fopen(p[i],"r"); if(!f)continue;
        long v=0; char u='K'; int rc=fscanf(f,"%ld%c",&v,&u);(void)rc;
        fclose(f); if(v<=0)continue;
        return (u=='M'||u=='m')?v*1024L*1024L:v*1024L;
    }
    return 2L*1024L*1024L;
}
static void init(double*A,double*B,int N){
    long t=(long)N*N;
    for(long i=0;i<t;i++){A[i]=(i%100)*0.01;B[i]=((i+1)%100)*0.01;}
}
static double csum(double*C,int N){
    double s=0;for(long i=0;i<(long)N*N;i++)s+=C[i];return s;
}

/* V0 sequential i,j,k */
static double v0(double*A,double*B,double*C,int N){
    memset(C,0,(long)N*N*sizeof(double));
    double t=now_sec();
    for(int i=0;i<N;i++) for(int j=0;j<N;j++){
        double s=0; for(int k=0;k<N;k++) s+=A[i*N+k]*B[k*N+j]; C[i*N+j]=s;
    }
    return now_sec()-t;
}
/* V1 naive parallel on i */
static double v1(double*A,double*B,double*C,int N){
    memset(C,0,(long)N*N*sizeof(double));
    double t=now_sec();
    #pragma omp parallel for schedule(static)
    for(int i=0;i<N;i++) for(int j=0;j<N;j++){
        double s=0; for(int k=0;k<N;k++) s+=A[i*N+k]*B[k*N+j]; C[i*N+j]=s;
    }
    return now_sec()-t;
}
/* V2 interchange sequential i,k,j */
static double v2(double*A,double*B,double*C,int N){
    memset(C,0,(long)N*N*sizeof(double));
    double t=now_sec();
    for(int i=0;i<N;i++) for(int k=0;k<N;k++) for(int j=0;j<N;j++)
        C[i*N+j]+=A[i*N+k]*B[k*N+j];
    return now_sec()-t;
}
/* V3 full pipeline */
static double v3(double*A,double*B,double*C,int N){
    memset(C,0,(long)N*N*sizeof(double));
    long ws=3L*N*N*(long)sizeof(double);
    long l2=get_l2_bytes();
    double t=now_sec();
    if(ws<=l2){
        #pragma omp parallel for schedule(static)
        for(int i=0;i<N;i++) for(int k=0;k<N;k++) for(int j=0;j<N;j++)
            C[i*N+j]+=A[i*N+k]*B[k*N+j];
    } else {
        #pragma omp parallel for schedule(dynamic,8)
        for(int i=0;i<N;i++) for(int k=0;k<N;k++) for(int j=0;j<N;j++)
            C[i*N+j]+=A[i*N+k]*B[k*N+j];
    }
    return now_sec()-t;
}
static double avg(double*a,int n){double s=0;for(int i=0;i<n;i++)s+=a[i];return s/n;}
static double sd(double*a,int n){double m=avg(a,n),s=0;
    for(int i=0;i<n;i++)s+=(a[i]-m)*(a[i]-m);return sqrt(s/n);}

int main(void){
    int Ns[]={128,256,512,1024}; int nN=4; int RUNS=5;
    int nth=omp_get_max_threads();
    long l2=get_l2_bytes();
    printf("N,threads,l2_kib,ws_mib,schedule,"
           "t_v0_avg,t_v0_std,t_v1_avg,t_v1_std,"
           "t_v2_avg,t_v2_std,t_v3_avg,t_v3_std,"
           "speedup_v1,speedup_v2,speedup_v3,"
           "cs_v0,cs_v1,cs_v2,cs_v3,correct_v1,correct_v2,correct_v3\n");
    for(int ni=0;ni<nN;ni++){
        int N=Ns[ni];
        long tot=(long)N*N;
        double*A=malloc(tot*sizeof(double));
        double*B=malloc(tot*sizeof(double));
        double*C=malloc(tot*sizeof(double));
        if(!A||!B||!C){fprintf(stderr,"OOM N=%d\n",N);continue;}
        init(A,B,N);
        double r0[5],r1[5],r2[5],r3[5];
        for(int r=0;r<RUNS;r++){
            r0[r]=v0(A,B,C,N);r1[r]=v1(A,B,C,N);
            r2[r]=v2(A,B,C,N);r3[r]=v3(A,B,C,N);
        }
        double cs0,cs1,cs2,cs3;
        v0(A,B,C,N);cs0=csum(C,N);
        v1(A,B,C,N);cs1=csum(C,N);
        v2(A,B,C,N);cs2=csum(C,N);
        v3(A,B,C,N);cs3=csum(C,N);
        double a0=avg(r0,RUNS),a1=avg(r1,RUNS),a2=avg(r2,RUNS),a3=avg(r3,RUNS);
        long ws=(long)(3L*N*N*sizeof(double));
        const char*sch=ws<=l2?"static":"dynamic";
        printf("%d,%d,%ld,%.3f,%s,"
               "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
               "%.4f,%.4f,%.4f,"
               "%.2f,%.2f,%.2f,%.2f,%s,%s,%s\n",
               N,nth,l2/1024,ws/1048576.0,sch,
               a0,sd(r0,RUNS),a1,sd(r1,RUNS),
               a2,sd(r2,RUNS),a3,sd(r3,RUNS),
               a0/a1,a0/a2,a0/a3,
               cs0,cs1,cs2,cs3,
               fabs(cs1-cs0)<1?"OK":"FAIL",
               fabs(cs2-cs0)<1?"OK":"FAIL",
               fabs(cs3-cs0)<1?"OK":"FAIL");
        fflush(stdout);
        free(A);free(B);free(C);
    }
}
