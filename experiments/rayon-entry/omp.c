// OpenMP parallel-region latency for comparison with src/main.rs. Build:
// clang -O2 -Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include \
//   -L$(brew --prefix libomp)/lib -lomp omp.c -o omp

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return t.tv_sec*1e6+t.tv_nsec*1e-3;}
static void spin(double us){double t=now();while(now()-t<us);}
static int cmp(const void*a,const void*b){double x=*(double*)a,y=*(double*)b;return x<y?-1:x>y;}
// Keep the region observable without shared writes: each worker passes its own
// thread number through an empty asm barrier, so there is no shared store (no
// data race, no cache-line contention) and the compiler cannot drop the region.
static inline void keep(int x){__asm__ volatile("" :: "r"(x) : "memory");}
int main(int argc,char**argv){
  int nt=atoi(argv[1]); omp_set_num_threads(nt);
  double gaps[]={0,100,1000,10000}; static double v[2000];
  for(int k=0;k<1000;k++){
    #pragma omp parallel
    { keep(omp_get_thread_num()); }
  }
  for(int g=0;g<4;g++){
    int reps=gaps[g]>=10000?100:2000;
    for(int r=0;r<reps;r++){ spin(gaps[g]); double t=now();
      #pragma omp parallel
      { keep(omp_get_thread_num()); }
      v[r]=now()-t; }
    qsort(v,reps,sizeof(double),cmp);
    printf("omp %2d thr gap %5.0fus: p10 %6.2f p50 %6.2f p90 %6.2f us\n",nt,gaps[g],v[reps/10],v[reps/2],v[reps*9/10]);
  }
}
