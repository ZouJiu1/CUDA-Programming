#include "error.cuh"
#include <stdio.h>

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int NUM_REPEATS = 20;
void timing(const real *x, const int N);
real reduce(const real *x, const int N);

int main(void)
{
    const int N = 100000000;
    const int M = sizeof(real) * N;
    real *x = (real *) malloc(M);
    for (int n = 0; n < N; ++n)
    {
        x[n] = 1.23;
    }

    timing(x, N);

    free(x);
    return 0;
}

void timing(const real *x, const int N)
{
    real sum = 0;

    for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));   //  产生事件用来记录起始时间
        CHECK(cudaEventCreate(&stop));    //  产生事件用来记录结束时间
        CHECK(cudaEventRecord(start));    //  记录起始时间
        cudaEventQuery(start);            //  查询起始时间

        sum = reduce(x, N);               //  对数组做累加操作

        CHECK(cudaEventRecord(stop));      //记录结束时间
        CHECK(cudaEventSynchronize(stop));    //同步
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));    // 计算运行的总时间
        printf("Time = %g ms.\n", elapsed_time);              

        CHECK(cudaEventDestroy(start));         //销毁起始结构体
        CHECK(cudaEventDestroy(stop));          //销毁结束结构体
    }

    printf("sum = %f.\n", sum);
}

real reduce(const real *x, const int N)              //  直接对数组做累加操作
{ 
    real sum = 0.0;
    for (int n = 0; n < N; ++n)
    {
        sum += x[n];
    }
    return sum;
}