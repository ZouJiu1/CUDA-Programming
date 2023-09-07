/*
nvcc -O3 -arch=compute_86 -code=sm_86  --ptxas-options=-v --maxrregcount=20 add1cpu.cu && ./a.out
Time = 244.885 ms.
Time = 93.1072 ms.
Time = 94.7384 ms.
Time = 94.8828 ms.
Time = 95.2146 ms.
Time = 96.6205 ms.
Time = 94.1804 ms.
Time = 93.9356 ms.
Time = 97.0496 ms.
Time = 97.9046 ms.
Time = 133.256 ms.
Time = 99.089 +- 11.4765 ms.
*/
#include <cuda_runtime.h>
#include <cuda.h>

#include "error.cuh"
#include <math.h>
#include <stdio.h>

#ifdef USE_DP
    typedef double real;
    const real EPSILON = 1.0e-15;
#else
    typedef float real;
    const real EPSILON = 1.0e-6f;
#endif

const int NUM_REPEATS = 10;
const real a = 1.23;
const real b = 2.34;
const real c = 3.57;
void add(const real *x, const real *y, real *z, const int N);
void check(const real *z, const int N);

int main(void)
{
    const int N = 100000000;
    const int M = sizeof(real) * N;
    real *x = (real*) malloc(M);
    real *y = (real*) malloc(M);
    real *z = (real*) malloc(M);

    for (int n = 0; n < N; ++n)
    {
        x[n] = a;
        y[n] = b;
    }

    float t_sum = 0;
    float t2_sum = 0;
    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;         //定义了两个CUDA事件，start和stop，类型是cudaEvent_t
        CHECK(cudaEventCreate(&start));   //初始化
        CHECK(cudaEventCreate(&stop));    //初始化
        CHECK(cudaEventRecord(start));    // record时间戳，表示code block的开始时间，之后的cudaEventRecord仅仅在WDDM模式才是必要的
        cudaEventQuery(start); // cannot use the macro function CHECK here

        // The code block to be timed

        CHECK(cudaEventRecord(stop)); //record时间戳，表示code block的结束时间
        CHECK(cudaEventSynchronize(stop));   //强制host等待上面的code执行完成
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop)); //算出两者的时间间隔，ms，micro second或者1/1000秒
        printf("Time = %g ms.\n", elapsed_time);

        CHECK(cudaEventDestroy(start));   //销毁resource
        CHECK(cudaEventDestroy(stop));    //销毁resource
    }

    const float t_ave = t_sum / NUM_REPEATS;
    const float t_err = sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave);
    printf("Time = %g +- %g ms.\n", t_ave, t_err);

    check(z, N);

    free(x);
    free(y);
    free(z);
    return 0;
}

void add(const real *x, const real *y, real *z, const int N)
{
    for (int n = 0; n < N; ++n)
    {
        z[n] = x[n] + y[n];
    }
}

void check(const real *z, const int N)
{
    bool has_error = false;
    for (int n = 0; n < N; ++n)
    {
        if (fabs(z[n] - c) > EPSILON)
        {
            has_error = true;
        }
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}


