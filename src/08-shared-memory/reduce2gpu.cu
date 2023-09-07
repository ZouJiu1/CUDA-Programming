#include "error.cuh"
#include <stdio.h>

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int NUM_REPEATS = 100;
const int N = 100000000;
const int M = sizeof(real) * N;
const int BLOCK_SIZE = 128;

void timing(real *h_x, real *d_x, const int method);

int main(void)
{
    real *h_x = (real *) malloc(M);
    for (int n = 0; n < N; ++n)
    {
        h_x[n] = 1.23;
    }
    real *d_x;
    CHECK(cudaMalloc(&d_x, M));

    printf("\nUsing global memory only:\n");
    timing(h_x, d_x, 0);
    printf("\nUsing static shared memory:\n");
    timing(h_x, d_x, 1);
    printf("\nUsing dynamic shared memory:\n");
    timing(h_x, d_x, 2);

    free(h_x);
    CHECK(cudaFree(d_x));
    return 0;
}

void __global__ reduce_global(real *d_x, real *d_y)
{
    const int tid = threadIdx.x;                       // 某个block内的线程标号 index
    real *x = d_x + blockDim.x * blockIdx.x;           // 某个block的线程起始地址
 
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)    //  一个block内的线程个数 / 2
    {
        if (tid < offset)                                           //  线程的标号tid要小于对应的边界offset
        {
            x[tid] += x[tid + offset];                              //  某个 block 内部对数组规约
        }
        __syncthreads();                                            //  block 线程块内部的同步函数
    }

    if (tid == 0)                                           //  一个block内只做一次操作，也就是第一个线程tid = 0
    { 
        d_y[blockIdx.x] = x[0];                             //  每个block累加的值复制到输出的全局内存
    }
}

void __global__ reduce_shared(real *d_x, real *d_y)
{
    const int tid = threadIdx.x;              //  某个block内的线程标号 index
    const int bid = blockIdx.x;               //  某个block在网格grid内的标号 index
    const int n = bid * blockDim.x + tid;      //  n 是某个线程的标号 index
    __shared__ real s_y[128];                  //  分配共享内存空间，不同的block都有共享内存变量的副本
    s_y[tid] = (n < N) ? d_x[n] : 0.0; //  每个block的共享内存变量副本，都用全局内存数组d_x来赋值，最后一个多出来的用0
    __syncthreads();  //  线程块内部直接同步

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) // 折半
    {

        if (tid < offset)           // 线程标号的index 不越界  折半
        {
            s_y[tid] += s_y[tid + offset];  //  某个block内的线程做折半规约
        }
        __syncthreads();        // 同步block内部的线程
    }

    if (tid == 0)        // 某个block只做一次操作
    {
        d_y[bid] = s_y[0];     //  复制共享内存变量累加的结果到全局内存
    }
}

void __global__ reduce_dynamic(real *d_x, real *d_y)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    extern __shared__ real s_y[];
    s_y[tid] = (n < N) ? d_x[n] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {

        if (tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        d_y[bid] = s_y[0];
    }
}

real reduce(real *d_x, const int method)
{
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int ymem = sizeof(real) * grid_size;
    const int smem = sizeof(real) * BLOCK_SIZE;
    real *d_y;
    CHECK(cudaMalloc(&d_y, ymem));
    real *h_y = (real *) malloc(ymem);

    switch (method)
    {
        case 0:
            reduce_global<<<grid_size, BLOCK_SIZE>>>(d_x, d_y);
            break;
        case 1:
            reduce_shared<<<grid_size, BLOCK_SIZE>>>(d_x, d_y);
            break;
        case 2:
            reduce_dynamic<<<grid_size, BLOCK_SIZE, smem>>>(d_x, d_y);
            break;
        default:
            printf("Error: wrong method\n");
            exit(1);
            break;
    }

    CHECK(cudaMemcpy(h_y, d_y, ymem, cudaMemcpyDeviceToHost));

    real result = 0.0;
    for (int n = 0; n < grid_size; ++n)
    {
        result += h_y[n];
    }

    free(h_y);
    CHECK(cudaFree(d_y));
    return result;
}

void timing(real *h_x, real *d_x, const int method)
{
    real sum = 0;

    for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
    {
        CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));

        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        sum = reduce(d_x, method);

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms.\n", elapsed_time);

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    printf("sum = %f.\n", sum);
}


