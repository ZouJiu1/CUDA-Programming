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

void timing(const real *d_x);

int main(void)
{
    real *h_x = (real *) malloc(M);
    for (int n = 0; n < N; ++n)
    {
        h_x[n] = 1.23;
    }
    real *d_x;
    CHECK(cudaMalloc(&d_x, M));
    CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));

    printf("\nusing atomicAdd:\n");
    timing(d_x);

    free(h_x);
    CHECK(cudaFree(d_x));
    return 0;
}

void __global__ reduce(const real *d_x, real *d_y, const int N)  //输入d_x，输出d_y，个数N
{
    const int tid = threadIdx.x;    // 某个block内的线程标号 index
    const int bid = blockIdx.x;     // 某个block在网格 grid 内的标号 index
    const int n = bid * blockDim.x + tid;  //  blockDim.x是某个block内的线程个数，n就是分配的线程内的标号 index
    extern __shared__ real s_y[];    //分配动态共享内存数组，s_y，大小在执行配置那的，每个block都有副本的
// s_y[tid]是某个block内共享内存数组的第tid个数值，赋值该block内的第tid个线程的值给它
    s_y[tid] = (n < N) ? d_x[n] : 0.0;  // 赋值 d_x 的数值到共享内存数组，每个block的副本复制该block的线程数值
    __syncthreads();  // 线程块 block 内的同步操作，同步该线程块内的所有线程，等待所有线程块的共享内存副本复制数据

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)  //折半规约的，每次缩小一倍
    {
        if (tid < offset)  //  block内部的前半部分线程，累加后半部分线程
        {
            s_y[tid] += s_y[tid + offset];  //  共享内存数组前半部分和后半部分累加，存放在前半部分
        }
        __syncthreads();  //  线程块 block 内同步所有线程，个个循环依次执行，等待这个循环全部做好
    }

    if (tid == 0)      //  保证某个线程块内只有一个线程运行，也就是只运行一次
    {
        atomicAdd(d_y, s_y[0]);   //atomic函数，累加到d_y指针所指的内存内
    }
}

real reduce(const real *d_x)
{
    const int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int smem = sizeof(real) * BLOCK_SIZE;

    real h_y[1] = {0};
    real *d_y;
    CHECK(cudaMalloc(&d_y, sizeof(real)));
    CHECK(cudaMemcpy(d_y, h_y, sizeof(real), cudaMemcpyHostToDevice));

    reduce<<<grid_size, BLOCK_SIZE, smem>>>(d_x, d_y, N);

    CHECK(cudaMemcpy(h_y, d_y, sizeof(real), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_y));

    return h_y[0];
}

void timing(const real *d_x)
{
    real sum = 0;

    for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        sum = reduce(d_x); 

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


