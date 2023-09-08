#include "error.cuh"
#include <stdio.h>
#include <cooperative_groups.h>
using namespace cooperative_groups;

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int NUM_REPEATS = 100;
const int N = 100000000;
const int M = sizeof(real) * N;
const int BLOCK_SIZE = 128;
const unsigned FULL_MASK = 0xffffffff;

void timing(const real *d_x, const int method);

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

    printf("\nusing syncwarp:\n");
    timing(d_x, 0);
    printf("\nusing shfl:\n");
    timing(d_x, 1);
    printf("\nusing cooperative group:\n");
    timing(d_x, 2);

    free(h_x);
    CHECK(cudaFree(d_x));
    return 0;
}

// 折半规约的，d_x是输入，d_y是输出，N是个数
void __global__ reduce_syncwarp(const real *d_x, real *d_y, const int N)  
{
    const int tid = threadIdx.x;    // 某个block内的线程标号 index
    const int bid = blockIdx.x;     // 某个block在网格 grid 内的标号 index
    const int n = bid * blockDim.x + tid;  //  blockDim.x是某个block内的线程个数，n就是分配的线程内的标号 index
    extern __shared__ real s_y[];    //分配动态共享内存数组，s_y，大小在执行配置那的，每个block都有副本的
// s_y[tid]是某个block内共享内存数组的第tid个数值，赋值该block内的第tid个线程的值给它
    s_y[tid] = (n < N) ? d_x[n] : 0.0;  // 赋值 d_x 的数值到共享内存数组，每个block的副本复制该block的线程数值
    __syncthreads();  // 线程块 block 内的同步操作，同步该线程块内的所有线程，等待所有线程块的共享内存副本复制数据

    //在所有线程块内部做累加，直到只剩下32个相邻线程的数组需要累加，也就是剩下一个线程束的线程需要累加
    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1)
    {
        if (tid < offset)  //  block内部的前半部分线程，累加后半部分线程
        {
            s_y[tid] += s_y[tid + offset];  //  共享内存数组前半部分和后半部分累加，存放在前半部分
        }
        __syncthreads();  //  线程块 block 内同步所有线程，个个循环依次执行，等待这个循环全部做好
    }
    //还剩下一个线程束内的线程需要做累加，使用束内同步函数__syncwarp()，更加高效的呢
    for (int offset = 16; offset > 0; offset >>= 1) 
    {
        if (tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncwarp(); //线程束内同步所有线程，等待某个循环全部累加完
    }

    if (tid == 0)      //  保证某个线程块内只有一个线程运行，也就是只运行一次
    {
        atomicAdd(d_y, s_y[0]);   //atomic函数，累加到d_y指针所指的内存内
    }
}

void __global__ reduce_shfl(const real *d_x, real *d_y, const int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    extern __shared__ real s_y[];
    s_y[tid] = (n < N) ? d_x[n] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1)
    {
        if (tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    real y = s_y[tid];

    for (int offset = 16; offset > 0; offset >>= 1)
    {
        y += __shfl_down_sync(FULL_MASK, y, offset);
    }

    if (tid == 0)
    {
        atomicAdd(d_y, y);
    }
}

void __global__ reduce_cp(const real *d_x, real *d_y, const int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    extern __shared__ real s_y[];
    s_y[tid] = (n < N) ? d_x[n] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1)
    {
        if (tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    real y = s_y[tid];

    thread_block_tile<32> g = tiled_partition<32>(this_thread_block());
    for (int i = g.size() >> 1; i > 0; i >>= 1)
    {
        y += g.shfl_down(y, i);
    }

    if (tid == 0)
    {
        atomicAdd(d_y, y);
    }
}

real reduce(const real *d_x, const int method)
{
    const int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int smem = sizeof(real) * BLOCK_SIZE;

    real h_y[1] = {0};
    real *d_y;
    CHECK(cudaMalloc(&d_y, sizeof(real)));
    CHECK(cudaMemcpy(d_y, h_y, sizeof(real), cudaMemcpyHostToDevice));

    switch (method)
    {
        case 0:
            reduce_syncwarp<<<grid_size, BLOCK_SIZE, smem>>>(d_x, d_y, N);
            break;
        case 1:
            reduce_shfl<<<grid_size, BLOCK_SIZE, smem>>>(d_x, d_y, N);
            break;
        case 2:
            reduce_cp<<<grid_size, BLOCK_SIZE, smem>>>(d_x, d_y, N);
            break;
        default:
            printf("Wrong method.\n");
            exit(1);
    }

    CHECK(cudaMemcpy(h_y, d_y, sizeof(real), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_y));

    return h_y[0];
}

void timing(const real *d_x, const int method)
{
    real sum = 0;
    
    for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
    {
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


