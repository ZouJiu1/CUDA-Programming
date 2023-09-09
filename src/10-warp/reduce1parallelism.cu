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
const int GRID_SIZE = 10240;

void timing(const real *h_x);

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

    timing(d_x);

    free(h_x);
    CHECK(cudaFree(d_x));
    return 0;
}

void __global__ reduce_cp(const real *d_x, real *d_y, const int N)
{
    const int tid = threadIdx.x;    // 某个block内的线程标号 index
    const int bid = blockIdx.x;     // 某个block在网格 grid 内的标号 index
    extern __shared__ real s_y[];    //分配动态共享内存数组，s_y，大小在执行配置那的，每个block都有副本的

    real y = 0.0;        //  分配寄存器变量的

//  累加的步长，累加步长是 block内线程的个数 * grid内block的个数
//  也就是一个网格内所有线程的数量，stride就是网格大小的，stride= 10230 * 128
    const int stride = blockDim.x * gridDim.x;        

// n初始值是所在的线程标号index，然后累加输入 d_x，步长是stride
// 很显然，执行配置grid size * block size = 10230 * 128 << 100000000，也就是分配的线程数量 小于 输入的数组长度
// 所以，一个线程需要累加很多次，约是76次的，也就是 100000000 / (10230 * 128)
    for (int n = bid * blockDim.x + tid; n < N; n += stride)
    {
        y += d_x[n];   // 该线程累加相应的输入，几十次运算
    }
    s_y[tid] = y;      // 线程的数值赋值给  共享内存数组，更加高效的
    __syncthreads();   //  同步线程块内的所有线程，保证共享内存数组复制到了线程的数值

    //在所有线程块内部做累加，直到只剩下32个相邻线程的数组需要累加，也就是剩下一个线程束的线程需要累加
    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1)
    {
        if (tid < offset)  //  block内部的前半部分线程，累加后半部分线程
        {
            s_y[tid] += s_y[tid + offset];  //  共享内存数组前半部分和后半部分累加，存放在前半部分
        }
        __syncthreads();  //  线程块 block 内同步所有线程，个个循环依次执行，等待这个循环全部做好
    }

    real y = s_y[tid]; // 从共享内存复制到寄存器，寄存器的读取和写入更加高效的，延迟更低的呢
    
    // 将线程块划分到线程组，每个线程组的分片大小是32，其实就是线程束的大小
    thread_block_tile<32> g = tiled_partition<32>(this_thread_block());
    for (int i = g.size() >> 1; i > 0; i >>= 1)   //线程组的大小整除以2
    {   //少了mask，线程组内所有线程都参与运算，少了最后的w，就是线程组的大小
        y += g.shfl_down(y, i);   //线程组内部的洗牌函数，向下平移操作的
    }
    
    if (tid == 0)    //  保证某个线程块内只有一个线程运行，也就是只运行一次
    {
        d_y[bid] = y;  //线程块累加的结果，存放到输出的数组内，用另外那个核函数累加
    }
}

real reduce(const real *d_x)
{
    const int ymem = sizeof(real) * GRID_SIZE;
    const int smem = sizeof(real) * BLOCK_SIZE;

    real h_y[1] = {0};
    real *d_y;
    CHECK(cudaMalloc(&d_y, ymem));

    reduce_cp<<<GRID_SIZE, BLOCK_SIZE, smem>>>(d_x, d_y, N);
    /*
第一次线程块内累加的结果，还需要再次累加，得到最后的结果，d_y的长度是grid size，
仍然调用上面的核函数，执行配置是grid size = 1, block size= 1024，
shared_memory = sizeof(real) * 1024，也就是1个block，这个block内有1024个线程，
共享内存大小是shared_memory
此时stride = 1x1024 = 1024，所以for(int n = bid * blockDim.x + tid; n < N; n += stride)
循环只进行了一次，也就是这个循环就起到了赋值的作用，没有累加了的，直接进入到后面的循环，
此时线程的使用率下降了很多，也就是正常规约。
    */
    reduce_cp<<<1, 1024, sizeof(real) * 1024>>>(d_y, d_y, GRID_SIZE);

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


