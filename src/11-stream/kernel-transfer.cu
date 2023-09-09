#include "error.cuh"
#include <math.h>
#include <stdio.h>

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int NUM_REPEATS = 10;
const int N = 1 << 22;
const int M = sizeof(real) * N;
const int MAX_NUM_STREAMS = 64;
cudaStream_t streams[MAX_NUM_STREAMS];

void timing
(
    const real *h_x, const real *h_y, real *h_z,
    real *d_x, real *d_y, real *d_z,
    const int num
);

int main(void)
{
    real *h_x, *h_y, *h_z;
    CHECK(cudaMallocHost(&h_x, M)); //host分配不可分页内存
    CHECK(cudaMallocHost(&h_y, M)); //host分配不可分页内存
    CHECK(cudaMallocHost(&h_z, M)); //host分配不可分页内存
    for (int n = 0; n < N; ++n)
    {
        h_x[n] = 1.23;
        h_y[n] = 2.34;
    }

    real *d_x, *d_y, *d_z;
    CHECK(cudaMalloc(&d_x, M));
    CHECK(cudaMalloc(&d_y, M));
    CHECK(cudaMalloc(&d_z, M));

    for (int i = 0; i < MAX_NUM_STREAMS; i++)
    {
        CHECK(cudaStreamCreate(&(streams[i])));    // 产生CUDA流
    }
    //给定不同的分段方案，num就是数据需要分开的段落，每次乘2
    for (int num = 1; num <= MAX_NUM_STREAMS; num *= 2)
    {
        timing(h_x, h_y, h_z, d_x, d_y, d_z, num);            //  在CUDA流内执行操作
    }

    for (int i = 0 ; i < MAX_NUM_STREAMS; i++)
    {            //  销毁CUDA流
        CHECK(cudaStreamDestroy(streams[i]));
    }

    CHECK(cudaFreeHost(h_x));
    CHECK(cudaFreeHost(h_y));
    CHECK(cudaFreeHost(h_z));
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));
    CHECK(cudaFree(d_z));

    return 0;
}

void __global__ add(const real *x, const real *y, real *z, int N)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N) //由于已经分配好了grid_size，所以多数的线程不会>N，只有最后那个分片有可能出现>N的线程
    {
        for (int i = 0; i < 40; ++i)
        {
            z[n] = x[n] + y[n];
        }
    }
}

void timing
(
    const real *h_x, const real *h_y, real *h_z,
    real *d_x, real *d_y, real *d_z,
    const int num
)
{   //  数据分成了num段，也就是需要num个流来处理的
    int N1 = N / num;
    int M1 = M / num;

    float t_sum = 0;
    float t2_sum = 0;
    
    //  重复的次数用来测量耗时
    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        //  交替执行就可以实现，数据传输和核函数并发执行
        for (int i = 0; i < num; i++) // 执行num个段落，异步
        {
            int offset = i * N1;
            CHECK(cudaMemcpyAsync(d_x + offset, h_x + offset, M1, 
                cudaMemcpyHostToDevice, streams[i]));  //第i个流异步传输输入段落，host不等待
            CHECK(cudaMemcpyAsync(d_y + offset, h_y + offset, M1, 
                cudaMemcpyHostToDevice, streams[i])); //第i个流异步传输输入段落，host不等待

            int block_size = 128;
            int grid_size = (N1 - 1) / block_size + 1; //算grid size
            //  在第i个流执行核函数
            add<<<grid_size, block_size, 0, streams[i]>>>
            (d_x + offset, d_y + offset, d_z + offset, N1);

            CHECK(cudaMemcpyAsync(h_z + offset, d_z + offset, M1, 
                cudaMemcpyDeviceToHost, streams[i]));//第i个流异步传输输入段落，host不等待
        }

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));

        if (repeat > 0)
        {
            t_sum += elapsed_time;
            t2_sum += elapsed_time * elapsed_time;
        }

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }
    // 输出数据分片的数量或者使用到的流的个数num，以及耗时的
    const float t_ave = t_sum / NUM_REPEATS;
    const float t_err = sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave);
    printf("%d %g\n", num, t_ave);
}