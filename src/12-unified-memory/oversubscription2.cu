#include "error.cuh"
#include <stdio.h>
#include <stdint.h>

const int N = 30;

__global__ void gpu_touch(uint64_t *x, const size_t size)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        x[i] = 0;
    }
}

int main(void)
{
    for (int n = 1; n <= N; ++n)
    {
        const size_t memory_size = size_t(n) * 1024 * 1024 * 1024;
        const size_t data_size = memory_size / sizeof(uint64_t);
        uint64_t *x;
        CHECK(cudaMallocManaged(&x, memory_size));
        //用到内存了的，不再是预定地址空间，会报错
        gpu_touch<<<(data_size - 1) / 1024 + 1, 1024>>>(x, data_size); 
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());         
   //同步host和device，核函数执行完以前，不会执行后续内容
        CHECK(cudaFree(x));
        printf("Allocated %d GB unified memory with GPU touch.\n", n);
    }
    return 0;
}


