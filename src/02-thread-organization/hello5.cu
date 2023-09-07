// nvcc -O3 -arch=compute_86 -code=sm_86  --ptxas-options=-v --maxrregcount=20 hello5.cu //GeForce 3080
#include <cuda_runtime.h>
#include <cuda.h>

#include <stdio.h>

__global__ void hello_from_gpu()
{
    const int b = blockIdx.x;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    printf("Hello World from block-%d and thread-(%d, %d)!\n", b, tx, ty);
}

int main(void)
{
    const dim3 block_size(3, 6);
    hello_from_gpu<<<1, block_size>>>();
    cudaDeviceSynchronize();         
   //同步host和device，核函数执行完以前，不会执行后续内容
    return 0;
}

/*
Hello World from block-0 and thread-(0, 0)!
Hello World from block-0 and thread-(1, 0)!
Hello World from block-0 and thread-(2, 0)!
Hello World from block-0 and thread-(0, 1)!
Hello World from block-0 and thread-(1, 1)!
Hello World from block-0 and thread-(2, 1)!
Hello World from block-0 and thread-(0, 2)!
Hello World from block-0 and thread-(1, 2)!
Hello World from block-0 and thread-(2, 2)!
Hello World from block-0 and thread-(0, 3)!
Hello World from block-0 and thread-(1, 3)!
Hello World from block-0 and thread-(2, 3)!
Hello World from block-0 and thread-(0, 4)!
Hello World from block-0 and thread-(1, 4)!
Hello World from block-0 and thread-(2, 4)!
Hello World from block-0 and thread-(0, 5)!
Hello World from block-0 and thread-(1, 5)!
Hello World from block-0 and thread-(2, 5)!
*/
