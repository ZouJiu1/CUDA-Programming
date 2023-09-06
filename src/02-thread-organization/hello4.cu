// nvcc -O3 -arch=compute_86 -code=sm_86  --ptxas-options=-v --maxrregcount=20 hello4.cu //GeForce 3080
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>

void __global__ hello_from_gpu()
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    printf("Hello World from block %d and thread %d!\n", bid, tid);
}

int main(void)
{
    hello_from_gpu<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
/*
Hello World from block 1 and thread 0!
Hello World from block 1 and thread 1!
Hello World from block 1 and thread 2!
Hello World from block 1 and thread 3!
Hello World from block 0 and thread 0!
Hello World from block 0 and thread 1!
Hello World from block 0 and thread 2!
Hello World from block 0 and thread 3!
*/