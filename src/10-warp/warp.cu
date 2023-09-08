#include "error.cuh"
#include <stdio.h>

const unsigned WIDTH = 8;
const unsigned BLOCK_SIZE = 16;
const unsigned FULL_MASK = 0xffffffff;

void __global__ test_warp_primitives(void);

int main(int argc, char **argv)
{
    test_warp_primitives<<<1, BLOCK_SIZE>>>();
    CHECK(cudaDeviceSynchronize());         
   //同步host和device，核函数执行完以前，不会执行后续内容
    return 0;
}

void __global__ test_warp_primitives(void)
{
    int tid = threadIdx.x;      //线程在block内部的
    int lane_id = tid % WIDTH; // 束内指标的

    if (tid == 0) printf("threadIdx.x: ");
    printf("%2d ", tid);
    if (tid == 0) printf("\n");

    if (tid == 0) printf("lane_id:     ");
    printf("%2d ", lane_id);
    if (tid == 0) printf("\n");

    //只有tid=0的线程二进制是0  (tid > 0)==False，其他0<tid<16的都返回1   (tid > 0)==True，返回fffe
    unsigned mask1 = __ballot_sync(FULL_MASK, tid > 0);
    //只有tid=0的线程二进制是1 (tid==0)==True，其他0<tid<16的都返回0  (tid==0)==False，返回1
    unsigned mask2 = __ballot_sync(FULL_MASK, tid == 0);
    if (tid == 0) printf("FULL_MASK = %x\n", FULL_MASK);
    if (tid == 1) printf("mask1     = %x\n", mask1);
    if (tid == 0) printf("mask2     = %x\n", mask2);

    //是选举函数，mask全1，16个线程内，第一个线程的tid = 0，不全是0，所以返回的0
    int result = __all_sync(FULL_MASK, tid);
    if (tid == 0) printf("all_sync (FULL_MASK): %d\n", result);
    
    //16个线程内的话，mask1=fffe二进制只有第一个是0，其他的都是1，所以第一个线程不考虑，其他线程的tid > 0，返回1
    result = __all_sync(mask1, tid);
    if (tid == 1) printf("all_sync     (mask1): %d\n", result);
    
    //是选举函数，mask全1，16个线程内，只有第一个线程的tid = 0，其他tid=1，所以至少有一个1，所以返回的1
    result = __any_sync(FULL_MASK, tid);
    if (tid == 0) printf("any_sync (FULL_MASK): %d\n", result);

    //是选举函数，mask2是1，所以只考虑了第一个线程，第一个线程的tid = 0，predicate都是0，所以直接返回的0
    result = __any_sync(mask2, tid);
    if (tid == 0) printf("any_sync     (mask2): %d\n", result);

    //  是广播函数的呢，mask全1，16个线程内，广播线程束内标号是2的线程内变量tid=2的数值到其他参与的线程内,
    //  线程束大小是WIDTH=16/2，
    //  第一个线程束0-7内的线程标号是2的线程内变量tid=2，也就是第3个线程，所以前半部分都是2，
    //  第二个线程束8-15内的线程标号是2的线程内变量tid=10，也就是第11个线程，所以后半部分都是10，
    //  2 2 2 2 2 2 2 2 10 10 10 10 10 10 10 10。
    int value = __shfl_sync(FULL_MASK, tid, 2, WIDTH);
    if (tid == 0) printf("shfl:      ");
    printf("%2d ", value);
    if (tid == 0) printf("\n");

    //  向上平移函数的呢，mask全1，标号是t的参与线程返回标号是t - 1的线程中的数值，t - 1<0的线程返回之前的值
    //  也就是 t = 0时返回之前的值，t >= 1时返回标号是 t - 1的线程中的值，也就是向上平移一个数字
    //  上面所述都是线程束内部的，所以返回值就是
    //  0 0 1 2 3 4 5 6 8 8 9 10 11 12 13 13 + 1
    value = __shfl_up_sync(FULL_MASK, tid, 1, WIDTH);
    if (tid == 0) printf("shfl_up:   ");
    printf("%2d ", value);
    if (tid == 0) printf("\n");

    //  向下平移函数的呢，mask全1，标号是t的参与线程返回标号是t + 1的线程中的数值，t + 1>=WIDTH的线程返回之前的值
    //  也就是 t = 7时返回之前的值，t <= 6时返回标号是 t + 1的线程中的值，也就是向下平移一个数字
    //  上面所述都是线程束内部的，所以返回值就是
    //  1 2 3 4 5 6 7 7 9 10 11 12 13 13+1 15 15 
    value = __shfl_down_sync(FULL_MASK, tid, 1, WIDTH);
    if (tid == 0) printf("shfl_down: ");
    printf("%2d ", value);
    if (tid == 0) printf("\n");

    //  异或函数的呢，mask全1，标号是t的参与线程变量v二进制和标号lanemark线程变量v的二进制做异或操作
    //  0^1=0000^0001=0001=1; 1^1=0001^0001=0000=0; 2^1=0010^0001=0011=3; 3^1=0011^0001=0010=2;
    //  4^1=0100^0001=0101=5; 5^1=0101^0001=0100=4; 6^1=0110^0001=0111=7; 7^1=0111^0001=0110=6;
    //  8^1=1000^0001=1001=9; 9^1=1001^0001=1000=8; 可见是前后两个数字调换位置的
    //  上面所述都是线程束内部的，所以返回值就是
    //  1 0 3 2 5 4 7 6 9 8 11 10 13 12 15 13+1
    value = __shfl_xor_sync(FULL_MASK, tid, 1, WIDTH);
    if (tid == 0) printf("shfl_xor:  ");
    printf("%2d ", value);
    if (tid == 0) printf("\n");
}