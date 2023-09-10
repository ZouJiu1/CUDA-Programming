#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <stdio.h>

int main(void)
{
    int N = 10;
    thrust::device_vector<int> x(N, 0);   // 分配设备端的向量
    thrust::device_vector<int> y(N, 0);   // 分配设备端的向量
    for (int i = 0; i < x.size(); ++i)
    {
        x[i] = i + 1; // 初始化
    }
    thrust::inclusive_scan(x.begin(), x.end(), y.begin());// 包含扫描也就是
    for (int i = 0; i < y.size(); ++i)
    {
        printf("%d ", (int) y[i]); // 强制类型转换的
    }
    printf("\n");
    return 0;
}

