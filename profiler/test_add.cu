#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <cub/cub.cuh>

int main()
{
    thrust::host_vector<int> vec_1(500, 10);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    thrust::device_vector<int> vec_2 = vec_1;
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("\n⏱️ Time taken: %.3f\n", milliseconds);

    thrust::host_vector<int> vec_3(500, 10);
    thrust::device_vector<int> d_vec_3(500);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 500; ++i)
    {
        d_vec_3[i] = vec_3[i];
    }
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("\n⏱️ Time taken: %.3f\n", milliseconds);
}