#include <stdio.h>

__global__ void print_kernel() {
    printf("Hello World. I am block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main() {
    print_kernel<<<10, 10>>>();
    cudaDeviceSynchronize();
}