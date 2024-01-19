#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <omp.h>
#define BLOCK_SIZE 256 // define the block size
__global__ void midpoint_rule_kernel(double a, double b, int n, double* result) 
{ 
 int tid = (blockIdx.x * blockDim.x) + threadIdx.x; // index of the current thread
 double delta = (b - a) / n; // width of each subinterval
 if (tid < n) { // check if the index is within the number of subintervals
 double x = exp(-(pow(a + delta * (tid + 0.5), 2))); // midpoint of tid-th subinterval
 result[tid] = x; // store the value of the function in result array
 } 
} 
int main() 
{ 
 // Host Input 
 double a = -2; // lower bound 
 double b = 2; // upper bound
 int n = 65500; // number of subintervals
 double* d_result; // device array to store the result of the kernel function
 // Number of blocks needed to process all subintervals 
 int num_threads = (n + BLOCK_SIZE - 1) / BLOCK_SIZE; // number of threads needed to process all 
subintervals
 size_t size = n * sizeof(double); // size of result array
 // Device Output 
 double* h_result; // host array to store the result of the kernel function
 cudaDeviceSynchronize(); 
 h_result = (double*)malloc(size); // allocate memory for each vector on GPU
 // Allocate memory on GPU and copy result array from the host
 cudaMalloc((void**)&d_result, size); 
 cudaMemcpy(d_result, h_result, size, cudaMemcpyHostToDevice); 
 double start = omp_get_wtime(); // Start Timing
 // Launch kernel function
 midpoint_rule_kernel <<< BLOCK_SIZE, num_threads >>> (a, b, n, d_result); 
 cudaDeviceSynchronize(); 
 // Copy the result array from the device to the host
 cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost); 
 // Free the device memory
 cudaFree(d_result); 
 // Compute the sum of the values in the result array
 double sum = 0; 
 for (int i = 0; i < n; i++) { 
 //printf("Operation %d: %f \n", i, h_result[i]);
 sum += h_result[i]; 
 } 
 // Approximation of the integral using midpoint rule
 double integral = sum * (b - a) / n; 
 printf("The approximate value of the integral is: %f \n", integral); 
 free(h_result); 
 double end = omp_get_wtime(); // End Timing
 printf("start = %.16g\nend = %.16g\ndiff = %.16g\n", start, end, end - start); //in secs
 
 return 0; 
} 
