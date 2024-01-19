#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <omp.h>

#define M_PI 3.14276 // PI
#define c 299792458 // speed of light in a vacuum
#define mu0 M_PI*4e-7 // magnetic permeability in a vacuum 
#define eta0 c*mu0 // wave impedance in free space 

using namespace std;

double** declare_array2D(int, int); // declare all the points in 2 Dimensions

ofstream gaussian_time("gaussian_excitation.out");



// Scatter 
__global__ void tlmScatter(int NX, int NY, double* V1, double* V2, double* V3, double* V4, double Z, double E0) {

   
    int Ein[] = { 10,10 };
    V1[Ein[0] * NY + Ein[1]] = V1[Ein[0] * NY + Ein[1]] + E0;
    V2[Ein[0] * NY + Ein[1]] = V2[Ein[0] * NY + Ein[1]] - E0;
    V3[Ein[0] * NY + Ein[1]] = V3[Ein[0] * NY + Ein[1]] - E0;
    V4[Ein[0] * NY + Ein[1]] = V4[Ein[0] * NY + Ein[1]] + E0;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;


    if (x < NX && y < NY) {
        double I = (2 * V1[x * NY + y] + 2 * V4[x * NY + y] - 2 * V2[x * NY + y] - 2 * V3[x * NY + y]) / (4 * Z);

        double V = 2 * V1[x * NY + y] - I * Z;         //port1
        V1[x * NY + y] = V - V1[x * NY + y];
        V = 2 * V2[x * NY + y] + I * Z;         //port2
        V2[x * NY + y] = V - V2[x * NY + y];
        V = 2 * V3[x * NY + y] + I * Z;         //port3
        V3[x * NY + y] = V - V3[x * NY + y];
        V = 2 * V4[x * NY + y] - I * Z;         //port4
        V4[x * NY + y] = V - V4[x * NY + y];
    }
}


__global__ void tlmConnect(int NX, int NY, double* V1, double* V2, double* V3, double* V4, double rXmin, double rXmax, double rYmin, double rYmax) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < NX && y < NY) {
        //connect
        if (x > 0 && y < NY - 1) {
            double tempV = V2[x * NY + y];
            V2[x * NY + y] = V4[(x - 1) * NY + y];
            V4[(x - 1) * NY + y] = tempV;
        }
        if (x < NX - 1 && y > 0) {
            double tempV = V1[x * NY + y];
            V1[x * NY + y] = V3[x * NY + (y - 1)];
            V3[x * NY + (y - 1)] = tempV;
        }
        //boundary
        if (x < NX && y == NY - 1) {
            V3[x * NY + y] = rYmax * V3[x * NY + y];
        }
        if (x < NX && y == 0) {
            V1[x * NY + y] = rYmin * V1[x * NY + y];
        }
        if (x == NX - 1 && y < NY) {
            V4[x * NY + y] = rXmax * V4[x * NY + y];
        }
        if (x == 0 && y < NY) {
            V2[x * NY + y] = rXmin * V2[x * NY + y];
        }
    }
}


int main()
{

    // Variable Declarations 
    std::clock_t start = std::clock();
    int NX = 100; // number of nodes horizontally 
    int NY = 100; // number of nodes vertically 
    int NT = 100; // number of time steps 
    double dl = 1; // node line segment length
    double dt = dl / (sqrt(2.) * c); // set time step duration


    //2D mesh variables
    double I = 0, tempV = 0, E0 = 0, V = 0;
    double** V1 = declare_array2D(NX, NY);
    double** V2 = declare_array2D(NX, NY);
    double** V3 = declare_array2D(NX, NY);
    double** V4 = declare_array2D(NX, NY);

    double Z = eta0 / sqrt(2.);

    //boundary coefficients
    double rXmin = -1;
    double rXmax = -1;
    double rYmin = -1;
    double rYmax = -1;

    double width = 20 * dt * sqrt(2.); // gaussian width
    double delay = 100 * dt * sqrt(2.); // set time delay before starting
    int Ein[] = { 10,10 };
    int Eout[] = { 15,15 };

    ofstream output("output.out");

    // allocate memory on the GPU
    double* d_V1;
    double* d_V2;
    double* d_V3;
    double* d_V4;

    cudaMalloc((void**)&d_V1, NX * NY * sizeof(double));
    cudaMalloc((void**)&d_V2, NX * NY * sizeof(double));
    cudaMalloc((void**)&d_V3, NX * NY * sizeof(double));
    cudaMalloc((void**)&d_V4, NX * NY * sizeof(double));

    cudaMemcpy(d_V1, V1, NX * NY * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V2, V2, NX * NY * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V3, V3, NX * NY * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V4, V4, NX * NY * sizeof(double), cudaMemcpyHostToDevice);

    // set the grid and block sizes for the kernel
    int blockSize = 32;
    dim3 block(blockSize, blockSize);
    dim3 grid((NX + blockSize - 1) / blockSize, (NY + blockSize - 1) / blockSize);



    for (int n = 0; n < NT; n++) {

        E0 = (1 / sqrt(2.)) * exp(-(n * dt - delay) * (n * dt - delay) / (width * width));
  

        tlmScatter << < grid, block >> > (NX, NY, d_V1, d_V2, d_V3, d_V4, Z, E0);
        tlmConnect << < grid, block >> > (NX, NY, d_V1, d_V2, d_V3, d_V4, rXmin, rXmax, rYmin, rYmax);

    }

    cudaDeviceSynchronize();

    cudaMemcpy(V1, d_V1, NX * NY * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(V2, d_V2, NX * NY * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(V3, d_V3, NX * NY * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(V4, d_V4, NX * NY * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_V1);
    cudaFree(d_V2);
    cudaFree(d_V3);
    cudaFree(d_V4);



    output.close();
    cout << "Done";
    std::cout << ((std::clock() - start) / (double)CLOCKS_PER_SEC) << '\n';
    cin.get();



    return 0;
}




double** declare_array2D(int NX, int NY) {
    double** V = new double* [NX];
    for (int x = 0; x < NX; x++) {
        V[x] = new double[NY];
    }

    for (int x = 0; x < NX; x++) {
        for (int y = 0; y < NY; y++) {
            V[x][y] = 0;
        }
    }
    return V;
}