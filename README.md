# GPU_Optimisation_using_CUDA
The fundamentals of GPU and CPU operations. Optimisation of CPU code to run on GPU

Project 1. Evaluation of an Integral using Midpoint Rule. 
Code files: CPUCode, GPUCode. 

The midpoint rule calculates the approximate value of an integral. It is done by obtaining the definite integral from a set of numerical values of the integrand. When there are many values it is hard to solve by hand. 

The test is first conducted using the CPU Code, with the expeonential amount of calculated values. The timings are written on the table. 

Then the same equation is ran on GPU Code. The GPU operations showed much faster runtime, especially with large amount of computations. 

The full report can be seen in project 1 file. 

Project 2. Involves a bit more complicated structure of 2D Transmission Line Matrix (TLM). It is a computationally intensive algorithm that is challenging to run on the CPU. To overcome this limitation the algorithm is ported into GPU. 
