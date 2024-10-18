/*
CS 4370 - Parallel Programming
Renee Paxson
Dr. Meilin Liu
Project 2 - Tiled Matrix Multiplication
30 September 2024
*/

#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <string>

#define MATRIX_WIDTH 8
#define BLOCK_SIZE 4
#define TILE_WIDTH 4

// declaring all functions
void matrix_mult_cpu(float* matrixM, float* matrixN, float* matrixP);
__global__ void tiled_matrix_mult_gpu(float* matrixM, float* matrixN, float* matrixP);
void compare(float *matrix, float *device_matrix);
void display(float *matrix);

void matrix_mult_cpu(float* matrixM, float* matrixN, float* matrixP)
{   
    for (int row = 0; row < MATRIX_WIDTH; ++row){

        for (int col = 0; col < MATRIX_WIDTH; ++col) {

            int sum = 0;

            for (int k = 0; k < MATRIX_WIDTH; ++k) {

                float a = matrixM[row * MATRIX_WIDTH + k];
                float b = matrixN[k * MATRIX_WIDTH + col];
                sum += a * b;

            }

            matrixP[row * MATRIX_WIDTH + col] = sum;
        }
    }
}


__global__ void tiled_matrix_mult_gpu(float* matrixM, float* matrixN, float* matrixP)
{

    // tile declaration
	__shared__ float M_tile[TILE_WIDTH * TILE_WIDTH];
	__shared__ float N_tile[TILE_WIDTH * TILE_WIDTH];

	int blockX = blockIdx.x;  
    int blockY = blockIdx.y;
	int threadX = threadIdx.x; 
    int threadY = threadIdx.y;

	// Identify the row and column of the matrixP element to work on
	int row = blockY * TILE_WIDTH + threadY;
	int col = blockX * TILE_WIDTH + threadX;
	
    int Pvalue = 0;

	
    // Loop over the matrixM and matrixN tiles required to compute the Pd element
    // "phase" refers to the phase of the computation that we're on
    // each phase is a new tile
    for (int phase = 0; phase < MATRIX_WIDTH/TILE_WIDTH; ++phase) {

	    // Collaborative loading of matrixM and matrixN tiles into shared memory
	    M_tile[threadY * phase + threadX] = (matrixM[(row * MATRIX_WIDTH) + (phase * TILE_WIDTH) + threadX]);
  	    N_tile[threadY * phase + threadX] = (matrixN[(phase * TILE_WIDTH + threadY) * MATRIX_WIDTH + col]);
  		
        __syncthreads();
	
        
        for (int k = 0; k < TILE_WIDTH; ++k) {
			Pvalue += M_tile[threadY * TILE_WIDTH + k] * N_tile[k * TILE_WIDTH + threadX];
        }

	 	__syncthreads();

	}

	matrixP[row * MATRIX_WIDTH + col] = Pvalue;

    

}

void display(float *matrix) 
{
    for (int row = 0; row < MATRIX_WIDTH; row++)
    { 
        for (int col = 0; col < MATRIX_WIDTH; col++)
        {
            printf("%9.0f ", matrix[row * MATRIX_WIDTH + col]);
        }

        printf("\n");

        if (MATRIX_WIDTH > 8) break;
    }
}

// compare arrays
void compare(float *matrix, float *device_matrix)
{
    for (int row = 0; row < MATRIX_WIDTH; row++)
    {
        for (int col = 0; col < MATRIX_WIDTH; col++)
        {
            if (matrix[row * MATRIX_WIDTH + col] != device_matrix[row * MATRIX_WIDTH + col])
            {
                printf("\nTest FAILED\n");
                return;
            }
        }
    }
    printf("\nTest PASSED\n");
}

int main() 
{

    // cpu array pointers
    float *matrixM, *matrixN, *matrixP, *device_output_P, *device_output_M, *device_output_N;
    matrixM = (float*) malloc(sizeof(int) * MATRIX_WIDTH * MATRIX_WIDTH);
    matrixN = (float*) malloc(sizeof(int) * MATRIX_WIDTH * MATRIX_WIDTH);
    matrixP = (float*) malloc(sizeof(int) * MATRIX_WIDTH * MATRIX_WIDTH);
    device_output_P = (float*) malloc(sizeof(int) * MATRIX_WIDTH * MATRIX_WIDTH);
    device_output_M = (float*) malloc(sizeof(int) * MATRIX_WIDTH * MATRIX_WIDTH);
    device_output_N = (float*) malloc(sizeof(int) * MATRIX_WIDTH * MATRIX_WIDTH);


    // device array pointers
    float *device_M, *device_N, *device_P;
    cudaMalloc((void **)(&device_M), MATRIX_WIDTH * MATRIX_WIDTH * sizeof(int));
    cudaMalloc((void **)(&device_N), MATRIX_WIDTH * MATRIX_WIDTH * sizeof(int));
    cudaMalloc((void **)(&device_P), MATRIX_WIDTH * MATRIX_WIDTH * sizeof(int));


    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid( ceil(double(MATRIX_WIDTH)/dimBlock.x), ceil(double (MATRIX_WIDTH) / dimBlock.y));

    // matrix initialization
    int init = 1325;
    for (int row = 0; row < MATRIX_WIDTH; row++) { 
        for (int col = 0; col < MATRIX_WIDTH; col++) {
            int index = row * MATRIX_WIDTH + col;
            init = (3125 * init) % 65536;
            
            matrixM[index] = (init - 1000) % 6553;
            matrixN[index] = (init % 251);
        }
    }

    // copy matrices matrixM and matrixN to device memory
    cudaMemcpy(device_M, matrixM, MATRIX_WIDTH * MATRIX_WIDTH * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_N, matrixN, MATRIX_WIDTH * MATRIX_WIDTH * sizeof(int), cudaMemcpyHostToDevice);
 

    // cpu matrix multiplication
    matrix_mult_cpu(matrixM, matrixN, matrixP);

    

    // device matrix multiplication
    tiled_matrix_mult_gpu<<<dimGrid, dimBlock>>>(device_M, device_N, device_P);

    // passing the computed matrix back to the CPU
    cudaMemcpy(device_output_P, device_P, MATRIX_WIDTH * MATRIX_WIDTH * sizeof(int), cudaMemcpyDeviceToHost);

    // also passing other matrices
    cudaMemcpy(device_output_M, device_M, MATRIX_WIDTH * MATRIX_WIDTH * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(device_output_N, device_N, MATRIX_WIDTH * MATRIX_WIDTH * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // print/compare M
    printf("CPU multiplication: \nCPU M: \n");
    display(matrixM);

    printf("\nGPU M\n");
    display(device_output_M);

    compare(matrixM, device_output_M);

    //print/compare N
    printf("\nCPU N: \n");
    display(matrixN);

    printf("\nGPU N\n");
    display(device_output_N);

    compare(matrixN, device_output_N);


    // print/compare P
    printf("\nCPU P: \n");
    display(matrixP);

    printf("\nGPU P: \n");
    display(device_output_P);

    compare(matrixP, device_output_P);

    

    free(matrixM);
    free(matrixN);
    free(matrixP);
    free(device_output_P);

    cudaFree(device_M);
    cudaFree(device_N);
    cudaFree(device_P);
    
    return 0;
}