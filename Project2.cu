// Students: Jahcorian Ivery, Bella Brickler, Renee Paxson
// Class: CS 4370 - Parallel Programming for Many-core Gpus
// Instructor: Meilin Liu
// Date - 10/18/2024
// Assignment: Project 2

#include <stdio.h>
#include <iostream>
#include <string>

#include <cuda.h>

#define MATRIX_WIDTH 1024
#define BLOCK_SIZE 8
#define TILE_WIDTH 8

void multiply_matrix_cpu(float *matrixA, float *matrixB, float *outputMatrix);
__global__ void multiply_matrix_gpu(float *matrixA, float *matrixB, float *outputMatrix);
void initArray(float *array);
bool matrixEqual(float *matrixA, float *matrixB);
void displayMatrix(float *matrix);

int main()
{
    // Define the matricies necessary for matrix addition
    int arraySize = MATRIX_WIDTH * MATRIX_WIDTH;
    float matrixA[arraySize];
    float matrixB[arraySize];
    float cpuOutput[arraySize];

    // Set all the entries in each matrix to 0
    initArray(matrixA);
    initArray(matrixB);
    initArray(cpuOutput);

    // Populate Matrix A and B with data
    int init = 1325;
    for (int i = 0; i < MATRIX_WIDTH; i++)
    {
        for (int j = 0; j < MATRIX_WIDTH; j++)
        {
            int index = (i * MATRIX_WIDTH) + j;

            init = 3125 * init % 6553;
            matrixA[index] = ((init - 1000) % 6553) % 100;
            matrixB[index] = (init % 251) % 100; // TODO change these back
        }
    }

    // Thread block size
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

    // Caluclate threads per block based of grid width and block size
    int threadX = std::ceil((double)MATRIX_WIDTH / dimBlock.x);
    int threadY = std::ceil((double)MATRIX_WIDTH / dimBlock.y);
    dim3 dimGrid(threadX, threadY, 1);

    // Print array information
    std::cout << "Array size: " << arraySize << std::endl;
    std::cout << "Thread block size: " << BLOCK_SIZE << std ::endl;
    std::cout << "Number of thread blocks: " << threadX * threadY << std::endl;

    clock_t start,
        end; // used to measure the execution time on CPU
    start = clock();

    // Do CPU matrix addition with matrixA and matrixB
    multiply_matrix_cpu(matrixA, matrixB, cpuOutput);

    end = clock();

    // Display how long it took the CPU to execute
    printf("\nCLOCKS_PER_SEC:%ld", CLOCKS_PER_SEC);
    printf("\nNumber of clock ticks:%ld", (end - start));
    printf("\nCPU execution time in seconds:%f\n", (double)(end - start) / CLOCKS_PER_SEC);

    std::cout << "Matrix A:" << std::endl;
    displayMatrix(matrixA);
    std::cout << std::endl;

    std::cout << "Matrix B:" << std::endl;
    displayMatrix(matrixB);
    std::cout << std::endl;

    std::cout << "CPU Output:" << std::endl;
    displayMatrix(cpuOutput);
    std::cout << std::endl;

    // Define GPU matrices
    float hostOutputMatrix[arraySize];
    float *deviceMatrixA, *deviceMatrixB, *deviceMatrixOutput;

    initArray(hostOutputMatrix);

    // Calculate Memory Size of Matrix
    int memSize = sizeof(int) * MATRIX_WIDTH * MATRIX_WIDTH;

    // Allocate memory on the GPU
    cudaMalloc((void **)&deviceMatrixA, memSize);
    cudaMalloc((void **)&deviceMatrixB, memSize);
    cudaMalloc((void **)&deviceMatrixOutput, memSize);

    // Copy matrix from host to device
    cudaMemcpy(deviceMatrixA, matrixA, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMatrixB, matrixB, memSize, cudaMemcpyHostToDevice);

    // Create timing variables
    float timeGPU; // Time the GPU method.
    cudaEvent_t gpuStart, gpuStop;

    // Start timing gpu execution
    cudaEventCreate(&gpuStart);
    cudaEventCreate(&gpuStop);
    cudaEventRecord(gpuStart, 0);

    multiply_matrix_gpu<<<dimGrid, dimBlock>>>(deviceMatrixA, deviceMatrixB, deviceMatrixOutput);

    // Stop timer
    cudaDeviceSynchronize();
    cudaEventRecord(gpuStop, 0);
    cudaEventSynchronize(gpuStop);
    cudaEventElapsedTime(&timeGPU, gpuStart, gpuStop);
    cudaEventDestroy(gpuStart);
    cudaEventDestroy(gpuStop);

    // Copy result back from GPU
    cudaMemcpy(hostOutputMatrix, deviceMatrixOutput, memSize, cudaMemcpyDeviceToHost);

    // Block until cuda operations complete
    cudaDeviceSynchronize();

    // Display results
    std::cout << "GPU Execution Time in seconds: " << timeGPU << std::endl;

    std::cout << "Matrix A:" << std::endl;
    displayMatrix(matrixA);
    std::cout << std::endl;

    std::cout << "Matrix B:" << std::endl;
    displayMatrix(matrixB);
    std::cout << std::endl;

    std::cout << "GPU Output:" << std::endl;
    displayMatrix(hostOutputMatrix);
    std::cout << std::endl;

    // If the matricies from CPU and GPU are equal then print "TEST PASSED"
    if (matrixEqual(cpuOutput, hostOutputMatrix))
    {
        std::cout << "TEST PASSED" << std::endl;
    }

    // Free all cuda memory
    cudaFree(deviceMatrixA);
    cudaFree(deviceMatrixB);
    cudaFree(deviceMatrixOutput);
}

/// @brief Displays the contents of a given array as a matrix, if the MATRIX_WIDTH is larger than 8 -> only displays first row
/// @param matrix The given array for the matrix
void displayMatrix(float *matrix)
{
    if (MATRIX_WIDTH <= 8)
    {
        // Small matrix print entire thing
        for (int i = 0; i < MATRIX_WIDTH; i++)
        {
            // Loop through columns of row
            for (int j = 0; j < MATRIX_WIDTH; j++)
            {
                // Display the value of the current entry
                int index = i * MATRIX_WIDTH + j;
                if (matrix[index] < 0)
                {
                    std::cout << matrix[index] << "  ";
                }
                else
                {
                    std::cout << matrix[index] << "   ";
                }
            }
            std::cout << std::endl;
        }
    }
    else
    {
        // large matrix only print first row
        for (int i = 0; i < MATRIX_WIDTH; i++)
        {
            // Display the value of the current entry
            std::cout << matrix[i] << " ";
        }

        std::cout << std::endl;
    }
}

/// @brief Sets all the values in an array to zero for initialization
/// @param array
void initArray(float *array)
{
    // Loop through rows and columns and init values to 0
    for (int i = 0; i < sizeof(array); i++)
    {
        array[i] = (float)0;
    }
}

/// @brief Determines if two given matricies are equal to one another
/// @param matrixA
/// @param matrixB
/// @return
bool matrixEqual(float *matrixA, float *matrixB)
{
    // Loop through rows and columns of matrix
    for (int i = 0; i < MATRIX_WIDTH; i++)
    {
        for (int j = 0; j < MATRIX_WIDTH; j++)
        {

            // If a single entry doesnt equal return false
            int index = i * MATRIX_WIDTH + j;
            if (matrixA[index] != matrixB[index])
            {
                return false;
            }
        }
    }

    // If we never hit false then the matricies are equal
    return true;
}

/// @brief Performs matrix addition on two matricies using the CPU
/// @param matrixA
/// @param matrixB
/// @param outputMatrix
/// @param matrixWidth
void multiply_matrix_cpu(float *matrixA, float *matrixB, float *outputMatrix)
{
    // Loop through all the rows of the matrix
    for (int i = 0; i < MATRIX_WIDTH; i++)
    {
        // Foreach row loop through each column
        for (int j = 0; j < MATRIX_WIDTH; j++)
        {
            // Declare a sum for the current entries output
            int sum = 0;

            // Loop over the matrix width summing up the matrix mul
            for (int k = 0; k < MATRIX_WIDTH; k++)
            {
                int aIndex = (i * MATRIX_WIDTH) + k;
                int bIndex = (k * MATRIX_WIDTH) + j;

                sum += matrixA[aIndex] * matrixB[bIndex];
            }

            outputMatrix[(i * MATRIX_WIDTH) + j] = sum;
        }
    }
}

/// @brief Performs tiled matrix multiplication on two matricies using the GPU
/// @param matrixA
/// @param matrixB
/// @param outputMatrix
/// @return
__global__ void multiply_matrix_gpu(float *matrixA, float *matrixB, float *outputMatrix)
{
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Calculate what row and column in the grid this thread is working ons
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float pValue = 0;

    for (int i = 0; i < MATRIX_WIDTH / TILE_WIDTH; i++)
    {
        // Loading data into the current tile
        ds_A[ty][tx] = matrixA[(row * MATRIX_WIDTH) + (i * TILE_WIDTH) + tx];
        ds_B[ty][tx] = matrixB[(i * TILE_WIDTH + ty) * MATRIX_WIDTH + col];
        __syncthreads();

        for (int j = 0; j < TILE_WIDTH; j++)
        {
            pValue += ds_A[ty][j] * ds_B[j][tx];
        }
        __syncthreads();
    }

    outputMatrix[row * MATRIX_WIDTH + col] = pValue;
}
