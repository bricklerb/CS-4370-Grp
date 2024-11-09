// Students: Jahcorian Ivery, Bella Brickler, Renee Paxson
// Class: CS 4370 - Parallel Programming for Many-core Gpus
// Instructor: Meilin Liu
// Date - 11/8/2024
// Assignment: Project 3

#include <stdio.h>
#include <iostream>
#include <string>

#include <cuda.h>

// #define MATRIX_WIDTH 4096
#define BLOCK_SIZE 512
#define ARRAY_SIZE 131072
// #define TILE_WIDTH 32

int sum_reduction(int *x, int N);
void initArray(int *, bool, int);
void displayArray(int *matrix, int);
__global__ void parallel_sum_reduction(int *input);

int main()
{
    // Define the matricies necessary for matrix addition
    int array[ARRAY_SIZE];
    int outputCpuArray[ARRAY_SIZE];

    // Init  each arry with data
    initArray(array, true, ARRAY_SIZE);
    initArray(outputCpuArray, true, ARRAY_SIZE);

    std::cout << "Array size: " << ARRAY_SIZE << std::endl;

    clock_t start, end; // used to measure the execution time on CPU
    start = clock();

    // Do Sum reduction on CPU
    sum_reduction(outputCpuArray, ARRAY_SIZE);

    end = clock();

    // Display how long it took the CPU to execute
    printf("\nCLOCKS_PER_SEC:%ld", CLOCKS_PER_SEC);
    printf("\nNumber of clock ticks:%ld", (end - start));
    printf("\nCPU execution time in seconds:%f\n", (double)(end - start) / CLOCKS_PER_SEC);

    std::cout << "Array:" << std::endl;
    displayArray(array, ARRAY_SIZE);
    std::cout << std::endl;

    std::cout << "CPU Output:" << std::endl;
    displayArray(outputCpuArray, ARRAY_SIZE);
    std::cout << std::endl;

    // Declare the array on the device
    int inputSize = ARRAY_SIZE;
    int *gpuInput;

    cudaMalloc((void **)&gpuInput, (inputSize * sizeof(int)));
    cudaMemcpy(gpuInput, array, inputSize * sizeof(int), cudaMemcpyHostToDevice);

    int dimBlock = BLOCK_SIZE;
    int dimGrid = ceil((double)ARRAY_SIZE / (2 * dimBlock));
    float totalGPUTime = 0;

    std::cout << "Number of thread blocks: " << dimGrid << std::endl;
    std::cout << "Thread block size: " << dimBlock << std::endl;

    // Loop!!
    while (true)
    {
        // Create timing variables
        float timeGPU; // Time the GPU method.
        cudaEvent_t gpuStart, gpuStop;

        cudaEventCreate(&gpuStart);
        cudaEventCreate(&gpuStop);
        cudaEventRecord(gpuStart, 0);

        parallel_sum_reduction<<<dimGrid, dimBlock>>>(gpuInput);

        cudaDeviceSynchronize();
        cudaEventRecord(gpuStop, 0);
        cudaEventSynchronize(gpuStop);
        cudaEventElapsedTime(&timeGPU, gpuStart, gpuStop);
        cudaEventDestroy(gpuStart);
        cudaEventDestroy(gpuStop);

        totalGPUTime += timeGPU;

        // If only one block was used no need to reduce more
        if (dimGrid <= 1)
        {
            break;
        }

        // we can reduce in one more execution
        int oldDimGrid = dimGrid;
        if (oldDimGrid < 513)
        {
            dimBlock = ceil(oldDimGrid / 2);
        }

        dimGrid = ceil((double)oldDimGrid / (2 * dimBlock));
    }

    int gpuOutputOnHost[ARRAY_SIZE];

    cudaMemcpy(gpuOutputOnHost, gpuInput, (inputSize * sizeof(int)), cudaMemcpyDeviceToHost);

    // Display results
    std::cout << "GPU Execution Time in seconds: " << totalGPUTime << std::endl;

    std::cout << "GPU Output:" << std::endl;
    displayArray(gpuOutputOnHost, ARRAY_SIZE);
    std::cout << std::endl;

    cudaFree(gpuInput);

    if (gpuOutputOnHost[0] == outputCpuArray[0])
    {
        std::cout << "TEST PASSED" << std::endl;
    }
}

/// @brief Displays the contents of a given array as a matrix, if the MATRIX_WIDTH is larger than 8 -> only displays first row
/// @param matrix The given array for the matrix
void displayArray(int *array, int arraySize)
{
    std::cout << "[ ";

    if (arraySize > 19)
    {
        arraySize = 20;
    }

    for (int i = 0; i < arraySize; i++)
    {
        std::cout << array[i] << " ";
    }

    std::cout << "]" << std::endl;
}

/// @brief Sets all the values in an array to zero for initialization
/// @param array
void initArray(int *array, bool initData, int arraySize)
{
    // Loop through rows and columns and init values to 0
    for (int i = 0; i < arraySize; i++)
    {
        array[i] = 0;
    }

    if (initData)
    {
        int init = 1325;
        for (int i = 0; i < arraySize; i++)
        {
            init = 3125 * init % 6553;
            array[i] = (init - 1000) % 97;
        }
    }
}

int sum_reduction(int *x, int N)
{
    for (int i = 1; i < N; i++)
        x[0] = x[0] + x[i];
    int overallSum = x[0];
    return overallSum;
}

__global__ void parallel_sum_reduction(int *input)
{
    __shared__ int partialSum[2 * BLOCK_SIZE];

    unsigned int tx = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * blockDim.x;

    partialSum[tx] = input[start + tx];
    partialSum[blockDim.x + tx] = input[start + blockDim.x + tx];

    for (unsigned int stride = blockDim.x; stride > 0; stride /= 2)
    {
        __syncthreads();
        if (tx < stride)
        {
            partialSum[tx] += partialSum[tx + stride];
        }
    }

    __syncthreads();
    input[blockIdx.x] = partialSum[0];
}