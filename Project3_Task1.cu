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
#define BLOCK_SIZE 2
#define ARRAY_SIZE 8
// #define TILE_WIDTH 32

int sum_reduction(int *x, int N);
void initArray(int *array);
void displayArray(int *matrix);
__global__ void parallel_sum_reduction(int *input, int *output);

int main()
{
    // Define the matricies necessary for matrix addition
    int array[ARRAY_SIZE];
    int outputCpuArray[ARRAY_SIZE];

    // Init  each arry with data
    initArray(array);
    initArray(outputCpuArray);

    // Thread block size
    dim3 dimBlock(BLOCK_SIZE);

    dim3 dimGrid = (double)ARRAY_SIZE / (2 * dimBlock.x);
    std::cout << dimGrid.x;

    // Print array information
    std::cout << "Array size: " << ARRAY_SIZE << std::endl;
    std::cout << "Thread block size: " << BLOCK_SIZE << std ::endl;
    std::cout << "Number of thread blocks: " << dimGrid.x << std::endl;

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
    displayArray(array);
    std::cout << std::endl;

    std::cout << "CPU Output:" << std::endl;
    displayArray(outputCpuArray);
    std::cout << std::endl;

    // // Define GPU arrays
    int *gpuInput, *gpuOutput;

    int currentOutputSize = dimGrid.x; // Number of thread blocks that will generate answers
    int gpuOutputOnHost[currentOutputSize];

    int inputSize = ARRAY_SIZE;

    float totalGPUTime = 0;

    // Loop through until only 1 thread block executes to get the final answer
    // while (currentOutputSize != 1)
    // {
    // Create arrays on device for this execution
    cudaMalloc((void **)&gpuInput, inputSize * sizeof(int));
    cudaMalloc((void **)&gpuOutput, currentOutputSize * sizeof(int));

    std::cout << currentOutputSize;

    // Copy over the current inputs and
    cudaMemcpy(gpuInput, array, inputSize * sizeof(int), cudaMemcpyHostToDevice);

    // // Create timing variables
    float timeGPU; // Time the GPU method.
    cudaEvent_t gpuStart, gpuStop;

    cudaEventCreate(&gpuStart);
    cudaEventCreate(&gpuStop);
    cudaEventRecord(gpuStart, 0);

    parallel_sum_reduction<<<dimGrid, dimBlock>>>(gpuInput, gpuOutput);

    cudaDeviceSynchronize();
    cudaEventRecord(gpuStop, 0);
    cudaEventSynchronize(gpuStop);
    cudaEventElapsedTime(&timeGPU, gpuStart, gpuStop);
    cudaEventDestroy(gpuStart);
    cudaEventDestroy(gpuStop);

    totalGPUTime += timeGPU;

    cudaMemcpy(gpuOutputOnHost, gpuOutput, currentOutputSize * sizeof(int), cudaMemcpyDeviceToHost);

    // Input to the next reduction is set to the output of the last run
    gpuInput = gpuOutputOnHost;
    inputSize = sizeof(gpuOutputOnHost);

    // Change block sizes
    dimBlock = (inputSize / 2);
    dimGrid = (double)inputSize / (2 * dimBlock.x);

    // update new output size
    currentOutputSize = dimGrid.x;

    // Clean up GPU resources
    cudaFree(gpuInput);
    cudaFree(gpuOutput);

    cudaDeviceSynchronize();
    // }

    // Display results
    std::cout << "GPU Execution Time in seconds: " << totalGPUTime << std::endl;

    std::cout << "GPU Output:" << std::endl;
    displayArray(gpuOutputOnHost);
    std::cout << std::endl;
}

/// @brief Displays the contents of a given array as a matrix, if the MATRIX_WIDTH is larger than 8 -> only displays first row
/// @param matrix The given array for the matrix
void displayArray(int *array)
{
    std::cout << "[ ";

    for (int i = 0; i < sizeof(array); i++)
    {
        std::cout << array[i] << " ";
    }

    std::cout << "]" << std::endl;

    // if (MATRIX_WIDTH <= 8)
    // {
    //     // Small matrix print entire thing
    //     for (int i = 0; i < MATRIX_WIDTH; i++)
    //     {
    //         // Loop through columns of row
    //         for (int j = 0; j < MATRIX_WIDTH; j++)
    //         {
    //             // Display the value of the current entry
    //             int index = i * MATRIX_WIDTH + j;
    //             if (matrix[index] < 0)
    //             {
    //                 std::cout << matrix[index] << "  ";
    //             }
    //             else
    //             {
    //                 std::cout << matrix[index] << "   ";
    //             }
    //         }
    //         std::cout << std::endl;
    //     }
    // }
    // else
    // {
    //     // large matrix only print first row
    //     for (int i = 0; i < MATRIX_WIDTH; i++)
    //     {
    //         // Display the value of the current entry
    //         std::cout << matrix[i] << " ";
    //     }

    //     std::cout << std::endl;
    // }
}

/// @brief Sets all the values in an array to zero for initialization
/// @param array
void initArray(int *array)
{
    // Loop through rows and columns and init values to 0
    for (int i = 0; i < sizeof(array); i++)
    {
        array[i] = (int)0;
    }

    int init = 1325;
    for (int i = 0; i < sizeof(array); i++)
    {
        init = 3125 * init % 6553;
        array[i] = (init - 1000) % 97;
    }
}

/// @brief Determines if two given matricies are equal to one another
/// @param matrixA
/// @param matrixB
/// @return
bool matrixEqual(float *matrixA, float *matrixB)
{
    // // Loop through rows and columns of matrix
    // for (int i = 0; i < MATRIX_WIDTH; i++)
    // {
    //     for (int j = 0; j < MATRIX_WIDTH; j++)
    //     {

    //         // If a single entry doesnt equal return false
    //         int index = i * MATRIX_WIDTH + j;
    //         if (matrixA[index] != matrixB[index])
    //         {
    //             return false;
    //         }
    //     }
    // }

    // // If we never hit false then the matricies are equal
    return true;
}

int sum_reduction(int *x, int N)
{
    for (int i = 1; i < N; i++)
        x[0] = x[0] + x[i];
    int overallSum = x[0];
    return overallSum;
}

__global__ void parallel_sum_reduction(int *input, int *output)
{
    printf("%d", sizeof(input));

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
    output[blockIdx.x] = partialSum[0];
}
