// Students: Jahcorian Ivery, Bella Brickler, Renee Paxson
// Class: CS 4370 - Parallel Programming for Many-core Gpus
// Instructor: Meilin Liu
// Date - 12/2/2024
// Assignment: Project 4

#include <stdio.h>
#include <iostream>
#include <string>

#include <cuda.h>

#define BLOCK_SIZE 1024
#define ARRAY_SIZE 4096

void histogram(unsigned int *buffer, unsigned int *histo);
void initArray(unsigned int *, bool, int);
void displayArray(unsigned int *array, int);
void displayHistogram(unsigned int *hist);
__global__ void histo_kernel(unsigned int *buffer, long size, unsigned int *histo);
bool arrayEqual(unsigned int *array1, unsigned int *array2, long size);

int main()
{
    // Define the arrays needed for calcs
    unsigned array[ARRAY_SIZE];
    unsigned int cpuHisto[256];
    unsigned int deviceHistoOnHost[256];

    // Init  each arry with data
    initArray(array, true, ARRAY_SIZE);
    initArray(cpuHisto, false, 256);

    std::cout << "Array size: " << ARRAY_SIZE << std::endl;

    clock_t start, end; // used to measure the execution time on CPU
    start = clock();

    // Do histo calc on cpu
    histogram(array, cpuHisto);

    end = clock();

    // Display how long it took the CPU to execute
    printf("\nCLOCKS_PER_SEC:%ld", CLOCKS_PER_SEC);
    printf("\nNumber of clock ticks:%ld", (end - start));
    printf("\nCPU execution time in seconds:%f\n", (double)(end - start) / CLOCKS_PER_SEC);

    std::cout << "Array:" << std::endl;
    displayArray(array, ARRAY_SIZE);
    std::cout << std::endl;

    std::cout << "CPU Output:" << std::endl;
    displayHistogram(cpuHisto);
    std::cout << std::endl;

    // Declare the array on the device
    unsigned int *deviceArray;
    unsigned int *deviceHisto;

    cudaMalloc((void **)&deviceArray, (ARRAY_SIZE * sizeof(unsigned int)));
    cudaMalloc((void **)&deviceHisto, (256 * sizeof(unsigned int)));

    // Reset data in histos
    initArray(deviceHistoOnHost, false, 256);

    cudaMemcpy(deviceArray, array, ARRAY_SIZE * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceHisto, deviceHistoOnHost, ARRAY_SIZE * sizeof(unsigned int), cudaMemcpyHostToDevice);

    int dimBlock = BLOCK_SIZE;
    int dimGrid = 1;

    std::cout << "Number of thread blocks: " << dimGrid << std::endl;
    std::cout << "Thread block size: " << dimBlock << std::endl;

    float timeGPU; // Time the GPU method.
    cudaEvent_t gpuStart, gpuStop;

    cudaEventCreate(&gpuStart);
    cudaEventCreate(&gpuStop);
    cudaEventRecord(gpuStart, 0);

    // Run threads
    histo_kernel<<<dimGrid, dimBlock>>>(deviceArray, ARRAY_SIZE, deviceHisto);

    cudaEventRecord(gpuStop, 0);
    cudaEventSynchronize(gpuStop);
    cudaEventElapsedTime(&timeGPU, gpuStart, gpuStop);
    cudaEventDestroy(gpuStart);
    cudaEventDestroy(gpuStop);

    cudaMemcpy(deviceHistoOnHost, deviceHisto, (256 * sizeof(unsigned int)), cudaMemcpyDeviceToHost);

    // Display results
    std::cout << "GPU Execution Time in seconds: " << timeGPU / 60 << std::endl;

    std::cout << "GPU Output:" << std::endl;
    displayHistogram(deviceHistoOnHost);
    std::cout << std::endl;

    // Free up gloabl mem
    cudaFree(deviceHisto);
    cudaFree(deviceArray);

    // Check if results are equal
    if (arrayEqual(deviceHistoOnHost, cpuHisto, 256))
    {
        std::cout << "TEST PASSED" << std::endl;
    }
}

/// @brief Displays the first 10 elements of an array
/// @param hist the 256 int long histo
void displayHistogram(unsigned int *hist)
{
    std::cout << "[ ";

    for (int i = 0; i < 10; i++)
    {
        std::cout << hist[i] << " ";
    }

    std::cout << "]" << std::endl;
}

/// @brief Displays the first 10 elements of an array
/// @param array
/// @param arraySize
void displayArray(unsigned int *array, int arraySize)
{
    std::cout << "[ ";

    if (arraySize > 10)
    {
        arraySize = 10;
    }

    for (int i = 0; i < arraySize; i++)
    {
        std::cout << array[i] << " ";
    }

    std::cout << "]" << std::endl;
}

/// @brief Sets all the values in an array to zero for initialization
/// @param array
/// @param initData
/// @param arraySize
void initArray(unsigned int *array, bool initData, int arraySize)
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
            init = 3125 * init % 65537;
            array[i] = init % 256;
        }
    }
}

/// @brief calcs a histogram of a given array on the cpu
/// @param buffer
/// @param histo
void histogram(unsigned int *buffer, unsigned int *histo)
{
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        histo[buffer[i]] += 1;
    }
}

/// @brief calcs a histogram of a given array on the device
/// @param buffer
/// @param size
/// @param histo
/// @return
__global__ void histo_kernel(unsigned int *buffer, long size, unsigned int *histo)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    // stride is total number of threads
    int stride = blockDim.x * gridDim.x; // All threads handle blockDim.x * gridDim.x
    // consecutive elements in one loop iteration
    while (i < size)
    {
        atomicAdd(&(histo[buffer[i]]), 1);
        i += stride;
    }
}

/// @brief Determines if two arrays contain the same elements in the same order to another
/// @param array1
/// @param array2
/// @param size
/// @return
bool arrayEqual(unsigned int *array1, unsigned int *array2, long size)
{
    for (long i = 0; i < size; i++)
    {
        if (array1[i] != array2[i])
        {
            return false;
        }
    }

    return true;
}
