#include <stdio.h>
#include <iostream>
#include <string>

#include <cuda.h>

#define BLOCK_SIZE 2
#define ARRAY_SIZE 8

void prefix_sum(int *arrayB, int *arrayA, int array_size);
__global__ void prefix_sum_kernel(int *arrayA, int *arrayB, int array_size);
void initArray(int *, bool, int);
void displayArray(int *, int);

int main()
{
    // define arrays
    int arrayA[ARRAY_SIZE];
    int arrayB[ARRAY_SIZE];

    // init arrays
    initArray(arrayA, true, ARRAY_SIZE);
    // i don't think b needs data
    initArray(arrayB, true, ARRAY_SIZE);

    displayArray(arrayA, ARRAY_SIZE);
    // CPU prefix sum
    prefix_sum(arrayB, arrayA, ARRAY_SIZE);

    displayArray(arrayB, ARRAY_SIZE);

    // // thread block size
    // dim3 dimBlock(BLOCK_SIZE);
    // dim3 dimGrid(ceil(double(ARRAY_SIZE)/(2 * dimBlock.x)));
    // prefix_sum_kernel<<<dimGrid,dimBlock>>>(arrayA, arrayB, ARRAY_SIZE);

    // std::cout << "Array size: " << ARRAY_SIZE << std::endl;

    // prefix_sum(arrayA, arrayB, ARRAY_SIZE);

    // cudaMalloc((void **)&arrayA, (ARRAY_SIZE * sizeof(int)));
    // cudaMalloc((void **)&arrayA, (ARRAY_SIZE * sizeof(int)));

    // cudaMemcpy(gpuInput, array, inputSize * sizeof(int), cudaMemcpyHostToDevice);
}

void prefix_sum(int *arrayB, int *arrayA, int array_size)
{
    arrayB[0] = arrayA[0];
    for (int i = 1; i < array_size; i++)
        arrayB[i] = arrayB[i - 1] + arrayA[i];
}

__global__ void prefix_sum_kernel(int *arrayA, int *arrayB, int array_size)
{

    __shared__ int scan_array[2 * BLOCK_SIZE];

    unsigned int t = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * blockDim.x;
    scan_array[t] = arrayA[start + t];
    scan_array[blockDim.x + t] = arrayA[start + blockDim.x + t];

    __syncthreads();

    //  Reduction step
    int stride = 1;
    int index;

    while (stride <= BLOCK_SIZE)
    {
        index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index < 2 * BLOCK_SIZE)
            scan_array[index] += scan_array[index - stride];
        stride = stride * 2;

        __syncthreads();
    }

    // Post Scan Step
    stride = BLOCK_SIZE / 2;
    while (stride > 0)
    {
        index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index + stride < 2 * BLOCK_SIZE)
        {
            scan_array[index + stride] += scan_array[index];
        }
        stride = stride / 2;
        __syncthreads();
    }

    __syncthreads();

    arrayA[start + t] = scan_array[t];
    arrayA[start + blockDim.x + t] = scan_array[blockDim.x + t];
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
