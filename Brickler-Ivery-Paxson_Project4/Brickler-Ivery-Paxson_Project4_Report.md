# CS 4370 - Parallel Programming
Bella Brickler, Jahcorian Ivery, Renee Paxson

Professor Meilin Liu

2 December 2024

## Project 4: Histogram

### Report 
For this lab, we were able to get our code fully functional and working. We executed the code using the following commands:
`nvcc Brickler-Ivery-Paxson_Project4_Task1.cu -o Project4_Task1 -arch=sm_30 && ./Project4_Task1`

`nvcc Brickler-Ivery-Paxson_Project4_Task2.cu -o Project4_Task2 -arch=sm_30 && ./Project4_Task12`

### Task 1 Runtime Table

Time | 131072 | 1048576 
CPU Comp Time (s) | X | X
GPU Comp Time (s) | X | X
GPU Mem Transfer time (s) | X | X

### Execution Results - Task 1
#### Array size 131072 with block size of 1024
![task1_131072]()

#### Array size 1048576 with block size of 1024 

![task1_1048576]()

### Task 2 Runtime Table
Time | 131072 | 1048576
CPU Comp Time (s) | X | X
GPU Comp Time (s) | X | X
GPU Mem Transfer time (s) | X | X

### Execution Results - Task 2
#### Array size 131072 with block size of 1024
![task2_131072]()

#### Array size 1048576 with block size of 1024

![task2_1048576]()
