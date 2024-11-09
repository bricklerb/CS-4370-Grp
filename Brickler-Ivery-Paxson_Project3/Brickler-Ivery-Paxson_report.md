# CS 4370 - Parallel Programming
Bella Brickler, Jahcorian Ivery, Renee Paxson

Professor Meilin Liu

8 November 2024

## Project 3: Sum Reduce and Prefix Sum - Project Report

### Report
For this lab, we were able to get our code fully functional and working. We executed the code using the following commands:
`nvcc Project3_Task1.cu -o Project 3_Task1 && ./Project3_Task1.out`
`nvcc Project3_Task2.cu -o Project3_Task2 && ./Project3_Task2.out`

### Task 1 Runtime Table

Time | 128 * 128 | 1024 * 1024 | 4096 * 4096
-----|-----------|-------------|-------------|
CPU Comp Time (s) | 0.011143 | 5.552852 | 1196.504552 |
GPU Comp Time (ms) | 0.075584 (Tile width of 8) / 0.06752 (Tile width of 16) | 6.23139 (Tile width of 8) / 5.28634 (Tile width of 16) | 388.023 (Tile width of 8) / 286.391 (Tile width of 16) / 203.53 (Tile width of 32) |
Speedup | -85% / -84% | -10% / 5% | 208% / 318%

### Execution Results
8*8 matrix with a tile width of 4:

![8*8 matrix with a tile width of 4](./images/8matrix-4tile-output.png)

128*128 matrix with a tile width of 8:

![128*128 matrix with a tile width of 8](./images/128matrix-8tile-output.png)

1024*1024 matrix with a tile width of 16:

![1024*1024 matrix with a tile width of 16](./images/1024matrix-16tile-output.JPG)

4096*4096 matrix with a tile width of 32 (had to zoom out a lot to capture a screenshot, but TEST PASSED can still be seen at the bottom of the image, and a textfile version of this output can be found in the [output-files folder](./output-files/4096matrix-32tile-output.txt)):

![4096*4096 matrix with a tile width of 32](./images/4096matrix-32tile-outtput.JPG)