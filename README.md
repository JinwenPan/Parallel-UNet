# Parallel U-Net

This project implements a parallelized version of the U-Net deep learning architecture using OpenMP. The goal is to accelerate U-Net's training and inference by utilizing multiple CPU cores efficiently.

## Project Structure

- **src/**: Contains the source code files.
- **weights/**: Contains kernel weights.
- **training_data/**: Directory to store images and annotations in .png and .csv formats (.csv files were used to run the program on the cluster).
- **Makefile**: Script to compile the project using `g++` with OpenMP support.
- **README.md**: Project documentation (this file).

## Features

- **Parallelization**: Utilizes OpenMP for multithreaded computation, taking advantage of multi-core CPUs.
- **Flexible Compilation**: Easily compile the project with optimization flags using the provided `Makefile`.
- **Efficient Execution**: Significant speedup compared to the serial version of U-Net, especially for large-scale data.

## Prerequisites

Before compiling the project, make sure the following dependencies are installed on your system:

1. **GCC Compiler**: Make sure `g++` is installed. You can check this using:
    ```bash
    g++ --version
    ```

2. **OpenMP**: Used for parallelism in the code. OpenMP support is included in most modern `g++` installations.

3. **X11 Development Libraries**:
    ```bash
    sudo apt-get install libx11-dev
    ```

4. **libpng Development Library**:
    ```bash
    sudo apt-get install libpng-dev
    ```

5. **libjpeg Development Library**:
    ```bash
    sudo apt-get install libjpeg-dev
    ```

6. **Other Common Libraries**:
    - **pthread**: Usually included in most systems by default.
    - **Math Library (libm)**: Included in the standard library.

## Installation and Compilation

1. Clone the repository:
    ```bash
    git clone https://github.com/JinwenPan/Parallel-UNet.git
    cd Parallel-UNet
    ```

2. Compile the code using the provided `Makefile`:
    ```bash
    make
    ```
    This will generate an executable named `unet`.
3. Clean the build flies:
    ```bash
    make clean
    ```

## Usage 

To run the program, 3 command line arguments are needed:
```bash
$ ./unet <imgsize> <batchsize> <num_of_batches>
```
For example:
```bash
$ ./unet 512 8 16
```