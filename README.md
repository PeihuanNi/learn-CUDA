# CUDA编程学习笔记

## CUDA基本概念

CUDA是由一个主机（CPU）和一个设备（GPU）组成的，并且各自拥有**独立的内存**

编程要做的事情，就是编写在CPU和GPU上运行的代码，并且根据代码的需要为CPU和GPU分配内存空间和拷贝数据。运行在GPU上的代码成为**核函数（Kernel）**，Kernel会由大量的硬件线程**并行执行**

### 典型的CUDA程序执行步骤

- 把数据从CPU内存拷贝到GPU
- 调用kernel对存储在GPU里的数据操作
- 将数据从GPU内存传回CPU

CUDA的特色功能：

- **层次结构组织线程**
- **层次结构组织内存**

### 线程层次

- **线程网格（Grid）**
- **线程块（Block）**
- **线程束（Warp）**
- **线程（Thread）**

![线程管理](https://github.com/PeihuanNi/learn-CUDA/blob/master/%E7%BA%BF%E7%A8%8B%E7%AE%A1%E7%90%86.png)

可以看到

- 由一个内核启动产生的所有线程就是Grid
- 同一个网格中的所有线程共享全局内存空间
- 一个网格由多个线程块（Block）组成
- 一个线程块由多个线程束（Warp）组成
- 一个线程数由多个线程（Threads）组成
- 在CUDA中，可以组织三维的Grid和三维的Block

### 内存层次：

- **寄存器（Regsister）：** 最快的内存空间，带宽为8TB/s，**延迟为一个cycle**，在kernel中没有特意声明的变量一般默认在reg中
- **共享内存：** 可受用户控制的一级缓存，可以由CUDA直接编程，带宽为1.5TB/s，**延迟为1～32个cycle**，当存在数据复用时，使用共享内存比较合适
- **常量内存**
- **全局内存：** 容量最大，延迟最高，最常使用的内存

![内存管理](https://github.com/PeihuanNi/learn-CUDA/blob/master/%E5%86%85%E5%AD%98%E7%AE%A1%E7%90%86.png)

## GPU硬件基础

### GPU硬件结构

每个GPU含有多个SM，每个SM又含有多个计算核

![Nvidia's Fermi GPU架构](https://github.com/PeihuanNi/learn-CUDA/blob/master/Fermi%E6%9E%B6%E6%9E%84.png)

可以看到每个SM（Streaming Multiprocesor）包含

- Instruction Cache（指令缓存）
- Warp Scheduler （线程束调度器）：负责将软件线程分布到计算核上
- Dispatch Unit（派遣器）
- Regsister File（寄存器）
- Core（计算核，或者SP，CUDA核，在Fermi架构中有32个）
- LD/ST（Load/Store Units）加载/存储单元，负责将值加载到内存或从内存上加载值
- SFU（Special Function Unit）特殊功能处理模块：用于计算sin，cos，求导或开平方操作
- Interconnect Network（内联网络）
- 64KB Shared Memory / L1 Cache（共享内存 / L1缓存）

每个SM支持数百个线程并发执行。

当启动一个Kernel时，它的线程块会被分布在可用的SM上来执行。当线程块一旦被调度到一个SM上，其中的线程只会在那个指定的SM上并发执行。多个线程块可能会被分配到同一个SM上，而且是根据SM资源的可用性进行调度的。

### 软硬件组织结构对比

CUDA采用单指令多线程架构（**SIMT**，Single Instruction Multiple Threads），每32个线程为1组，称为线程束。

**线程束中所有线程同时执行相同的指令，每个线程有自己的指令地址计数器和寄存器状态**，使用自身的数据执行相同的指令。每个SM都会把自己的线程块划分到有32个线程的线程束中，然后在可用硬件资源上执行。

线程和硬件的关系如下图

![Software with corresponding hardware hierarchy in CUDA](https://github.com/PeihuanNi/learn-CUDA/blob/master/%E8%BD%AF%E4%BB%B6vs%E7%A1%AC%E4%BB%B6.png)

需要注意的是，一个Block只能在一个SM上调度，一旦被调度，就会保存在这个SM上直到执行完成。并且，一个SM可以容纳多个Block。

当线程束闲置的时候（比如等待从内存中读数）是，会执行同一个SM上的常驻线程块中其他可用的线程束。由于资源已经分配到了SM上的所有线程和线程块中，所以切换并发的线程束之间不会有额外开销

## 编程：矩阵加法

```c++
//CPU对照组，用于对比加速比
void sumMatrix2DonCPU(float *MatA, float *MatB, float *MatC, int nx, int ny)
  // 参数为MatA，MatB，MatC的指针；
  // nx：矩阵有多少列，ny：矩阵有多少行
{
    float *a = MatA;
    float *b = MatB;
    float *c = MatC;
    for(int j=0; j<ny; j++)
    {
        for(int i=0; i<nx; i++)
        {
          c[i] = a[i] + b[i];
        }
      	// ptr += nx 就是指针直接移到下一行
        c += nx;
        b += nx;
        a += nx;
    }
}
```
这部分定义了一个在CPU上执行的矩阵加法

这个函数通过指针的方式访问每一个元素，nx表示有多少列（可以理解成矩阵的横坐标最大值），ny表示有多少行（矩阵的列坐标最大值）

所以第一个for要计算的是每一行的向量相加，然后直接让指针加nx，即移到下一行

第二个for循环就是每一行的行向量中的元素按元素相加

```c++
//核函数，每一个线程计算矩阵中的一个元素
__global__ void sumMatrix(float * MatA,float * MatB,float * MatC,int nx,int ny)
{
  	// 核函数就是每一个线程要干的事情，在这里每一个线程要做的就是计算一次加法
  	// ix, iy找的就是线程在所有线程中的的索引
    int ix = threadIdx.x+blockDim.x*blockIdx.x;
    int iy = threadIdx.y+blockDim.y*blockIdx.y;
    int idx = ix+iy*nx;
    if (ix<nx && iy<ny)
    {
        MatC[idx] = MatA[idx]+MatB[idx];
    }
}
```

![线程管理](https://github.com/PeihuanNi/learn-CUDA/blob/master/%E7%BA%BF%E7%A8%8B%E7%AE%A1%E7%90%86.png)

这部分按照这个图来理解

比如我们最终要定位的是`Thread(2,1)`

`threadIdx.x`就是`thread`在这个`block`里的横坐标位置，在这里就是2

`blockDim.x`就是指一个`block`在x轴方向有多少`threads`，在这里是5

`blockIdx.x`就是`block`的索引了，在这里就是`Block(1, 1)`，也就是1

`threadIdx.y, blockDim.y, blockIdx.y`都同理，分别是1，2，1

也就是说，当把所有的线程都按照图示这样排列的话
$$
ix = blockIdx.x \times blcokDim.x + threadIdx.x
$$

$$
iy = blockIdx.y \times blockDim.y + threadIdx.y
$$

$$
idx = ix + iy \times nx
$$

拓展到三维的情况：

- 每个Grid有三维的block

$$
blockId=blockIdx.x+blockIdx.y \times gridDim.x+blockIdx.z \times gridDim.x \times gridDim.y
$$

- 每个Block有三维的Thread

$$
threadId=threadIdx.x+threadIdx.y \times blockDim.x + threadIdx.z \times blcokDim.x \times blockDim.y
$$

- 每个block的线程数M

$$
M=blockDim.x \times blcokDim.y \times blockDim.z
$$

- 所以线程序列号idx

$$
idx=threadId+M \times blockId
$$



**注意，nx是前面说的，这个矩阵的列数，也就是这个矩阵的一行有多少个元素**

**这样最终得到的idx就是在内存中这个线程所需要的数（因为在内存中矩阵也是按照这个顺序一维排列的，因此要把二维坐标转换为一位坐标，并和thread对应上）**

在写main函数之前，先介绍一下cuda的错误处理

一般来说，cuda相关的函数返回值都是`cudaError_t`的枚举类型数据，该类型不仅包含了**成功执行的状态**，还包含了**各种错误状态**

常见的cuda函数返回值有

- `cudaSuccess`，这是`cudaError_t`枚举中的一个值，表示函数成功执行
- `cudaErrorMemoryAllocation`， ` cudaErrorInvalidValue`， `cudaErrorLaunchFailure`

`const char* cudaGetErrorString(cudaError_t error);`这个函数将`cudaError_t`的错误码转换成字符串输出，这样就能看到错误信息是什么了

```c++
#include <cuda_runtime.h>
#include <stdio.h>
#include "cudastart.h"

void initital data(float *matrix, int size);//这个函数在cudastart.h中
//主函数
int main(int argc,char** argv)
{
    //设备初始化
    printf("strating...\n");
    initDevice(0);

    //输入二维矩阵，4096*4096，单精度浮点型。
    int nx = 1<<12;
    int ny = 1<<12;
    int nBytes = nx*ny*sizeof(float);//开辟内存是以字节为单位

    //Malloc，开辟主机内存
    float* A_host = (float*)malloc(nBytes);
    float* B_host = (float*)malloc(nBytes);
    float* C_host = (float*)malloc(nBytes);
  	// 开辟GPU内存，这个内存是在cpu上，存储的是cuda计算完后写回cpu的数据地址
    float* C_from_gpu = (float*)malloc(nBytes);
  	//这个函数在cuda_start.h里，把A和B矩阵初始化
    initialData(A_host, nx*ny);
    initialData(B_host, nx*ny);

    //cudaMalloc，开辟设备内存
    float* A_dev = NULL;
    float* B_dev = NULL;
    float* C_dev = NULL;
  	// 在A，B，C指针的地址（gpu上）开辟出nBytes大小的空间
    CHECK(cudaMalloc((void**)&A_dev, nBytes));
    CHECK(cudaMalloc((void**)&B_dev, nBytes));
    CHECK(cudaMalloc((void**)&C_dev, nBytes));

    //输入数据从主机内存拷贝到设备内存并检查
    CHECK(cudaMemcpy(A_dev, A_host, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_dev, B_host, nBytes, cudaMemcpyHostToDevice));

    //定义二维线程块，一个block程块有32×32的threads
    dim3 block(32, 32);
    //定义二维线程网格，一个grid有128×128的blocks
    dim3 grid((nx-1)/block.x+1, (ny-1)/block.y+1);

    //测试GPU执行时间
    double gpuStart = cpuSecond();
    //将核函数放在线程网格中执行
  	// <<<grid, block, (sharedmMemSize, stream)>>> 这个是cuda的专用语法，就是用于指定核函数的配置
    sumMatrix<<<grid,block>>>(A_dev, B_dev, C_dev, nx, ny);
    CHECK(cudaDeviceSynchronize());//让所有线程同步
    double gpuTime = cpuSecond() - gpuStart;
    printf("GPU Execution Time: %f sec\n", gpuTime);

  
    double cpuStart=cpuSecond();
  	// 在cpu上计算相同的任务
    sumMatrix2DonCPU(A_host, B_host, C_host, nx, ny);
    double cpuTime = cpuSecond() - cpuStart;
    printf("CPU Execution Time: %f sec\n", cpuTime);

    //检查GPU与CPU计算结果是否相同
  	//把gpu输出的数据复制到cpu
    CHECK(cudaMemcpy(C_from_gpu, C_dev, nBytes, cudaMemcpyDeviceToHost));
    checkResult(C_host, C_from_gpu, nx*ny);

    //释放内存
    //gpu端内存
    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C_dev);
  	//cpu端内存
    free(A_host);
    free(B_host);
    free(C_host);
    free(C_from_gpu);
  
    cudaDeviceReset();
  
    return 0;
}
```

cuda_start.h（这个函数是对设备进行初始化）

**`\`是续行符，因为`define`命令需要都在一行，当需要define一个函数的时候，就需要通过`\`提高代码的可读性**

```c++
#ifndef CUDASTART_H
#define CUDASTART_H
#define CHECK(call)\ //
{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
  }\
}
// 解释一下CHECK函数
// 在main函数中可以看到输入的call变量实际上就是cuda函数返回的cudaError_t类型的变量
// 那么CHECK函数的作用就很明显了，判断返回值是不是cudaSuccess
// 如果是，就什么也不敢
// 如果不是，就打印出来错误在哪个文件的第几行，并用 char* cudaGetErroeString(cudaError_t error)函/数返回错误信息输出

#include <time.h>
#ifdef _WIN32
#	include <windows.h>
#else
#	include <sys/time.h>
#endif

double cpuSecond()
{
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return((double)tp.tv_sec+(double)tp.tv_usec*1e-6);

}

void initialData(float* ip,int size)
{
  time_t t;
  srand((unsigned )time(&t));
  for(int i=0;i<size;i++)
  {
    ip[i]=(float)(rand()&0xffff)/1000.0f;
  }
}

void initDevice(int devNum)
{
  int dev = devNum;
  // 下面是获取gpu信息的，并设置使用的设备
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp,dev));
  printf("Using device %d: %s\n",dev,deviceProp.name);
  CHECK(cudaSetDevice(dev));

}
void checkResult(float * hostRef,float * gpuRef,const int N)
{
  double epsilon=1.0E-8;
  for(int i=0;i<N;i++)
  {
    if(abs(hostRef[i]-gpuRef[i])>epsilon)
    {
      printf("Results don\'t match!\n");
      printf("%f(hostRef[%d] )!= %f(gpuRef[%d])\n",hostRef[i],i,gpuRef[i],i);
      return;
    }
  }
  printf("Check result success!\n");
}

#endif

```

