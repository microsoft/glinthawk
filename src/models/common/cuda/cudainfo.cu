#include "cudainfo.cuh"

#include <iostream>

#include <cuda_runtime.h>
#include <glog/logging.h>

using namespace std;
using namespace glinthawk::gpu;

CUDAInfo::CUDAInfo()
{
  cudaDeviceProp prop;
  int device;

  cudaGetDevice( &device );
  cudaGetDeviceProperties( &prop, device );

  max_threads_per_block_ = prop.maxThreadsPerBlock;
}
