#include <stdio.h>

__global__ void helloFromGPU()
{

}

void launchHelloFromGPU()
{
  helloFromGPU<<<1, 10>>>();
  cudaDeviceSynchronize();
}
