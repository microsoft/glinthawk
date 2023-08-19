#include "ops.cuh"

#include <source_location>
#include <string>

#include <cublas_v2.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include "util/exception.hh"

using namespace std;

namespace glinthawk::models::common::cuda::ops {

namespace {
cublasHandle_t cublas_handle;
}

void CHECK_CUBLAS( const cublasStatus_t err, const source_location location = source_location::current() )
{
  if ( err != CUBLAS_STATUS_SUCCESS ) {
    throw runtime_error( "CUBLAS error "s + cublasGetStatusName( err ) + ": " + cublasGetStatusString( err ) + " ("
                         + location.file_name() + ":" + std::to_string( location.line() ) + ")" );
  }
}

void init() { cublasCreate( &cublas_handle ); }
void destroy() { cublasDestroy( cublas_handle ); }

__global__ void normalize_and_scale_full( float* output,
                                          const float* x,
                                          const float* weight,
                                          const int size,
                                          const float ss )
{
  const int i = threadIdx.x + blockIdx.x * 1024;
  if (i < size)
    output[i] = weight[i] * ss * x[i];
}

__global__ void normalize_and_scale_half( __half* output,
                                          const __half* x,
                                          const __half* weight,
                                          const int size,
                                          const float ss )
{
  const int i = threadIdx.x + blockIdx.x * 1024;
  if (i < size)
    output[i] = weight[i] * __float2half( ss * __half2float( x[i] ) );
}

template<>
void rmsnorm<float>( float* output, const float* x, const float* weight, const int size )
{
  // calculate sum of squares
  float ss = 0.0f;

  CHECK_CUBLAS( cublasSdot( cublas_handle, size, x, 1, x, 1, &ss ) );
  ss /= size;
  ss += 1e-5f;
  ss = 1.0f / sqrtf( ss );

  normalize_and_scale_full<<<(size+1023)/1024, 1024>>>( output, x, weight, size, ss );
}

template<>
void rmsnorm<__half>( __half* output, const __half* x, const __half* weight, const int size )
{
  // calculate sum of squares
  __half ss_half = __half();

  CHECK_CUBLAS(
    cublasNrm2Ex( cublas_handle, size, x, CUDA_R_16F, 1, &ss_half, CUDA_R_16F, CUDA_R_32F ) );
  float ss = __half2float(ss_half) * __half2float(ss_half) / size;
  ss += 1e-5f;
  ss = 1.0f / sqrtf( ss );

  normalize_and_scale_half<<<(size+1023)/1024, 1024>>>( output, x, weight, size, ss );
}

template<>
void softmax<float>( float* _x, const int size )
{
  thrust::device_ptr<float> x { _x };

  const float max_val = *thrust::max_element( x, x + size );
  const float sum = thrust::transform_reduce(
    x, x + size, [max_val] __device__( const float x ) { return expf( x - max_val ); }, 0.0f, thrust::plus<float>() );
  thrust::transform( x, x + size, x, [sum] __device__( const float x ) { return x / sum; } );
}

template<>
void softmax( __half* _x, const int size )
{
  thrust::device_ptr<__half> x { _x };

  const __half max_val = *thrust::max_element( x, x + size );
  const __half sum = thrust::transform_reduce(
    x,
    x + size,
    [max_val] __device__( const __half x ) { return hexp( x - max_val ); },
    __half(),
    thrust::plus<__half>() );
  thrust::transform( x, x + size, x, [sum] __device__( const __half x ) { return x / sum; } );
}

template<typename DType>
void sample( const DType* probabilities, const int n, int* output )
{
  throw runtime_error( "not implemented" );
}

template<typename DType>
void argmax( const DType* _v, const int n, int* _output )
{
  thrust::device_ptr<const DType> v { _v };
  thrust::device_ptr<int> output { _output };

  const auto it = thrust::max_element( v, v + n );
  *output = thrust::distance( v, it );
}

template<>
void accum<float>( float* a, const float* b, const int size )
{
  float alpha = 1.0f;
  cublasSaxpy( cublas_handle, size, &alpha, b, 1, a, 1 );
}

template<>
void accum<__half>( __half* a, const __half* b, const int size )
{
  float alpha = 1.0f;
  CHECK_CUBLAS(
    cublasAxpyEx( cublas_handle, size, &alpha, CUDA_R_32F, b, CUDA_R_16F, 1, a, CUDA_R_16F, 1, CUDA_R_32F ) );
}

// void rmsnorm( float* o, const float* x, const float* weight, const int size );
// void softmax( float* x, const int size );

template<>
void matmul<float>( float* xout, const float* x, const float* W, const int n, const int d )
{
  float alpha = 1.0f;
  float beta = 0.0f;

  // W(d,n) @ x(n,) -> xout(d,)
  CHECK_CUBLAS( cublasSgemv( cublas_handle, CUBLAS_OP_T, n, d, &alpha, W, n, x, 1, &beta, xout, 1 ) );
}

template<>
void matmul<__half>( __half* xout, const __half* x, const __half* W, const int s, const int r )
{
  float alpha = 1.0f;
  float beta = 0.0f;

  // W(r,s) @ x(s,) -> xout(r,)
  const int m = 1;
  const int n = r;
  const int k = s;
  const int lda = m;
  const int ldb = k;
  const int ldc = m;

  CHECK_CUBLAS(
    cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, x, CUDA_R_16F, lda, W, CUDA_R_16F, ldb, &beta, xout, CUDA_R_16F, ldc, CUDA_R_32F, CUBLAS_GEMM_DEFAULT ) );
}

__global__ void silu_direct(float* _hb, const float* _hb2, const int hidden_dim) {
  const int i = threadIdx.x + blockIdx.x * 1024;
  if (i < hidden_dim){
    const float x = _hb[i];
    _hb[i] = x / (1.0f + expf(-x)) * _hb2[i];
  }
}

__global__ void silu_direct(__half* _hb, const __half* _hb2, const int hidden_dim) {
  const int i = threadIdx.x + blockIdx.x * 1024;
  if (i < hidden_dim){
    const __half x = _hb[i];
    _hb[i] = x / (__half(1.0f) + hexp(-x)) * _hb2[i];
  }
}

template<>
void silu<float>( float* _hb, float* _hb2, const int hidden_dim )
{
  silu_direct<<<(hidden_dim+1023)/1024, 1024>>>( _hb, _hb2, hidden_dim );
}

template<>
void silu<__half>( __half* _hb, __half* _hb2, const int hidden_dim )
{
  silu_direct<<<(hidden_dim+1023)/1024, 1024>>>( _hb, _hb2, hidden_dim );
}

template void matmul<float>( float* xout, const float* x, const float* w, const int n, const int d );
template void matmul<__half>( __half* xout, const __half* x, const __half* w, const int n, const int d );

template void rmsnorm<float>( float* output, const float* x, const float* weight, const int size );
template void rmsnorm<__half>( __half* output, const __half* x, const __half* weight, const int size );

template void argmax<float>( const float* v, const int n, int* output );
template void argmax<__half>( const __half* v, const int n, int* output );

template void sample<float>( const float* probabilities, const int n, int* output );
template void sample<__half>( const __half* probabilities, const int n, int* output );

template void accum<float>( float* a, const float* b, const int size );
template void accum<__half>( __half* a, const __half* b, const int size );

template void softmax<float>( float* x, const int size );
template void softmax<__half>( __half* x, const int size );

template void matmul<float>( float* xout, const float* x, const float* w, const int n, const int d );
template void matmul<__half>( __half* xout, const __half* x, const __half* w, const int n, const int d );

} // namespace glinthawk::models::common::cuda
