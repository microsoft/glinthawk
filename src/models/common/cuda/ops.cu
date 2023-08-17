#include "ops.cuh"

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

void init() { cublasCreate( &cublas_handle ); }

template<typename DType>
__global__ void normalize_and_scale( DType* output,
                                     const DType* x,
                                     const DType* weight,
                                     const int size,
                                     const DType ss )
{
  const int i = threadIdx.x;
  output[i] = weight[i] * ss * x[i];
}

template<>
void rmsnorm<float>( float* output, const float* x, const float* weight, const int size )
{
  // calculate sum of squares
  float ss = 0.0f;

  cublasSdot( cublas_handle, size, x, 1, x, 1, &ss );
  ss /= size;
  ss += 1e-5f;
  ss = 1.0f / sqrtf( ss );

  normalize_and_scale<<<1, size>>>( output, x, weight, size, ss );
}

template<>
void rmsnorm<__half>( __half* output, const __half* x, const __half* weight, const int size )
{
  throw runtime_error( "not implemented" );
}

template<typename DType>
void softmax( DType* _x, const int size )
{
  thrust::device_ptr<DType> x { _x };

  const DType max_val = *thrust::max_element( x, x + size );
  const DType sum = thrust::transform_reduce(
    x, x + size, [max_val] __device__( const DType x ) { return expf( x - max_val ); }, 0.0f, thrust::plus<DType>() );
  thrust::transform( x, x + size, x, [sum] __device__( const DType x ) { return x / sum; } );
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
  throw runtime_error( "not implemented" );
}

// void rmsnorm( float* o, const float* x, const float* weight, const int size );
// void softmax( float* x, const int size );

template<>
void matmul<float>( float* xout, const float* x, const float* W, const int n, const int d )
{
  float alpha = 1.0f;
  float beta = 0.0f;

  // W(d,n) @ x(n,) -> xout(d,)
  cublasSgemv( cublas_handle, CUBLAS_OP_T, n, d, &alpha, W, n, x, 1, &beta, xout, 1 );
}

template<>
void matmul<__half>( __half* xout, const __half* x, const __half* W, const int n, const int d )
{
  __half alpha = 1.0f;
  __half beta = 0.0f;

  // W(d,n) @ x(n,) -> xout(d,)
  cublasHgemm( cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, n, d, 1, &alpha, W, n, x, 1, &beta, xout, 1 );
}

template<>
void silu<float>( float* _hb, float* _hb2, const int hidden_dim )
{
  thrust::device_ptr<float> hb { _hb };
  thrust::device_ptr<float> hb2 { _hb2 };

  thrust::transform(
    hb, hb + hidden_dim, hb, [] __device__( float x ) { return x * ( 1.0f / ( 1.0f + expf( -x ) ) ); } );
  thrust::transform( hb, hb + hidden_dim, hb2, hb, thrust::multiplies<float>() );
}

template<>
void silu<__half>( __half* _hb, __half* _hb2, const int hidden_dim )
{
  thrust::device_ptr<__half> hb { _hb };
  thrust::device_ptr<__half> hb2 { _hb2 };

  thrust::transform( hb, hb + hidden_dim, hb, [] __device__( __half x ) {
    return x * ( static_cast<__half>( 1.0f ) / ( static_cast<__half>( 1.0f ) + hexp( -x ) ) );
  } );

  thrust::transform( hb, hb + hidden_dim, hb2, hb, thrust::multiplies<__half>() );
}

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
