#include "ops.cuh"

#include <cublas_v2.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include "util/exception.hh"

using namespace std;

namespace glinthawk::models::common::cuda
{

namespace {
cublasHandle_t cublas_handle;
}

__global__ void normalize_and_scale( float* output,
                                     const float* x,
                                     const float* weight,
                                     const int size,
                                     const float ss )
{
  const int i = threadIdx.x;
  output[i] = weight[i] * ss * x[i];
}

void rmsnorm( float* output, const float* x, const float* weight, const int size )
{
  // calculate sum of squares
  float ss = 0.0f;

  cublasSdot( cublas_handle, size, x, 1, x, 1, &ss );
  ss /= size;
  ss += 1e-5f;
  ss = 1.0f / sqrtf( ss );

  normalize_and_scale<<<1, size>>>( output, x, weight, size, ss );
}

void softmax( float* _x, const int size )
{
  thrust::device_ptr<float> x { _x };

  const float max_val = *thrust::max_element( x, x + size );
  const float sum = thrust::transform_reduce(
    x, x + size, [max_val] __device__( const float x ) { return expf( x - max_val ); }, 0.0f, thrust::plus<float>() );
  thrust::transform( x, x + size, x, [sum] __device__( const float x ) { return x / sum; } );
}

void sample( const float* probabilities, const int n, int* output ) { throw runtime_error( "not implemented" ); }

void argmax( const float* _v, const int n, int* _output )
{
  thrust::device_ptr<const float> v { _v };
  thrust::device_ptr<int> output { _output };

  const auto it = thrust::max_element( v, v + n );
  *output = thrust::distance( v, it );
}

void accum( float* a, const float* b, const int size )
{
  float alpha = 1.0f;
  cublasSaxpy( cublas_handle, size, &alpha, b, 1, a, 1 );
}

// void rmsnorm( float* o, const float* x, const float* weight, const int size );
// void softmax( float* x, const int size );

void matmul( float* xout, const float* x, const float* W, const int n, const int d )
{
  float alpha = 1.0f;
  float beta = 0.0f;

  // W(d,n) @ x(n,) -> xout(d,)
  cublasSgemv( cublas_handle, CUBLAS_OP_T, n, d, &alpha, W, n, x, 1, &beta, xout, 1 );
}

void silu( float* _hb, float* _hb2, const int hidden_dim )
{
  thrust::device_ptr<float> hb { _hb };
  thrust::device_ptr<float> hb2 { _hb2 };

  thrust::transform(
    hb, hb + hidden_dim, hb, [] __device__( float x ) { return x * ( 1.0f / ( 1.0f + expf( -x ) ) ); } );
  thrust::transform( hb, hb + hidden_dim, hb2, hb, thrust::multiplies<float>() );
}

}  // namespace glinthawk::models::common::cuda
