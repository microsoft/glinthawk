#include "ops.cuh"

#include <concepts>
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
constexpr size_t TPB = 64; /* threads per block */

template<std::unsigned_integral T>
T div_ceil( const T x, const T y )
{
  return x / y + ( x % y != 0 );
}

} // namespace

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
                                          const uint64_t size,
                                          const float ss )
{
  const uint64_t i = threadIdx.x + blockIdx.x * TPB;

  if ( i < size ) {
    output[i] = weight[i] * ss * x[i];
  }
}

__global__ void normalize_and_scale_half( __half* output,
                                          const __half* x,
                                          const __half* weight,
                                          const uint64_t size,
                                          const float ss )
{
  const uint64_t i = threadIdx.x + blockIdx.x * TPB;

  if ( i < size ) {
    output[i] = weight[i] * __float2half( ss * __half2float( x[i] ) );
  }
}

template<>
void rmsnorm<float>( float* output, const float* x, const float* weight, const uint64_t size )
{
  float ss = 0.0f;

  CHECK_CUBLAS( cublasSdot( cublas_handle, size, x, 1, x, 1, &ss ) );
  ss /= size;
  ss += 1e-5f;
  ss = 1.0f / sqrtf( ss );

  normalize_and_scale_full<<<div_ceil( size, TPB ), TPB>>>( output, x, weight, size, ss );
}

struct square : public thrust::unary_function<__half, float>
{
  __host__ __device__ float operator()( const __half& x ) const
  {
    const float x_f = __half2float( x );
    return x_f * x_f;
  }
};

template<>
void rmsnorm<__half>( __half* output, const __half* x, const __half* weight, const uint64_t size )
{
  thrust::device_ptr<__half> thrust_x { const_cast<__half*>( x ) };
  float ss = thrust::transform_reduce( thrust_x, thrust_x + size, square(), 0.0f, thrust::plus<float>() );
  ss /= size;
  ss += 1e-5f;
  ss = 1.0f / sqrtf( ss );

  normalize_and_scale_half<<<div_ceil( size, TPB ), TPB>>>( output, x, weight, size, ss );
}

template<>
void softmax<float>( float* _x, const uint64_t size )
{
  thrust::device_ptr<float> x { _x };

  const float max_val = *thrust::max_element( x, x + size );
  const float sum = thrust::transform_reduce(
    x, x + size, [max_val] __device__( const float x ) { return expf( x - max_val ); }, 0.0f, thrust::plus<float>() );
  thrust::transform( x, x + size, x, [sum] __device__( const float x ) { return x / sum; } );
}

template<>
void softmax( __half* _x, const uint64_t size )
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
void attention_0_gemm( const DType* query,
                       const DType* key,
                       DType* att,
                       const uint64_t n_layers,
                       const uint64_t seq_len,
                       const uint64_t head_size,
                       const uint64_t n_heads,
                       const uint64_t n_tokens )
{
  const cudaDataType_t cuda_arg_type = is_same_v<DType, __half> ? CUDA_R_16F : CUDA_R_32F;

  const float alpha = 1.0f / sqrtf( head_size );
  const float beta = 0.0f;

  const uint64_t m = 1;
  const uint64_t n = n_tokens;
  const uint64_t k = head_size;

  const uint64_t lda = m;
  const uint64_t ldb = n_layers * n_heads * head_size * 2;
  const uint64_t ldc = m;

  const uint64_t strideA = head_size;
  const uint64_t strideB = head_size;
  const uint64_t strideC = seq_len;

  const uint64_t batchCount = n_heads;

  CHECK_CUBLAS( cublasGemmStridedBatchedEx( cublas_handle,
                                            CUBLAS_OP_N,
                                            CUBLAS_OP_N,
                                            m,
                                            n,
                                            k,
                                            &alpha,
                                            query,
                                            cuda_arg_type,
                                            lda,
                                            strideA,
                                            key,
                                            cuda_arg_type,
                                            ldb,
                                            strideB,
                                            &beta,
                                            att,
                                            cuda_arg_type,
                                            ldc,
                                            strideC,
                                            batchCount,
                                            CUDA_R_32F,
                                            CUBLAS_GEMM_DEFAULT ) );
}

template<typename DType>
void attention_2_gemm( const DType* att,
                       const DType* value,
                       DType* xb,
                       const uint64_t n_layers,
                       const uint64_t seq_len,
                       const uint64_t head_size,
                       const uint64_t n_heads,
                       const uint64_t n_tokens )
{
  const cudaDataType_t cuda_arg_type = is_same_v<DType, __half> ? CUDA_R_16F : CUDA_R_32F;

  const float alpha = 1.0f;
  const float beta = 0.0f;

  const uint64_t m = head_size;
  const uint64_t n = 1;
  const uint64_t k = n_tokens;

  const uint64_t lda = n_layers * n_heads * head_size * 2;
  const uint64_t ldb = k;
  const uint64_t ldc = m;

  const uint64_t strideA = head_size;
  const uint64_t strideB = seq_len;
  const uint64_t strideC = head_size;

  const uint64_t batchCount = n_heads;

  CHECK_CUBLAS( cublasGemmStridedBatchedEx( cublas_handle,
                                            CUBLAS_OP_N,
                                            CUBLAS_OP_N,
                                            m,
                                            n,
                                            k,
                                            &alpha,
                                            value,
                                            cuda_arg_type,
                                            lda,
                                            strideA,
                                            att,
                                            cuda_arg_type,
                                            ldb,
                                            strideB,
                                            &beta,
                                            xb,
                                            cuda_arg_type,
                                            ldc,
                                            strideC,
                                            batchCount,
                                            CUDA_R_32F,
                                            CUBLAS_GEMM_DEFAULT ) );
}

template<typename DType>
uint32_t sample( const DType* probabilities, const uint64_t n )
{
  throw runtime_error( "not implemented" );
}

template<typename DType>
uint32_t argmax( const DType* _v, const uint64_t n )
{
  thrust::device_ptr<const DType> v { _v };
  const auto it = thrust::max_element( v, v + n );
  return thrust::distance( v, it );
}

template<>
void accum<float>( float* a, const float* b, const uint64_t size )
{
  float alpha = 1.0f;
  cublasSaxpy( cublas_handle, size, &alpha, b, 1, a, 1 );
}

__global__ void accum_cuda( __half* a, const __half* b, const uint64_t size )
{
  const uint64_t i = blockIdx.x * TPB + threadIdx.x;
  if ( i < size ) {
    a[i] += b[i];
  }
}

template<>
void accum<__half>( __half* a, const __half* b, const uint64_t size )
{
  accum_cuda<<<div_ceil( size, TPB ), TPB>>>( a, b, size );
}

template<>
void matmul<float>( float* xout, const float* x, const float* W, const uint64_t n, const uint64_t d )
{
  float alpha = 1.0f;
  float beta = 0.0f;

  // W(d,n) @ x(n,) -> xout(d,)
  CHECK_CUBLAS( cublasSgemv( cublas_handle, CUBLAS_OP_T, n, d, &alpha, W, n, x, 1, &beta, xout, 1 ) );
}

template<>
void matmul<__half>( __half* xout, const __half* x, const __half* W, const uint64_t s, const uint64_t r )
{
  float alpha = 1.0f;
  float beta = 0.0f;

  // W(r,s) @ x(s,) -> xout(r,)
  const uint64_t m = 1;
  const uint64_t n = r;
  const uint64_t k = s;
  const uint64_t lda = m;
  const uint64_t ldb = k;
  const uint64_t ldc = m;

  CHECK_CUBLAS( cublasGemmEx( cublas_handle,
                              CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              m,
                              n,
                              k,
                              &alpha,
                              x,
                              CUDA_R_16F,
                              lda,
                              W,
                              CUDA_R_16F,
                              ldb,
                              &beta,
                              xout,
                              CUDA_R_16F,
                              ldc,
                              CUDA_R_32F,
                              CUBLAS_GEMM_DEFAULT ) );
}

__global__ void silu_direct( float* _hb, const float* _hb2, const uint64_t hidden_dim )
{
  const uint64_t i = threadIdx.x + blockIdx.x * TPB;
  if ( i < hidden_dim ) {
    const float x = _hb[i];
    _hb[i] = x / ( 1.0f + expf( -x ) ) * _hb2[i];
  }
}

__global__ void silu_direct( __half* _hb, const __half* _hb2, const uint64_t hidden_dim )
{
  const uint64_t i = threadIdx.x + blockIdx.x * TPB;
  if ( i < hidden_dim ) {
    const __half x = _hb[i];
    _hb[i] = x / ( __half( 1.0f ) + hexp( -x ) ) * _hb2[i];
  }
}

template<>
void silu<float>( float* _hb, float* _hb2, const uint64_t hidden_dim )
{
  silu_direct<<<div_ceil( hidden_dim, TPB ), TPB>>>( _hb, _hb2, hidden_dim );
}

template<>
void silu<__half>( __half* _hb, __half* _hb2, const uint64_t hidden_dim )
{
  silu_direct<<<div_ceil( hidden_dim, TPB ), TPB>>>( _hb, _hb2, hidden_dim );
}

template void matmul<float>( float* xout, const float* x, const float* w, const uint64_t n, const uint64_t d );
template void matmul<__half>( __half* xout, const __half* x, const __half* w, const uint64_t n, const uint64_t d );

template void rmsnorm<float>( float* output, const float* x, const float* weight, const uint64_t size );
template void rmsnorm<__half>( __half* output, const __half* x, const __half* weight, const uint64_t size );

template uint32_t argmax<float>( const float* v, const uint64_t n );
template uint32_t argmax<__half>( const __half* v, const uint64_t n );

template uint32_t sample<float>( const float* probabilities, const uint64_t n );
template uint32_t sample<__half>( const __half* probabilities, const uint64_t n );

template void accum<float>( float* a, const float* b, const uint64_t size );
template void accum<__half>( __half* a, const __half* b, const uint64_t size );

template void softmax<float>( float* x, const uint64_t size );
template void softmax<__half>( __half* x, const uint64_t size );

template void matmul<float>( float* xout, const float* x, const float* w, const uint64_t n, const uint64_t d );
template void matmul<__half>( __half* xout, const __half* x, const __half* w, const uint64_t n, const uint64_t d );

template void attention_0_gemm<float>( const float* query,
                                       const float* key,
                                       float* att,
                                       const uint64_t n_layers,
                                       const uint64_t seq_len,
                                       const uint64_t head_size,
                                       const uint64_t n_heads,
                                       const uint64_t n_tokens );

template void attention_0_gemm<__half>( const __half* query,
                                        const __half* key,
                                        __half* att,
                                        const uint64_t n_layers,
                                        const uint64_t seq_len,
                                        const uint64_t head_size,
                                        const uint64_t n_heads,
                                        const uint64_t n_tokens );

template void attention_2_gemm<float>( const float* query,
                                       const float* key,
                                       float* att,
                                       const uint64_t n_layers,
                                       const uint64_t seq_len,
                                       const uint64_t head_size,
                                       const uint64_t n_heads,
                                       const uint64_t n_tokens );

template void attention_2_gemm<__half>( const __half* query,
                                        const __half* key,
                                        __half* att,
                                        const uint64_t n_layers,
                                        const uint64_t seq_len,
                                        const uint64_t head_size,
                                        const uint64_t n_heads,
                                        const uint64_t n_tokens );

} // namespace glinthawk::models::common::cuda
