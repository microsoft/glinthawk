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

cudaStream_t* streams;
cublasHandle_t cublas_handle_default;
cublasHandle_t* cublas_handle_array;
int cublas_handle_count;

cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
//cublasComputeType_t computeType = CUBLAS_COMPUTE_32F_PEDANTIC;
const float alpha = 1.0f;
const float beta = 0.0f;

// cublasComputeType_t computeType = CUBLAS_COMPUTE_16F;
// cublasComputeType_t computeType = CUBLAS_COMPUTE_16F_PEDANTIC;
// const __half alpha = __half(1.0f);
// const __half beta = __half(0.0f);

template<unsigned_integral T>
T div_ceil( const T x, const T y )
{
  return x / y + ( x % y != 0 );
}

} // namespace

void CHECK_CUBLAS( const cublasStatus_t err, const source_location location = source_location::current() )
{
  if ( err != CUBLAS_STATUS_SUCCESS ) {
    throw runtime_error( "CUBLAS error "s + cublasGetStatusName( err ) + ": " + cublasGetStatusString( err ) + " ("
                         + location.file_name() + ":" + to_string( location.line() ) + ")" );
  }
}

void CHECK_CUDA( const cudaError_t err, const source_location location )
{
  if ( err != cudaSuccess ) {
    throw runtime_error( "CUDA error " + string( cudaGetErrorName( err ) ) + ": " + string( cudaGetErrorString( err ) )
                         + " (" + location.file_name() + ":" + to_string( location.line() ) + ")" );
  }
}

void init( const int num_streams )
{
  cublasCreate( &cublas_handle_default );
  cublas_handle_count = num_streams;
//  cublasSetMathMode(cublas_handle_default, CUBLAS_PEDANTIC_MATH);
  streams = (cudaStream_t*)malloc( num_streams * sizeof( cudaStream_t ) );
  cublas_handle_array = (cublasHandle_t*)malloc( num_streams * sizeof( cublasHandle_t ) );
  for ( int i = 0; i < num_streams; i++ ) {
    cudaStreamCreate( &( streams[i] ) );
    cublasCreate( &( cublas_handle_array[i] ) );
//    cublasSetMathMode(cublas_handle_array[i], CUBLAS_PEDANTIC_MATH);
    cublasSetStream( cublas_handle_array[i], streams[i] );
  }
}

void destroy()
{
  cublasDestroy( cublas_handle_default );
  for ( int i = 0; i < cublas_handle_count; i++ ) {
    cublasDestroy( cublas_handle_array[i] );
    cudaStreamDestroy( streams[i] );
  }
  free( cublas_handle_array );
  free( streams );
}

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
void rmsnorm<float>( float* output,
                     const float* x,
                     const float* weight,
                     const uint64_t size,
                     const uint64_t batch_size )
{
  // TODO: optimize batching
  // Doing the dumbest batching possible and optimizing later
  for ( size_t i = 0; i < batch_size; i++ ) {
    float ss = 0.0f;

    CHECK_CUBLAS( cublasSdot( cublas_handle_default, size, x + i * size, 1, x + i * size, 1, &ss ) );
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf( ss );

    normalize_and_scale_full<<<div_ceil( size, TPB ), TPB>>>( output + i * size, x + i * size, weight, size, ss );
  }
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
void rmsnorm<__half>( __half* output,
                      const __half* x,
                      const __half* weight,
                      const uint64_t size,
                      const uint64_t batch_size )
{
  // TODO: optimize batching
  // Doing the dumbest batching possible and optimizing later
  for ( size_t i = 0; i < batch_size; i++ ) {
    thrust::device_ptr<__half> thrust_x { const_cast<__half*>( x + i * size ) };
    float ss = thrust::transform_reduce( thrust_x, thrust_x + size, square(), 0.0f, thrust::plus<float>() );
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf( ss );

    normalize_and_scale_half<<<div_ceil( size, TPB ), TPB>>>( output + i * size, x + i * size, weight, size, ss );
  }
}

template<>
void softmax<float>( float* _x, const uint64_t size, const uint64_t batch_size )
{
  // optimize batching
  // Doing the dumbest batching possible and optimizing later
  for ( size_t i = 0; i < batch_size; i++ ) {
    thrust::device_ptr<float> x { _x + i * size };

    const float max_val = *thrust::max_element( x, x + size );
    const float sum = thrust::transform_reduce(
      x, x + size, [max_val] __device__( const float x ) { return expf( x - max_val ); }, 0.0f, thrust::plus<float>() );
    thrust::transform( x, x + size, x, [sum] __device__( const float x ) { return x / sum; } );
  }
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
void softmax( __half* _x, const uint64_t size, const uint64_t batch_size )
{
  // optimize batching
  // Doing the dumbest batching possible and optimizing later
  for ( size_t i = 0; i < batch_size; i++ ) {
    thrust::device_ptr<__half> x { _x + i * size };
    const __half max_val = *thrust::max_element( x, x + size );
    const __half sum = thrust::transform_reduce(
      x,
      x + size,
      [max_val] __device__( const __half x ) { return hexp( x - max_val ); },
      __half(),
      thrust::plus<__half>() );
    thrust::transform( x, x + size, x, [sum] __device__( const __half x ) { return x / sum; } );
  }
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
void copy_kv_cache( const DType* state_k,
                    const DType* state_v,
                    DType* key_base,
                    DType* value_base,
                    const uint64_t dim,
                    const uint64_t n_layers,
                    const uint64_t batch_size,
                    const uint64_t max_batch_size,
                    vector<uint64_t> id_alloc_s,
                    vector<uint64_t> token_pos_s )
{
  const uint64_t large_base = n_layers * dim * max_batch_size * 2;
  const uint64_t small_base = dim;
  for ( size_t i = 0; i < batch_size; i++ ) {
    DType* k_cache_pos = key_base + token_pos_s[i] * large_base + id_alloc_s[i] * small_base;
    DType* v_cache_pos = value_base + token_pos_s[i] * large_base + id_alloc_s[i] * small_base;
    ops::CHECK_CUDA(
      cudaMemcpyAsync( k_cache_pos, state_k + i * dim, dim * sizeof( DType ), cudaMemcpyDeviceToDevice, streams[i] ) );
    ops::CHECK_CUDA(
      cudaMemcpyAsync( v_cache_pos, state_v + i * dim, dim * sizeof( DType ), cudaMemcpyDeviceToDevice, streams[i] ) );
  }
}

template<typename DType>
void attention_0_gemm( const DType* query,
                       const DType* key_base,
                       DType* att,
                       const uint64_t n_layers,
                       const uint64_t seq_len,
                       const uint64_t head_size,
                       const uint64_t n_heads,
                       const uint64_t batch_size,
                       const uint64_t max_batch_size,
                       vector<uint64_t> id_alloc_s,
                       vector<uint64_t> token_pos_s )
{
  const cudaDataType_t cuda_arg_type = is_same_v<DType, __half> ? CUDA_R_16F : CUDA_R_32F;
  const float scale = 1.0f / sqrtf( head_size );

  const uint64_t m = 1;
  const uint64_t k = head_size;

  const uint64_t lda = m;
  const uint64_t ldb = n_layers * max_batch_size * n_heads * head_size * 2;
  const uint64_t ldc = m;

  const uint64_t strideA = head_size;
  const uint64_t strideB = head_size;
  const uint64_t strideC = seq_len;

  const uint64_t batchCount = n_heads;

  const uint64_t dim_ = head_size * n_heads;
  const uint64_t att_dim_ = seq_len * n_heads;

  for ( size_t i = 0; i < batch_size; i++ ) {
    const uint64_t n = token_pos_s[i] + 1;
    CHECK_CUBLAS( cublasGemmStridedBatchedEx( cublas_handle_array[i],
                                              CUBLAS_OP_N,
                                              CUBLAS_OP_N,
                                              m,
                                              n,
                                              k,
                                              &scale,
                                              query + i * dim_,
                                              cuda_arg_type,
                                              lda,
                                              strideA,
                                              key_base + id_alloc_s[i] * dim_,
                                              cuda_arg_type,
                                              ldb,
                                              strideB,
                                              &beta,
                                              att + i * att_dim_,
                                              cuda_arg_type,
                                              ldc,
                                              strideC,
                                              batchCount,
                                              computeType,
                                              CUBLAS_GEMM_DEFAULT ) );
  }
}

template<typename DType>
void attention_2_gemm( const DType* att,
                       const DType* value_base,
                       DType* xb,
                       const uint64_t n_layers,
                       const uint64_t seq_len,
                       const uint64_t head_size,
                       const uint64_t n_heads,
                       const uint64_t batch_size,
                       const uint64_t max_batch_size,
                       vector<uint64_t> id_alloc_s,
                       vector<uint64_t> token_pos_s )
{
  const cudaDataType_t cuda_arg_type = is_same_v<DType, __half> ? CUDA_R_16F : CUDA_R_32F;

  const uint64_t m = head_size;
  const uint64_t n = 1;

  const uint64_t lda = n_layers * max_batch_size * n_heads * head_size * 2;
  const uint64_t ldc = m;

  const uint64_t strideA = head_size;
  const uint64_t strideB = seq_len;
  const uint64_t strideC = head_size;

  const uint64_t batchCount = n_heads;

  const uint64_t dim_ = head_size * n_heads;
  const uint64_t att_dim_ = seq_len * n_heads;

  for ( size_t i = 0; i < batch_size; i++ ) {
    const uint64_t k = token_pos_s[i] + 1;
    const uint64_t ldb = k;

    CHECK_CUBLAS( cublasGemmStridedBatchedEx( cublas_handle_array[i],
                                              CUBLAS_OP_N,
                                              CUBLAS_OP_N,
                                              m,
                                              n,
                                              k,
                                              &alpha,
                                              value_base + id_alloc_s[i] * dim_,
                                              cuda_arg_type,
                                              lda,
                                              strideA,
                                              att + i * att_dim_,
                                              cuda_arg_type,
                                              ldb,
                                              strideB,
                                              &beta,
                                              xb + i * dim_,
                                              cuda_arg_type,
                                              ldc,
                                              strideC,
                                              batchCount,
                                              computeType,
                                              CUBLAS_GEMM_DEFAULT ) );
  }
}

// template<>
// void attention_0_gemm_gemv( const __half* query,
//                             const __half* key_base,
//                             __half* att,
//                             const uint64_t n_layers,
//                             const uint64_t seq_len,
//                             const uint64_t head_size,
//                             const uint64_t n_heads,
//                             const uint64_t batch_size,
//                             const uint64_t max_batch_size,
//                             vector<uint64_t> id_alloc_s,
//                             vector<uint64_t> token_pos_s )
//{
//   const float scale = 1.0f / sqrtf( head_size );
//   const float temp_beta = 0.0f;
//
//   const __half* query_p = query;
//   __half* att_p = att;
//
//   for ( size_t i = 0; i < batch_size; i++ ) {
//     const uint64_t m = head_size;
//     const uint64_t n = token_pos_s[i] + 1;
//
//     const uint64_t lda = n_layers * max_batch_size * n_heads * head_size * 2;
//     const uint64_t incx = 1;
//     const uint64_t incy = 1;
//
//     const uint64_t strideA = head_size;
//     const uint64_t stridex = head_size;
//     const uint64_t stridey = seq_len;
//
//     const uint64_t batchCount = n_heads;
//
//     const __half* key = key_base + id_alloc_s[i] * head_size * n_heads;
//
//     CHECK_CUBLAS( cublasHSHgemvStridedBatched( cublas_handle_default,
//                                                CUBLAS_OP_T,
//                                                m,
//                                                n,
//
//                                                &scale,
//                                                key,
//                                                lda,
//                                                strideA,
//
//                                                query_p,
//                                                incx,
//                                                stridex,
//                                                &temp_beta,
//
//                                                att_p,
//                                                incy,
//                                                stridey,
//                                                batchCount ) );
//
//     query_p += head_size * n_heads;
//     att_p += seq_len * n_heads;
//   }
// }
//
// template<>
// void attention_0_gemm_gemv( const float* query,
//                             const float* key_base,
//                             float* att,
//                             const uint64_t n_layers,
//                             const uint64_t seq_len,
//                             const uint64_t head_size,
//                             const uint64_t n_heads,
//                             const uint64_t batch_size,
//                             const uint64_t max_batch_size,
//                             vector<uint64_t> id_alloc_s,
//                             vector<uint64_t> token_pos_s )
//{
//   const float scale = 1.0f / sqrtf( head_size );
//   //  const __half scale = __half(1.0f / sqrtf( head_size ));
//
//   const float* query_p = query;
//   float* att_p = att;
//
//   for ( size_t i = 0; i < batch_size; i++ ) {
//     const uint64_t m = head_size;
//     const uint64_t n = token_pos_s[i] + 1;
//
//     const uint64_t lda = n_layers * max_batch_size * n_heads * head_size * 2;
//     const uint64_t incx = 1;
//     const uint64_t incy = 1;
//
//     const uint64_t strideA = head_size;
//     const uint64_t stridex = head_size;
//     const uint64_t stridey = seq_len;
//
//     const uint64_t batchCount = n_heads;
//
//     const float* key = key_base + id_alloc_s[i] * head_size * n_heads;
//
//     CHECK_CUBLAS( cublasSgemvStridedBatched( cublas_handle_array[i],
//                                              CUBLAS_OP_T,
//                                              m,
//                                              n,
//                                              &scale,
//                                              key,
//                                              lda,
//                                              strideA,
//                                              query_p,
//                                              incx,
//                                              stridex,
//                                              &beta,
//                                              att_p,
//                                              incy,
//                                              stridey,
//                                              batchCount ) );
//
//     query_p += head_size * n_heads;
//     att_p += seq_len * n_heads;
//   }
// }
//
// template<>
// void attention_2_gemm_gemv( const __half* att,
//                             const __half* value_base,
//                             __half* xb,
//                             const uint64_t n_layers,
//                             const uint64_t seq_len,
//                             const uint64_t head_size,
//                             const uint64_t n_heads,
//                             const uint64_t batch_size,
//                             const uint64_t max_batch_size,
//                             vector<uint64_t> id_alloc_s,
//                             vector<uint64_t> token_pos_s )
//{
//   const __half* att_p = att;
//   __half* xb_p = xb;
//
//   for ( size_t i = 0; i < batch_size; i++ ) {
//     const uint64_t m = head_size;
//     const uint64_t n = token_pos_s[i] + 1;
//
//     const uint64_t lda = n_layers * max_batch_size * n_heads * head_size * 2;
//     const uint64_t incx = 1;
//     const uint64_t incy = 1;
//
//     const uint64_t strideA = head_size;
//     const uint64_t stridex = seq_len;
//     const uint64_t stridey = head_size;
//
//     const uint64_t batchCount = n_heads;
//
//     const __half* value = value_base + id_alloc_s[i] * head_size * n_heads;
//
//     CHECK_CUBLAS( cublasHSHgemvStridedBatched( cublas_handle_array[i],
//                                                CUBLAS_OP_T,
//                                                m,
//                                                n,
//                                                &alpha,
//                                                value,
//                                                lda,
//                                                strideA,
//                                                att_p,
//                                                incx,
//                                                stridex,
//                                                &beta,
//                                                xb_p,
//                                                incy,
//                                                stridey,
//                                                batchCount ) );
//
//     att_p += seq_len * n_heads;
//     xb_p += head_size * n_heads;
//   }
// }
//
// template<>
// void attention_2_gemm_gemv( const float* att,
//                             const float* value_base,
//                             float* xb,
//                             const uint64_t n_layers,
//                             const uint64_t seq_len,
//                             const uint64_t head_size,
//                             const uint64_t n_heads,
//                             const uint64_t batch_size,
//                             const uint64_t max_batch_size,
//                             vector<uint64_t> id_alloc_s,
//                             vector<uint64_t> token_pos_s )
//{
//   const float* att_p = att;
//   float* xb_p = xb;
//
//   for ( size_t i = 0; i < batch_size; i++ ) {
//     const uint64_t m = head_size;
//     const uint64_t n = token_pos_s[i] + 1;
//
//     const uint64_t lda = n_layers * max_batch_size * n_heads * head_size * 2;
//     const uint64_t incx = 1;
//     const uint64_t incy = 1;
//
//     const uint64_t strideA = head_size;
//     const uint64_t stridex = seq_len;
//     const uint64_t stridey = head_size;
//
//     const uint64_t batchCount = n_heads;
//
//     const float* value = value_base + id_alloc_s[i] * head_size * n_heads;
//
//     CHECK_CUBLAS( cublasSgemvStridedBatched( cublas_handle_array[i],
//                                              CUBLAS_OP_T,
//                                              m,
//                                              n,
//                                              &alpha,
//                                              value,
//                                              lda,
//                                              strideA,
//                                              att_p,
//                                              incx,
//                                              stridex,
//                                              &beta,
//                                              xb_p,
//                                              incy,
//                                              stridey,
//                                              batchCount ) );
//
//     att_p += seq_len * n_heads;
//     xb_p += head_size * n_heads;
//   }
// }

template<typename DType>
vector<uint32_t> sample( const DType* probabilities, const uint64_t n, const uint64_t batch_size )
{
  throw runtime_error( "not implemented" );
}

template<typename DType>
uint32_t sample( const DType* probabilities, const uint64_t n )
{
  throw runtime_error( "not implemented" );
}

template<typename DType>
vector<uint32_t> argmax( const DType* _v, const uint64_t n, const uint64_t batch_size )
{
  // optimize batching
  // Doing the dumbest batching possible and optimizing later
  vector<uint32_t> arg_maxes;
  for ( size_t i = 0; i < batch_size; i++ ) {
    thrust::device_ptr<const DType> v { _v + i * n };
    const auto it = thrust::max_element( v, v + n );
    arg_maxes.push_back( thrust::distance( v, it ) );
  }
  return arg_maxes;
}

template<typename DType>
uint32_t argmax( const DType* _v, const uint64_t n )
{
  thrust::device_ptr<const DType> v { _v };
  const auto it = thrust::max_element( v, v + n );
  return thrust::distance( v, it );
}

template<>
void accum<float>( float* a, const float* b, const uint64_t size, const uint64_t batch_size )
{
  float alpha = 1.0f;
  cublasSaxpy( cublas_handle_default, size * batch_size, &alpha, b, 1, a, 1 );
}

__global__ void accum_cuda( __half* a, const __half* b, const uint64_t size )
{
  const uint64_t i = blockIdx.x * TPB + threadIdx.x;
  if ( i < size ) {
    a[i] += b[i];
  }
}

template<>
void accum<__half>( __half* a, const __half* b, const uint64_t size, const uint64_t batch_size )
{
  accum_cuda<<<div_ceil( size * batch_size, TPB ), TPB>>>( a, b, size * batch_size );
}

template<typename DType>
void matmul( DType* xout, const DType* x, const DType* W, const uint64_t b, const uint64_t s, const uint64_t r )
{
  const cudaDataType_t cuda_arg_type = is_same_v<DType, __half> ? CUDA_R_16F : CUDA_R_32F;

  // x(b,s) @ W(s,r) -> xout(b,r)
  // OR
  // W(r,s) @ x(s,b) -> xout(r,b)
  // A(m,k) @ B(k,n) ->    C(m,n)
  const uint64_t m = r;
  const uint64_t n = b;
  const uint64_t k = s;
  const uint64_t lda = k;
  const uint64_t ldb = k;
  const uint64_t ldc = m;

  CHECK_CUBLAS( cublasGemmEx( cublas_handle_default,
                              CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              m,
                              n,
                              k,
                              &alpha,
                              W,
                              cuda_arg_type,
                              lda,
                              x,
                              cuda_arg_type,
                              ldb,
                              &beta,
                              xout,
                              cuda_arg_type,
                              ldc,
                              computeType,
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

template<typename DType>
void silu( DType* _hb, DType* _hb2, const uint64_t hidden_dim, const uint64_t batch_size )
{
  silu_direct<<<div_ceil( hidden_dim * batch_size, TPB ), TPB>>>( _hb, _hb2, hidden_dim * batch_size );
}

template<typename DType>
__global__ void do_rope( const uint64_t head_size,
                         const DType* freq_cis_real_row,
                         const DType* freq_cis_imag_row,
                         DType* state_q,
                         DType* state_k )
{
  const uint64_t head_num = blockIdx.x;
  const uint64_t elem_idx = 2 * threadIdx.x;

  // apply RoPE rotation to the q and k vectors for each head
  // get the q and k vectors for this head
  DType* q = state_q + head_num * head_size;
  DType* k = state_k + head_num * head_size;

  // rotate q and k by the freq_cis_real and freq_cis_imag
  const DType q0 = q[elem_idx];
  const DType q1 = q[elem_idx + 1];
  const DType k0 = k[elem_idx];
  const DType k1 = k[elem_idx + 1];
  const DType fcr = freq_cis_real_row[elem_idx / 2];
  const DType fci = freq_cis_imag_row[elem_idx / 2];
  q[elem_idx] = q0 * fcr - q1 * fci;
  q[elem_idx + 1] = q0 * fci + q1 * fcr;
  k[elem_idx] = k0 * fcr - k1 * fci;
  k[elem_idx + 1] = k0 * fci + k1 * fcr;
}

template<typename DType>
void apply_rope( const uint64_t head_size,
                 const uint64_t n_heads,
                 const uint64_t curr_batch_size,
                 vector<uint64_t> token_pos_s,
                 const DType* freq_cis_real,
                 const DType* freq_cis_imag,
                 DType* state_q,
                 DType* state_k )
{
  for ( uint64_t i = 0; i < curr_batch_size; i++ ) {
    do_rope<<<n_heads, head_size / 2, 0, streams[i]>>>( head_size,
                                                        freq_cis_real + token_pos_s[i] * head_size / 2,
                                                        freq_cis_imag + token_pos_s[i] * head_size / 2,
                                                        state_q + i * n_heads * head_size,
                                                        state_k + i * n_heads * head_size );
  }
}

template<typename DType>
__global__ void find_max_for_rows( const DType* att, DType* output, const uint64_t token_pos, const uint64_t seq_len )
{
  const uint64_t head_num = threadIdx.x;
  att += head_num * seq_len;

  DType max_value = att[0];
  for ( uint64_t i = 1; i <= token_pos; i++ ) {
    if constexpr ( is_same_v<DType, __half> ) {
      max_value = __hmax( max_value, att[i] );
    } else {
      max_value = max( max_value, att[i] );
    }
  }

  output[head_num] = max_value;
}

template<typename DType>
__global__ void subtract_and_expf( const DType* values, DType* att, const uint64_t seq_len )
{
  const uint64_t head_num = threadIdx.x;
  const uint64_t token_pos = blockIdx.x;

  att += head_num * seq_len;

  if constexpr ( is_same_v<DType, __half> ) {
    att[token_pos] = hexp( att[token_pos] - values[head_num] );
  } else {
    att[token_pos] = expf( att[token_pos] - values[head_num] );
  }
}

template<typename DType>
__global__ void sum_rows( DType* att, DType* output, const uint64_t token_pos, const uint64_t seq_len )
{
  const uint64_t head_num = threadIdx.x;
  att += head_num * seq_len;

  DType sum = 0.0;
  for ( uint64_t i = 0; i <= token_pos; i++ ) {
    sum += att[i];
  }

  output[head_num] = sum;
}

template<typename DType>
__global__ void normalize_by_sum( DType* att, const DType* sums, const uint64_t seq_len )
{
  const uint64_t head_num = threadIdx.x;
  const uint64_t token_pos = blockIdx.x;

  att += head_num * seq_len;
  att[token_pos] /= sums[head_num];
}

template<typename DType>
void attention_softmax( DType* att,
                        vector<uint64_t> token_pos_s,
                        const uint64_t seq_len,
                        const uint64_t n_heads,
                        DType* temp_buffer,
                        const uint64_t batch_size )
{
  for ( uint64_t i = 0; i < batch_size; i++ ) {
    DType* this_att = att + i * n_heads * seq_len;
    DType* this_buff = temp_buffer + i * n_heads;

    // (1) find the max value for each head (each row)
    find_max_for_rows<<<1, n_heads, 0, streams[i]>>>( this_att, this_buff, token_pos_s[i], seq_len );

    // (2) exp(att - max)
    subtract_and_expf<<<token_pos_s[i] + 1, n_heads, 0, streams[i]>>>( this_buff, this_att, seq_len );

    // (3) sum each row
    sum_rows<<<1, n_heads, 0, streams[i]>>>( this_att, this_buff, token_pos_s[i], seq_len );

    // (4) normalize each row by its sum
    normalize_by_sum<<<token_pos_s[i] + 1, n_heads, 0, streams[i]>>>( this_att, this_buff, seq_len );
  }
}

template void rmsnorm<float>( float* output,
                              const float* x,
                              const float* weight,
                              const uint64_t size,
                              const uint64_t batch_size );
template void rmsnorm<__half>( __half* output,
                               const __half* x,
                               const __half* weight,
                               const uint64_t size,
                               const uint64_t batch_size );

 template vector<uint32_t> argmax<float>( const float* v, const uint64_t n, const uint64_t batch_size );
 template vector<uint32_t> argmax<__half>( const __half* v, const uint64_t n, const uint64_t batch_size );

template uint32_t argmax<float>( const float* v, const uint64_t n );
template uint32_t argmax<__half>( const __half* v, const uint64_t n );

template vector<uint32_t> sample<float>( const float* probabilities, const uint64_t n, const uint64_t batch_size );
template vector<uint32_t> sample<__half>( const __half* probabilities, const uint64_t n, const uint64_t batch_size );

template uint32_t sample<float>( const float* probabilities, const uint64_t n );
template uint32_t sample<__half>( const __half* probabilities, const uint64_t n );

template void accum<float>( float* a, const float* b, const uint64_t size, const uint64_t batch_size );
template void accum<__half>( __half* a, const __half* b, const uint64_t size, const uint64_t batch_size );

 template void softmax<float>( float* x, const uint64_t size, const uint64_t batch_size );
 template void softmax<__half>( __half* x, const uint64_t size, const uint64_t batch_size );

 template void softmax<float>( float* x, const uint64_t size );
 template void softmax<__half>( __half* x, const uint64_t size );

template void matmul<float>( float* xout,
                             const float* x,
                             const float* w,
                             const uint64_t b,
                             const uint64_t s,
                             const uint64_t r );
template void matmul<__half>( __half* xout,
                              const __half* x,
                              const __half* w,
                              const uint64_t b,
                              const uint64_t s,
                              const uint64_t r );

template void silu<float>( float* _hb, float* _hb2, const uint64_t hidden_dim, const uint64_t batch_size );
template void silu<__half>( __half* _hb, __half* _hb2, const uint64_t hidden_dim, const uint64_t batch_size );

template void attention_0_gemm<float>( const float* query,
                                       const float* key_base,
                                       float* att,
                                       const uint64_t n_layers,
                                       const uint64_t seq_len,
                                       const uint64_t head_size,
                                       const uint64_t n_heads,
                                       const uint64_t batch_size,
                                       const uint64_t max_batch_size,
                                       vector<uint64_t> id_alloc_s,
                                       vector<uint64_t> token_pos_s );

template void attention_0_gemm<__half>( const __half* query,
                                        const __half* key_base,
                                        __half* att,
                                        const uint64_t n_layers,
                                        const uint64_t seq_len,
                                        const uint64_t head_size,
                                        const uint64_t n_heads,
                                        const uint64_t batch_size,
                                        const uint64_t max_batch_size,
                                        vector<uint64_t> id_alloc_s,
                                        vector<uint64_t> token_pos_s );

template void attention_2_gemm<float>( const float* att,
                                       const float* value_base,
                                       float* xb,
                                       const uint64_t n_layers,
                                       const uint64_t seq_len,
                                       const uint64_t head_size,
                                       const uint64_t n_heads,
                                       const uint64_t batch_size,
                                       const uint64_t max_batch_size,
                                       vector<uint64_t> id_alloc_s,
                                       vector<uint64_t> token_pos_s );

template void attention_2_gemm<__half>( const __half* att,
                                        const __half* value_base,
                                        __half* xb,
                                        const uint64_t n_layers,
                                        const uint64_t seq_len,
                                        const uint64_t head_size,
                                        const uint64_t n_heads,
                                        const uint64_t batch_size,
                                        const uint64_t max_batch_size,
                                        vector<uint64_t> id_alloc_s,
                                        vector<uint64_t> token_pos_s );

template void attention_softmax<float>( float* att,
                                        vector<uint64_t> token_pos_s,
                                        const uint64_t seq_len,
                                        const uint64_t n_heads,
                                        float* temp_buffer,
                                        const uint64_t batch_size );

template void attention_softmax<__half>( __half* att,
                                         vector<uint64_t> token_pos_s,
                                         const uint64_t seq_len,
                                         const uint64_t n_heads,
                                         __half* temp_buffer,
                                         const uint64_t batch_size );

template void apply_rope<float>( const uint64_t head_size,
                                 const uint64_t n_heads,
                                 const uint64_t curr_batch_size,
                                 vector<uint64_t> token_pos_s,
                                 const float* freq_cis_real,
                                 const float* freq_cis_imag,
                                 float* state_q,
                                 float* state_k );

template void apply_rope<__half>( const uint64_t head_size,
                                  const uint64_t n_heads,
                                  const uint64_t curr_batch_size,
                                  vector<uint64_t> token_pos_s,
                                  const __half* freq_cis_real,
                                  const __half* freq_cis_imag,
                                  __half* state_q,
                                  __half* state_k );

template void copy_kv_cache<float>( const float* state_k,
                                    const float* state_v,
                                    float* key_base,
                                    float* value_base,
                                    const uint64_t dim,
                                    const uint64_t n_layers,
                                    const uint64_t batch_size,
                                    const uint64_t max_batch_size,
                                    vector<uint64_t> id_alloc_s,
                                    vector<uint64_t> token_pos_s );

template void copy_kv_cache<__half>( const __half* state_k,
                                     const __half* state_v,
                                     __half* key_base,
                                     __half* value_base,
                                     const uint64_t dim,
                                     const uint64_t n_layers,
                                     const uint64_t batch_size,
                                     const uint64_t max_batch_size,
                                     vector<uint64_t> id_alloc_s,
                                     vector<uint64_t> token_pos_s );

} // namespace glinthawk::models::common::cuda
