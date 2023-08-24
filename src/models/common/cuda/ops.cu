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

void CHECK_CUDA( const cudaError_t err, const source_location location )
{
  if ( err != cudaSuccess ) {
    throw runtime_error( "CUDA error " + string( cudaGetErrorName( err ) ) + ": " + string( cudaGetErrorString( err ) )
                         + " (" + location.file_name() + ":" + to_string( location.line() ) + ")" );
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
void rmsnorm<float>( float* output, const float* x, const float* weight, const uint64_t size, const uint64_t batch_size )
{
  // TODO: optimize batching
  // Doing the dumbest batching possible and optimizing later
  for (size_t i = 0; i < batch_size; i++){
    float ss = 0.0f;

    CHECK_CUBLAS( cublasSdot( cublas_handle, size, x + i * size, 1, x + i * size, 1, &ss ) );
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
void rmsnorm<__half>( __half* output, const __half* x, const __half* weight, const uint64_t size, const uint64_t batch_size )
{
  // TODO: optimize batching
  // Doing the dumbest batching possible and optimizing later
  for (size_t i = 0; i < batch_size; i++){
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
  // TODO: optimize batching
  // Doing the dumbest batching possible and optimizing later
  for (size_t i = 0; i < batch_size; i++){
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
  // TODO: optimize batching
  // Doing the dumbest batching possible and optimizing later
  for (size_t i = 0; i < batch_size; i++){
    thrust::device_ptr<__half> x { _x + i * size};
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
void fill_pointers_init( DType** q_p,
                         std::vector<DType*> query_p_cpu,
                         DType* query,
                         DType* att,
                         DType* xb,
                         const uint64_t seq_len,
                         const uint64_t head_size,
                         const uint64_t n_heads,
                         const uint64_t max_batch_size )
{
  for (size_t i = 0; i < max_batch_size * n_heads; i++) {
    query_p_cpu[i] = query + head_size * i;
    query_p_cpu[i + max_batch_size * n_heads] = att + seq_len * i;
    query_p_cpu[i + 2 * max_batch_size * n_heads] = xb + head_size * i;
  }
  CHECK_CUDA( cudaMemcpy( q_p, query_p_cpu.data(), 3 * max_batch_size * n_heads * sizeof(DType*), cudaMemcpyHostToDevice ) );
}

template<typename DType>
void fill_pointers_kv( DType** k_p,
                       std::vector<DType*> key_p_cpu,
                       DType* key,
                       DType* value,
                       const uint64_t* id_allocation,
                       const uint64_t head_size,
                       const uint64_t n_heads,
                       const uint64_t batch_size,
                       const uint64_t max_batch_size )
{
  for (size_t i = 0; i < batch_size; i++)
    for (size_t j = 0; j < n_heads; j++) {
      key_p_cpu[i * n_heads + j] = key + head_size * (id_allocation[i] * n_heads + j);
      key_p_cpu[i * n_heads + j + max_batch_size * n_heads] = value + head_size * (id_allocation[i] * n_heads + j);
    }
  CHECK_CUDA( cudaMemcpy( k_p, key_p_cpu.data(), 2 * max_batch_size * n_heads * sizeof(DType*), cudaMemcpyHostToDevice ) );
}

template<typename DType>
void attention_0_gemm( const DType* const* query_p,
                       const DType* const* key_p,
                       DType** att_p,
                       const uint64_t n_layers,
                       const uint64_t seq_len,
                       const uint64_t head_size,
                       const uint64_t n_heads,
                       const uint64_t n_tokens, 
                       const uint64_t batch_size,
                       const uint64_t max_batch_size )
{
  const cudaDataType_t cuda_arg_type = is_same_v<DType, __half> ? CUDA_R_16F : CUDA_R_32F;

  const float alpha = 1.0f / sqrtf( head_size );
  const float beta = 0.0f;

  const uint64_t m = 1;
  const uint64_t n = n_tokens;
  const uint64_t k = head_size;

  const uint64_t lda = m;
  const uint64_t ldb = n_layers * max_batch_size * n_heads * head_size * 2;
  const uint64_t ldc = m;

  const uint64_t batchCount = n_heads * batch_size;
  
  CHECK_CUBLAS( cublasGemmBatchedEx( cublas_handle,
                                     CUBLAS_OP_N,
                                     CUBLAS_OP_N,
                                     m,
                                     n,
                                     k,
                                     &alpha,
                                     reinterpret_cast<const void* const*>(query_p),
                                     cuda_arg_type,
                                     lda,
                                     reinterpret_cast<const void* const*>(key_p),
                                     cuda_arg_type,
                                     ldb,
                                     &beta,
                                     reinterpret_cast<void**>(att_p),
                                     cuda_arg_type,
                                     ldc,
                                     batchCount,
                                     CUBLAS_COMPUTE_32F,
                                     CUBLAS_GEMM_DEFAULT ) );
}

template<typename DType>
void attention_2_gemm( const DType* const* att_p,
                       const DType* const* value_p,
                       DType** xb_p,
                       const uint64_t n_layers,
                       const uint64_t seq_len,
                       const uint64_t head_size,
                       const uint64_t n_heads,
                       const uint64_t n_tokens, 
                       const uint64_t batch_size,
                       const uint64_t max_batch_size )
{
  const cudaDataType_t cuda_arg_type = is_same_v<DType, __half> ? CUDA_R_16F : CUDA_R_32F;

  const float alpha = 1.0f;
  const float beta = 0.0f;

  const uint64_t m = head_size;
  const uint64_t n = 1;
  const uint64_t k = n_tokens;

  const uint64_t lda = n_layers * max_batch_size * n_heads * head_size * 2;
  const uint64_t ldb = k;
  const uint64_t ldc = m;

  const uint64_t batchCount = n_heads * batch_size;

  CHECK_CUBLAS( cublasGemmBatchedEx( cublas_handle,
                                     CUBLAS_OP_N,
                                     CUBLAS_OP_N,
                                     m,
                                     n,
                                     k,
                                     &alpha,
                                     reinterpret_cast<const void* const*>(value_p),
                                     cuda_arg_type,
                                     lda,
                                     reinterpret_cast<const void* const*>(att_p),
                                     cuda_arg_type,
                                     ldb,
                                     &beta,
                                     reinterpret_cast<void**>(xb_p),
                                     cuda_arg_type,
                                     ldc,
                                     batchCount,
                                     CUBLAS_COMPUTE_32F,
                                     CUBLAS_GEMM_DEFAULT ) );
}

template<typename DType>
std::vector<uint32_t> sample( const DType* probabilities, const uint64_t n, const uint64_t batch_size )
{
  throw runtime_error( "not implemented" );
}

template<typename DType>
uint32_t sample( const DType* probabilities, const uint64_t n )
{
  throw runtime_error( "not implemented" );
}

template<typename DType>
std::vector<uint32_t> argmax( const DType* _v, const uint64_t n, const uint64_t batch_size )
{
  // TODO: optimize batching
  // Doing the dumbest batching possible and optimizing later
  std::vector<uint32_t> arg_maxes;
  for (size_t i = 0; i < batch_size; i++){
    thrust::device_ptr<const DType> v { _v + i * n };
    const auto it = thrust::max_element( v, v + n );
    arg_maxes.push_back(thrust::distance( v, it ));
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
  cublasSaxpy( cublas_handle, size * batch_size, &alpha, b, 1, a, 1 );
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

  float alpha = 1.0f;
  float beta = 0.0f;

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

  CHECK_CUBLAS( cublasGemmEx( cublas_handle,
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
                 const DType* freq_cis_real_row,
                 const DType* freq_cis_imag_row,
                 DType* state_q,
                 DType* state_k )
{
  do_rope<<<n_heads * curr_batch_size, head_size / 2>>>( head_size,
                                                         freq_cis_real_row,
                                                         freq_cis_imag_row,
                                                         state_q,
                                                         state_k );
}

template<typename DType>
__global__ void find_max_for_rows( const DType* att,
                                   DType* output,
                                   const uint64_t token_pos,
                                   const uint64_t seq_len )
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
  att[token_pos] = expf( att[token_pos] - values[head_num] );
}

template<typename DType>
__global__ void sum_rows( DType* att,
                          DType* output,
                          const uint64_t token_pos,
                          const uint64_t seq_len )
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
                        const uint64_t token_pos,
                        const uint64_t seq_len,
                        const uint64_t n_heads,
                        DType* temp_buffer,
                        const uint64_t batch_size )
{
  // TODO: fix thread and block assignments, threads might overflow
  // (1) find the max value for each head (each row)
  find_max_for_rows<<<1, n_heads * batch_size>>>( att, temp_buffer, token_pos, seq_len );

  // (2) exp(att - max)
  subtract_and_expf<<<token_pos + 1, n_heads * batch_size>>>( temp_buffer, att, seq_len );

  // (3) sum each row
  sum_rows<<<1, n_heads * batch_size>>>( att, temp_buffer, token_pos, seq_len );

  // (4) normalize each row by its sum
  normalize_by_sum<<<token_pos + 1, n_heads * batch_size>>>( att, temp_buffer, seq_len );
}

template void rmsnorm<float>( float* output, const float* x, const float* weight, const uint64_t size, const uint64_t batch_size );
template void rmsnorm<__half>( __half* output, const __half* x, const __half* weight, const uint64_t size, const uint64_t batch_size );

template std::vector<uint32_t> argmax<float>( const float* v, const uint64_t n, const uint64_t batch_size );
template std::vector<uint32_t> argmax<__half>( const __half* v, const uint64_t n, const uint64_t batch_size );

template uint32_t argmax<float>( const float* v, const uint64_t n );
template uint32_t argmax<__half>( const __half* v, const uint64_t n );

template std::vector<uint32_t> sample<float>( const float* probabilities, const uint64_t n, const uint64_t batch_size );
template std::vector<uint32_t> sample<__half>( const __half* probabilities, const uint64_t n, const uint64_t batch_size );

template uint32_t sample<float>( const float* probabilities, const uint64_t n );
template uint32_t sample<__half>( const __half* probabilities, const uint64_t n );

template void accum<float>( float* a, const float* b, const uint64_t size, const uint64_t batch_size );
template void accum<__half>( __half* a, const __half* b, const uint64_t size, const uint64_t batch_size );

template void softmax<float>( float* x, const uint64_t size, const uint64_t batch_size );
template void softmax<__half>( __half* x, const uint64_t size, const uint64_t batch_size );

template void softmax<float>( float* x, const uint64_t size );
template void softmax<__half>( __half* x, const uint64_t size );

template void matmul<float>( float* xout, const float* x, const float* w, const uint64_t b, const uint64_t s, const uint64_t r);
template void matmul<__half>( __half* xout, const __half* x, const __half* w, const uint64_t b, const uint64_t s, const uint64_t r);

template void silu<float>( float* _hb, float* _hb2, const uint64_t hidden_dim, const uint64_t batch_size );
template void silu<__half>( __half* _hb, __half* _hb2, const uint64_t hidden_dim, const uint64_t batch_size );

template void attention_0_gemm<float>( const float* const* query_p,
                                       const float* const* key_p,
                                       float** att_p,
                                       const uint64_t n_layers,
                                       const uint64_t seq_len,
                                       const uint64_t head_size,
                                       const uint64_t n_heads,
                                       const uint64_t n_tokens, 
                                       const uint64_t batch_size,
                                       const uint64_t max_batch_size );

template void attention_0_gemm<__half>( const __half* const* query_p,
                                        const __half* const* key_p,
                                        __half** att_p,
                                        const uint64_t n_layers,
                                        const uint64_t seq_len,
                                        const uint64_t head_size,
                                        const uint64_t n_heads,
                                        const uint64_t n_tokens, 
                                        const uint64_t batch_size,
                                        const uint64_t max_batch_size );

template void attention_2_gemm<float>( const float* const* att_p,
                                       const float* const* value_p,
                                       float** xb_p,
                                       const uint64_t n_layers,
                                       const uint64_t seq_len,
                                       const uint64_t head_size,
                                       const uint64_t n_heads,
                                       const uint64_t n_tokens, 
                                       const uint64_t batch_size,
                                       const uint64_t max_batch_size );

template void attention_2_gemm<__half>( const __half* const* att_p,
                                        const __half* const* value_p,
                                        __half** xb_p,
                                        const uint64_t n_layers,
                                        const uint64_t seq_len,
                                        const uint64_t head_size,
                                        const uint64_t n_heads,
                                        const uint64_t n_tokens, 
                                        const uint64_t batch_size,
                                        const uint64_t max_batch_size );

template void attention_softmax<float>( float* att,
                                        const uint64_t token_pos,
                                        const uint64_t seq_len,
                                        const uint64_t n_heads,
                                        float* temp_buffer,
                                        const uint64_t batch_size );

template void attention_softmax<__half>( __half* att,
                                         const uint64_t token_pos,
                                         const uint64_t seq_len,
                                         const uint64_t n_heads,
                                         __half* temp_buffer,
                                         const uint64_t batch_size );

template void apply_rope<float>( const uint64_t head_size,
                                 const uint64_t n_heads,
                                 const uint64_t curr_batch_size,
                                 const float* freq_cis_real_row,
                                 const float* freq_cis_imag_row,
                                 float* state_q,
                                 float* state_k );

template void apply_rope<__half>( const uint64_t head_size,
                                  const uint64_t n_heads,
                                  const uint64_t curr_batch_size,
                                  const __half* freq_cis_real_row,
                                  const __half* freq_cis_imag_row,
                                  __half* state_q,
                                  __half* state_k );

template void fill_pointers_init<float>( float** q_p,
                                         std::vector<float*> query_p_cpu,
                                         float* query,
                                         float* att,
                                         float* xb,
                                         const uint64_t seq_len,
                                         const uint64_t head_size,
                                         const uint64_t n_heads,
                                         const uint64_t max_batch_size );

template void fill_pointers_init<__half>( __half** q_p,
                                          std::vector<__half*> query_p_cpu,
                                          __half* query,
                                          __half* att,
                                          __half* xb,
                                          const uint64_t seq_len,
                                          const uint64_t head_size,
                                          const uint64_t n_heads,
                                          const uint64_t max_batch_size );

template void fill_pointers_kv<float>( float** k_p,
                                       std::vector<float*> key_p_cpu,
                                       float* key,
                                       float* value,
                                       const uint64_t* id_allocation,
                                       const uint64_t head_size,
                                       const uint64_t n_heads,
                                       const uint64_t batch_size,
                                       const uint64_t max_batch_size );

template void fill_pointers_kv<__half>( __half** k_p,
                                        std::vector<__half*> key_p_cpu,
                                        __half* key,
                                        __half* value,
                                        const uint64_t* id_allocation,
                                        const uint64_t head_size,
                                        const uint64_t n_heads,
                                        const uint64_t batch_size,
                                        const uint64_t max_batch_size );

} // namespace glinthawk::models::common::cuda
