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

cudaStream_t* streams;
cublasHandle_t cublas_handle_default;
cublasHandle_t* cublas_handle_array;
int cublas_handle_count;

cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
const float alpha = 1.0f;
const float beta = 0.0f;

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
  streams = (cudaStream_t*)malloc( num_streams * sizeof( cudaStream_t ) );
  cublas_handle_array = (cublasHandle_t*)malloc( num_streams * sizeof( cublasHandle_t ) );
  for ( int i = 0; i < num_streams; i++ ) {
    cudaStreamCreate( &( streams[i] ) );
    cublasCreate( &( cublas_handle_array[i] ) );
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
                                          const float* ss )
{
  const uint64_t i = threadIdx.x + blockIdx.x * TPB;

  if ( i < size ) {
    const float denom = sqrtf( *ss / size + 1e-5f );
    output[i] = weight[i] * x[i] / denom;
  }
}

__global__ void normalize_and_scale_half( __half* output,
                                          const __half* x,
                                          const __half* weight,
                                          const uint64_t size,
                                          const float* ss )
{
  const uint64_t gb_i = threadIdx.x + blockIdx.x * TPB + blockIdx.y * size;
  const uint64_t i = threadIdx.x + blockIdx.x * TPB;

  if ( i < size ) {
    const float denom = sqrtf( ss[blockIdx.y] / size + 1e-5f );
    output[gb_i] = weight[i] * __float2half( __half2float( x[gb_i] ) / denom );
  }
}

template<>
void rmsnorm<float>( float* output,
                     const float* x,
                     float* temp,
                     const float* weight,
                     const uint64_t size,
                     const uint64_t batch_size )
{
  for ( size_t i = 0; i < batch_size; i++ ) {
    CHECK_CUBLAS( cublasSdot( cublas_handle_default, size, x + i * size, 1, x + i * size, 1, temp + i ) );
    normalize_and_scale_full<<<div_ceil( size, TPB ), TPB>>>( output + i * size, x + i * size, weight, size, temp + i );
  }
}

__global__ void reduce_norm_v2_square_batched( float* output, const __half* x, const uint64_t size )
{
  extern __shared__ float s_out[];

  const uint64_t global_tid = size * blockIdx.y + NRBS * 2 * blockIdx.x + threadIdx.x; // index within whole batch
  const uint64_t local_tid = NRBS * 2 * blockIdx.x + threadIdx.x;                      // index within array
  const uint64_t tid = threadIdx.x;                                                    // index within block

  if ( local_tid < size ) {
    const float _x_f = __half2float( x[global_tid] );
    s_out[tid] = _x_f * _x_f;
  } else {
    s_out[tid] = 0;
  }
  if ( local_tid + NRBS < size ) {
    const float _x_f = __half2float( x[global_tid + NRBS] );
    s_out[tid + NRBS] = _x_f * _x_f;
  } else {
    s_out[tid + NRBS] = 0;
  }

  for ( unsigned int s = NRBS; s > 1; s >>= 1 ) {
    if ( tid < s ) {
      s_out[tid] += s_out[tid + s];
    }
    __syncthreads();
  }

  if ( tid == 0 )
    output[blockIdx.y * gridDim.x + blockIdx.x] = s_out[0] + s_out[1];
}

__global__ void reduce_norm_v2_sum_batched( float* output, const float* x, const uint64_t size )
{
  extern __shared__ float s_out[];

  const uint64_t global_tid = size * blockIdx.y + NRBS * 2 * blockIdx.x + threadIdx.x; // index within whole batch
  const uint64_t local_tid = NRBS * 2 * blockIdx.x + threadIdx.x;                      // index within array
  const uint64_t tid = threadIdx.x;                                                    // index within block

  if ( local_tid < size ) {
    s_out[tid] = x[global_tid];
  } else {
    s_out[tid] = 0;
  }
  if ( local_tid + NRBS < size ) {
    s_out[tid + NRBS] = x[global_tid + NRBS];
  } else {
    s_out[tid + NRBS] = 0;
  }

  for ( unsigned int s = NRBS; s > 1; s >>= 1 ) {
    if ( tid < s ) {
      s_out[tid] += s_out[tid + s];
    }
    __syncthreads();
  }

  if ( tid == 0 )
    output[blockIdx.y * gridDim.x + blockIdx.x] = s_out[0] + s_out[1];
}

void square_reduce_step_2( float* output,
                           const float* x,
                           float* temp_1,
                           float* temp_2,
                           const uint64_t size,
                           const uint64_t batch_size )
{
  const uint64_t max_elems_per_block = NRBS * 2;
  const uint64_t shmem_size = sizeof( float ) * max_elems_per_block;

  const uint64_t grid_size = div_ceil( size, max_elems_per_block );

  dim3 grids( grid_size, batch_size );
  if ( grid_size == 1 ) {
    reduce_norm_v2_sum_batched<<<grids, NRBS, shmem_size>>>( output, x, size );
  } else {
    reduce_norm_v2_sum_batched<<<grids, NRBS, shmem_size>>>( temp_1, x, size );
    square_reduce_step_2( output, temp_1, temp_2, temp_1, grid_size, batch_size );
  }
}

void square_reduce_step_1( float* output, const __half* x, const uint64_t size, const uint64_t batch_size )
{
  const uint64_t max_elems_per_block = NRBS * 2;
  const uint64_t shmem_size = sizeof( float ) * max_elems_per_block;

  const uint64_t grid_size = div_ceil( size, max_elems_per_block );

  dim3 grids( grid_size, batch_size );
  if ( grid_size == 1 ) {
    reduce_norm_v2_square_batched<<<grids, NRBS, shmem_size>>>( output, x, size );
  } else {
    float* temp_1 = output + batch_size;
    float* temp_2 = temp_1 + batch_size * grid_size;
    reduce_norm_v2_square_batched<<<grids, NRBS, shmem_size>>>( temp_1, x, size );
    square_reduce_step_2( output, temp_1, temp_2, temp_1, grid_size, batch_size );
  }
}

template<>
void rmsnorm<__half>( __half* output,
                      const __half* x,
                      __half* temp,
                      const __half* weight,
                      const uint64_t size,
                      const uint64_t batch_size )
{
  square_reduce_step_1( reinterpret_cast<float*>( temp ), x, size, batch_size );

  dim3 grid( div_ceil( size, TPB ), batch_size );
  normalize_and_scale_half<<<grid, TPB>>>( output, x, weight, size, reinterpret_cast<float*>( temp ) );
}

template<typename DType>
void copy_kv_cache( DType* context_pointers[],
                    const DType* state_k,
                    const DType* state_v,
                    const uint64_t kv_dim,
                    const uint64_t batch_size,
                    const uint32_t* token_positions )
{
  for ( size_t i = 0; i < batch_size; i++ ) {
    DType* k_cache_pos = context_pointers[i] + token_positions[i] * kv_dim * 2;
    DType* v_cache_pos = k_cache_pos + kv_dim;

    ops::CHECK_CUDA(
      cudaMemcpyAsync( k_cache_pos, state_k + i * kv_dim, kv_dim * sizeof( DType ), cudaMemcpyDeviceToDevice ) );
    ops::CHECK_CUDA(
      cudaMemcpyAsync( v_cache_pos, state_v + i * kv_dim, kv_dim * sizeof( DType ), cudaMemcpyDeviceToDevice ) );
  }
}

template<typename DType>
void attention_0_gemm( const DType* query,
                       const DType* const context_pointers[],
                       DType* att,
                       const uint64_t seq_len,
                       const uint64_t head_size,
                       const uint64_t n_kv_heads,
                       const uint64_t gqa_size,
                       const uint64_t batch_size,
                       const uint32_t* token_positions )
{
  const cudaDataType_t cuda_arg_type = is_same_v<DType, __half> ? CUDA_R_16F : CUDA_R_32F;
  const float scale = 1.0f / sqrtf( head_size );

  const uint64_t k = head_size;
  const uint64_t n = gqa_size;

  const uint64_t lda = n_kv_heads * head_size * 2;
  const uint64_t ldb = k;
  const uint64_t ldc = seq_len;

  const uint64_t strideA = head_size;
  const uint64_t strideB = head_size * gqa_size;
  const uint64_t strideC = seq_len * gqa_size;

  const uint64_t batchCount = n_kv_heads;

  const uint64_t dim_ = head_size * n_kv_heads * gqa_size;
  const uint64_t att_dim_ = seq_len * n_kv_heads * gqa_size;

  for ( size_t i = 0; i < batch_size; i++ ) {
    const uint64_t m = token_positions[i] + 1;
    CHECK_CUBLAS( cublasGemmStridedBatchedEx( cublas_handle_array[i],
                                              CUBLAS_OP_T,
                                              CUBLAS_OP_N,
                                              m,
                                              n,
                                              k,
                                              &scale,
                                              context_pointers[i],
                                              cuda_arg_type,
                                              lda,
                                              strideA,
                                              query + i * dim_,
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
                       const DType* const context_pointers[],
                       DType* xb,
                       const uint64_t seq_len,
                       const uint64_t head_size,
                       const uint64_t n_kv_heads,
                       const uint64_t gqa_size,
                       const uint64_t batch_size,
                       const uint32_t* token_positions )
{
  const cudaDataType_t cuda_arg_type = is_same_v<DType, __half> ? CUDA_R_16F : CUDA_R_32F;

  const uint64_t m = head_size;
  const uint64_t n = gqa_size;

  const uint64_t lda = n_kv_heads * head_size * 2;
  const uint64_t ldb = seq_len;
  const uint64_t ldc = m;

  const uint64_t strideA = head_size;
  const uint64_t strideB = seq_len * gqa_size;
  const uint64_t strideC = head_size * gqa_size;

  const uint64_t batchCount = n_kv_heads;

  const uint64_t kv_dim_ = head_size * n_kv_heads;
  const uint64_t dim_ = head_size * n_kv_heads * gqa_size;
  const uint64_t att_dim_ = seq_len * n_kv_heads * gqa_size;

  for ( size_t i = 0; i < batch_size; i++ ) {
    const uint64_t k = token_positions[i] + 1;

    CHECK_CUBLAS( cublasGemmStridedBatchedEx( cublas_handle_array[i],
                                              CUBLAS_OP_N,
                                              CUBLAS_OP_N,
                                              m,
                                              n,
                                              k,
                                              &alpha,
                                              context_pointers[i] + kv_dim_,
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

template<typename DType>
__global__ void argmax_batched_init( uint32_t* output_arg, DType* output, const DType* x, const uint64_t size )
{
  extern __shared__ float smem[];

  DType* s_out = reinterpret_cast<DType*>( &smem[0] );
  uint32_t* a_out = reinterpret_cast<uint32_t*>( s_out + AMRBS * 2 );

  const uint64_t global_tid = size * blockIdx.y + AMRBS * 2 * blockIdx.x + threadIdx.x; // index within whole batch
  const uint64_t local_tid = AMRBS * 2 * blockIdx.x + threadIdx.x;                      // index within array
  const uint64_t tid = threadIdx.x;                                                     // index within block

  if ( local_tid < size ) {
    s_out[tid] = x[global_tid];
    a_out[tid] = local_tid;
  } else {
    if constexpr ( is_same_v<DType, __half> ) {
      s_out[tid] = -CUDART_INF_FP16;
    } else {
      s_out[tid] = -INFINITY;
    }
  }
  if ( local_tid + AMRBS < size ) {
    s_out[tid + AMRBS] = x[global_tid + AMRBS];
    a_out[tid + AMRBS] = local_tid + AMRBS;
  } else {
    if constexpr ( is_same_v<DType, __half> ) {
      s_out[tid + AMRBS] = -CUDART_INF_FP16;
    } else {
      s_out[tid + AMRBS] = -INFINITY;
    }
  }

  for ( unsigned int s = AMRBS; s > 1; s >>= 1 ) {
    if ( tid < s ) {
      if ( s_out[tid + s] > s_out[tid] ) {
        s_out[tid] = s_out[tid + s];
        a_out[tid] = a_out[tid + s];
      }
    }
    __syncthreads();
  }

  if ( tid == 0 ) {
    if ( s_out[1] > s_out[0] ) {
      output[blockIdx.y * gridDim.x + blockIdx.x] = s_out[1];
      output_arg[blockIdx.y * gridDim.x + blockIdx.x] = a_out[1];
    } else {
      output[blockIdx.y * gridDim.x + blockIdx.x] = s_out[0];
      output_arg[blockIdx.y * gridDim.x + blockIdx.x] = a_out[0];
    }
  }
}

template<typename DType>
__global__ void argmax_batched_next( uint32_t* output_arg,
                                     DType* output,
                                     const uint32_t* x_arg,
                                     const DType* x,
                                     const uint64_t size )
{
  extern __shared__ float smem[];

  DType* s_out = reinterpret_cast<DType*>( &smem[0] );
  uint32_t* a_out = reinterpret_cast<uint32_t*>( s_out + AMRBS * 2 );

  const uint64_t global_tid = size * blockIdx.y + AMRBS * 2 * blockIdx.x + threadIdx.x; // index within whole batch
  const uint64_t local_tid = AMRBS * 2 * blockIdx.x + threadIdx.x;                      // index within array
  const uint64_t tid = threadIdx.x;                                                     // index within block

  if ( local_tid < size ) {
    s_out[tid] = x[global_tid];
    a_out[tid] = x_arg[global_tid];
  } else {
    if constexpr ( is_same_v<DType, __half> ) {
      s_out[tid] = -CUDART_INF_FP16;
    } else {
      s_out[tid] = -INFINITY;
    }
  }
  if ( local_tid + AMRBS < size ) {
    s_out[tid + AMRBS] = x[global_tid + AMRBS];
    a_out[tid + AMRBS] = x_arg[global_tid + AMRBS];
  } else {
    if constexpr ( is_same_v<DType, __half> ) {
      s_out[tid + AMRBS] = -CUDART_INF_FP16;
    } else {
      s_out[tid + AMRBS] = -INFINITY;
    }
  }

  for ( unsigned int s = AMRBS; s > 1; s >>= 1 ) {
    if ( tid < s ) {
      if ( s_out[tid + s] > s_out[tid] ) {
        s_out[tid] = s_out[tid + s];
        a_out[tid] = a_out[tid + s];
      }
    }
    __syncthreads();
  }

  if ( tid == 0 ) {
    if ( s_out[1] > s_out[0] ) {
      output[blockIdx.y * gridDim.x + blockIdx.x] = s_out[1];
      output_arg[blockIdx.y * gridDim.x + blockIdx.x] = a_out[1];
    } else {
      output[blockIdx.y * gridDim.x + blockIdx.x] = s_out[0];
      output_arg[blockIdx.y * gridDim.x + blockIdx.x] = a_out[0];
    }
  }
}

template<typename DType>
void argmax_step_2( uint32_t* output_arg,
                    const uint32_t* x_arg,
                    const DType* x,
                    uint32_t* temp_1_arg,
                    DType* temp_1,
                    uint32_t* temp_2_arg,
                    DType* temp_2,
                    const uint64_t size,
                    const uint64_t batch_size )
{
  const uint64_t max_elems_per_block = AMRBS * 2;
  const uint64_t shmem_size = ( sizeof( DType ) + sizeof( uint32_t ) ) * max_elems_per_block;

  const uint64_t grid_size = div_ceil( size, max_elems_per_block );

  dim3 grids( grid_size, batch_size );
  if ( grid_size == 1 ) {
    argmax_batched_next<<<grids, AMRBS, shmem_size>>>( output_arg, temp_1, x_arg, x, size );
  } else {
    argmax_batched_next<<<grids, AMRBS, shmem_size>>>( temp_1_arg, temp_1, x_arg, x, size );
    argmax_step_2( output_arg, temp_1_arg, temp_1, temp_2_arg, temp_2, temp_1_arg, temp_1, grid_size, batch_size );
  }
}

template<typename DType>
void argmax_step_1( uint32_t* output_arg, const DType* x, const uint64_t size, const uint64_t batch_size )
{
  const uint64_t max_elems_per_block = AMRBS * 2;
  const uint64_t shmem_size = ( sizeof( DType ) + sizeof( uint32_t ) ) * max_elems_per_block;

  const uint64_t grid_size = div_ceil( size, max_elems_per_block );

  dim3 grids( grid_size, batch_size );
  if ( grid_size == 1 ) {
    DType* output = reinterpret_cast<DType*>( output_arg + batch_size );
    argmax_batched_init<<<grids, AMRBS, shmem_size>>>( output_arg, output, x, size );
  } else {
    DType* temp_1 = reinterpret_cast<DType*>( output_arg + batch_size );
    DType* temp_2 = temp_1 + batch_size * grid_size;
    uint32_t* temp_1_arg = reinterpret_cast<uint32_t*>( temp_2 + batch_size * grid_size );
    uint32_t* temp_2_arg = temp_1_arg + batch_size * grid_size;
    argmax_batched_init<<<grids, AMRBS, shmem_size>>>( temp_1_arg, temp_1, x, size );
    argmax_step_2( output_arg, temp_1_arg, temp_1, temp_2_arg, temp_2, temp_1_arg, temp_1, grid_size, batch_size );
  }
}

template<typename DType>
void argmax( uint32_t* output_cpu, const DType* _v, DType* temp, const uint64_t n, const uint64_t batch_size )
{
  argmax_step_1( reinterpret_cast<uint32_t*>( temp ), _v, n, batch_size );
  ops::CHECK_CUDA( cudaMemcpy( output_cpu, temp, batch_size * sizeof( uint32_t ), cudaMemcpyDeviceToHost ) );
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
  CHECK_CUBLAS( cublasSaxpy( cublas_handle_default, size * batch_size, &alpha, b, 1, a, 1 ) );
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
                         const uint64_t gqa_size,
                         const DType* freq_cis_real_row,
                         const DType* freq_cis_imag_row,
                         DType* state_q,
                         DType* state_k )
{
  const uint64_t head_q_num = gqa_size * blockIdx.x;
  const uint64_t head_k_num = blockIdx.x;
  const uint64_t elem_idx = 2 * threadIdx.x;

  // apply RoPE rotation to the q and k vectors for each head
  // get the q and k vectors for this head
  DType* q = state_q + head_q_num * head_size;
  DType* k = state_k + head_k_num * head_size;

  // rotate q and k by the freq_cis_real and freq_cis_imag
  const DType fcr = freq_cis_real_row[elem_idx / 2];
  const DType fci = freq_cis_imag_row[elem_idx / 2];

  const DType k0 = k[elem_idx];
  const DType k1 = k[elem_idx + 1];
  k[elem_idx] = k0 * fcr - k1 * fci;
  k[elem_idx + 1] = k0 * fci + k1 * fcr;

  for ( uint64_t i = 0; i < gqa_size; i++ ) {
    const DType q0 = q[i * head_size + elem_idx];
    const DType q1 = q[i * head_size + elem_idx + 1];
    q[i * head_size + elem_idx] = q0 * fcr - q1 * fci;
    q[i * head_size + elem_idx + 1] = q0 * fci + q1 * fcr;
  }
}

template<typename DType>
void apply_rope( const uint64_t head_size,
                 const uint64_t n_kv_heads,
                 const uint64_t gqa_size,
                 const uint64_t curr_batch_size,
                 const uint32_t* token_positions,
                 const DType* freq_cis_real,
                 const DType* freq_cis_imag,
                 DType* state_q,
                 DType* state_k )
{
  for ( uint64_t i = 0; i < curr_batch_size; i++ ) {
    do_rope<<<n_kv_heads, head_size / 2, 0, streams[i]>>>( head_size,
                                                           gqa_size,
                                                           freq_cis_real + token_positions[i] * head_size / 2,
                                                           freq_cis_imag + token_positions[i] * head_size / 2,
                                                           state_q + i * n_kv_heads * gqa_size * head_size,
                                                           state_k + i * n_kv_heads * head_size );
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
                        const uint32_t* token_positions,
                        const uint64_t seq_len,
                        const uint64_t n_heads,
                        DType* temp_buffer,
                        const uint64_t batch_size )
{
  for ( uint64_t i = 0; i < batch_size; i++ ) {
    DType* this_att = att + i * n_heads * seq_len;
    DType* this_buff = temp_buffer + i * n_heads;

    // (1) find the max value for each head (each row)
    find_max_for_rows<<<1, n_heads, 0, streams[i]>>>( this_att, this_buff, token_positions[i], seq_len );

    // (2) exp(att - max)
    subtract_and_expf<<<token_positions[i] + 1, n_heads, 0, streams[i]>>>( this_buff, this_att, seq_len );

    // (3) sum each row
    sum_rows<<<1, n_heads, 0, streams[i]>>>( this_att, this_buff, token_positions[i], seq_len );

    // (4) normalize each row by its sum
    normalize_by_sum<<<token_positions[i] + 1, n_heads, 0, streams[i]>>>( this_att, this_buff, seq_len );
  }
}

__global__ void setup_kernel( curandState* state, unsigned long seed )
{
  int id = threadIdx.x + blockIdx.x * TPB;
  curand_init( seed, id, 0, &state[id] );
}

void setup_rng( curandState* rng_state, unsigned long seed, const uint64_t size, const uint64_t batch_size )
{
  for ( uint64_t i = 0; i < batch_size; i++ ) {
    setup_kernel<<<div_ceil( size, TPB ), TPB, 0, streams[i]>>>( rng_state + i * size, 1234 );
  }
}

template<typename DType>
__global__ void gumbel_fix( DType* array, float temp, const uint64_t vocab_size, curandState* rng_state )
{
  const uint64_t i = threadIdx.x + blockIdx.x * TPB;

  if ( i < vocab_size ) {
    float myrandf = curand_uniform( rng_state + i );
    myrandf = logf( -logf( myrandf ) );
    if constexpr ( is_same_v<DType, __half> ) {
      array[i] = __float2half( __half2float( array[i] ) / temp - myrandf );
    } else {
      array[i] = array[i] / temp - myrandf;
    }
  }
}

template<typename DType>
void soft_sample( DType* v,
                  const vector<float>& temp_s,
                  curandState* rng_state,
                  const uint64_t vocab_size,
                  const uint64_t batch_size )
{
  for ( uint64_t i = 0; i < batch_size; i++ ) {
    if ( temp_s[i] > 0 ) {
      gumbel_fix<<<div_ceil( vocab_size, TPB ), TPB, 0, streams[i]>>>(
        v + i * vocab_size, temp_s[i], vocab_size, rng_state + i * vocab_size );
    }
  }
}

template void rmsnorm<float>( float* output,
                              const float* x,
                              float* temp,
                              const float* weight,
                              const uint64_t size,
                              const uint64_t batch_size );
template void rmsnorm<__half>( __half* output,
                               const __half* x,
                               __half* temp,
                               const __half* weight,
                               const uint64_t size,
                               const uint64_t batch_size );

template void argmax<float>( uint32_t* output,
                             const float* v,
                             float* temp,
                             const uint64_t n,
                             const uint64_t batch_size );
template void argmax<__half>( uint32_t* output,
                              const __half* v,
                              __half* temp,
                              const uint64_t n,
                              const uint64_t batch_size );

template uint32_t argmax<float>( const float* v, const uint64_t n );
template uint32_t argmax<__half>( const __half* v, const uint64_t n );

template void soft_sample<float>( float* v,
                                  const vector<float>& temp_s,
                                  curandState* rng_state,
                                  const uint64_t vocab_size,
                                  const uint64_t batch_size );
template void soft_sample<__half>( __half* v,
                                   const vector<float>& temp_s,
                                   curandState* rng_state,
                                   const uint64_t vocab_size,
                                   const uint64_t batch_size );

template void accum<float>( float* a, const float* b, const uint64_t size, const uint64_t batch_size );
template void accum<__half>( __half* a, const __half* b, const uint64_t size, const uint64_t batch_size );

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
                                       const float* const context_pointers[],
                                       float* att,
                                       const uint64_t seq_len,
                                       const uint64_t head_size,
                                       const uint64_t n_kv_heads,
                                       const uint64_t gqa_size,
                                       const uint64_t batch_size,
                                       const uint32_t* token_positions );

template void attention_0_gemm<__half>( const __half* query,
                                        const __half* const context_pointers[],
                                        __half* att,
                                        const uint64_t seq_len,
                                        const uint64_t head_size,
                                        const uint64_t n_kv_heads,
                                        const uint64_t gqa_size,
                                        const uint64_t batch_size,
                                        const uint32_t* token_positions );

template void attention_2_gemm<float>( const float* att,
                                       const float* const context_pointers[],
                                       float* xb,
                                       const uint64_t seq_len,
                                       const uint64_t head_size,
                                       const uint64_t n_kv_heads,
                                       const uint64_t gqa_size,
                                       const uint64_t batch_size,
                                       const uint32_t* token_positions );

template void attention_2_gemm<__half>( const __half* att,
                                        const __half* const context_pointers[],
                                        __half* xb,
                                        const uint64_t seq_len,
                                        const uint64_t head_size,
                                        const uint64_t n_kv_heads,
                                        const uint64_t gqa_size,
                                        const uint64_t batch_size,
                                        const uint32_t* token_positions );

template void attention_softmax<float>( float* att,
                                        const uint32_t* token_positions,
                                        const uint64_t seq_len,
                                        const uint64_t n_heads,
                                        float* temp_buffer,
                                        const uint64_t batch_size );

template void attention_softmax<__half>( __half* att,
                                         const uint32_t* token_positions,
                                         const uint64_t seq_len,
                                         const uint64_t n_heads,
                                         __half* temp_buffer,
                                         const uint64_t batch_size );

template void apply_rope<float>( const uint64_t head_size,
                                 const uint64_t n_kv_heads,
                                 const uint64_t gqa_size,
                                 const uint64_t curr_batch_size,
                                 const uint32_t* token_positions,
                                 const float* freq_cis_real,
                                 const float* freq_cis_imag,
                                 float* state_q,
                                 float* state_k );

template void apply_rope<__half>( const uint64_t head_size,
                                  const uint64_t n_kv_heads,
                                  const uint64_t gqa_size,
                                  const uint64_t curr_batch_size,
                                  const uint32_t* token_positions,
                                  const __half* freq_cis_real,
                                  const __half* freq_cis_imag,
                                  __half* state_q,
                                  __half* state_k );

template void copy_kv_cache<float>( float* context_pointers[],
                                    const float* state_k,
                                    const float* state_v,
                                    const uint64_t dim,
                                    const uint64_t batch_size,
                                    const uint32_t* token_positions );

template void copy_kv_cache<__half>( __half* context_pointers[],
                                     const __half* state_k,
                                     const __half* state_v,
                                     const uint64_t dim,
                                     const uint64_t batch_size,
                                     const uint32_t* token_positions );

} // namespace glinthawk::models::common::cuda
