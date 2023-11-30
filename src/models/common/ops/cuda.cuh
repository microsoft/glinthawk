#pragma once

#include "concept.hh"

#if !defined( GLINTHAWK_CUDA_ENABLED )
#warning "CUDA is not enabled"
#endif

#include <random>
#include <glog/logging.h>
#include <source_location>

#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

namespace glinthawk::models::common::cuda {

constexpr size_t TPB = 64;    /* threads per block */
constexpr size_t NRBS = 32;   /* norm reduce block size */
constexpr size_t AMRBS = 128; /* argmax reduce block size */

template<typename DType>
class Operations
{
protected:
  void CHECK_CUBLAS( const cublasStatus_t err, const std::source_location location = std::source_location::current() );
  void CHECK_CUDA( const cudaError_t err, const source_location location = std::source_location::current() );

  cudaStream_t* streams;
  cublasHandle_t cublas_handle_default;
  cublasHandle_t* cublas_handle_array;
  int cublas_handle_count;

  constexpr cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;

public:
  Operations( const int num_streams );
  ~Operations();

  template<uint64_t size>
  void accum( DType* a, const DType* b, const uint64_t batch_size );

  template<uint64_t size>
  void rmsnorm( DType* o, const DType* x, const DType* weight, const uint64_t batch_size );

  template<uint64_t n>
  void argmax( uint32_t* output, const DType* v, const uint64_t batch_size );

  template<uint64_t hidden_dim>
  void silu( DType* hb, DType* hb2, const uint64_t batch_size );

  template<uint64_t s, uint64_t r>
  void matmul( DType* xo, const DType* x, const DType* w, const uint64_t b );

  template<uint64_t vocab_size>
  void soft_sample( DType* v, const std::vector<float>& temp_s, const uint64_t batch_size );

  void copy( DType* dst,
             const DType* src,
             const uint64_t batch_size,
             const bool async = false,
             const CopyType type = CopyType::HostToHost );

  void randomize_buffer( DType* buffer, const uint64_t len, const float min, const float max );
};

static_assert( OperationsConcept<Operations<float>, float> );
static_assert( OperationsConcept<Operations<__half>, __half> );

// helper functions are in this anonymous namespace
namespace {

template<std::unsigned_integral T>
constexpr T div_ceil( const T x, const T y )
{
  return x / y + ( x % y != 0 );
}

__global__ void accum_cuda( __half* a, const __half* b, const uint64_t size )
{
  const uint64_t i = blockIdx.x * TPB + threadIdx.x;
  if ( i < size ) {
    a[i] += b[i];
  }
}

namespace { // rmsnorm

template<uint64_t size>
__global__ void normalize_and_scale_full( float* output, const float* x, const float* weight, const float* ss )
{
  const uint64_t i = threadIdx.x + blockIdx.x * TPB;

  if ( i < size ) {
    const float denom = sqrtf( *ss / size + 1e-5f );
    output[i] = weight[i] * x[i] / denom;
  }
}

template<uint64_t size>
__global__ void normalize_and_scale_half( __half* output, const __half* x, const __half* weight, const float* ss )
{
  const uint64_t gb_i = threadIdx.x + blockIdx.x * TPB + blockIdx.y * size;
  const uint64_t i = threadIdx.x + blockIdx.x * TPB;

  if ( i < size ) {
    const float denom = sqrtf( ss[blockIdx.y] / size + 1e-5f );
    output[gb_i] = weight[i] * __float2half( __half2float( x[gb_i] ) / denom );
  }
}

template<uint64_t size>
__global__ void reduce_norm_v2_square_batched( float* output, const __half* x )
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

template<uint64_t size>
__global__ void reduce_norm_v2_sum_batched( float* output, const float* x )
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

template<uint64_t size>
void square_reduce_step_2( float* output, const float* x, float* temp_1, float* temp_2, const uint64_t batch_size )
{
  const uint64_t max_elems_per_block = NRBS * 2;
  const uint64_t shmem_size = sizeof( float ) * max_elems_per_block;

  const uint64_t grid_size = div_ceil( size, max_elems_per_block );

  dim3 grids( grid_size, batch_size );
  if ( grid_size == 1 ) {
    reduce_norm_v2_sum_batched<size><<<grids, NRBS, shmem_size>>>( output, x, size );
  } else {
    reduce_norm_v2_sum_batched<size><<<grids, NRBS, shmem_size>>>( temp_1, x, size );
    square_reduce_step_2<size>( output, temp_1, temp_2, temp_1, grid_size, batch_size );
  }
}

template<uint64_t size>
void square_reduce_step_1( float* output, const __half* x, const uint64_t batch_size )
{
  const uint64_t max_elems_per_block = NRBS * 2;
  const uint64_t shmem_size = sizeof( float ) * max_elems_per_block;

  const uint64_t grid_size = div_ceil( size, max_elems_per_block );

  dim3 grids( grid_size, batch_size );
  if ( grid_size == 1 ) {
    reduce_norm_v2_square_batched<size><<<grids, NRBS, shmem_size>>>( output, x, size );
  } else {
    float* temp_1 = output + batch_size;
    float* temp_2 = temp_1 + batch_size * grid_size;
    reduce_norm_v2_square_batched<size><<<grids, NRBS, shmem_size>>>( temp_1, x, size );
    square_reduce_step_2<size>( output, temp_1, temp_2, temp_1, grid_size, batch_size );
  }
}

} // namespace rmsnorm

namespace { // argmax

template<typename DType, uint64_t size>
__global__ void argmax_batched_init( uint32_t* output_arg, DType* output, const DType* x )
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

template<typename DType, uint64_t size>
__global__ void argmax_batched_next( uint32_t* output_arg, DType* output, const uint32_t* x_arg, const DType* x )
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

template<typename DType, uint64_t size>
void argmax_step_2( uint32_t* output_arg,
                    const uint32_t* x_arg,
                    const DType* x,
                    uint32_t* temp_1_arg,
                    DType* temp_1,
                    uint32_t* temp_2_arg,
                    DType* temp_2,
                    const uint64_t batch_size )
{
  constexpr uint64_t max_elems_per_block = AMRBS * 2;
  constexpr uint64_t shmem_size = ( sizeof( DType ) + sizeof( uint32_t ) ) * max_elems_per_block;
  constexpr uint64_t grid_size = div_ceil( size, max_elems_per_block );

  dim3 grids( grid_size, batch_size );

  if constexpr ( grid_size == 1 ) {
    argmax_batched_next<DType, size><<<grids, AMRBS, shmem_size>>>( output_arg, temp_1, x_arg, x );
  } else {
    argmax_batched_next<DType, size><<<grids, AMRBS, shmem_size>>>( temp_1_arg, temp_1, x_arg, x );
    argmax_step_2<grid_size>( output_arg, temp_1_arg, temp_1, temp_2_arg, temp_2, temp_1_arg, temp_1, batch_size );
  }
}

template<typename DType, uint64_t size>
void argmax_step_1( uint32_t* output_arg, const DType* x, const uint64_t batch_size )
{
  constexpr uint64_t max_elems_per_block = AMRBS * 2;
  constexpr uint64_t shmem_size = ( sizeof( DType ) + sizeof( uint32_t ) ) * max_elems_per_block;
  constexpr uint64_t grid_size = div_ceil( size, max_elems_per_block );

  dim3 grids( grid_size, batch_size );
  if constexpr ( grid_size == 1 ) {
    DType* output = reinterpret_cast<DType*>( output_arg + batch_size );
    argmax_batched_init<DType, size><<<grids, AMRBS, shmem_size>>>( output_arg, output, x );
  } else {
    DType* temp_1 = reinterpret_cast<DType*>( output_arg + batch_size );
    DType* temp_2 = temp_1 + batch_size * grid_size;
    uint32_t* temp_1_arg = reinterpret_cast<uint32_t*>( temp_2 + batch_size * grid_size );
    uint32_t* temp_2_arg = temp_1_arg + batch_size * grid_size;
    argmax_batched_init<DType, size><<<grids, AMRBS, shmem_size>>>( temp_1_arg, temp_1, x );
    argmax_step_2<grid_size>( output_arg, temp_1_arg, temp_1, temp_2_arg, temp_2, temp_1_arg, temp_1, batch_size );
  }
}

} // namespace argmax

namespace { // silu

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

}

namespace { // soft_sample

template<typename DType, uint64_t vocab_size>
__global__ void gumbel_fix( DType* array, float temp, curandState* rng_state )
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

}

}

template<typename DType>
Operations<DType>::Operations( const int num_streams )
{
  CHECK_CUBLAS( cublasCreate( &cublas_handle_default ) );
  cublas_handle_count = num_streams;
  streams = (cudaStream_t*)malloc( num_streams * sizeof( cudaStream_t ) );
  cublas_handle_array = (cublasHandle_t*)malloc( num_streams * sizeof( cublasHandle_t ) );
  for ( int i = 0; i < num_streams; i++ ) {
    cudaStreamCreate( &( streams[i] ) );
    CHECK_CUBLAS( cublasCreate( &( cublas_handle_array[i] ) ) );
    CHECK_CUBLAS( cublasSetStream( cublas_handle_array[i], streams[i] ) );
  }
}

template<typename DType>
Operations<DType>::~Operations()
{
  CHECK_CUBLAS( cublasDestroy( cublas_handle_default ) );
  for ( int i = 0; i < cublas_handle_count; i++ ) {
    CHECK_CUBLAS( cublasDestroy( cublas_handle_array[i] ) );
    cudaStreamDestroy( streams[i] );
  }
  free( streams );
  free( cublas_handle_array );
}

template<>
template<uint64_t size>
void Operations<float>::accum( float* a, const float* b, const uint64_t batch_size )
{
  const float alpha = 1.0f;
  CHECK_CUBLAS( cublasSaxpy( cublas_handle_default, size * batch_size, &alpha, b, 1, a, 1 ) );
}

template<>
template<uint64_t size>
void Operations<__half>::accum( __half* a, const __half* b, const uint64_t batch_size )
{
  accum_cuda<<<div_ceil( size * batch_size, TPB ), TPB>>>( a, b, size * batch_size );
}

template<>
template<uint64_t size>
void Operations<__half>::rmsnorm( __half* o, const __half* x, const __half* weight, const uint64_t batch_size )
{
  square_reduce_step_1<size>( reinterpret_cast<float*>( temp ), x, batch_size );

  dim3 grid { div_ceil( size, TPB ), batch_size };
  normalize_and_scale_half<size><<<grid, TPB>>>( output, x, weight, reinterpret_cast<float*>( temp ) );
}

template<>
template<uint64_t size>
void Operations<float>::rmsnorm( float* o, const float* x, const float* weight, const uint64_t batch_size )
{
  for ( size_t i = 0; i < batch_size; i++ ) {
    CHECK_CUBLAS( cublasSdot( cublas_handle_default, size, x + i * size, 1, x + i * size, 1, temp + i ) );
    normalize_and_scale_full<size><<<div_ceil( size, TPB ), TPB>>>( output + i * size, x + i * size, weight, temp + i );
  }
}

template<typename DType>
template<uint64_t n>
void Operations<DType>::argmax( uint32_t* output, const DType* v, const uint64_t batch_size )
{
  argmax_step_1<DType, n>( reinterpret_cast<uint32_t*>( temp ), v, batch_size );
  CHECK_CUDA( cudaMemcpy( output_cpu, temp, batch_size * sizeof( uint32_t ), cudaMemcpyDeviceToHost ) );
}

template<typename DType>
template<uint64_t hidden_dim>
void Operations<DType>::silu( DType* hb, DType* hb2, const uint64_t batch_size )
{
  silu_direct<<<div_ceil( hidden_dim * batch_size, TPB ), TPB>>>( _hb, _hb2, hidden_dim * batch_size );
}

template<typename DType>
template<uint64_t s, uint64_t r>
void Operations<DType>::matmul( DType* xo, const DType* x, const DType* w, const uint64_t b )
{
  constexpr cudaDataType_t cuda_arg_type = is_same_v<DType, __half> ? CUDA_R_16F : CUDA_R_32F;

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

template<typename DType>
template<uint64_t vocab_size>
void Operations<DType>::soft_sample( DType* v, const std::vector<float>& temp_s, const uint64_t batch_size )
{
  for ( uint64_t i = 0; i < batch_size; i++ ) {
    if ( temp_s[i] > 0 ) {
      gumbel_fix<vocab_size><<<div_ceil( vocab_size, TPB ), TPB, 0, streams[i]>>>(
        v + i * vocab_size, temp_s[i], vocab_size, rng_state + i * vocab_size );
    }
  }
}

template<typename DType>
void Operations<DType>::copy( DType* dst,
                              const DType* src,
                              const uint64_t batch_size,
                              const bool async,
                              const CopyType type )
{
  auto convert_to_cuda = []( const CopyType type ) {
    switch ( type ) {
      case CopyType::HostToHost: return cudaMemcpyHostToHost;
      case CopyType::HostToDevice: return cudaMemcpyHostToDevice;
      case CopyType::DeviceToHost: return cudaMemcpyDeviceToHost;
      case CopyType::DeviceToDevice: return cudaMemcpyDeviceToDevice;
    }
  };

  if ( async ) {
    CHECK_CUDA( cudaMemcpyAsync( dst, src, batch_size * sizeof( DType ), convert_to_cuda( type ) ) );
  } else {
    CHECK_CUDA( cudaMemcpy( dst, src, batch_size * sizeof( DType ), convert_to_cuda( type ) ) );
  }
}

template<typename DType>
void Operations<DType>::randomize_buffer( DType* buffer, const uint64_t len, const float min, const float max )
{
  static thread_local std::mt19937 generator { std::random_device {}() };
  std::uniform_real_distribution<float> distribution( min, max );

  size_t i;
#pragma omp parallel for schedule( static ) private( i )
  for ( i = 0; i < len; i++ ) {
    if constexpr ( std::is_same_v<DType, float> ) {
      buffer[i] = distribution( generator );
    } else {
      buffer[i] = static_cast<DType>( distribution( generator ) );
    }
  }
}

} // namespace glinthawk::models::common::cuda
