#pragma once

#include <memory>
#include <vector>

#include "../common/ops.cuh"
#include "models/llama2/base.hh"
#include "models/llama2/ops/concept.hh"

namespace glinthawk::models::llama2::cuda {

template<typename Config, typename DType>
class LlamaOperations : public common::cuda::Operations<DType>
{
public:
  using common::cuda::Operations<DType>::DeviceUniquePtr;

private:
  std::unique_ptr<curandState, models::common::cuda::CUDADeleter<curandState>> rng_state { nullptr };
  void setup_rng( unsigned long seed, const uint64_t size, const uint64_t batch_size );

public:
  LlamaOperations( const Settings<Config>& settings );
  ~LlamaOperations() {}

  void attention_0_gemm( const DType* query,
                         const DType* const context_pointers[],
                         DType* att,
                         const uint64_t batch_size,
                         const uint32_t* token_positions );

  void attention_2_gemm( const DType* att,
                         const DType* const context_pointers[],
                         DType* xb,
                         const uint64_t batch_size,
                         const uint32_t* token_positions );

  void attention_softmax( DType* att, const uint32_t* token_positions, DType* temp_buffer, const uint64_t batch_size );

  void apply_rope( const uint64_t curr_batch_size,
                   const uint32_t* token_positions,
                   const DType* freq_cis_real,
                   const DType* freq_cis_imag,
                   DType* state_q,
                   DType* context_pointers[] );

  void soft_sample( DType* v, const std::vector<float>& temp_s, const uint64_t batch_size );

  void copy_kv_cache( DType* context_pointers[],
                      const DType* state_k,
                      const DType* state_v,
                      const uint64_t batch_size,
                      const uint32_t* token_positions );

  template<typename DTypeDst, typename DTypeSrc>
  void convert_and_copy( DTypeDst* dst, const DTypeSrc* src, const uint64_t size, const CopyType );
};

static_assert(
  LlamaOperationsConcept<LlamaOperations<configs::Stories_110M, float>, float, Settings<configs::Stories_110M>> );

static_assert(
  LlamaOperationsConcept<LlamaOperations<configs::Stories_110M, __half>, __half, Settings<configs::Stories_110M>> );

template<typename Config, typename DType>
class Context : public InferenceContext<Config, DType>
{
private:
  typename LlamaOperations<Config, DType>::DeviceUniquePtr storage_;

public:
  Context( const Settings<Config>& settings );
};

namespace {

template<std::unsigned_integral T>
constexpr T div_ceil( const T x, const T y )
{
  return x / y + ( x % y != 0 );
}

constexpr size_t TPB = 64;    /* threads per block */
constexpr size_t NRBS = 32;   /* norm reduce block size */
constexpr size_t AMRBS = 128; /* argmax reduce block size */

}

template<typename Config, typename DType>
Context<Config, DType>::Context( const Settings<Config>& settings )
  : storage_( [&]() -> decltype( storage_ ) {
    DType* ptr;
    const cudaError_t err = cudaMalloc( &ptr, Context<Config, DType>::context_size( settings ) );

    if ( err == cudaSuccess ) {
      return decltype( storage_ ) { ptr };
    } else {
      return decltype( storage_ ) { nullptr };
    }
  }() )
{
  this->buffer_ = storage_.get();
}

// all helper functions are defined in this anonymous namespace
namespace {

namespace { // attention_softmax

template<typename DType, uint64_t seq_len>
__global__ void find_max_for_rows( const DType* att, DType* output, const uint64_t token_pos )
{
  const uint64_t head_num = threadIdx.x;
  att += head_num * seq_len;

  DType max_value = att[0];
  for ( uint64_t i = 1; i <= token_pos; i++ ) {
    if constexpr ( std::is_same_v<DType, __half> ) {
      max_value = __hmax( max_value, att[i] );
    } else {
      max_value = max( max_value, att[i] );
    }
  }

  output[head_num] = max_value;
}

template<typename DType, uint64_t seq_len>
__global__ void subtract_and_expf( const DType* values, DType* att )
{
  const uint64_t head_num = threadIdx.x;
  const uint64_t token_pos = blockIdx.x;

  att += head_num * seq_len;

  if constexpr ( std::is_same_v<DType, __half> ) {
    att[token_pos] = hexp( att[token_pos] - values[head_num] );
  } else {
    att[token_pos] = expf( att[token_pos] - values[head_num] );
  }
}

template<typename DType, uint64_t seq_len>
__global__ void sum_rows( DType* att, DType* output, const uint64_t token_pos )
{
  const uint64_t head_num = threadIdx.x;
  att += head_num * seq_len;

  DType sum = 0.0;
  for ( uint64_t i = 0; i <= token_pos; i++ ) {
    sum += att[i];
  }

  output[head_num] = sum;
}

template<typename DType, uint64_t seq_len>
__global__ void normalize_by_sum( DType* att, const DType* sums )
{
  const uint64_t head_num = threadIdx.x;
  const uint64_t token_pos = blockIdx.x;

  att += head_num * seq_len;
  att[token_pos] /= sums[head_num];
}

}

namespace { // rope

template<typename DType, uint64_t head_size, uint64_t gqa_size>
__global__ void do_rope( const DType* freq_cis_real_row,
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

}

namespace { // soft_sample

template<typename DType, uint64_t vocab_size>
__global__ void gumbel_fix( DType* array, const float temp, curandState* rng_state )
{
  const uint64_t i = threadIdx.x + blockIdx.x * TPB;

  if ( i < vocab_size ) {
    float myrandf = curand_uniform( rng_state + i );
    myrandf = logf( -logf( myrandf ) );
    if constexpr ( std::is_same_v<DType, __half> ) {
      array[i] = __float2half( __half2float( array[i] ) / temp - myrandf );
    } else {
      array[i] = array[i] / temp - myrandf;
    }
  }
}

}

namespace { // setup_rng

__global__ void setup_rng_kernel( curandState* state, unsigned long seed )
{
  int id = threadIdx.x + blockIdx.x * TPB;
  curand_init( seed, id, 0, &state[id] );
}

}

}

template<typename Config, typename DType>
LlamaOperations<Config, DType>::LlamaOperations( const Settings<Config>& settings )
  : common::cuda::Operations<DType>( settings.concurrency_limit )
{
  setup_rng( 1234ul, Config::vocab_size, settings.concurrency_limit );

  // Summary of Checks:
  // (a) TPB must not exceed 1024. Threads per block cannot surpass 1024.
  // (b) Config::n_heads must not exceed 1024.
  // (c) Config::dim / Config::n_heads / 2 must not exceed 1024.
  // (d) Config::n_heads must not exceed (1 << 16) - 1. RoPE has n_heads blocks, which cannot surpass 2^16.
  // (e) Config::seq_len must not exceed (1 << 16) - 1. Attention softmax has seq_len blocks, which cannot surpass 2^16.
  // (f) Accum blocks must not exceed (1 << 16) - 1.
  // (g) Silu blocks must not exceed (1 << 16) - 1.
  // (h) CuRAND blocks must not exceed (1 << 16) - 1.
  // (i) RMS Norm blocks must not exceed (1 << 16) - 1.
  // (j) RMS Norm scratch pad must have enough space for calculations.
  // (k) Argmax scratch pad must have enough space.

  static_assert( 1024 >= TPB );                                                                  // (a)
  static_assert( 1024 >= Config::n_heads );                                                      // (b)
  static_assert( 1024 >= Config::dim / Config::n_heads / 2 );                                    // (c)
  static_assert( ( 1 << 16 ) - 1 >= Config::n_heads );                                           // (d)
  static_assert( ( 1 << 16 ) - 1 >= Config::seq_len );                                           // (e)
  CHECK_GE( ( 1 << 16 ) - 1, div_ceil( Config::dim * settings.concurrency_limit, TPB ) );        // (f)
  CHECK_GE( ( 1 << 16 ) - 1, div_ceil( Config::hidden_dim * settings.concurrency_limit, TPB ) ); // (g)
  static_assert( ( 1 << 16 ) - 1 >= div_ceil( Config::vocab_size, TPB ) );                       // (h)
  static_assert( ( 1 << 16 ) - 1 >= div_ceil( Config::dim, NRBS ) );                             // (i)

  static_assert(
    sizeof( DType ) * Config::dim
    >= sizeof( float )
         * ( div_ceil( Config::dim, 2 * NRBS ) + div_ceil( div_ceil( Config::dim, 2 * NRBS ), 2 * NRBS ) + 1 ) ); // (j)

  static_assert( sizeof( DType ) * ( 4 * Config::dim + 2 * Config::hidden_dim )
                 >= ( sizeof( uint32_t )
                        * ( div_ceil( Config::vocab_size, 2 * AMRBS )
                            + div_ceil( div_ceil( Config::vocab_size, 2 * AMRBS ), 2 * AMRBS ) + 1 )
                      + sizeof( DType )
                          * ( div_ceil( Config::vocab_size, 2 * AMRBS )
                              + div_ceil( div_ceil( Config::vocab_size, 2 * AMRBS ), 2 * AMRBS ) ) ) ); // (k)
}

template<typename Config, typename DType>
void LlamaOperations<Config, DType>::attention_0_gemm( const DType* query,
                                                       const DType* const context_pointers[],
                                                       DType* att,
                                                       const uint64_t batch_size,
                                                       const uint32_t* token_positions )
{
  constexpr cudaDataType_t cuda_arg_type = std::is_same_v<DType, __half> ? CUDA_R_16F : CUDA_R_32F;
  const float scale = 1.0f / sqrtf( Config::head_size );
  constexpr uint64_t k = Config::head_size;
  constexpr uint64_t n = Config::gqa_size;
  constexpr uint64_t lda = Config::n_kv_heads * Config::head_size * 2;
  constexpr uint64_t ldb = k;
  constexpr uint64_t ldc = Config::seq_len;
  constexpr uint64_t strideA = Config::head_size;
  constexpr uint64_t strideB = Config::head_size * Config::gqa_size;
  constexpr uint64_t strideC = Config::seq_len * Config::gqa_size;
  constexpr uint64_t gemm_batch_count = Config::n_kv_heads;
  constexpr uint64_t dim = Config::head_size * Config::n_kv_heads * Config::gqa_size;
  constexpr uint64_t att_dim = Config::seq_len * Config::n_kv_heads * Config::gqa_size;

  for ( size_t i = 0; i < batch_size; i++ ) {
    const uint64_t m = token_positions[i] + 1;
    common::cuda::CHECK_CUBLAS( cublasGemmStridedBatchedEx( this->cublas_handle_array[i],
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
                                                            query + i * dim,
                                                            cuda_arg_type,
                                                            ldb,
                                                            strideB,
                                                            &this->beta,
                                                            att + i * att_dim,
                                                            cuda_arg_type,
                                                            ldc,
                                                            strideC,
                                                            gemm_batch_count,
                                                            this->computeType,
                                                            CUBLAS_GEMM_DEFAULT ) );
  }
}

template<typename Config, typename DType>
void LlamaOperations<Config, DType>::attention_2_gemm( const DType* att,
                                                       const DType* const context_pointers[],
                                                       DType* xb,
                                                       const uint64_t batch_size,
                                                       const uint32_t* token_positions )
{
  constexpr cudaDataType_t cuda_arg_type = std::is_same_v<DType, __half> ? CUDA_R_16F : CUDA_R_32F;

  constexpr uint64_t m = Config::head_size;
  constexpr uint64_t n = Config::gqa_size;

  constexpr uint64_t lda = Config::n_kv_heads * Config::head_size * 2;
  constexpr uint64_t ldb = Config::seq_len;
  constexpr uint64_t ldc = m;

  constexpr uint64_t strideA = Config::head_size;
  constexpr uint64_t strideB = Config::seq_len * Config::gqa_size;
  constexpr uint64_t strideC = Config::head_size * Config::gqa_size;

  constexpr uint64_t gemm_batch_count = Config::n_kv_heads;

  constexpr uint64_t kv_dim = Config::head_size * Config::n_kv_heads;
  constexpr uint64_t dim = Config::head_size * Config::n_kv_heads * Config::gqa_size;
  constexpr uint64_t att_dim = Config::seq_len * Config::n_kv_heads * Config::gqa_size;

  for ( size_t i = 0; i < batch_size; i++ ) {
    const uint64_t k = token_positions[i] + 1;

    common::cuda::CHECK_CUBLAS( cublasGemmStridedBatchedEx( this->cublas_handle_array[i],
                                                            CUBLAS_OP_N,
                                                            CUBLAS_OP_N,
                                                            m,
                                                            n,
                                                            k,
                                                            &this->alpha,
                                                            context_pointers[i] + kv_dim,
                                                            cuda_arg_type,
                                                            lda,
                                                            strideA,
                                                            att + i * att_dim,
                                                            cuda_arg_type,
                                                            ldb,
                                                            strideB,
                                                            &this->beta,
                                                            xb + i * dim,
                                                            cuda_arg_type,
                                                            ldc,
                                                            strideC,
                                                            gemm_batch_count,
                                                            this->computeType,
                                                            CUBLAS_GEMM_DEFAULT ) );
  }
}

template<typename Config, typename DType>
void LlamaOperations<Config, DType>::attention_softmax( DType* att,
                                                        const uint32_t* token_positions,
                                                        DType* temp_buffer,
                                                        const uint64_t batch_size )
{
  for ( uint64_t i = 0; i < batch_size; i++ ) {
    DType* this_att = att + i * Config::n_heads * Config::seq_len;
    DType* this_buff = temp_buffer + i * Config::n_heads;

    // (1) find the max value for each head (each row)
    find_max_for_rows<DType, Config::seq_len>
      <<<1, Config::n_heads, 0, this->streams[i]>>>( this_att, this_buff, token_positions[i] );

    // (2) exp(att - max)
    subtract_and_expf<DType, Config::seq_len>
      <<<token_positions[i] + 1, Config::n_heads, 0, this->streams[i]>>>( this_buff, this_att );

    // (3) sum each row
    sum_rows<DType, Config::seq_len>
      <<<1, Config::n_heads, 0, this->streams[i]>>>( this_att, this_buff, token_positions[i] );

    // (4) normalize each row by its sum
    normalize_by_sum<DType, Config::seq_len>
      <<<token_positions[i] + 1, Config::n_heads, 0, this->streams[i]>>>( this_att, this_buff );
  }
}

template<typename Config, typename DType>
void LlamaOperations<Config, DType>::apply_rope( const uint64_t curr_batch_size,
                                                 const uint32_t* token_positions,
                                                 const DType* freq_cis_real,
                                                 const DType* freq_cis_imag,
                                                 DType* state_q,
                                                 DType* context_pointers[] )
{
  for ( uint64_t i = 0; i < curr_batch_size; i++ ) {
    do_rope<DType, Config::head_size, Config::gqa_size>
      <<<Config::n_kv_heads, Config::head_size / 2, 0, this->streams[i]>>>(
        freq_cis_real + token_positions[i] * Config::head_size / 2,
        freq_cis_imag + token_positions[i] * Config::head_size / 2,
        state_q + i * Config::n_kv_heads * Config::gqa_size * Config::head_size,
        context_pointers[i] + token_positions[i] * Config::n_kv_heads * Config::head_size * 2 );
  }
}

template<typename Config, typename DType>
void LlamaOperations<Config, DType>::soft_sample( DType* v,
                                                  const std::vector<float>& temp_s,
                                                  const uint64_t batch_size )
{
  for ( uint64_t i = 0; i < batch_size; i++ ) {
    if ( temp_s[i] > 0 ) {
      gumbel_fix<DType, Config::vocab_size><<<div_ceil( Config::vocab_size, TPB ), TPB, 0, this->streams[i]>>>(
        v + i * Config::vocab_size, temp_s[i], rng_state.get() + i * Config::vocab_size );
    }
  }
}

template<typename Config, typename DType>
void LlamaOperations<Config, DType>::copy_kv_cache( DType* context_pointers[],
                                                    const DType* state_k,
                                                    const DType* state_v,
                                                    const uint64_t batch_size,
                                                    const uint32_t* token_positions )
{
  for ( size_t i = 0; i < batch_size; i++ ) {
    if ( context_pointers[i] == nullptr ) {
      continue;
    }

    DType* k_cache_pos = context_pointers[i] + token_positions[i] * Config::kv_dim * 2;
    DType* v_cache_pos = k_cache_pos + Config::kv_dim;

    // XXX why not just one memcpy?
    common::cuda::CHECK_CUDA( cudaMemcpyAsync(
      k_cache_pos, state_k + i * Config::kv_dim, Config::kv_dim * sizeof( DType ), cudaMemcpyDeviceToDevice ) );

    common::cuda::CHECK_CUDA( cudaMemcpyAsync(
      v_cache_pos, state_v + i * Config::kv_dim, Config::kv_dim * sizeof( DType ), cudaMemcpyDeviceToDevice ) );
  }
}

template<typename Config, typename DType>
template<typename DTypeDst, typename DTypeSrc>
void LlamaOperations<Config, DType>::convert_and_copy( DTypeDst* dst,
                                                       const DTypeSrc* src,
                                                       const uint64_t size,
                                                       const CopyType type )
{
  switch ( type ) {
    case CopyType::DeviceToHost: {
      if constexpr ( std::is_same_v<DTypeSrc, DTypeDst> ) {
        common::cuda::CHECK_CUDA( cudaMemcpy( dst, src, size * sizeof( DTypeSrc ), cudaMemcpyDeviceToHost ) );
      } else {
        std::unique_ptr<DTypeSrc> src_cpu { reinterpret_cast<DTypeSrc*>( new uint8_t[sizeof( DTypeSrc ) * size] ) };
        common::cuda::CHECK_CUDA( cudaMemcpy( src_cpu.get(), src, size * sizeof( DTypeSrc ), cudaMemcpyDeviceToHost ) );
        for ( uint64_t i = 0; i < size; i++ ) {
          dst[i] = static_cast<DTypeDst>( src_cpu.get()[i] );
        }
      }
      break;
    }

    case CopyType::HostToDevice: {
      if constexpr ( std::is_same_v<DTypeSrc, DTypeDst> ) {
        common::cuda::CHECK_CUDA( cudaMemcpy( dst, src, size * sizeof( DTypeSrc ), cudaMemcpyHostToDevice ) );
      } else {
        std::unique_ptr<DTypeSrc> dst_cpu { reinterpret_cast<DTypeSrc*>( new uint8_t[sizeof( DTypeDst ) * size] ) };
        for ( uint64_t i = 0; i < size; i++ ) {
          dst_cpu.get()[i] = static_cast<DTypeDst>( src[i] );
        }
        common::cuda::CHECK_CUDA( cudaMemcpy( dst, dst_cpu.get(), size * sizeof( DTypeSrc ), cudaMemcpyHostToDevice ) );
      }
    } break;
  }

  LOG( FATAL ) << "Invalid copy type";
}

template<typename Config, typename DType>
void LlamaOperations<Config, DType>::setup_rng( unsigned long seed, const uint64_t size, const uint64_t batch_size )
{
  curandState* rng_state_ptr = nullptr;
  common::cuda::CHECK_CUDA( cudaMalloc( &rng_state_ptr, size * batch_size * sizeof( curandState ) ) );
  rng_state.reset( rng_state_ptr );

  for ( uint64_t i = 0; i < batch_size; i++ ) {
    setup_rng_kernel<<<div_ceil( size, TPB ), TPB, 0, this->streams[i]>>>( rng_state_ptr + i * size, seed );
  }
}

} // namespace glinthawk::models::llama2::cuda
