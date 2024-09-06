#pragma once

#include <list>
#include <memory>
#include <optional>
#include <vector>

#include "../common/ops.cuh"
#include "../common/vmem.cuh"
#include "models/llama2/base.hh"
#include "models/llama2/ops/concept.hh"

namespace glinthawk::models::llama2::cuda {

// platform-specific context
template<typename Config, typename DType>
class Context;

template<typename Config, typename DType>
class DynamicContext;

template<typename Config, typename DType, typename Ctx = Context<Config, DType>>
requires ModelConfig<Config> && ContextConcept<Ctx, DType>
class LlamaOperations : public common::cuda::Operations<DType>
{
public:
  using common::cuda::Operations<DType>::DeviceUniquePtr;
  using ContextType = Ctx;

public:
  LlamaOperations( const ConfigRuntime<Config>& settings );
  ~LlamaOperations() {}

  void attention_0_gemm( const DType* query,
                         const typename ContextType::LayerContextType layer_contexts[],
                         DType* att,
                         const uint64_t batch_size,
                         const uint32_t* token_positions ) const;

  void attention_2_gemm( const DType* att,
                         const typename ContextType::LayerContextType layer_contexts[],
                         DType* xb,
                         const uint64_t batch_size,
                         const uint32_t* token_positions ) const;

  void attention_softmax( DType* att,
                          const uint32_t* token_positions,
                          DType* temp_buffer,
                          const uint64_t batch_size ) const;

  void apply_rope( const uint64_t curr_batch_size,
                   const uint32_t* token_positions,
                   const DType* freq_cis_real,
                   const DType* freq_cis_imag,
                   DType* state_q,
                   typename ContextType::TokenContextType token_contexts[] ) const;

  void copy_kv_cache( typename ContextType::TokenContextType token_contexts[],
                      const DType* state_kv,
                      const uint64_t batch_size ) const;

  template<typename DTypeDst, typename DTypeSrc>
  void convert_and_copy( DTypeDst* dst, const DTypeSrc* src, const uint64_t size, const CopyType ) const;
};

template<typename Config, typename DType>
class Context : public llama2::Context<Config, DType>
{
private:
  typename common::cuda::Operations<DType>::DeviceUniquePtr storage_;

public:
  using llama2::Context<Config, DType>::Context;
  Context( const ConfigRuntime<Config>& settings, const bool make_empty = false );
};

template<typename Config, typename DType>
class DynamicContext : public llama2::Context<Config, DType>
{
private:
  std::optional<common::cuda::VirtualMemoryRegion> storage_;
  std::vector<size_t> layer_allocated_offsets_;

public:
  using llama2::Context<Config, DType>::Context;

  DynamicContext( const ConfigRuntime<Config>& settings, const bool make_empty = false );
  ~DynamicContext() {}

  bool prepare( const size_t layer_num, const size_t token_pos );
};

static_assert( LlamaOperationsConcept<LlamaOperations<configs::Stories_110M, glinthawk::float32_t>,
                                      glinthawk::float32_t,
                                      ConfigRuntime<configs::Stories_110M>> );

static_assert( LlamaOperationsConcept<LlamaOperations<configs::Stories_110M, glinthawk::float16_t>,
                                      glinthawk::float16_t,
                                      ConfigRuntime<configs::Stories_110M>> );

static_assert( LlamaOperationsConcept<LlamaOperations<configs::Stories_110M, glinthawk::bfloat16_t>,
                                      glinthawk::bfloat16_t,
                                      ConfigRuntime<configs::Stories_110M>> );

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
Context<Config, DType>::Context( const ConfigRuntime<Config>& settings, const bool make_empty )
  : llama2::Context<Config, DType>()
  , storage_( [&]() -> decltype( storage_ ) {
    DType* ptr;
    if ( make_empty ) {
      ptr = nullptr;
    } else {
      const cudaError_t err
        = cudaMalloc( &ptr, Context<Config, DType>::max_size( settings.num_attention_layers_hosted() ) );
      CHECK_EQ( err, cudaSuccess ) << "Failed to create context vector on CUDA device";
      return decltype( storage_ ) { ptr };
    }
    return decltype( storage_ ) { ptr };
  }() )
{
  llama2::Context<Config, DType>::set_buffer( settings, storage_.get() );
}

template<typename Config, typename DType>
DynamicContext<Config, DType>::DynamicContext( const ConfigRuntime<Config>& settings, const bool make_empty )
  : llama2::Context<Config, DType>()
  , storage_( [&]() -> decltype( storage_ ) {
    if ( make_empty ) {
      return std::nullopt;
    }

    return common::cuda::VirtualMemoryRegion(
      Context<Config, DType>::max_size( settings.num_attention_layers_hosted() ) );
  }() )
  , layer_allocated_offsets_( Config::n_layers, 0 )
{
  llama2::Context<Config, DType>::set_buffer(
    settings, storage_.has_value() ? reinterpret_cast<DType*>( storage_->ptr() ) : nullptr );
}

template<typename Config, typename DType>
bool DynamicContext<Config, DType>::prepare( const size_t layer_num, const size_t token_pos )
{
  constexpr size_t BLOCK_SIZE = 8 * 1024 * 1024; // 8 MiB

  // we need to make sure that for this layer and this token, there is memory allocated
  const auto layer = this->layer( layer_num );
  const DType* layer_start_addr = layer.token( 0 ).key();
  const DType* layer_end_addr = layer.token( token_pos + 1 ).key();

  // (1) do we have enough memory allocated?
  const size_t layer_allocated_len = layer_allocated_offsets_.at( layer_num );
  const size_t needed_allocated_len = ( layer_end_addr - layer_start_addr ) * sizeof( DType );

  if ( needed_allocated_len > layer_allocated_len ) {
    storage_->allocate_span( layer_start_addr + layer_allocated_len / sizeof( DType ), BLOCK_SIZE );
    layer_allocated_offsets_.at( layer_num ) += BLOCK_SIZE;
  }

  return true;
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
    if constexpr ( std::is_same_v<DType, glinthawk::float16_t> or std::is_same_v<DType, glinthawk::bfloat16_t> ) {
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

  if constexpr ( std::is_same_v<DType, glinthawk::float16_t> or std::is_same_v<DType, glinthawk::bfloat16_t> ) {
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

} // end of anonymous namespace for helper functions

template<typename Config, typename DType, typename ContextType>
LlamaOperations<Config, DType, ContextType>::LlamaOperations( const ConfigRuntime<Config>& settings )
  : common::cuda::Operations<DType>( settings.concurrency_limit,
                                     Config::vocab_size,
                                     settings.hosts( Config::n_layers - 1, InferenceStage::Classification ),
                                     settings.concurrency_limit )
{
  // Summary of Checks:
  // (a) TPB must not exceed 1024. Threads per block cannot surpass 1024.
  // (b) Config::n_heads must not exceed 1024.
  // (c) Config::dim / Config::n_heads / 2 must not exceed 1024.
  // (d) Config::n_heads must not exceed (1 << 31) - 1. RoPE has n_heads blocks, which cannot surpass 2^31.
  // (e) Config::seq_len must not exceed (1 << 31) - 1. Attention softmax has seq_len blocks, which cannot surpass 2^31.
  // (f) Accum blocks must not exceed (1 << 31) - 1.
  // (g) Silu blocks must not exceed (1 << 31) - 1.
  // (h) CuRAND blocks must not exceed (1 << 31) - 1.
  // (i) RMS Norm blocks must not exceed (1 << 31) - 1.
  // (j) RMS Norm scratch pad must have enough space for calculations.
  // (k) Argmax scratch pad must have enough space.

  static_assert( 1024 >= TPB );                                                                   // (a)
  static_assert( 1024 >= Config::n_heads );                                                       // (b)
  static_assert( 1024 >= Config::dim / Config::n_heads / 2 );                                     // (c)
  static_assert( ( 1l << 31 ) - 1 >= Config::n_heads );                                           // (d)
  static_assert( ( 1l << 31 ) - 1 >= Config::seq_len );                                           // (e)
  CHECK_GE( ( 1l << 31 ) - 1, div_ceil( Config::dim * settings.concurrency_limit, TPB ) );        // (f)
  CHECK_GE( ( 1l << 31 ) - 1, div_ceil( Config::hidden_dim * settings.concurrency_limit, TPB ) ); // (g)
  static_assert( ( 1l << 31 ) - 1 >= div_ceil( Config::vocab_size, TPB ) );                       // (h)
  static_assert( ( 1l << 31 ) - 1 >= div_ceil( Config::dim, NRBS ) );                             // (i)

  static_assert(
    sizeof( DType ) * Config::dim
    >= sizeof( glinthawk::float32_t )
         * ( div_ceil( Config::dim, 2 * NRBS ) + div_ceil( div_ceil( Config::dim, 2 * NRBS ), 2 * NRBS ) + 1 ) ); // (j)

  static_assert( sizeof( DType ) * ( 4 * Config::dim + 2 * Config::hidden_dim )
                 >= ( sizeof( uint32_t )
                        * ( div_ceil( Config::vocab_size, 2 * AMRBS )
                            + div_ceil( div_ceil( Config::vocab_size, 2 * AMRBS ), 2 * AMRBS ) + 1 )
                      + sizeof( DType )
                          * ( div_ceil( Config::vocab_size, 2 * AMRBS )
                              + div_ceil( div_ceil( Config::vocab_size, 2 * AMRBS ), 2 * AMRBS ) ) ) ); // (k)
}

template<typename Config, typename DType, typename ContextType>
void LlamaOperations<Config, DType, ContextType>::attention_0_gemm(
  const DType* query,
  const typename ContextType::LayerContextType layer_contexts[],
  DType* att,
  const uint64_t batch_size,
  const uint32_t* token_positions ) const
{
  static_assert( ContextType::LayerContextType::is_contiguous(), "ContextType::LayerContextType must be contiguous" );

  const glinthawk::float32_t scale = 1.0f / sqrtf( Config::head_size );
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
    if ( layer_contexts[i].empty() ) {
      continue;
    }
    const uint64_t m = token_positions[i] + 1;
    common::cuda::CHECK_CUBLAS( cublasGemmStridedBatchedEx( this->cublas_handle_array[i],
                                                            CUBLAS_OP_T,
                                                            CUBLAS_OP_N,
                                                            m,
                                                            n,
                                                            k,
                                                            &scale,
                                                            layer_contexts[i].token( 0 ).key(),
                                                            common::cuda::get_cuda_data_type<DType>(),
                                                            lda,
                                                            strideA,
                                                            query + i * dim,
                                                            common::cuda::get_cuda_data_type<DType>(),
                                                            ldb,
                                                            strideB,
                                                            &this->beta,
                                                            att + i * att_dim,
                                                            common::cuda::get_cuda_data_type<DType>(),
                                                            ldc,
                                                            strideC,
                                                            gemm_batch_count,
                                                            common::cuda::Operations<DType>::GH_CUDA_COMPUTE_TYPE,
                                                            CUBLAS_GEMM_DEFAULT ) );
  }
}

template<typename Config, typename DType, typename ContextType>
void LlamaOperations<Config, DType, ContextType>::attention_2_gemm(
  const DType* att,
  const ContextType::LayerContextType layer_contexts[],
  DType* xb,
  const uint64_t batch_size,
  const uint32_t* token_positions ) const
{
  static_assert( ContextType::LayerContextType::is_contiguous(), "ContextType::LayerContextType must be contiguous" );

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
    if ( layer_contexts[i].empty() ) {
      continue;
    }
    const uint64_t k = token_positions[i] + 1;

    common::cuda::CHECK_CUBLAS( cublasGemmStridedBatchedEx( this->cublas_handle_array[i],
                                                            CUBLAS_OP_N,
                                                            CUBLAS_OP_N,
                                                            m,
                                                            n,
                                                            k,
                                                            &this->alpha,
                                                            layer_contexts[i].token( 0 ).value(),
                                                            common::cuda::get_cuda_data_type<DType>(),
                                                            lda,
                                                            strideA,
                                                            att + i * att_dim,
                                                            common::cuda::get_cuda_data_type<DType>(),
                                                            ldb,
                                                            strideB,
                                                            &this->beta,
                                                            xb + i * dim,
                                                            common::cuda::get_cuda_data_type<DType>(),
                                                            ldc,
                                                            strideC,
                                                            gemm_batch_count,
                                                            common::cuda::Operations<DType>::GH_CUDA_COMPUTE_TYPE,
                                                            CUBLAS_GEMM_DEFAULT ) );
  }
}

template<typename Config, typename DType, typename ContextType>
void LlamaOperations<Config, DType, ContextType>::attention_softmax( DType* att,
                                                                     const uint32_t* token_positions,
                                                                     DType* temp_buffer,
                                                                     const uint64_t batch_size ) const
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

template<typename Config, typename DType, typename ContextType>
void LlamaOperations<Config, DType, ContextType>::apply_rope(
  const uint64_t curr_batch_size,
  const uint32_t* token_positions,
  const DType* freq_cis_real,
  const DType* freq_cis_imag,
  DType* state_q,
  typename ContextType::TokenContextType token_contexts[] ) const
{
  for ( uint64_t i = 0; i < curr_batch_size; i++ ) {
    if ( token_contexts[i].empty() ) {
      continue;
    }
    do_rope<DType, Config::head_size, Config::gqa_size>
      <<<Config::n_kv_heads, Config::head_size / 2, 0, this->streams[i]>>>(
        freq_cis_real + token_positions[i] * Config::head_size / 2,
        freq_cis_imag + token_positions[i] * Config::head_size / 2,
        state_q + i * Config::n_kv_heads * Config::gqa_size * Config::head_size,
        token_contexts[i].key() );
  }
}

template<typename Config, typename DType, typename ContextType>
void LlamaOperations<Config, DType, ContextType>::copy_kv_cache(
  typename ContextType::TokenContextType token_contexts[],
  const DType* state_kv,
  const uint64_t batch_size ) const
{
  for ( size_t i = 0; i < batch_size; i++ ) {
    if ( token_contexts[i].empty() ) {
      continue;
    }

    DType* kv_cache_pos = token_contexts[i].key();

    common::cuda::CHECK_CUDA( cudaMemcpyAsync( kv_cache_pos,
                                               state_kv + i * Config::kv_dim * 2,
                                               Config::kv_dim * sizeof( DType ) * 2,
                                               cudaMemcpyDeviceToDevice ) );
  }
}

template<typename Config, typename DType, typename ContextType>
template<typename DTypeDst, typename DTypeSrc>
void LlamaOperations<Config, DType, ContextType>::convert_and_copy( DTypeDst* dst,
                                                                    const DTypeSrc* src,
                                                                    const uint64_t size,
                                                                    const CopyType type ) const
{
  switch ( type ) {
    case CopyType::DeviceToHost: {
      if constexpr ( std::is_same_v<DTypeSrc, DTypeDst> ) {
        common::cuda::CHECK_CUDA( cudaMemcpy( dst, src, size * sizeof( DTypeSrc ), cudaMemcpyDeviceToHost ) );
      } else {
        std::unique_ptr<DTypeSrc[]> src_host { new DTypeSrc[size] };
        common::cuda::CHECK_CUDA(
          cudaMemcpy( src_host.get(), src, size * sizeof( DTypeSrc ), cudaMemcpyDeviceToHost ) );
        for ( uint64_t i = 0; i < size; i++ ) {
          dst[i] = static_cast<DTypeDst>( src_host.get()[i] );
        }
      }
      break;
    }

    case CopyType::HostToDevice: {
      if constexpr ( std::is_same_v<DTypeSrc, DTypeDst> ) {
        common::cuda::CHECK_CUDA( cudaMemcpy( dst, src, size * sizeof( DTypeSrc ), cudaMemcpyHostToDevice ) );
      } else {
        std::unique_ptr<DTypeDst[]> dst_host { new DTypeDst[size] };
        for ( uint64_t i = 0; i < size; i++ ) {
          dst_host.get()[i] = static_cast<DTypeDst>( src[i] );
        }
        common::cuda::CHECK_CUDA(
          cudaMemcpy( dst, dst_host.get(), size * sizeof( DTypeSrc ), cudaMemcpyHostToDevice ) );
      }
    } break;

    default: LOG( FATAL ) << "Invalid copy type";
  }
}

} // namespace glinthawk::models::llama2::cuda
