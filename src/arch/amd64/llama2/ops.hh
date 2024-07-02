#pragma once

#include <glog/logging.h>
#include <random>
#include <type_traits>

#include "../common/ops.hh"
#include "arch/float.hh"
#include "models/llama2/base.hh"
#include "models/llama2/ops/concept.hh"
#include "models/llama2/variants.hh"

namespace glinthawk::models::llama2::amd64 {

template<typename Config, typename DType>
class Context;

template<typename Config, typename DType>
using DynamicContext = Context<Config, DType>;

template<typename Config, typename DType, typename Ctx = Context<Config, DType>>
requires ModelConfig<Config>
class LlamaOperations : public common::amd64::Operations<DType>
{
public:
  using common::amd64::Operations<DType>::DeviceUniquePtr;
  using ContextType = Ctx;

public:
  LlamaOperations( const ConfigRuntime<Config>& ) {}
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

  void soft_sample( DType* v, const std::vector<glinthawk::float32_t>& temp_s, const uint64_t batch_size ) const;

  void copy_kv_cache( typename ContextType::TokenContextType token_contexts[],
                      const DType* state_kv,
                      const uint64_t batch_size ) const;

  template<typename DTypeDst, typename DTypeSrc>
  void convert_and_copy( DTypeDst* dst, const DTypeSrc* src, const uint64_t size, const CopyType ) const;
};

// platform-specific context
template<typename Config, typename DType>
class Context : public llama2::Context<Config, DType>
{
private:
  typename common::amd64::Operations<DType>::DeviceUniquePtr storage_ {};

public:
  using llama2::Context<Config, DType>::Context;
  Context( const ConfigRuntime<Config>& settings, const bool make_empty = false );
};

static_assert( LlamaOperationsConcept<LlamaOperations<configs::Stories_110M, glinthawk::float32_t>,
                                      glinthawk::float32_t,
                                      ConfigRuntime<configs::Stories_110M>> );

static_assert( LlamaOperationsConcept<LlamaOperations<configs::Stories_110M, glinthawk::float16_t>,
                                      glinthawk::float16_t,
                                      ConfigRuntime<configs::Stories_110M>> );

template<typename Config, typename DType>
Context<Config, DType>::Context( const ConfigRuntime<Config>& settings, const bool make_empty )
  : llama2::Context<Config, DType>( settings )
  , storage_( [&]() -> decltype( storage_ ) {
    DType* ptr;
    if ( make_empty ) {
      ptr = nullptr;
    } else {
      ptr = reinterpret_cast<DType*>( new ( std::nothrow )
                                        uint8_t[Context<Config, DType>::max_size( settings.n_layers_loaded() )] );
    }
    return decltype( storage_ ) { ptr };
  }() )
{
  this->buffer_ = storage_.get();
}

// helper functions are in this anonymous namespace`
namespace {

namespace { // attetion_softmax

template<typename DType>
void softmax( DType* x, const uint64_t size )
{
  // find max value (for numerical stability)
  DType max_val = x[0];
  for ( uint64_t i = 1; i < size; i++ ) {
    if ( x[i] > max_val ) {
      max_val = x[i];
    }
  }

  // exp and sum
  glinthawk::float32_t sum = 0.0f;
  for ( uint64_t i = 0; i < size; i++ ) {
    x[i] = static_cast<DType>( expf( static_cast<glinthawk::float32_t>( x[i] - max_val ) ) );
    sum += static_cast<glinthawk::float32_t>( x[i] );
  }

  // normalize
  for ( uint64_t i = 0; i < size; i++ ) {
    x[i] = static_cast<DType>( static_cast<glinthawk::float32_t>( x[i] ) / sum );
  }
}

}

namespace { // rope

template<typename DType, uint64_t head_size, uint64_t gqa_size>
inline void do_rope( const DType* freq_cis_real_row,
                     const DType* freq_cis_imag_row,
                     DType* state_q,
                     DType* state_k,
                     const uint64_t head_q_num,
                     const uint64_t head_k_num,
                     const uint64_t elem_idx )
{
  DType* q = state_q + head_q_num * head_size;
  DType* k = state_k + head_k_num * head_size;

  const glinthawk::float32_t fcr = freq_cis_real_row[elem_idx / 2];
  const glinthawk::float32_t fci = freq_cis_imag_row[elem_idx / 2];

  const glinthawk::float32_t k0 = k[elem_idx];
  const glinthawk::float32_t k1 = k[elem_idx + 1];
  k[elem_idx] = static_cast<DType>( k0 * fcr - k1 * fci );
  k[elem_idx + 1] = static_cast<DType>( k0 * fci + k1 * fcr );

  for ( uint64_t i = 0; i < gqa_size; i++ ) {
    const glinthawk::float32_t q0 = q[i * head_size + elem_idx];
    const glinthawk::float32_t q1 = q[i * head_size + elem_idx + 1];
    q[i * head_size + elem_idx] = static_cast<DType>( q0 * fcr - q1 * fci );
    q[i * head_size + elem_idx + 1] = static_cast<DType>( q0 * fci + q1 * fcr );
  }
}

}

namespace { // soft_sample

template<typename DType, uint64_t vocab_size>
void gumbel_fix( DType* array, glinthawk::float32_t temp )
{
  for ( uint64_t i = 0; i < vocab_size; i++ ) {
    glinthawk::float32_t myrandf = static_cast<glinthawk::float32_t>( rand() ) / RAND_MAX;
    myrandf = logf( -logf( myrandf ) );
    array[i] = static_cast<DType>( static_cast<glinthawk::float32_t>( array[i] ) / temp - myrandf );
  }
}

}

}

template<typename Config, typename DType, typename ContextType>
void LlamaOperations<Config, DType, ContextType>::attention_0_gemm(
  const DType* query,
  const typename ContextType::LayerContextType layer_contexts[],
  DType* att,
  const uint64_t batch_size,
  const uint32_t* token_positions ) const
{
  const glinthawk::float32_t scale = 1.0f / sqrtf( Config::head_size );

  constexpr uint64_t ld_qry = Config::head_size;
  constexpr uint64_t ld_att = Config::seq_len;

  constexpr uint64_t stride_qry = Config::head_size * Config::gqa_size;
  constexpr uint64_t stride_att = Config::seq_len * Config::gqa_size;

  constexpr uint64_t dim_ = Config::head_size * Config::n_kv_heads * Config::gqa_size;
  constexpr uint64_t att_dim_ = Config::seq_len * Config::n_kv_heads * Config::gqa_size;

  uint64_t i;
  uint64_t kv_head;

#pragma omp parallel for private( i, kv_head ) shared( token_positions, layer_contexts, att, query ) collapse( 2 )
  for ( i = 0; i < batch_size; i++ ) {
    for ( kv_head = 0; kv_head < Config::n_kv_heads; kv_head++ ) {
      const DType* current_query = query + i * dim_ + kv_head * stride_qry;
      DType* current_att = att + i * att_dim_ + kv_head * stride_att;

      for ( uint64_t key_pos = 0; key_pos < token_positions[i] + 1; key_pos++ ) {
        const DType* current_key = layer_contexts[i].token( key_pos ).key_head( kv_head );

        glinthawk::float32_t sum_s[Config::gqa_size] = { 0.0 };

        for ( uint64_t p = 0; p < Config::head_size; ++p ) {
          const glinthawk::float32_t a_value = current_key[p];

          for ( uint64_t query_gqa_head = 0; query_gqa_head < Config::gqa_size; query_gqa_head++ ) {
            const glinthawk::float32_t b_value = current_query[query_gqa_head * ld_qry + p];
            sum_s[query_gqa_head] += a_value * b_value;
          }
        }

        for ( uint64_t query_gqa_head = 0; query_gqa_head < Config::gqa_size; query_gqa_head++ ) {
          current_att[query_gqa_head * ld_att] = DType( scale * sum_s[query_gqa_head] );
        }

        current_att += 1;
      }
    }
  }
}

template<typename Config, typename DType, typename ContextType>
void LlamaOperations<Config, DType, ContextType>::attention_2_gemm(
  const DType* att,
  const typename ContextType::LayerContextType layer_contexts[],
  DType* xb,
  const uint64_t batch_size,
  const uint32_t* token_positions ) const
{
  constexpr uint64_t ld_att = Config::seq_len;

  constexpr uint64_t stride_val = Config::head_size;
  constexpr uint64_t stride_att = Config::seq_len * Config::gqa_size;
  constexpr uint64_t stride_xb = Config::head_size * Config::gqa_size;

  constexpr uint64_t dim_ = Config::head_size * Config::n_kv_heads * Config::gqa_size;
  constexpr uint64_t att_dim_ = Config::seq_len * Config::n_kv_heads * Config::gqa_size;

  uint64_t i;
  uint64_t kv_head;

  static_assert( Config::n_kv_heads % Config::attention_rounds == 0, "Remainders are bad" );

#pragma omp parallel for private( i, kv_head ) shared( xb, token_positions, layer_contexts, att ) collapse( 2 )
  for ( i = 0; i < batch_size; i++ ) {
    for ( kv_head = 0; kv_head < Config::n_kv_heads; kv_head += Config::attention_rounds ) {

      glinthawk::float32_t sum_s[Config::attention_rounds * Config::gqa_size * Config::head_size];
      std::memset(
        sum_s, 0, sizeof( glinthawk::float32_t ) * Config::attention_rounds * Config::gqa_size * Config::head_size );
      const DType* current_att = att + i * att_dim_ + kv_head * stride_att;

      for ( uint64_t p = 0; p < token_positions[i] + 1; ++p ) {
        const DType* current_value = layer_contexts[i].token( p ).value_head( kv_head );

        for ( uint64_t round_index = 0; round_index < Config::attention_rounds; round_index++ ) {
          for ( uint64_t att_gqa_head = 0; att_gqa_head < Config::gqa_size; att_gqa_head++ ) {
            const glinthawk::float32_t b_value = current_att[round_index * stride_att + att_gqa_head * ld_att + p];

            for ( uint64_t val_pos = 0; val_pos < Config::head_size; val_pos++ ) {
              const glinthawk::float32_t a_value = current_value[round_index * stride_val + val_pos];
              sum_s[round_index * stride_xb + att_gqa_head * Config::head_size + val_pos] += a_value * b_value;
            }
          }
        }
      }

      DType* current_xb = xb + i * dim_ + kv_head * stride_xb;
      for ( uint64_t val_pos = 0; val_pos < Config::attention_rounds * Config::gqa_size * Config::head_size;
            val_pos++ ) {
        current_xb[val_pos] = DType( sum_s[val_pos] );
      }
    }
  }
}

template<typename Config, typename DType, typename ContextType>
void LlamaOperations<Config, DType, ContextType>::attention_softmax( DType* att,
                                                                     const uint32_t* token_positions,
                                                                     DType*,
                                                                     const uint64_t batch_size ) const
{
  uint64_t i;
  uint64_t j;
#pragma omp parallel for private( i, j ) collapse( 2 )
  for ( i = 0; i < batch_size; i++ ) {
    for ( j = 0; j < Config::n_heads; j++ ) {
      DType* this_att = att + i * Config::n_heads * Config::seq_len + j * Config::seq_len;
      softmax( this_att, token_positions[i] + 1 );
    }
  }
}

template<typename Config, typename DType, typename ContextType>
void LlamaOperations<Config, DType, ContextType>::apply_rope(
  const uint64_t batch_size,
  const uint32_t* token_positions,
  const DType* freq_cis_real,
  const DType* freq_cis_imag,
  DType* state_q,
  typename ContextType::TokenContextType token_contexts[] ) const
{
  uint64_t i;
  uint64_t j;
#pragma omp parallel for private( i, j ) collapse( 2 )
  for ( i = 0; i < batch_size; i++ ) {
    for ( j = 0; j < Config::n_kv_heads; j++ ) {
      for ( uint64_t k = 0; k < Config::head_size / 2; k++ ) {
        const uint64_t head_q_num = Config::gqa_size * j;
        const uint64_t head_k_num = j;
        const uint64_t elem_idx = 2 * k;

        do_rope<DType, Config::head_size, Config::gqa_size>(
          freq_cis_real + token_positions[i] * Config::head_size / 2,
          freq_cis_imag + token_positions[i] * Config::head_size / 2,
          state_q + i * Config::n_kv_heads * Config::gqa_size * Config::head_size,
          token_contexts[i].key(),
          head_q_num,
          head_k_num,
          elem_idx );
      }
    }
  }
}

template<typename Config, typename DType, typename ContextType>
void LlamaOperations<Config, DType, ContextType>::soft_sample( DType* v,
                                                               const std::vector<float>& temp_s,
                                                               const uint64_t batch_size ) const
{
  uint64_t i;
#pragma omp parallel for private( i )
  for ( i = 0; i < batch_size; i++ ) {
    if ( temp_s[i] > 0 ) {
      gumbel_fix<DType, Config::vocab_size>( v + i * Config::vocab_size, temp_s[i] );
    }
  }
}

template<typename Config, typename DType, typename ContextType>
void LlamaOperations<Config, DType, ContextType>::copy_kv_cache(
  typename ContextType::TokenContextType context_pointers[],
  const DType* state_kv,
  const uint64_t batch_size ) const
{
  uint64_t i;
#pragma omp parallel for private( i )
  for ( i = 0; i < batch_size; i++ ) {
    if ( context_pointers[i].empty() ) {
      continue;
    }

    DType* kv_cache_pos = context_pointers[i].key();

    memcpy( kv_cache_pos, state_kv + i * Config::kv_dim * 2, Config::kv_dim * sizeof( DType ) * 2 );
  }
}

template<typename Config, typename DType, typename ContextType>
template<typename DTypeDst, typename DTypeSrc>
void LlamaOperations<Config, DType, ContextType>::convert_and_copy( DTypeDst* dst,
                                                                    const DTypeSrc* src,
                                                                    const uint64_t size,
                                                                    const CopyType ) const
{
  if constexpr ( std::is_same_v<DTypeSrc, DTypeDst> ) {
    memcpy( dst, src, sizeof( DTypeSrc ) * size );
  } else {
    for ( uint64_t i = 0; i < size; i++ ) {
      dst[i] = static_cast<DTypeDst>( src[i] );
    }
  }
}

} // namespace glinthawk::models::llama2::amd64
