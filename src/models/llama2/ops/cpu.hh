#pragma once

#include <random>
#include <type_traits>
#include <glog/logging.h>

#include "concept.hh"
#include "models/common/ops/cpu.hh"

namespace glinthawk::models::llama2::cpu {

template<typename DType>
class LlamaOperations : public common::cpu::Operations<DType>
{
public:
  using common::cpu::Operations<DType>::Operations;

  ~LlamaOperations() {}

  template<uint64_t seq_len, uint64_t head_size, uint64_t n_kv_heads, uint64_t gqa_size>
  void attention_0_gemm( const DType* query,
                         const DType* const context_pointers[],
                         DType* att,
                         const uint64_t batch_size,
                         const uint32_t* token_positions );

  template<uint64_t seq_len, uint64_t head_size, uint64_t n_kv_heads, uint64_t gqa_size, uint64_t rounds>
  void attention_2_gemm( const DType* att,
                         const DType* const context_pointers[],
                         DType* xb,
                         const uint64_t batch_size,
                         const uint32_t* token_positions );

  template<uint64_t seq_len, uint64_t n_heads>
  void attention_softmax( DType* att, const uint32_t* token_positions, DType* temp_buffer, const uint64_t batch_size );

  template<uint64_t head_size, uint64_t n_kv_heads, uint64_t gqa_size>
  void apply_rope( const uint64_t curr_batch_size,
                   const uint32_t* token_positions,
                   const DType* freq_cis_real,
                   const DType* freq_cis_imag,
                   DType* state_q,
                   DType* context_pointers[] );

  template<uint64_t dim>
  void copy_kv_cache( DType* context_pointers[],
                      const DType* state_k,
                      const DType* state_v,
                      const uint64_t batch_size,
                      const uint32_t* token_positions );

  template<typename DTypeDst, typename DTypeSrc>
  void convert_and_copy( DTypeDst* dst, const DTypeSrc* src, const uint64_t size, const CopyType );
};

static_assert( LlamaOperationsConcept<LlamaOperations<float>, float, float, _Float16> );
static_assert( LlamaOperationsConcept<LlamaOperations<_Float16>, _Float16, _Float16, float> );

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
  float sum = 0.0f;
  for ( uint64_t i = 0; i < size; i++ ) {
    x[i] = DType( expf( float( x[i] - max_val ) ) );
    sum += float( x[i] );
  }

  // normalize
  for ( uint64_t i = 0; i < size; i++ ) {
    x[i] = DType( float( x[i] ) / sum );
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

  const float fcr = freq_cis_real_row[elem_idx / 2];
  const float fci = freq_cis_imag_row[elem_idx / 2];

  const float k0 = k[elem_idx];
  const float k1 = k[elem_idx + 1];
  k[elem_idx] = DType( k0 * fcr - k1 * fci );
  k[elem_idx + 1] = DType( k0 * fci + k1 * fcr );

  for ( uint64_t i = 0; i < gqa_size; i++ ) {
    const float q0 = q[i * head_size + elem_idx];
    const float q1 = q[i * head_size + elem_idx + 1];
    q[i * head_size + elem_idx] = DType( q0 * fcr - q1 * fci );
    q[i * head_size + elem_idx + 1] = DType( q0 * fci + q1 * fcr );
  }
}

}

}

template<typename DType>
template<uint64_t seq_len, uint64_t head_size, uint64_t n_kv_heads, uint64_t gqa_size>
void LlamaOperations<DType>::attention_0_gemm( const DType* query,
                                               const DType* const context_pointers[],
                                               DType* att,
                                               const uint64_t batch_size,
                                               const uint32_t* token_positions )
{
  const float scale = 1.0f / sqrtf( head_size );

  constexpr uint64_t ld_key = n_kv_heads * head_size * 2;
  constexpr uint64_t ld_qry = head_size;
  constexpr uint64_t ld_att = seq_len;

  constexpr uint64_t stride_key = head_size;
  constexpr uint64_t stride_qry = head_size * gqa_size;
  constexpr uint64_t stride_att = seq_len * gqa_size;

  constexpr uint64_t dim_ = head_size * n_kv_heads * gqa_size;
  constexpr uint64_t att_dim_ = seq_len * n_kv_heads * gqa_size;

  uint64_t i;
  uint64_t kv_head;

#pragma omp parallel for private( i, kv_head ) shared( token_positions, context_pointers, att, query ) collapse( 2 )
  for ( i = 0; i < batch_size; i++ ) {
    for ( kv_head = 0; kv_head < n_kv_heads; kv_head++ ) {
      const DType* current_query = query + i * dim_ + kv_head * stride_qry;
      DType* current_att = att + i * att_dim_ + kv_head * stride_att;
      const DType* current_key = context_pointers[i] + kv_head * stride_key;

      for ( uint64_t key_pos = 0; key_pos < token_positions[i] + 1; key_pos++ ) {

        float sum_s[gqa_size] = { 0.0 };

        for ( uint64_t p = 0; p < head_size; ++p ) {
          const float a_value = current_key[p];

          for ( uint64_t query_gqa_head = 0; query_gqa_head < gqa_size; query_gqa_head++ ) {
            const float b_value = current_query[query_gqa_head * ld_qry + p];
            sum_s[query_gqa_head] += a_value * b_value;
          }
        }

        for ( uint64_t query_gqa_head = 0; query_gqa_head < gqa_size; query_gqa_head++ ) {
          current_att[query_gqa_head * ld_att] = DType( scale * sum_s[query_gqa_head] );
        }

        current_att += 1;
        current_key += ld_key;
      }
    }
  }
}

template<typename DType>
template<uint64_t seq_len, uint64_t head_size, uint64_t n_kv_heads, uint64_t gqa_size, uint64_t rounds>
void LlamaOperations<DType>::attention_2_gemm( const DType* att,
                                               const DType* const context_pointers[],
                                               DType* xb,
                                               const uint64_t batch_size,
                                               const uint32_t* token_positions )
{
  constexpr uint64_t ld_val = n_kv_heads * head_size * 2;
  constexpr uint64_t ld_att = seq_len;

  constexpr uint64_t stride_val = head_size;
  constexpr uint64_t stride_att = seq_len * gqa_size;
  constexpr uint64_t stride_xb = head_size * gqa_size;

  constexpr uint64_t kv_dim_ = head_size * n_kv_heads;
  constexpr uint64_t dim_ = head_size * n_kv_heads * gqa_size;
  constexpr uint64_t att_dim_ = seq_len * n_kv_heads * gqa_size;

  uint64_t i;
  uint64_t kv_head;

  CHECK_EQ( n_kv_heads % rounds, 0 ) << "Remainders are bad";

#pragma omp parallel for private( i, kv_head ) shared( xb, token_positions, context_pointers, att ) collapse( 2 )
  for ( i = 0; i < batch_size; i++ ) {
    for ( kv_head = 0; kv_head < n_kv_heads; kv_head += rounds ) {

      float sum_s[rounds * gqa_size * head_size];
      std::memset( sum_s, 0, sizeof( float ) * rounds * gqa_size * head_size );
      const DType* current_att = att + i * att_dim_ + kv_head * stride_att;
      const DType* current_value = context_pointers[i] + kv_dim_ + kv_head * stride_val;

      for ( uint64_t p = 0; p < token_positions[i] + 1; ++p ) {
        for ( uint64_t round_index = 0; round_index < rounds; round_index++ ) {
          for ( uint64_t att_gqa_head = 0; att_gqa_head < gqa_size; att_gqa_head++ ) {
            const float b_value = current_att[round_index * stride_att + att_gqa_head * ld_att + p];

            for ( uint64_t val_pos = 0; val_pos < head_size; val_pos++ ) {
              const float a_value = current_value[round_index * stride_val + val_pos];
              sum_s[round_index * stride_xb + att_gqa_head * head_size + val_pos] += a_value * b_value;
            }
          }
        }
        current_value += ld_val;
      }
      DType* current_xb = xb + i * dim_ + kv_head * stride_xb;
      for ( uint64_t val_pos = 0; val_pos < rounds * gqa_size * head_size; val_pos++ ) {
        current_xb[val_pos] = DType( sum_s[val_pos] );
      }
    }
  }
}

template<typename DType>
template<uint64_t seq_len, uint64_t n_heads>
void LlamaOperations<DType>::attention_softmax( DType* att,
                                                const uint32_t* token_positions,
                                                DType*, /* CPU doesn't use the temp buffer */
                                                const uint64_t batch_size )
{
  uint64_t i;
  uint64_t j;
#pragma omp parallel for private( i, j ) collapse( 2 )
  for ( i = 0; i < batch_size; i++ ) {
    for ( j = 0; j < n_heads; j++ ) {
      DType* this_att = att + i * n_heads * seq_len + j * seq_len;
      softmax( this_att, token_positions[i] + 1 );
    }
  }
}

template<typename DType>
template<uint64_t head_size, uint64_t n_kv_heads, uint64_t gqa_size>
void LlamaOperations<DType>::apply_rope( const uint64_t batch_size,
                                         const uint32_t* token_positions,
                                         const DType* freq_cis_real,
                                         const DType* freq_cis_imag,
                                         DType* state_q,
                                         DType* context_pointers[] )
{
  uint64_t i;
  uint64_t j;
#pragma omp parallel for private( i, j ) collapse( 2 )
  for ( i = 0; i < batch_size; i++ ) {
    for ( j = 0; j < n_kv_heads; j++ ) {
      for ( uint64_t k = 0; k < head_size / 2; k++ ) {
        const uint64_t head_q_num = gqa_size * j;
        const uint64_t head_k_num = j;
        const uint64_t elem_idx = 2 * k;

        do_rope<DType, head_size, gqa_size>( freq_cis_real + token_positions[i] * head_size / 2,
                                             freq_cis_imag + token_positions[i] * head_size / 2,
                                             state_q + i * n_kv_heads * gqa_size * head_size,
                                             context_pointers[i] + token_positions[i] * n_kv_heads * head_size * 2,
                                             head_q_num,
                                             head_k_num,
                                             elem_idx );
      }
    }
  }
}

template<typename DType>
template<uint64_t dim>
void LlamaOperations<DType>::copy_kv_cache( DType* context_pointers[],
                                            const DType* state_k,
                                            const DType* state_v,
                                            const uint64_t batch_size,
                                            const uint32_t* token_positions )
{
  uint64_t i;
#pragma omp parallel for private( i )
  for ( i = 0; i < batch_size; i++ ) {
    if ( context_pointers[i] == nullptr ) {
      continue;
    }

    DType* k_cache_pos = context_pointers[i] + token_positions[i] * dim * 2;
    DType* v_cache_pos = k_cache_pos + dim;

    memcpy( k_cache_pos, state_k + i * dim, dim * sizeof( DType ) );
    memcpy( v_cache_pos, state_v + i * dim, dim * sizeof( DType ) );
  }
}

template<typename DType>
template<typename DTypeDst, typename DTypeSrc>
void LlamaOperations<DType>::convert_and_copy( DTypeDst* dst, const DTypeSrc* src, const uint64_t size, const CopyType )
{
  if constexpr ( std::is_same_v<DTypeSrc, DTypeDst> ) {
    memcpy( dst, src, sizeof( DTypeSrc ) * size );
  } else {
    for ( uint64_t i = 0; i < size; i++ ) {
      dst[i] = static_cast<DTypeDst>( src[i] );
    }
  }
}

} // namespace glinthawk::models::llama2::cpu
