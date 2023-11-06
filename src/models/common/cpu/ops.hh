#pragma once

#include <concepts>
#include <cstdint>
#include <source_location>
#include <vector>

namespace glinthawk::models::common::cpu::ops {

template<std::unsigned_integral T>
T div_ceil( const T x, const T y )
{
  return x / y + ( x % y != 0 );
}

template<typename DType>
void accum( DType* a, const DType* b, const uint64_t size, const uint64_t batch_size );

template<typename DType>
void rmsnorm( DType* o, const DType* x, const DType* weight, const uint64_t size, const uint64_t batch_size );

template<typename DType>
void matmul( DType* xout, const DType* x, const DType* w, const uint64_t b, const uint64_t s, const uint64_t r );

template<typename DType>
void soft_sample( DType* v, const std::vector<float>& temp_s, const uint64_t vocab_size, const uint64_t batch_size );

template<typename DType>
void argmax( uint32_t* output, const DType* v, const uint64_t n, const uint64_t batch_size );

template<typename DType>
void silu( DType* hb, DType* hb2, const uint64_t hidden_dim, const uint64_t batch_size );

template<typename DType>
void attention_0_gemm( const DType* query,
                       const DType* const context_pointers[],
                       DType* att,
                       const uint64_t seq_len,
                       const uint64_t head_size,
                       const uint64_t n_kv_heads,
                       const uint64_t gqa_size,
                       const uint64_t batch_size,
                       const uint32_t* token_positions );

template<typename DType>
void attention_2_gemm( const DType* att,
                       const DType* const context_pointers[],
                       DType* xb,
                       const uint64_t seq_len,
                       const uint64_t head_size,
                       const uint64_t n_kv_heads,
                       const uint64_t gqa_size,
                       const uint64_t batch_size,
                       const uint32_t* token_positions );

template<typename DType>
void attention_0_gemm_fast( const DType* query,
                            const DType* const context_pointers[],
                            DType* att,
                            const uint64_t batch_size,
                            const uint32_t* token_positions );

template<typename DType>
void attention_2_gemm_fast( const DType* att,
                            const DType* const context_pointers[],
                            DType* xb,
                            const uint64_t batch_size,
                            const uint32_t* token_positions );

template<typename DType>
void attention_softmax( DType* att, const uint32_t* token_positions, const uint64_t batch_size );

template<typename DType>
void apply_rope( const uint64_t head_size,
                 const uint64_t n_kv_heads,
                 const uint64_t gqa_size,
                 const uint64_t curr_batch_size,
                 const uint32_t* token_positions,
                 const DType* freq_cis_real,
                 const DType* freq_cis_imag,
                 DType* state_q,
                 DType* state_k );

template<typename DType>
void copy_kv_cache( DType* context_pointers[],
                    const DType* state_k,
                    const DType* state_v,
                    const uint64_t dim,
                    const uint64_t batch_size,
                    const uint32_t* token_positions );

} // namespace glinthawk::models::common::cpu::ops
