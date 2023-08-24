#pragma once

#include <cstdint>
#include <source_location>

namespace glinthawk::models::common::cuda::ops {

constexpr size_t TPB = 64; /* threads per block */

void init();
void destroy();

void CHECK_CUDA( const cudaError_t err, const std::source_location location = std::source_location::current() );

template<typename DType>
void accum( DType* a, const DType* b, const uint64_t size, const uint64_t batch_size );

template<typename DType>
void rmsnorm( DType* o, const DType* x, const DType* weight, const uint64_t size, const uint64_t batch_size );

template<typename DType>
void softmax( DType* x, const uint64_t size, const uint64_t batch_size );

template<typename DType>
void softmax( DType* x, const uint64_t size );

template<typename DType>
void matmul( DType* xout, const DType* x, const DType* w, const uint64_t b, const uint64_t s, const uint64_t r);

template<typename DType>
std::vector<uint32_t> sample( const DType* probabilities, const uint64_t n, const uint64_t batch_size );

template<typename DType>
uint32_t sample( const DType* probabilities, const uint64_t n);

template<typename DType>
std::vector<uint32_t> argmax( const DType* v, const uint64_t n, const uint64_t batch_size );

template<typename DType>
uint32_t argmax( const DType* v, const uint64_t n );

template<typename DType>
void silu( DType* hb, DType* hb2, const uint64_t hidden_dim, const uint64_t batch_size );

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
                       const uint64_t max_batch_size );

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
                       const uint64_t max_batch_size );

template<typename DType>
void attention_softmax( DType* att,
                        const uint64_t token_pos,
                        const uint64_t seq_len,
                        const uint64_t n_heads,
                        DType* temp_buffer,
                        const uint64_t batch_size );

template<typename DType>
void apply_rope( const uint64_t head_size,
                 const uint64_t n_heads,
                 const uint64_t curr_batch_size,
                 const DType* freq_cis_real_row,
                 const DType* freq_cis_imag_row,
                 DType* state_q,
                 DType* state_k );

template<typename DType>
void fill_pointers_init( DType** q_p,
                         std::vector<DType*> query_p_cpu,
                         DType* query,
                         DType* att,
                         DType* xb,
                         const uint64_t seq_len,
                         const uint64_t head_size,
                         const uint64_t n_heads,
                         const uint64_t max_batch_size );

template<typename DType>
void fill_pointers_kv( DType** k_p,
                       std::vector<DType*> key_p_cpu,
                       DType* key,
                       DType* value,
                       const uint64_t* id_allocation,
                       const uint64_t head_size,
                       const uint64_t n_heads,
                       const uint64_t batch_size,
                       const uint64_t max_batch_size );

}
