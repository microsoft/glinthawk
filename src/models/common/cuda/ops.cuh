#pragma once

#include <cstdint>
#include <curand.h>
#include <curand_kernel.h>
#include <source_location>

namespace glinthawk::models::common::cuda::ops {

constexpr size_t TPB = 64; /* threads per block */

void init( const int num_streams );
void destroy();

void CHECK_CUDA( const cudaError_t err, const std::source_location location = std::source_location::current() );

__global__ void setup_kernel( curandState* state, unsigned long seed );

template<typename DType>
void accum( DType* a, const DType* b, const uint64_t size, const uint64_t batch_size );

template<typename DType>
void rmsnorm( DType* o, const DType* x, const DType* weight, const uint64_t size, const uint64_t batch_size );

template<typename DType>
void softmax( DType* x, const uint64_t size, const uint64_t batch_size );

template<typename DType>
void softmax( DType* x, const uint64_t size );

template<typename DType>
void matmul( DType* xout, const DType* x, const DType* w, const uint64_t b, const uint64_t s, const uint64_t r );

template<typename DType>
void soft_sample( DType* v,
                  const std::vector<float>& temp_s,
                  curandState* rng_state,
                  const uint64_t vocab_size,
                  const uint64_t batch_size );

template<typename DType>
std::vector<uint32_t> argmax( const DType* v, const uint64_t n, const uint64_t batch_size );

template<typename DType>
uint32_t argmax( const DType* v, const uint64_t n );

template<typename DType>
void silu( DType* hb, DType* hb2, const uint64_t hidden_dim, const uint64_t batch_size );

template<typename DType>
void attention_0_gemm( const DType* query,
                       const DType* key_base,
                       DType* att,
                       const uint64_t n_layers,
                       const uint64_t seq_len,
                       const uint64_t head_size,
                       const uint64_t n_kv_heads,
                       const uint64_t gqa_size,
                       const uint64_t batch_size,
                       const uint64_t max_batch_size,
                       const std::vector<uint64_t>& id_alloc_s,
                       const std::vector<uint64_t>& token_pos_s );

template<typename DType>
void attention_2_gemm( const DType* att,
                       const DType* value_base,
                       DType* xb,
                       const uint64_t n_layers,
                       const uint64_t seq_len,
                       const uint64_t head_size,
                       const uint64_t n_kv_heads,
                       const uint64_t gqa_size,
                       const uint64_t batch_size,
                       const uint64_t max_batch_size,
                       const std::vector<uint64_t>& id_alloc_s,
                       const std::vector<uint64_t>& token_pos_s );

template<typename DType>
void attention_softmax( DType* att,
                        const std::vector<uint64_t>& token_pos_s,
                        const uint64_t seq_len,
                        const uint64_t n_heads,
                        DType* temp_buffer,
                        const uint64_t batch_size );

template<typename DType>
void apply_rope( const uint64_t head_size,
                 const uint64_t n_kv_heads,
                 const uint64_t gqa_size,
                 const uint64_t curr_batch_size,
                 const std::vector<uint64_t>& token_pos_s,
                 const DType* freq_cis_real,
                 const DType* freq_cis_imag,
                 DType* state_q,
                 DType* state_k );

template<typename DType>
void copy_kv_cache( const DType* state_k,
                    const DType* state_v,
                    DType* key_base,
                    DType* value_base,
                    const uint64_t dim,
                    const uint64_t n_layers,
                    const uint64_t batch_size,
                    const uint64_t max_batch_size,
                    const std::vector<uint64_t>& id_alloc_s,
                    const std::vector<uint64_t>& token_pos_s );

}
