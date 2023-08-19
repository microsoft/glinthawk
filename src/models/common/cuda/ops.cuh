#pragma once

#include <cstdint>

namespace glinthawk::models::common::cuda::ops {

void init();
void destroy();

template<typename DType>
void accum( DType* a, const DType* b, const uint64_t size );

template<typename DType>
void rmsnorm( DType* o, const DType* x, const DType* weight, const uint64_t size );

template<typename DType>
void softmax( DType* x, const uint64_t size );

template<typename DType>
void matmul( DType* xout, const DType* x, const DType* w, const uint64_t n, const uint64_t d );

template<typename DType>
uint32_t sample( const DType* probabilities, const uint64_t n );

template<typename DType>
uint32_t argmax( const DType* v, const uint64_t n );

template<typename DType>
void silu( DType* hb, DType* hb2, const uint64_t hidden_dim );

template<typename DType>
void attention_0_gemm( const DType* q,
                       const DType* k,
                       DType* att,
                       const uint64_t n_layers,
                       const uint64_t seq_len,
                       const uint64_t head_size,
                       const uint64_t n_heads,
                       const uint64_t n_tokens );

template<typename DType>
void attention_2_gemm( const DType* q,
                       const DType* k,
                       DType* att,
                       const uint64_t n_layers,
                       const uint64_t seq_len,
                       const uint64_t head_size,
                       const uint64_t n_heads,
                       const uint64_t n_tokens );

}
