#pragma once

namespace glinthawk::models::common::cuda::ops {

void init();
void destroy();

template<typename DType>
void accum( DType* a, const DType* b, const int size );

template<typename DType>
void rmsnorm( DType* o, const DType* x, const DType* weight, const int size );

template<typename DType>
void softmax( DType* x, const int size );

template<typename DType>
void matmul( DType* xout, const DType* x, const DType* w, const int n, const int d );

template<typename DType>
void sample( const DType* probabilities, const int n, int* output );

template<typename DType>
void argmax( const DType* v, const int n, int* output );

template<typename DType>
void silu( DType* hb, DType* hb2, const int hidden_dim );

template<typename DType>
void attention_0_gemm(const DType* q,
                            const DType* k,
                            DType* att,
                            const int n_layers,
                            const int seq_len,
                            const int head_size,
                            const int n_heads,
                            const int n_tokens);
template<typename DType>
void attention_2_gemm(const DType* q,
                       const DType* k,
                       DType* att,
                       const int n_layers,
                       const int seq_len,
                       const int head_size,
                       const int n_heads,
                       const int n_tokens);


}
