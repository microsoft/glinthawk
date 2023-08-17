#pragma once

namespace glinthawk::models::common::cuda::ops {

void init();

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

}
