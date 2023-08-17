#pragma once

namespace glinthawk::models::common::cuda::ops {

template<typename DType>
void accum( DType* a, const DType* b, const int size );

template<typename DType>
void rmsnorm( DType* o, const DType* x, const DType* weight, const int size );

template<typename DType>
void softmax( DType* x, const int size );

template<typename DType>
void matmul( DType* xout, const DType* x, const DType* w, const int n, const int d );

template<typename DType>
int sample( const DType* probabilities, const int n );

template<typename DType>
int argmax( const DType* v, const int n );

template<typename DType>
void silu( DType* hb, DType* hb2, const int hidden_dim );

}
