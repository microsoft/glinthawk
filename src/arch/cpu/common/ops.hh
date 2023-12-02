#pragma once

#include <memory>
#include <random>

#include "models/common/ops/concept.hh"

namespace glinthawk::models::common::cpu {

template<typename DType>
class Operations
{
public:
  using DeviceUniquePtr = std::unique_ptr<DType>;
  using Float16 = _Float16;
  using Float32 = float;

public:
  Operations() {}
  ~Operations() {}

  Operations( const Operations& ) = delete;
  Operations& operator=( const Operations& ) = delete;
  Operations( Operations&& ) = default;
  Operations& operator=( Operations&& ) = default;

  template<uint64_t size>
  void accum( DType* a, const DType* b, const uint64_t batch_size );

  template<uint64_t size>
  void rmsnorm( DType* o, const DType* x, DType* temp, const DType* weight, const uint64_t batch_size );

  template<uint64_t n>
  void argmax( uint32_t* output, const DType* v, DType* temp, const uint64_t batch_size );

  template<uint64_t hidden_dim>
  void silu( DType* hb, DType* hb2, const uint64_t batch_size );

  template<uint64_t s, uint64_t r>
  void matmul( DType* xo, const DType* x, const DType* w, const uint64_t b );

  template<uint64_t vocab_size>
  void soft_sample( DType* v, const std::vector<float>& temp_s, const uint64_t batch_size );

  DeviceUniquePtr device_allocate( const uint64_t size_bytes );

  void copy( DType* dst, const DType* src, const uint64_t len_bytes, const CopyType type, const bool async = false );
};

static_assert( OperationsConcept<Operations<float>, float> );
static_assert( OperationsConcept<Operations<_Float16>, _Float16> );

// helper functions are in this anonymous namespace
namespace {

namespace { // matmul

template<typename DType, uint64_t m, uint64_t k, uint64_t lda, uint64_t ldb, uint64_t ldc>
void fast_matmul_row_major( uint64_t n, const DType* A, const DType* B, DType* C )
{
  uint64_t row;
  uint64_t col;
#pragma omp parallel for private( row, col ) shared( A, B, C ) collapse( 2 )
  for ( row = 0; row < m; row++ ) {
    for ( col = 0; col < n; col++ ) {
      float sum = 0.0;

      for ( uint64_t p = 0; p < k; ++p ) {
        const float a_value = A[row * lda + p];
        const float b_value = B[col * ldb + p];
        sum += a_value * b_value;
      }

      C[col * ldc + row] = DType( sum );
    }
  }
}

}

}

template<typename DType>
template<uint64_t size>
void Operations<DType>::accum( DType* a, const DType* b, const uint64_t batch_size )
{
  uint64_t b_idx;
  uint64_t i;
#pragma omp parallel for private( b_idx, i ) collapse( 2 )
  for ( b_idx = 0; b_idx < batch_size; b_idx++ ) {
    for ( i = 0; i < size; i++ ) {
      a[b_idx * size + i] = DType( float( a[b_idx * size + i] ) + float( b[b_idx * size + i] ) );
    }
  }
}

template<typename DType>
template<uint64_t size>
void Operations<DType>::rmsnorm( DType* output, const DType* x, DType*, const DType* weight, const uint64_t batch_size )
{
  uint64_t b;
#pragma omp parallel for private( b )
  for ( b = 0; b < batch_size; b++ ) {
    const DType* X = x + b * size;
    DType* O = output + b * size;

    // calculate sum of squares
    float ss = 0.0f;
    for ( uint64_t j = 0; j < size; j++ ) {
      ss += float( X[j] ) * float( X[j] );
    }

    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf( ss );

    // normalize and scale
    for ( uint64_t j = 0; j < size; j++ ) {
      O[j] = DType( float( weight[j] ) * ( ss * float( X[j] ) ) );
    }
  }
}

template<typename DType>
template<uint64_t n>
void Operations<DType>::argmax( uint32_t* output, const DType* v, DType*, const uint64_t batch_size )
{
  uint64_t b;
#pragma omp parallel for private( b )
  for ( b = 0; b < batch_size; b++ ) {
    const DType* this_v = v + b * n;

    uint64_t max_i = 0;
    float max_p = this_v[0];

    for ( uint64_t i = 1; i < n; i++ ) {
      if ( this_v[i] > max_p ) {
        max_i = i;
        max_p = this_v[i];
      }
    }

    output[b] = max_i;
  }
}

template<typename DType>
template<uint64_t hidden_dim>
void Operations<DType>::silu( DType* hb, DType* hb2, const uint64_t batch_size )
{
  uint64_t b;
#pragma omp parallel for private( b )
  for ( b = 0; b < batch_size; b++ ) {
    DType* current_hb = hb + b * hidden_dim;
    DType* current_hb2 = hb2 + b * hidden_dim;

    for ( size_t i = 0; i < hidden_dim; i++ ) {
      const float x = current_hb[i];
      current_hb[i] = DType( x * ( 1.0f / ( 1.0f + expf( -x ) ) ) * float( current_hb2[i] ) );
    }
  }
}

template<typename DType>
template<uint64_t s, uint64_t r>
void Operations<DType>::matmul( DType* xout, const DType* x, const DType* w, const uint64_t b )
{
  // x(b,s) @ W(s,r) -> xout(b,r)
  // OR
  // W(r,s) @ x(s,b) -> xout(r,b)
  // A(m,k) @ B(k,n) ->    C(m,n)

  constexpr uint64_t m = r;
  constexpr uint64_t k = s;
  const uint64_t n = b;
  constexpr uint64_t lda = k;
  constexpr uint64_t ldb = k;
  constexpr uint64_t ldc = m;

  fast_matmul_row_major<DType, m, k, lda, ldb, ldc>( n, w, x, xout );
}

template<typename DType>
Operations<DType>::DeviceUniquePtr Operations<DType>::device_allocate( const uint64_t size )
{
  return DeviceUniquePtr { reinterpret_cast<DType*>( new uint8_t[size] ) };
}

template<typename DType>
void Operations<DType>::copy( DType* dst, const DType* src, const uint64_t len_bytes, const CopyType, const bool )
{
  std::memcpy( dst, src, len_bytes );
}

} // namespace glinthawk::models::common::cpu
