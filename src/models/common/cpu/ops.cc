#include "ops.hh"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace std;

namespace glinthawk::models::common::cpu::ops {

/// @brief Accumulate the values in b into a (a += b)
template<typename DType>
void accum( DType* a, const DType* b, const uint64_t size, const uint64_t batch_size )
{
  size_t b_idx;
#pragma omp parallel for private( b_idx )
  for ( b_idx = 0; b_idx < batch_size; b_idx++ ) {
    for ( uint64_t i = 0; i < size; i++ ) {
      a[b_idx * size + i] += b[b_idx * size + i];
    }
  }
}

template<typename DType>
void rmsnorm( DType* output, const DType* x, const DType* weight, const uint64_t size, const uint64_t batch_size )
{
  size_t b;
#pragma omp parallel for private( b )
  for ( b = 0; b < batch_size; b++ ) {
    const DType* X = x + b * size;
    const DType* W = weight + b * size;
    DType* O = output + b * size;

    // calculate sum of squares
    DType ss = 0.0f;
    for ( uint64_t j = 0; j < size; j++ ) {
      ss += X[j] * X[j];
    }

    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf( ss );

    // normalize and scale
    for ( uint64_t j = 0; j < size; j++ ) {
      O[j] = W[j] * ( ss * X[j] );
    }
  }
}

template<typename DType>
void matmul( DType* xout, const DType* x, const DType* w, const uint64_t b, const uint64_t s, const uint64_t r )
{
  // x(b,s) @ W(s,r) -> xout(b,r)

  size_t i;
#pragma omp parallel for private( i )
  for ( i = 0; i < b; ++i ) {
    size_t j;
#pragma omp parallel for private( j )
    for ( j = 0; j < r; ++j ) {
      DType sum = 0;
      for ( uint64_t k = 0; k < s; ++k ) {
        sum += x[i * s + k] * w[k * r + j];
      }
      xout[i * r + j] = sum;
    }
  }
}

template<typename DType>
void silu( DType* hb, DType* hb2, const uint64_t hidden_dim, const uint64_t batch_size )
{
  size_t b;
#pragma omp parallel for private( b )
  for ( b = 0; b < batch_size; b++ ) {
    size_t i;
#pragma omp parallel for private( i )
    for ( i = 0; i < hidden_dim; i++ ) {
      hb[b * hidden_dim + i] = hb2[b * hidden_dim + i] * ( 1.0f / ( 1.0f + expf( -hb2[b * hidden_dim + i] ) ) );
    }
  }
}

template<typename DType>
void simple_gemm_strided_batch( int m,
                                int n,
                                int k,
                                bool transpose_a,
                                bool transpose_b,
                                float alpha,
                                const DType* A,
                                int lda,
                                uint64_t strideA,
                                const DType* B,
                                int ldb,
                                uint64_t strideB,
                                float beta,
                                DType* C,
                                int ldc,
                                uint64_t strideC,
                                uint64_t batch_count )
{
  for ( uint64_t batch = 0; batch < batch_count; batch++ ) {
    const DType* current_A = A + batch * strideA;
    const DType* current_B = B + batch * strideB;
    DType* current_C = C + batch * strideC;

    for ( int row = 0; row < m; row++ ) {
      for ( int col = 0; col < n; col++ ) {
        DType sum = 0.0;

        for ( int p = 0; p < k; ++p ) {
          const DType a_value = ( not transpose_a ) ? current_A[row * lda + p] : current_A[p * lda + row];
          const DType b_value = ( not transpose_b ) ? current_B[p * ldb + col] : current_B[col * ldb + p];
          sum += a_value * b_value;
        }

        current_C[row * ldc + col] = alpha * sum + beta * current_C[row * ldc + col];
      }
    }
  }
}

template<typename DType>
void attention_0_gemm( const DType* query,
                       const DType* const context_pointers[],
                       DType* att,
                       const uint64_t n_layers,
                       const uint64_t seq_len,
                       const uint64_t head_size,
                       const uint64_t n_kv_heads,
                       const uint64_t gqa_size,
                       const uint64_t batch_size,
                       const uint32_t* token_positions )
{
  const float scale = 1.0f / sqrtf( head_size );

  const uint64_t k = head_size;
  const uint64_t n = gqa_size;

  const uint64_t lda = n_layers * n_kv_heads * head_size * 2;
  const uint64_t ldb = k;
  const uint64_t ldc = seq_len;

  const uint64_t strideA = head_size;
  const uint64_t strideB = head_size * gqa_size;
  const uint64_t strideC = seq_len * gqa_size;

  const uint64_t batchCount = n_kv_heads;

  const uint64_t dim_ = head_size * n_kv_heads * gqa_size;
  const uint64_t att_dim_ = seq_len * n_kv_heads * gqa_size;

  for ( size_t i = 0; i < batch_size; i++ ) {
    const uint64_t m = token_positions[i] + 1;
    simple_gemm_strided_batch( m,
                               n,
                               k,
                               true,
                               false,
                               scale,
                               context_pointers[i],
                               lda,
                               strideA,
                               query + i * dim_,
                               ldb,
                               strideB,
                               0.0f,
                               att + i * att_dim_,
                               ldc,
                               strideC,
                               batchCount );
  }
}

template<typename DType>
void attention_2_gemm( const DType* att,
                       const DType* const context_pointers[],
                       DType* xb,
                       const uint64_t n_layers,
                       const uint64_t seq_len,
                       const uint64_t head_size,
                       const uint64_t n_kv_heads,
                       const uint64_t gqa_size,
                       const uint64_t batch_size,
                       const uint32_t* token_positions )
{
  const uint64_t m = head_size;
  const uint64_t n = gqa_size;

  const uint64_t lda = n_layers * n_kv_heads * head_size * 2;
  const uint64_t ldb = seq_len;
  const uint64_t ldc = m;

  const uint64_t strideA = head_size;
  const uint64_t strideB = seq_len * gqa_size;
  const uint64_t strideC = head_size * gqa_size;

  const uint64_t batchCount = n_kv_heads;

  const uint64_t dim_ = head_size * n_kv_heads * gqa_size;
  const uint64_t att_dim_ = seq_len * n_kv_heads * gqa_size;

  for ( size_t i = 0; i < batch_size; i++ ) {
    const uint64_t k = token_positions[i] + 1;

    simple_gemm_strided_batch( m,
                               n,
                               k,
                               false,
                               false,
                               1,
                               context_pointers[i] + dim_,
                               lda,
                               strideA,
                               att + i * att_dim_,
                               ldb,
                               strideB,
                               0,
                               xb + i * dim_,
                               ldc,
                               strideC,
                               batchCount );
  }
}

template<typename DType>
void softmax( DType* x, const uint64_t size )
{
  // find max value (for numerical stability)
  float max_val = x[0];
  for ( uint64_t i = 1; i < size; i++ ) {
    if ( x[i] > max_val ) {
      max_val = x[i];
    }
  }

  // exp and sum
  float sum = 0.0f;
  for ( uint64_t i = 0; i < size; i++ ) {
    x[i] = expf( x[i] - max_val );
    sum += x[i];
  }

  // normalize
  for ( uint64_t i = 0; i < size; i++ ) {
    x[i] /= sum;
  }
}

template<typename DType>
void attention_softmax( DType* att,
                        const uint32_t* token_positions,
                        const uint64_t seq_len,
                        const uint64_t n_heads,
                        [[maybe_unused]] DType* temp_buffer,
                        const uint64_t batch_size )
{
  for ( uint64_t i = 0; i < batch_size; i++ ) {
    DType* this_att = att + i * n_heads * seq_len;
    softmax( this_att, token_positions[i] + 1 );
  }
}

template<typename DType>
void do_rope( const uint64_t head_size,
              const uint64_t gqa_size,
              const DType* freq_cis_real_row,
              const DType* freq_cis_imag_row,
              DType* state_q,
              DType* state_k,
              const uint64_t head_q_num,
              const uint64_t head_k_num,
              const uint64_t elem_idx )
{
  // apply RoPE rotation to the q and k vectors for each head
  // get the q and k vectors for this head
  DType* q = state_q + head_q_num * head_size;
  DType* k = state_k + head_k_num * head_size;

  // rotate q and k by the freq_cis_real and freq_cis_imag
  const DType q0 = q[elem_idx];
  const DType q1 = q[elem_idx + 1];
  const DType k0 = k[elem_idx];
  const DType k1 = k[elem_idx + 1];
  const DType fcr = freq_cis_real_row[elem_idx / 2];
  const DType fci = freq_cis_imag_row[elem_idx / 2];
  k[elem_idx] = k0 * fcr - k1 * fci;
  k[elem_idx + 1] = k0 * fci + k1 * fcr;
  for ( uint64_t i = 0; i < gqa_size; i++ ) {
    q[i * head_size + elem_idx] = q0 * fcr - q1 * fci;
    q[i * head_size + elem_idx + 1] = q0 * fci + q1 * fcr;
  }
}

template<typename DType>
void apply_rope( const uint64_t head_size,
                 const uint64_t n_kv_heads,
                 const uint64_t gqa_size,
                 const uint64_t batch_size,
                 const uint32_t* token_positions,
                 const DType* freq_cis_real,
                 const DType* freq_cis_imag,
                 DType* state_q,
                 DType* state_k )
{
  for ( uint64_t i = 0; i < batch_size; i++ ) {
    for ( uint64_t j = 0; j < n_kv_heads; j++ ) {
      for ( uint64_t k = 0; k < head_size / 2; k++ ) {
        const uint64_t head_q_num = gqa_size * j;
        const uint64_t head_k_num = j;
        const uint64_t elem_idx = 2 * k;

        do_rope( head_size,
                 gqa_size,
                 freq_cis_real + token_positions[i] * head_size / 2,
                 freq_cis_imag + token_positions[i] * head_size / 2,
                 state_q + i * n_kv_heads * gqa_size * head_size,
                 state_k + i * n_kv_heads * head_size,
                 head_q_num,
                 head_k_num,
                 elem_idx );
      }
    }
  }
}

template<typename DType>
void copy_kv_cache( DType* context_pointers[],
                    const DType* state_k,
                    const DType* state_v,
                    const uint64_t dim,
                    const uint64_t n_layers,
                    const uint64_t batch_size,
                    const uint32_t* token_positions )
{
  for ( size_t i = 0; i < batch_size; i++ ) {
    DType* k_cache_pos = context_pointers[i] + token_positions[i] * n_layers * dim * 2;
    DType* v_cache_pos = k_cache_pos + dim;

    memcpy( k_cache_pos, state_k + i * dim, dim * sizeof( DType ) );
    memcpy( v_cache_pos, state_v + i * dim, dim * sizeof( DType ) );
  }
}

template<typename DType>
void gumbel_fix( DType* array, float temp, const uint64_t vocab_size )
{
  for ( uint64_t i = 0; i < vocab_size; i++ ) {
    float myrandf = static_cast<float>( rand() ) / RAND_MAX;
    myrandf = logf( -logf( myrandf ) );
    array[i] = array[i] / temp - myrandf;
  }
}

template<typename DType>
void soft_sample( DType* v, const vector<float>& temp_s, const uint64_t vocab_size, const uint64_t batch_size )
{
  for ( uint64_t i = 0; i < batch_size; i++ ) {
    if ( temp_s[i] > 0 ) {
      gumbel_fix( v + i * vocab_size, temp_s[i], vocab_size );
    }
  }
}

template<typename DType>
void argmax( uint32_t* output, const DType* v, const uint64_t n, const uint64_t batch_size )
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

template void accum<_Float16>( _Float16* a, const _Float16* b, const uint64_t size, const uint64_t batch_size );
template void argmax<_Float16>( uint32_t* output, const _Float16* v, const uint64_t n, const uint64_t batch_size );
template void silu<_Float16>( _Float16* hb, _Float16* hb2, const uint64_t hidden_dim, const uint64_t batch_size );

template void rmsnorm<_Float16>( _Float16* o,
                                 const _Float16* x,
                                 const _Float16* weight,
                                 const uint64_t size,
                                 const uint64_t batch_size );

template void matmul<_Float16>( _Float16* xout,
                                const _Float16* x,
                                const _Float16* w,
                                const uint64_t b,
                                const uint64_t s,
                                const uint64_t r );

template void soft_sample<_Float16>( _Float16* v,
                                     const std::vector<float>& temp_s,
                                     const uint64_t vocab_size,
                                     const uint64_t batch_size );

template void attention_0_gemm<_Float16>( const _Float16* query,
                                          const _Float16* const context_pointers[],
                                          _Float16* att,
                                          const uint64_t n_layers,
                                          const uint64_t seq_len,
                                          const uint64_t head_size,
                                          const uint64_t n_kv_heads,
                                          const uint64_t gqa_size,
                                          const uint64_t batch_size,
                                          const uint32_t* token_positions );

template void attention_2_gemm<_Float16>( const _Float16* att,
                                          const _Float16* const context_pointers[],
                                          _Float16* xb,
                                          const uint64_t n_layers,
                                          const uint64_t seq_len,
                                          const uint64_t head_size,
                                          const uint64_t n_kv_heads,
                                          const uint64_t gqa_size,
                                          const uint64_t batch_size,
                                          const uint32_t* token_positions );

template void attention_softmax<_Float16>( _Float16* att,
                                           const uint32_t* token_positions,
                                           const uint64_t seq_len,
                                           const uint64_t n_heads,
                                           _Float16* temp_buffer,
                                           const uint64_t batch_size );

template void apply_rope<_Float16>( const uint64_t head_size,
                                    const uint64_t n_kv_heads,
                                    const uint64_t gqa_size,
                                    const uint64_t curr_batch_size,
                                    const uint32_t* token_positions,
                                    const _Float16* freq_cis_real,
                                    const _Float16* freq_cis_imag,
                                    _Float16* state_q,
                                    _Float16* state_k );

template void copy_kv_cache<_Float16>( _Float16* context_pointers[],
                                       const _Float16* state_k,
                                       const _Float16* state_v,
                                       const uint64_t dim,
                                       const uint64_t n_layers,
                                       const uint64_t batch_size,
                                       const uint32_t* token_positions );

template void accum<float>( float* a, const float* b, const uint64_t size, const uint64_t batch_size );
template void argmax<float>( uint32_t* output, const float* v, const uint64_t n, const uint64_t batch_size );
template void silu<float>( float* hb, float* hb2, const uint64_t hidden_dim, const uint64_t batch_size );

template void rmsnorm<float>( float* o,
                              const float* x,
                              const float* weight,
                              const uint64_t size,
                              const uint64_t batch_size );

template void matmul<float>( float* xout,
                             const float* x,
                             const float* w,
                             const uint64_t b,
                             const uint64_t s,
                             const uint64_t r );

template void soft_sample<float>( float* v,
                                  const std::vector<float>& temp_s,
                                  const uint64_t vocab_size,
                                  const uint64_t batch_size );

template void attention_0_gemm<float>( const float* query,
                                       const float* const context_pointers[],
                                       float* att,
                                       const uint64_t n_layers,
                                       const uint64_t seq_len,
                                       const uint64_t head_size,
                                       const uint64_t n_kv_heads,
                                       const uint64_t gqa_size,
                                       const uint64_t batch_size,
                                       const uint32_t* token_positions );

template void attention_2_gemm<float>( const float* att,
                                       const float* const context_pointers[],
                                       float* xb,
                                       const uint64_t n_layers,
                                       const uint64_t seq_len,
                                       const uint64_t head_size,
                                       const uint64_t n_kv_heads,
                                       const uint64_t gqa_size,
                                       const uint64_t batch_size,
                                       const uint32_t* token_positions );

template void attention_softmax<float>( float* att,
                                        const uint32_t* token_positions,
                                        const uint64_t seq_len,
                                        const uint64_t n_heads,
                                        float* temp_buffer,
                                        const uint64_t batch_size );

template void apply_rope<float>( const uint64_t head_size,
                                 const uint64_t n_kv_heads,
                                 const uint64_t gqa_size,
                                 const uint64_t curr_batch_size,
                                 const uint32_t* token_positions,
                                 const float* freq_cis_real,
                                 const float* freq_cis_imag,
                                 float* state_q,
                                 float* state_k );

template void copy_kv_cache<float>( float* context_pointers[],
                                    const float* state_k,
                                    const float* state_v,
                                    const uint64_t dim,
                                    const uint64_t n_layers,
                                    const uint64_t batch_size,
                                    const uint32_t* token_positions );

} // namespace glinthawk::models::common::cpu
