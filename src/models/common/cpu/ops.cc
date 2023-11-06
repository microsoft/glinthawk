#include "ops.hh"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <glog/logging.h>

using namespace std;

namespace glinthawk::models::common::cpu::ops {

/// @brief Accumulate the values in b into a (a += b)
template<typename DType>
void accum( DType* a, const DType* b, const uint64_t size, const uint64_t batch_size )
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
void rmsnorm( DType* output, const DType* x, const DType* weight, const uint64_t size, const uint64_t batch_size )
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
void simple_gemm_strided_batch( uint64_t m,
                                uint64_t n,
                                uint64_t k,
                                const bool transpose_a,
                                const bool transpose_b,
                                float alpha,
                                const DType* A,
                                uint64_t lda,
                                uint64_t strideA,
                                const DType* B,
                                uint64_t ldb,
                                uint64_t strideB,
                                float beta,
                                DType* C,
                                uint64_t ldc,
                                uint64_t strideC,
                                uint64_t batch_count )
{
  uint64_t batch;
  uint64_t row;
#pragma omp parallel for private( batch, row ) shared( A, B, C ) collapse( 2 )
  for ( batch = 0; batch < batch_count; batch++ ) {
    for ( row = 0; row < m; row++ ) {
      uint64_t col;
      const DType* current_A = A + batch * strideA;
      const DType* current_B = B + batch * strideB;
      DType* current_C = C + batch * strideC;
      for ( col = 0; col < n; col++ ) {
        float sum = 0.0;

        for ( uint64_t p = 0; p < k; ++p ) {
          const float a_value = ( not transpose_a ) ? current_A[p * lda + row] : current_A[row * lda + p];
          const float b_value = ( not transpose_b ) ? current_B[col * ldb + p] : current_B[p * ldb + col];
          sum += a_value * b_value;
        }

        current_C[col * ldc + row] = DType( alpha * sum + beta * float( current_C[col * ldc + row] ) );
      }
    }
  }
}

template<typename DType>
void fast_matmul_row_major( uint64_t m,
                            uint64_t n,
                            uint64_t k,
                            const DType* A,
                            uint64_t lda,
                            const DType* B,
                            uint64_t ldb,
                            DType* C,
                            uint64_t ldc )
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

template<typename DType>
void matmul( DType* xout, const DType* x, const DType* w, const uint64_t b, const uint64_t s, const uint64_t r )
{
  // x(b,s) @ W(s,r) -> xout(b,r)
  // OR
  // W(r,s) @ x(s,b) -> xout(r,b)
  // A(m,k) @ B(k,n) ->    C(m,n)

  const uint64_t m = r;
  const uint64_t n = b;
  const uint64_t k = s;
  const uint64_t lda = k;
  const uint64_t ldb = k;
  const uint64_t ldc = m;

  fast_matmul_row_major( m, n, k, w, lda, x, ldb, xout, ldc );
}

template<typename DType>
void silu( DType* hb, DType* hb2, const uint64_t hidden_dim, const uint64_t batch_size )
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
void attention_0_gemm_fast( const DType* query,
                            const DType* const context_pointers[],
                            DType* att,
                            const uint64_t batch_size,
                            const uint32_t* token_positions )
{

  const uint64_t seq_len = 2048;
  const uint64_t head_size = 128;
  const uint64_t n_kv_heads = 8;
  const uint64_t gqa_size = 8;
  const float scale = 1.0f / sqrtf( head_size );

  const uint64_t ld_key = n_kv_heads * head_size * 2;
  const uint64_t ld_qry = head_size;
  const uint64_t ld_att = seq_len;

  const uint64_t stride_key = head_size;
  const uint64_t stride_qry = head_size * gqa_size;
  const uint64_t stride_att = seq_len * gqa_size;

  const uint64_t dim_ = head_size * n_kv_heads * gqa_size;
  const uint64_t att_dim_ = seq_len * n_kv_heads * gqa_size;

  uint64_t i;
  uint64_t kv_head;
  uint64_t key_pos;
  uint64_t query_gqa_head;
  uint64_t p;
  const size_t sum_len = 8;
  float sum_s[sum_len];
  CHECK_GE( sum_len, gqa_size ) << "Accounting for GQAs";

#pragma omp parallel for private( i, kv_head, p, sum_s, key_pos, query_gqa_head ) collapse( 3 )
  for ( i = 0; i < batch_size; i++ ) {
    for ( kv_head = 0; kv_head < n_kv_heads; kv_head++ ) {
      for ( key_pos = 0; key_pos < seq_len; key_pos++ ) {

        if ( key_pos >= token_positions[i] + 1 ) {
          continue;
        }

        std::memset( sum_s, 0, sizeof( float ) * gqa_size );
        const DType* current_key = context_pointers[i] + kv_head * stride_key + key_pos * ld_key;
        const DType* current_query = query + i * dim_ + kv_head * stride_qry;
        DType* current_att = att + i * att_dim_ + kv_head * stride_att;

        for ( p = 0; p < head_size; ++p ) {
          const float a_value = current_key[p];

          for ( query_gqa_head = 0; query_gqa_head < gqa_size; query_gqa_head++ ) {
            const float b_value = current_query[query_gqa_head * ld_qry + p];
            sum_s[query_gqa_head] += a_value * b_value;
          }
        }

        for ( query_gqa_head = 0; query_gqa_head < gqa_size; query_gqa_head++ ) {
          current_att[query_gqa_head * ld_att + key_pos] = DType( scale * sum_s[query_gqa_head] );
        }
      }
    }
  }
}

template<typename DType>
void attention_2_gemm_fast( const DType* att,
                            const DType* const context_pointers[],
                            DType* xb,
                            const uint64_t batch_size,
                            const uint32_t* token_positions )
{
  const uint64_t seq_len = 2048;
  const uint64_t head_size = 128;
  const uint64_t n_kv_heads = 8;
  const uint64_t gqa_size = 8;
  const uint64_t ld_val = n_kv_heads * head_size * 2;
  const uint64_t ld_att = seq_len;
  const uint64_t ld_xb = head_size;

  const uint64_t stride_val = head_size;
  const uint64_t stride_att = seq_len * gqa_size;
  const uint64_t stride_xb = head_size * gqa_size;

  const uint64_t kv_dim_ = head_size * n_kv_heads;
  const uint64_t dim_ = head_size * n_kv_heads * gqa_size;
  const uint64_t att_dim_ = seq_len * n_kv_heads * gqa_size;

  uint64_t i;
  uint64_t kv_head;
  uint64_t val_pos;
  uint64_t p;
  uint64_t att_gqa_head;
  uint64_t round_index;

  const size_t sum_len = 16;
  float sum_s[sum_len];
  uint64_t rounds = head_size / sum_len;
  CHECK_EQ( head_size % sum_len, 0 ) << "Remainders are bad";

#pragma omp parallel for private( i, kv_head, att_gqa_head, round_index, sum_s ) schedule( dynamic )                   \
  shared( token_positions, context_pointers, att, xb ) collapse( 4 )
  for ( i = 0; i < batch_size; i++ ) {
    for ( kv_head = 0; kv_head < n_kv_heads; kv_head++ ) {
      for ( att_gqa_head = 0; att_gqa_head < gqa_size; att_gqa_head++ ) {
        for ( round_index = 0; round_index < rounds; round_index++ ) {

          std::memset( sum_s, 0, sizeof( float ) * sum_len );
          const DType* current_att = att + i * att_dim_ + kv_head * stride_att + att_gqa_head * ld_att;
          const DType* current_value = context_pointers[i] + kv_dim_ + kv_head * stride_val + round_index * sum_len;

          for ( p = 0; p < token_positions[i] + 1; ++p ) {
            const float b_value = current_att[p];

            for ( val_pos = 0; val_pos < sum_len; val_pos++ ) {
              const float a_value = current_value[val_pos];
              sum_s[val_pos] += a_value * b_value;
            }
            current_value += ld_val;
          }
          DType* current_xb = xb + i * dim_ + kv_head * stride_xb + att_gqa_head * ld_xb + round_index * sum_len;
          for ( val_pos = 0; val_pos < sum_len; val_pos++ ) {
            current_xb[val_pos] = DType( sum_s[val_pos] );
          }
        }
      }
    }
  }
}

template<typename DType>
void attention_0_gemm( const DType* query,
                       const DType* const context_pointers[],
                       DType* att,
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

  const uint64_t lda = n_kv_heads * head_size * 2;
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
                       const uint64_t seq_len,
                       const uint64_t head_size,
                       const uint64_t n_kv_heads,
                       const uint64_t gqa_size,
                       const uint64_t batch_size,
                       const uint32_t* token_positions )
{
  const uint64_t m = head_size;
  const uint64_t n = gqa_size;

  const uint64_t lda = n_kv_heads * head_size * 2;
  const uint64_t ldb = seq_len;
  const uint64_t ldc = m;

  const uint64_t strideA = head_size;
  const uint64_t strideB = seq_len * gqa_size;
  const uint64_t strideC = head_size * gqa_size;

  const uint64_t batchCount = n_kv_heads;

  const uint64_t kv_dim_ = head_size * n_kv_heads;
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
                               context_pointers[i] + kv_dim_,
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

template<typename DType>
void attention_softmax( DType* att, const uint32_t* token_positions, const uint64_t batch_size )
{
  const uint64_t seq_len = 2048;
  const uint64_t n_heads = 64;
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
inline void do_rope( const uint64_t head_size,
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
  uint64_t i;
  uint64_t j;
#pragma omp parallel for private( i, j ) collapse( 2 )
  for ( i = 0; i < batch_size; i++ ) {
    for ( j = 0; j < n_kv_heads; j++ ) {
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
                    const uint64_t batch_size,
                    const uint32_t* token_positions )
{
  uint64_t i;
#pragma omp parallel for private( i )
  for ( i = 0; i < batch_size; i++ ) {
    DType* k_cache_pos = context_pointers[i] + token_positions[i] * dim * 2;
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
    array[i] = DType( float( array[i] ) / temp - myrandf );
  }
}

template<typename DType>
void soft_sample( DType* v, const vector<float>& temp_s, const uint64_t vocab_size, const uint64_t batch_size )
{
  uint64_t i;
#pragma omp parallel for private( i )
  for ( i = 0; i < batch_size; i++ ) {
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
                                          const uint64_t seq_len,
                                          const uint64_t head_size,
                                          const uint64_t n_kv_heads,
                                          const uint64_t gqa_size,
                                          const uint64_t batch_size,
                                          const uint32_t* token_positions );

template void attention_2_gemm<_Float16>( const _Float16* att,
                                          const _Float16* const context_pointers[],
                                          _Float16* xb,
                                          const uint64_t seq_len,
                                          const uint64_t head_size,
                                          const uint64_t n_kv_heads,
                                          const uint64_t gqa_size,
                                          const uint64_t batch_size,
                                          const uint32_t* token_positions );

template void attention_0_gemm_fast<_Float16>( const _Float16* query,
                                               const _Float16* const context_pointers[],
                                               _Float16* att,
                                               const uint64_t batch_size,
                                               const uint32_t* token_positions );

template void attention_2_gemm_fast<_Float16>( const _Float16* att,
                                               const _Float16* const context_pointers[],
                                               _Float16* xb,
                                               const uint64_t batch_size,
                                               const uint32_t* token_positions );

template void attention_softmax<_Float16>( _Float16* att, const uint32_t* token_positions, const uint64_t batch_size );

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
                                       const uint64_t seq_len,
                                       const uint64_t head_size,
                                       const uint64_t n_kv_heads,
                                       const uint64_t gqa_size,
                                       const uint64_t batch_size,
                                       const uint32_t* token_positions );

template void attention_2_gemm<float>( const float* att,
                                       const float* const context_pointers[],
                                       float* xb,
                                       const uint64_t seq_len,
                                       const uint64_t head_size,
                                       const uint64_t n_kv_heads,
                                       const uint64_t gqa_size,
                                       const uint64_t batch_size,
                                       const uint32_t* token_positions );

template void attention_0_gemm_fast<float>( const float* query,
                                            const float* const context_pointers[],
                                            float* att,
                                            const uint64_t batch_size,
                                            const uint32_t* token_positions );

template void attention_2_gemm_fast<float>( const float* att,
                                            const float* const context_pointers[],
                                            float* xb,
                                            const uint64_t batch_size,
                                            const uint32_t* token_positions );

template void attention_softmax<float>( float* att, const uint32_t* token_positions, const uint64_t batch_size );

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
                                    const uint64_t batch_size,
                                    const uint32_t* token_positions );

} // namespace glinthawk::models::common::cpu
