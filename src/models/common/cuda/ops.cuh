#pragma once

#include <concepts>
#include <cstdint>
#include <curand.h>
#include <curand_kernel.h>
#include <source_location>
#include <vector>
#include <iostream>

namespace glinthawk::models::common::cuda::ops {

constexpr size_t TPB = 64;    /* threads per block */
constexpr size_t NRBS = 32;   /* norm reduce block size */
constexpr size_t AMRBS = 128; /* argmax reduce block size */

template<std::unsigned_integral T>
T div_ceil( const T x, const T y )
{
  return x / y + ( x % y != 0 );
}

template<typename DType>
struct CUDADeleter
{
  void operator()( DType* ptr ) const
  {
    if ( ptr )
      cudaFree( ptr );
  }
};

void init( const int num_streams );
void destroy();

void CHECK_CUDA( const cudaError_t err, const std::source_location location = std::source_location::current() );

__global__ void setup_kernel( curandState* state, unsigned long seed );

template<typename DType>
void accum( DType* a, const DType* b, const uint64_t size, const uint64_t batch_size );

template<typename DType>
void rmsnorm( DType* o, const DType* x, DType* a, const DType* weight, const uint64_t size, const uint64_t batch_size );

template<typename DType>
void matmul( DType* xout, const DType* x, const DType* w, const uint64_t b, const uint64_t s, const uint64_t r );

template<typename DType>
void soft_sample( DType* v,
                  const std::vector<float>& temp_s,
                  curandState* rng_state,
                  const uint64_t vocab_size,
                  const uint64_t batch_size );

template<typename DType>
void argmax( uint32_t* output, const DType* v, DType* temp, const uint64_t n, const uint64_t batch_size );

template<typename DType>
uint32_t argmax( const DType* v, const uint64_t n );

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
void attention_softmax( DType* att,
                        const uint32_t* token_positions,
                        const uint64_t seq_len,
                        const uint64_t n_heads,
                        DType* temp_buffer,
                        const uint64_t batch_size );

template<typename DType>
void apply_rope( const uint64_t head_size,
                 const uint64_t n_kv_heads,
                 const uint64_t gqa_size,
                 const uint64_t curr_batch_size,
                 const uint32_t* token_positions,
                 const DType* freq_cis_real,
                 const DType* freq_cis_imag,
                 DType* state_q,
                 DType* context_pointers[] );

template<typename DType>
void copy_kv_cache( DType* context_pointers[],
                    const DType* state_kv,
                    const uint64_t dim,
                    const uint64_t batch_size,
                    const uint32_t* token_positions );

template<typename DType_dst, typename DType_src>
void cvt_and_copy_to_cuda( DType_dst* dst_cuda, const DType_src* src_cpu, const uint64_t size );

template<typename DType_dst, typename DType_src>
void cvt_and_copy_from_cuda( DType_dst* dst_cpu, const DType_src* src_cuda, const uint64_t size );

void setup_rng( curandState* rng_state, unsigned long seed, const uint64_t size, const uint64_t batch_size );

}
