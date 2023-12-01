#pragma once

#include "concept.hh"
#include "models/common/ops/cuda.hh"

namespace glinthawk::models::llama2::cuda {

template<typename DType>
class LlamaOperations : public common::cuda::Operations<DType>
{
public:
  using common::cuda::Operations<DType>::Operations;
  using common::cuda::Operations<DType>::~Operations;

  template<uint64_t seq_len, uint64_t head_size, uint64_t n_kv_heads, uint64_t gqa_size>
  void attention_0_gemm( const DType* query,
                         const DType* const context_pointers[],
                         DType* att,
                         const uint64_t batch_size,
                         const uint32_t* token_positions );

  template<uint64_t seq_len, uint64_t head_size, uint64_t n_kv_heads, uint64_t gqa_size, uint64_t rounds>
  void attention_2_gemm( const DType* att,
                         const DType* const context_pointers[],
                         DType* xb,
                         const uint64_t batch_size,
                         const uint32_t* token_positions );

  template<uint64_t seq_len, uint64_t n_heads>
  void attention_softmax( DType* att, const uint32_t* token_positions, DType* temp_buffer, const uint64_t batch_size );

  template<uint64_t head_size, uint64_t n_kv_heads, uint64_t gqa_size>
  void apply_rope( const uint64_t curr_batch_size,
                   const uint32_t* token_positions,
                   const DType* freq_cis_real,
                   const DType* freq_cis_imag,
                   DType* state_q,
                   DType* context_pointers[] );

  template<uint64_t dim>
  void copy_kv_cache( DType* context_pointers[],
                      const DType* state_k,
                      const DType* state_v,
                      const uint64_t batch_size,
                      const uint32_t* token_positions );

  template<typename DTypeDst, typename DTypeSrc>
  void convert_and_copy( DTypeDst* dst, const DTypeSrc* src, const uint64_t size, const CopyType );
};

static_assert( LlamaOperationsConcept<LlamaOperations<float>, float, float, __half> );
static_assert( LlamaOperationsConcept<LlamaOperations<__half>, __half, __half, float> );

namespace {

namespace { // attention_softmax

template<typename DType, uint64_t seq_len>
__global__ void find_max_for_rows( const DType* att, DType* output, const uint64_t token_pos )
{
  const uint64_t head_num = threadIdx.x;
  att += head_num * seq_len;

  DType max_value = att[0];
  for ( uint64_t i = 1; i <= token_pos; i++ ) {
    if constexpr ( std::is_same_v<DType, __half> ) {
      max_value = __hmax( max_value, att[i] );
    } else {
      max_value = max( max_value, att[i] );
    }
  }

  output[head_num] = max_value;
}

template<typename DType, uint64_t seq_len>
__global__ void subtract_and_expf( const DType* values, DType* att )
{
  const uint64_t head_num = threadIdx.x;
  const uint64_t token_pos = blockIdx.x;

  att += head_num * seq_len;

  if constexpr ( std::is_same_v<DType, __half> ) {
    att[token_pos] = hexp( att[token_pos] - values[head_num] );
  } else {
    att[token_pos] = expf( att[token_pos] - values[head_num] );
  }
}

template<typename DType, uint64_t seq_len>
__global__ void sum_rows( DType* att, DType* output, const uint64_t token_pos )
{
  const uint64_t head_num = threadIdx.x;
  att += head_num * seq_len;

  DType sum = 0.0;
  for ( uint64_t i = 0; i <= token_pos; i++ ) {
    sum += att[i];
  }

  output[head_num] = sum;
}

template<typename DType, uint64_t seq_len>
__global__ void normalize_by_sum( DType* att, const DType* sums )
{
  const uint64_t head_num = threadIdx.x;
  const uint64_t token_pos = blockIdx.x;

  att += head_num * seq_len;
  att[token_pos] /= sums[head_num];
}

}

namespace { // rope

template<typename DType, uint64_t head_size, uint64_t gqa_size>
__global__ void do_rope( const DType* freq_cis_real_row,
                         const DType* freq_cis_imag_row,
                         DType* state_q,
                         DType* state_k )
{
  const uint64_t head_q_num = gqa_size * blockIdx.x;
  const uint64_t head_k_num = blockIdx.x;
  const uint64_t elem_idx = 2 * threadIdx.x;

  // apply RoPE rotation to the q and k vectors for each head
  // get the q and k vectors for this head
  DType* q = state_q + head_q_num * head_size;
  DType* k = state_k + head_k_num * head_size;

  // rotate q and k by the freq_cis_real and freq_cis_imag
  const DType fcr = freq_cis_real_row[elem_idx / 2];
  const DType fci = freq_cis_imag_row[elem_idx / 2];

  const DType k0 = k[elem_idx];
  const DType k1 = k[elem_idx + 1];
  k[elem_idx] = k0 * fcr - k1 * fci;
  k[elem_idx + 1] = k0 * fci + k1 * fcr;

  for ( uint64_t i = 0; i < gqa_size; i++ ) {
    const DType q0 = q[i * head_size + elem_idx];
    const DType q1 = q[i * head_size + elem_idx + 1];
    q[i * head_size + elem_idx] = q0 * fcr - q1 * fci;
    q[i * head_size + elem_idx + 1] = q0 * fci + q1 * fcr;
  }
}

}

}

template<typename DType>
template<uint64_t seq_len, uint64_t head_size, uint64_t n_kv_heads, uint64_t gqa_size>
void LlamaOperations<DType>::attention_0_gemm( const DType* query,
                                               const DType* const context_pointers[],
                                               DType* att,
                                               const uint64_t batch_size,
                                               const uint32_t* token_positions )
{
  constexpr cudaDataType_t cuda_arg_type = is_same_v<DType, __half> ? CUDA_R_16F : CUDA_R_32F;
  constexpr float scale = 1.0f / sqrtf( head_size );
  constexpr uint64_t k = head_size;
  constexpr uint64_t n = gqa_size;
  constexpr uint64_t lda = n_kv_heads * head_size * 2;
  constexpr uint64_t ldb = k;
  constexpr uint64_t ldc = seq_len;
  constexpr uint64_t strideA = head_size;
  constexpr uint64_t strideB = head_size * gqa_size;
  constexpr uint64_t strideC = seq_len * gqa_size;
  constexpr uint64_t gemm_batch_count = n_kv_heads;
  constexpr uint64_t dim = head_size * n_kv_heads * gqa_size;
  constexpr uint64_t att_dim = seq_len * n_kv_heads * gqa_size;

  for ( size_t i = 0; i < batch_size; i++ ) {
    const uint64_t m = token_positions[i] + 1;
    CHECK_CUBLAS( cublasGemmStridedBatchedEx( cublas_handle_array[i],
                                              CUBLAS_OP_T,
                                              CUBLAS_OP_N,
                                              m,
                                              n,
                                              k,
                                              &scale,
                                              context_pointers[i],
                                              cuda_arg_type,
                                              lda,
                                              strideA,
                                              query + i * dim,
                                              cuda_arg_type,
                                              ldb,
                                              strideB,
                                              &beta,
                                              att + i * att_dim,
                                              cuda_arg_type,
                                              ldc,
                                              strideC,
                                              gemm_batch_count,
                                              computeType,
                                              CUBLAS_GEMM_DEFAULT ) );
  }
}

template<typename DType>
template<uint64_t seq_len, uint64_t head_size, uint64_t n_kv_heads, uint64_t gqa_size, uint64_t rounds>
void LlamaOperations<DType>::attention_2_gemm( const DType* att,
                                               const DType* const context_pointers[],
                                               DType* xb,
                                               const uint64_t batch_size,
                                               const uint32_t* token_positions )
{
  constexpr cudaDataType_t cuda_arg_type = is_same_v<DType, __half> ? CUDA_R_16F : CUDA_R_32F;

  constexpr uint64_t m = head_size;
  constexpr uint64_t n = gqa_size;

  constexpr uint64_t lda = n_kv_heads * head_size * 2;
  constexpr uint64_t ldb = seq_len;
  constexpr uint64_t ldc = m;

  constexpr uint64_t strideA = head_size;
  constexpr uint64_t strideB = seq_len * gqa_size;
  constexpr uint64_t strideC = head_size * gqa_size;

  constexpr uint64_t gemm_batch_count = n_kv_heads;

  constexpr uint64_t kv_dim = head_size * n_kv_heads;
  constexpr uint64_t dim = head_size * n_kv_heads * gqa_size;
  constexpr uint64_t att_dim = seq_len * n_kv_heads * gqa_size;

  for ( size_t i = 0; i < batch_size; i++ ) {
    const uint64_t k = token_positions[i] + 1;

    CHECK_CUBLAS( cublasGemmStridedBatchedEx( cublas_handle_array[i],
                                              CUBLAS_OP_N,
                                              CUBLAS_OP_N,
                                              m,
                                              n,
                                              k,
                                              &alpha,
                                              context_pointers[i] + kv_dim,
                                              cuda_arg_type,
                                              lda,
                                              strideA,
                                              att + i * att_dim,
                                              cuda_arg_type,
                                              ldb,
                                              strideB,
                                              &beta,
                                              xb + i * dim,
                                              cuda_arg_type,
                                              ldc,
                                              strideC,
                                              gemm_batch_count,
                                              computeType,
                                              CUBLAS_GEMM_DEFAULT ) );
  }
}

template<typename DType>
template<uint64_t seq_len, uint64_t n_heads>
void LlamaOperations<DType>::attention_softmax( DType* att,
                                                const uint32_t* token_positions,
                                                DType* temp_buffer,
                                                const uint64_t batch_size )
{
  for ( uint64_t i = 0; i < batch_size; i++ ) {
    DType* this_att = att + i * n_heads * seq_len;
    DType* this_buff = temp_buffer + i * n_heads;

    // (1) find the max value for each head (each row)
    find_max_for_rows<<<1, n_heads, 0, streams[i]>>>( this_att, this_buff, token_positions[i], seq_len );

    // (2) exp(att - max)
    subtract_and_expf<<<token_positions[i] + 1, n_heads, 0, streams[i]>>>( this_buff, this_att, seq_len );

    // (3) sum each row
    sum_rows<<<1, n_heads, 0, streams[i]>>>( this_att, this_buff, token_positions[i], seq_len );

    // (4) normalize each row by its sum
    normalize_by_sum<<<token_positions[i] + 1, n_heads, 0, streams[i]>>>( this_att, this_buff, seq_len );
  }
}

template<typename DType>
template<uint64_t head_size, uint64_t n_kv_heads, uint64_t gqa_size>
void LlamaOperations<DType>::apply_rope( const uint64_t curr_batch_size,
                                         const uint32_t* token_positions,
                                         const DType* freq_cis_real,
                                         const DType* freq_cis_imag,
                                         DType* state_q,
                                         DType* context_pointers[] )
{
  for ( uint64_t i = 0; i < curr_batch_size; i++ ) {
    do_rope<<<n_kv_heads, head_size / 2, 0, streams[i]>>>( head_size,
                                                           gqa_size,
                                                           freq_cis_real + token_positions[i] * head_size / 2,
                                                           freq_cis_imag + token_positions[i] * head_size / 2,
                                                           state_q + i * n_kv_heads * gqa_size * head_size,
                                                           context_pointers[i]
                                                             + token_positions[i] * n_kv_heads * head_size * 2 );
  }
}

template<typename DType>
template<uint64_t kv_dim>
void LlamaOperations<DType>::copy_kv_cache( DType* context_pointers[],
                                            const DType* state_k,
                                            const DType* state_v,
                                            const uint64_t batch_size,
                                            const uint32_t* token_positions )
{
  for ( size_t i = 0; i < batch_size; i++ ) {
    if ( context_pointers[i] == nullptr ) {
      continue;
    }

    DType* k_cache_pos = context_pointers[i] + token_positions[i] * kv_dim * 2;
    DType* v_cache_pos = k_cache_pos + kv_dim;

    // XXX why not just one memcpy?
    CHECK_CUDA(
      cudaMemcpyAsync( k_cache_pos, state_k + i * kv_dim, kv_dim * sizeof( DType ), cudaMemcpyDeviceToDevice ) );
    CHECK_CUDA(
      cudaMemcpyAsync( v_cache_pos, state_v + i * kv_dim, kv_dim * sizeof( DType ), cudaMemcpyDeviceToDevice ) );
  }
}

template<typename DType>
template<typename DTypeDst, typename DTypeSrc>
void LlamaOperations<DType>::convert_and_copy( DTypeDst* dst,
                                               const DTypeSrc* src,
                                               const uint64_t size,
                                               const CopyType type )
{
  switch ( type ) {
    case CopyType::DeviceToHost: {
      if constexpr ( std::is_same_v<DTypeSrc, DTypeDst> ) {
        CHECK_CUDA( cudaMemcpy( dst_cuda, src_cpu, size * sizeof( DTypeSrc ), cudaMemcpyDeviceToHost ) );
      } else {
        std::unique_ptr<DTypeDst> dst_cpu { reinterpret_cast<DTypeDst*>( new uint8_t[sizeof( DTypeDst ) * size] ) };
        for ( uint64_t i = 0; i < size; i++ ) {
          dst_cpu[i] = static_cast<DTypeDst>( src_cpu[i] );
        }
        CHECK_CUDA( cudaMemcpy( dst_cuda, dst_cpu, size * sizeof( DTypeDst ), cudaMemcpyDeviceToHost ) );
      }
      break;
    }

    case CopyType::HostToDevice: {
      if constexpr ( std::is_same_v<DTypeSrc, DTypeDst> ) {
        CHECK_CUDA( cudaMemcpy( dst_cpu, src_cuda, size * sizeof( DTypeSrc ), cudaMemcpyHostToDevice ) );
      } else {
        std::unique_ptr<DTypeSrc> src_cpu { reinterpret_cast<DTypeSrc*>( new uint8_t[sizeof( DTypeDst ) * size] ) };
        CHECK_CUDA( cudaMemcpy( src_cpu, src_cuda, size * sizeof( DTypeSrc ), cudaMemcpyHostToDevice ) );
        for ( uint64_t i = 0; i < size; i++ ) {
          dst_cpu[i] = static_cast<DTypeDst>( src_cpu[i] );
        }
      }
    } break;
  }

  throw runtime_error( "convert_and_copy: Invalid copy type" );
}

} // namespace glinthawk::models::llama2::cuda
