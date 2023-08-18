#include "model.cuh"

#include <fcntl.h>
#include <memory>
#include <source_location>

#include <cuda_fp16.h>

#include "util/exception.hh"
#include "util/file_descriptor.hh"
#include "util/ring_buffer.hh"

#include "models/common/cuda/ops.cuh"
#include "models/llama2/base.hh"

using namespace std;
using namespace glinthawk::models;
using namespace glinthawk::models::common::cuda;

namespace glinthawk::models::llama2::cuda {

void CHECK_CUDA( const cudaError_t err, const source_location location = source_location::current() )
{
  if ( err != cudaSuccess ) {
    throw runtime_error( "CUDA error " + string( cudaGetErrorName( err ) ) + ": " + string( cudaGetErrorString( err ) )
                         + " (" + location.file_name() + ":" + to_string( location.line() ) + ")" );
  }
}

template<typename DType>
void cuda_deleter( DType* ptr )
{
  cudaFree( ptr );
}

template<typename DType>
std::string dtype_str()
{
  if constexpr ( is_same_v<DType, float> ) {
    return "FP32";
  } else if constexpr ( is_same_v<DType, __half> ) {
    return "FP16";
  } else {
    return "unknown";
  }
}

template<typename DType>
Llama2<DType> Llama2<DType>::load( const filesystem::path& model_path,
                                   const int32_t start_layer,
                                   const int32_t end_layer_raw )
{
  ops::init();

  const string filename_suffix = "_" + dtype_str<DType>();
  const auto config_path = model_path / "CONFIG";
  const auto base_path = model_path / ( "BASEWEIGHTS" + filename_suffix );

  llama2::Config config { config_path };

  const int32_t end_layer = ( end_layer_raw == -1 ) ? ( config.n_layers - 1 ) : end_layer_raw;
  const int32_t layer_count = end_layer - start_layer + 1;

  CHECK_GE( start_layer, 0 ) << "Start layer must be non-negative.";
  CHECK_LE( start_layer, end_layer ) << "Start layer must be less than or equal to end layer.";
  CHECK_LT( end_layer, config.n_layers ) << "End layer must be less than the number of layers.";

  const auto run_state_size = RunState<DType>::state_size( config );
  const auto base_size = BaseWeights<DType>::base_size( config );
  const auto layer_size = LayerWeights<DType>::layer_size( config );
  const auto kv_cache_size = KVCache<DType>::cache_size( config, start_layer, end_layer );

  DType* base_raw_ptr;
  DType* layers_raw_ptr;
  DType* run_state_raw_ptr;
  DType* kv_cache_raw_ptr;

  // Allocate memory for the base weights
  CHECK_CUDA( cudaMalloc( &base_raw_ptr, base_size ) );
  unique_ptr<DType, void ( * )( DType* )> base { base_raw_ptr, cuda_deleter };

  // Allocate memory for the layers
  CHECK_CUDA( cudaMalloc( &layers_raw_ptr, layer_size * layer_count ) );
  unique_ptr<DType, void ( * )( DType* )> layers { layers_raw_ptr, cuda_deleter };

  // Allocate memory for the run state
  CHECK_CUDA( cudaMalloc( &run_state_raw_ptr, run_state_size ) );
  unique_ptr<DType, void ( * )( DType* )> run_state { run_state_raw_ptr, cuda_deleter };

  // Allocate memory for the kv cache
  CHECK_CUDA( cudaMalloc( &kv_cache_raw_ptr, kv_cache_size ) );
  unique_ptr<DType, void ( * )( DType* )> kv_cache { kv_cache_raw_ptr, cuda_deleter };

  // Load the model

  // (1) loading the base weights
  {
    CHECK_EQ( filesystem::file_size( base_path ), base_size ) << "Base weights are not the expected size.";

    FileDescriptor base_fd { CHECK_SYSCALL( "open", open( base_path.c_str(), O_RDONLY ) ) };
    MMap_Region base_mmap { nullptr, base_size, PROT_READ, MAP_PRIVATE, base_fd.fd_num(), 0 };
    CHECK_CUDA( cudaMemcpy( base.get(), base_mmap.addr(), base_size, cudaMemcpyHostToDevice ) );

    LOG( INFO ) << "Loaded base weights from " << base_path << " (" << base_size << " bytes).";
  }

  // (2) load the layers
  for ( auto i = start_layer; i <= end_layer; i++ ) {
    const auto layer_path = model_path / ( "LAYER" + to_string( i ) + filename_suffix );

    CHECK_EQ( filesystem::file_size( layer_path ), layer_size ) << "Layer " << i << " is not the expected size.";

    FileDescriptor layer_fd { CHECK_SYSCALL( "open", open( layer_path.c_str(), O_RDONLY ) ) };
    MMap_Region layer_mmap { nullptr, layer_size, PROT_READ, MAP_PRIVATE, layer_fd.fd_num(), 0 };

    CHECK_CUDA( cudaMemcpy( reinterpret_cast<uint8_t*>( layers.get() ) + ( i - start_layer ) * layer_size,
                            layer_mmap.addr(),
                            layer_size,
                            cudaMemcpyHostToDevice ) );

    LOG( INFO ) << "Loaded layer " << i << " from " << layer_path << " (" << layer_size << " bytes).";
  }

  return { config, move( base ), move( layers ), move( run_state ), move( kv_cache ), start_layer, end_layer };
}

template<typename DType>
__global__ void do_rope( const int head_size,
                         const int n_heads,
                         const DType* freq_cis_real_row,
                         const DType* freq_cis_imag_row,
                         DType* state_q,
                         DType* state_k )
{
  const int head_num = blockIdx.x;
  const int elem_idx = 2 * threadIdx.x;

  // apply RoPE rotation to the q and k vectors for each head
  // get the q and k vectors for this head
  DType* q = state_q + head_num * head_size;
  DType* k = state_k + head_num * head_size;

  // rotate q and k by the freq_cis_real and freq_cis_imag
  const DType q0 = q[elem_idx];
  const DType q1 = q[elem_idx + 1];
  const DType k0 = k[elem_idx];
  const DType k1 = k[elem_idx + 1];
  const DType fcr = freq_cis_real_row[elem_idx / 2];
  const DType fci = freq_cis_imag_row[elem_idx / 2];
  q[elem_idx] = q0 * fcr - q1 * fci;
  q[elem_idx + 1] = q0 * fci + q1 * fcr;
  k[elem_idx] = k0 * fcr - k1 * fci;
  k[elem_idx + 1] = k0 * fci + k1 * fcr;
}

template<typename DType>
__global__ void attention_0( const DType* all_q,
                             const DType* kv_cache,
                             DType* att,
                             const int layer_num,
                             const int n_layers,
                             const int seq_len,
                             const int head_size,
                             const int dim )
{
  const int head_num = threadIdx.x;
  const int token_pos = blockIdx.x;

  att += head_num * seq_len;
  const DType* q = all_q + head_num * head_size;
  const DType* k = kv_cache + token_pos * ( n_layers * dim * 2 ) + layer_num * ( dim * 2 ) + head_num * head_size;

  DType score = 0.0f;
  for ( int i = 0; i < head_size; i++ ) {
    score += q[i] * k[i];
  }

  if constexpr ( is_same_v<DType, __half> ) {
    score /= hsqrt( head_size );
  } else {
    score /= sqrtf( head_size );
  }

  // save the score to the attention buffer
  att[token_pos] = score;
}

template<typename DType>
__global__ void find_max_for_rows( const DType* att,
                                   DType* output,
                                   const int token_pos,
                                   const int n_heads,
                                   const int seq_len )
{
  const int head_num = threadIdx.x;
  att += head_num * seq_len;

  DType max_value = att[0];
  for ( int i = 1; i <= token_pos; i++ ) {
    if constexpr ( is_same_v<DType, __half> ) {
      max_value = __hmax( max_value, att[i] );
    } else {
      max_value = max( max_value, att[i] );
    }
  }

  output[head_num] = max_value;
}

template<typename DType>
__global__ void subtract_and_expf( const DType* values, DType* att, const int n_heads, const int seq_len )
{
  const int head_num = threadIdx.x;
  const int token_pos = blockIdx.x;

  att += head_num * seq_len;
  att[token_pos] = expf( att[token_pos] - values[head_num] );
}

template<typename DType>
__global__ void sum_rows( DType* att, DType* output, const int token_pos, const int n_heads, const int seq_len )
{
  const int head_num = threadIdx.x;
  att += head_num * seq_len;

  DType sum = 0.0;
  for ( int i = 0; i <= token_pos; i++ ) {
    sum += att[i];
  }

  output[head_num] = sum;
}

template<typename DType>
__global__ void normalize_by_sum( DType* att, const DType* sums, const int n_heads, const int seq_len )
{
  const int head_num = threadIdx.x;
  const int token_pos = blockIdx.x;

  att += head_num * seq_len;
  att[token_pos] /= sums[head_num];
}

template<typename DType>
void attention_softmax( DType* att, const int token_pos, const int seq_len, const int n_heads, DType* temp_buffer )
{
  DType* head_values = temp_buffer;

  // (1) find the max value for each head (each row)
  find_max_for_rows<<<1, n_heads>>>( att, head_values, token_pos, n_heads, seq_len );

  // (2) exp(att - max)
  subtract_and_expf<<<token_pos + 1, n_heads>>>( head_values, att, n_heads, seq_len );

  // (3) sum each row
  sum_rows<<<1, n_heads>>>( att, head_values, token_pos, n_heads, seq_len );

  // (4) normalize each row by its sum
  normalize_by_sum<<<token_pos + 1, n_heads>>>( att, head_values, n_heads, seq_len );
}

template<typename DType>
__global__ void attention_2( DType* att,
                             const DType* kv_cache,
                             DType* xb,
                             const int layer_num,
                             const int n_layers,
                             const int seq_len,
                             const int head_size,
                             const int dim )
{
  const int head_num = threadIdx.x;
  const int token_pos = blockIdx.x;

  att += head_num * seq_len;
  xb += head_num * head_size;

  const DType a = att[token_pos];
  const DType* v = kv_cache + token_pos * ( n_layers * dim * 2 ) + layer_num * ( dim * 2 ) + head_num * head_size + dim;

  for ( int i = 0; i < head_size; i++ ) {
    atomicAdd( &xb[i], a * v[i] );
  }
}

template<typename DType>
__global__ void attention_2_v2( DType* att,
                                const DType* kv_cache,
                                DType* xb,
                                const int layer_num,
                                const int n_layers,
                                const int seq_len,
                                const int head_size,
                                const int dim,
                                const int n_tokens )
{
  const int head_num = threadIdx.x;
  const int i_head = blockIdx.x;

  att += head_num * seq_len;
  xb += head_num * head_size;

  const DType* v = kv_cache + layer_num * ( dim * 2 ) + head_num * head_size + dim + i_head;
  DType res = DType();

  for ( int i = 0; i < n_tokens; i++ ) {
    res += att[i] * v[i * ( n_layers * dim * 2 )];
  }

  xb[i_head] = res;
}

template<typename DType>
void Llama2<DType>::pass_begin( const int token )
{
  // copy the token embedding into the state
  const DType* content_row = this->base_weights_.token_embedding_table + token * this->config_.dim;
  CHECK_CUDA(
    cudaMemcpy( this->state_.x, content_row, this->config_.dim * sizeof( DType ), cudaMemcpyDeviceToDevice ) );
}

template<typename DType>
void Llama2<DType>::transformer_layer( const int32_t layer_num, const int token_pos )
{
  DType* const x = this->state_.x;
  const int dim = this->config_.dim;
  const int hidden_dim = this->config_.hidden_dim;
  const int head_size = dim / this->config_.n_heads;

  // pluck out the "pos" row of freq_cis_real and freq_cis_imag
  const DType* freq_cis_real_row = this->base_weights_.freq_cis_real + token_pos * head_size / 2;
  const DType* freq_cis_imag_row = this->base_weights_.freq_cis_imag + token_pos * head_size / 2;

  const auto& layer_weights = this->layer_weights_[layer_num];

  // attention rmsnorm
  ops::rmsnorm( this->state_.xb, x, layer_weights.rms_att_weight, dim );

  // qkv matmuls for this position
  ops::matmul( this->state_.q, this->state_.xb, layer_weights.wq, dim, dim );
  ops::matmul( this->state_.k, this->state_.xb, layer_weights.wk, dim, dim );
  ops::matmul( this->state_.v, this->state_.xb, layer_weights.wv, dim, dim );

  do_rope<<<this->config_.n_heads, head_size / 2>>>(
    head_size, this->config_.n_heads, freq_cis_real_row, freq_cis_imag_row, this->state_.q, this->state_.k );

  DType* k_cache_pos = this->kv_cache_.key( layer_num, token_pos );
  DType* v_cache_pos = this->kv_cache_.value( layer_num, token_pos );

  // save key,value at this time step (pos) to our kv cache
  CHECK_CUDA( cudaMemcpy( k_cache_pos, this->state_.k, dim * sizeof( DType ), cudaMemcpyDeviceToDevice ) );
  CHECK_CUDA( cudaMemcpy( v_cache_pos, this->state_.v, dim * sizeof( DType ), cudaMemcpyDeviceToDevice ) );

  // multihead attention. for each head and for each token up to and including the current one
  attention_0<<<token_pos + 1, this->config_.n_heads>>>( this->state_.q,
                                                         this->kv_cache_.buffer_,
                                                         this->state_.att,
                                                         layer_num,
                                                         this->config_.n_layers,
                                                         this->config_.seq_len,
                                                         head_size,
                                                         dim );

  // softmax
  attention_softmax(
    this->state_.att, token_pos, this->config_.seq_len, this->config_.n_heads, this->state_.temp_softmax );

  CHECK_CUDA( cudaMemset( this->state_.xb, 0, dim * sizeof( DType ) ) );

  //  attention_2<<<token_pos + 1, this->config_.n_heads>>>( this->state_.att,
  //                                                         this->kv_cache_.buffer_,
  //                                                         this->state_.xb,
  //                                                         layer_num,
  //                                                         this->config_.n_layers,
  //                                                         this->config_.seq_len,
  //                                                         head_size,
  //                                                         dim );

  attention_2_v2<<<head_size, this->config_.n_heads>>>( this->state_.att,
                                                        this->kv_cache_.buffer_,
                                                        this->state_.xb,
                                                        layer_num,
                                                        this->config_.n_layers,
                                                        this->config_.seq_len,
                                                        head_size,
                                                        dim,
                                                        token_pos + 1 );
  // end of multihead attention

  // final matmul to get the output of the attention
  ops::matmul( this->state_.xb2, this->state_.xb, layer_weights.wo, dim, dim );

  // residual connection back into x
  ops::accum( x, this->state_.xb2, dim );

  // ffn rmsnorm
  ops::rmsnorm( this->state_.xb, x, layer_weights.rms_ffn_weight, dim );

  // now for ffn in we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
  // first calculate self.w1(x) and self.w3(x)
  ops::matmul( this->state_.hb, this->state_.xb, layer_weights.w1, dim, hidden_dim );
  ops::matmul( this->state_.hb2, this->state_.xb, layer_weights.w3, dim, hidden_dim );

  ops::silu( this->state_.hb, this->state_.hb2, hidden_dim );

  // final matmul to get the output of the ffn
  ops::matmul( this->state_.xb, this->state_.hb, layer_weights.w2, hidden_dim, dim );

  // residual connection
  ops::accum( x, this->state_.xb, dim );
}

template<typename DType>
void Llama2<DType>::pass_end()
{
  DType* x = this->state_.x;

  // final rmsnorm
  ops::rmsnorm( this->state_.x, this->state_.x, this->base_weights_.rms_final_weight, this->config_.dim );

  // classifier into logits
  ops::matmul(
    this->state_.logits, this->state_.x, this->base_weights_.wcls, this->config_.dim, this->config_.vocab_size );
}

template<typename DType>
int Llama2<DType>::forward( const int token )
{
  if ( token_pos_ >= this->config_.seq_len ) {
    return 2; /* EOS */
  }

  pass_begin( token );

  for ( int layer_num = this->start_layer_num_; layer_num <= this->end_layer_num_; layer_num++ ) {
    transformer_layer( layer_num, token_pos_ );
  }

  pass_end();

  int next_token;
  int* next_token_device;
  CHECK_CUDA( cudaMalloc( &next_token_device, sizeof( int ) ) );

  if ( temperature_ == 0.0f ) {
    // greedy argmax sampling
    ops::argmax( this->state_.logits, this->config_.vocab_size, next_token_device );
  } else {
    // apply the temperature to the logits
    for ( int q = 0; q < this->config_.vocab_size; q++ ) {
      this->state_.logits[q] /= temperature_;
    }

    // apply softmax to the logits to get the probabilities for next token
    ops::softmax( this->state_.logits, this->config_.vocab_size );

    // we now want to sample from this distribution to get the next token
    ops::sample( this->state_.logits, this->config_.vocab_size, next_token_device );
  }

  token_pos_++;
  CHECK_CUDA( cudaMemcpy( &next_token, next_token_device, sizeof( int ), cudaMemcpyDeviceToHost ) );
  return next_token;
}

template class Llama2<float>;
template class Llama2<__half>;

} // namespace glinthawk::models::llama2::cuda
