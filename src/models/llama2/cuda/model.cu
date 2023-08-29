#include "model.cuh"

#include <cstdint>
#include <fcntl.h>
#include <memory>
#include <optional>
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
Context<DType>::Context( const Config& config )
  : storage_( [&]() -> decltype( storage_ ) {
    DType* ptr;
    CHECK_CUDA( cudaMalloc( &ptr, InferenceContext<DType>::context_size( config ) ) );
    return { ptr, cuda_deleter };
  }() )
{
  this->buffer_ = storage_.get();
}

template<typename DType>
Llama2<DType>::~Llama2()
{
  ops::destroy();
}

template<typename DType>
Llama2<DType> Llama2<DType>::load_model( const filesystem::path& model_path,
                                         const int32_t start_layer,
                                         const int32_t end_layer )
{
  ops::init();

  const string filename_suffix = "_" + dtype_str<DType>();
  const auto config_path = model_path / "CONFIG";
  const auto base_path = model_path / ( "BASEWEIGHTS" + filename_suffix );

  llama2::Config config { config_path, start_layer, end_layer };

  const int32_t layer_count = end_layer - start_layer + 1;

  const auto run_state_size = RunState<DType>::state_size( config );
  const auto base_size = BaseWeights<DType>::base_size( config );
  const auto layer_size = LayerWeights<DType>::layer_size( config );

  DType* base_raw_ptr;
  DType* layers_raw_ptr;
  DType* run_state_raw_ptr;

  // Allocate memory for the base weights
  CHECK_CUDA( cudaMalloc( &base_raw_ptr, base_size ) );
  unique_ptr<DType, void ( * )( DType* )> base { base_raw_ptr, cuda_deleter };

  // Allocate memory for the layers
  CHECK_CUDA( cudaMalloc( &layers_raw_ptr, layer_size * layer_count ) );
  unique_ptr<DType, void ( * )( DType* )> layers { layers_raw_ptr, cuda_deleter };

  // Allocate memory for the run state
  CHECK_CUDA( cudaMalloc( &run_state_raw_ptr, run_state_size ) );
  unique_ptr<DType, void ( * )( DType* )> run_state { run_state_raw_ptr, cuda_deleter };

  // Load the model

  // (1) loading the base weights
  {
    CHECK_EQ( filesystem::file_size( base_path ), base_size ) << "Base weights are not the expected size.";

    FileDescriptor base_fd { CHECK_SYSCALL( "open", open( base_path.c_str(), O_RDONLY ) ) };
    MMap_Region base_mmap { nullptr, base_size, PROT_READ, MAP_PRIVATE, base_fd.fd_num(), 0 };
    CHECK_CUDA( cudaMemcpy( base.get(), base_mmap.addr(), base_size, cudaMemcpyHostToDevice ) );

    LOG( INFO ) << "Loaded base weights (" << base_size << " bytes).";
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

    LOG( INFO ) << "Loaded layer " << i << " (" << layer_size << " bytes).";
  }

  return { config, move( base ), move( layers ), move( run_state ) };
}

template<typename DType>
__global__ void do_rope( const uint64_t head_size,
                         const uint64_t n_heads,
                         const DType* freq_cis_real_row,
                         const DType* freq_cis_imag_row,
                         DType* state_q,
                         DType* state_k )
{
  const uint64_t head_num = blockIdx.x;
  const uint64_t elem_idx = 2 * threadIdx.x;

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
__global__ void find_max_for_rows( const DType* att,
                                   DType* output,
                                   const uint64_t token_pos,
                                   const uint64_t n_heads,
                                   const uint64_t seq_len )
{
  const uint64_t head_num = threadIdx.x;
  att += head_num * seq_len;

  DType max_value = att[0];
  for ( uint64_t i = 1; i <= token_pos; i++ ) {
    if constexpr ( is_same_v<DType, __half> ) {
      max_value = __hmax( max_value, att[i] );
    } else {
      max_value = max( max_value, att[i] );
    }
  }

  output[head_num] = max_value;
}

template<typename DType>
__global__ void subtract_and_expf( const DType* values, DType* att, const uint64_t n_heads, const uint64_t seq_len )
{
  const uint64_t head_num = threadIdx.x;
  const uint64_t token_pos = blockIdx.x;

  att += head_num * seq_len;
  att[token_pos] = expf( att[token_pos] - values[head_num] );
}

template<typename DType>
__global__ void sum_rows( DType* att,
                          DType* output,
                          const uint64_t token_pos,
                          const uint64_t n_heads,
                          const uint64_t seq_len )
{
  const uint64_t head_num = threadIdx.x;
  att += head_num * seq_len;

  DType sum = 0.0;
  for ( uint64_t i = 0; i <= token_pos; i++ ) {
    sum += att[i];
  }

  output[head_num] = sum;
}

template<typename DType>
__global__ void normalize_by_sum( DType* att, const DType* sums, const uint64_t n_heads, const uint64_t seq_len )
{
  const uint64_t head_num = threadIdx.x;
  const uint64_t token_pos = blockIdx.x;

  att += head_num * seq_len;
  att[token_pos] /= sums[head_num];
}

template<typename DType>
void attention_softmax( DType* att,
                        const uint64_t token_pos,
                        const uint64_t seq_len,
                        const uint64_t n_heads,
                        DType* temp_buffer )
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
void Llama2<DType>::pass_begin( const uint32_t token )
{
  // copy the token embedding into the state
  const DType* content_row = this->base_weights_.token_embedding_table + token * this->config_.dim;
  CHECK_CUDA(
    cudaMemcpy( this->state_.x, content_row, this->config_.dim * sizeof( DType ), cudaMemcpyDeviceToDevice ) );
}

template<typename DType>
void Llama2<DType>::transformer_layer( const int32_t layer_num, const uint64_t token_pos, ContextType& context )
{
  DType* const x = this->state_.x;
  const uint64_t dim = this->config_.dim;
  const uint64_t hidden_dim = this->config_.hidden_dim;
  const uint64_t head_size = dim / this->config_.n_heads;

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

  DType* k_cache_pos = context.key( this->config_, layer_num, token_pos, 0 );
  DType* v_cache_pos = context.value( this->config_, layer_num, token_pos, 0 );

  // save key,value at this time step (pos) to our kv cache
  CHECK_CUDA( cudaMemcpy( k_cache_pos, this->state_.k, dim * sizeof( DType ), cudaMemcpyDeviceToDevice ) );
  CHECK_CUDA( cudaMemcpy( v_cache_pos, this->state_.v, dim * sizeof( DType ), cudaMemcpyDeviceToDevice ) );

  // multihead attention. for each head and for each token up to and including the current one
  ops::attention_0_gemm( this->state_.q,
                         context.buffer_ + ( layer_num - this->config_.start_layer_num ) * ( dim * 2 ),
                         this->state_.att,
                         this->config_.end_layer_num - this->config_.start_layer_num + 1,
                         this->config_.seq_len,
                         head_size,
                         this->config_.n_heads,
                         token_pos + 1 );

  // softmax
  attention_softmax(
    this->state_.att, token_pos, this->config_.seq_len, this->config_.n_heads, this->state_.temp_softmax );

  ops::attention_2_gemm( this->state_.att,
                         context.buffer_ + ( layer_num - this->config_.start_layer_num ) * ( dim * 2 ) + dim,
                         this->state_.xb,
                         this->config_.end_layer_num - this->config_.start_layer_num + 1,
                         this->config_.seq_len,
                         head_size,
                         this->config_.n_heads,
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
  // final rmsnorm
  ops::rmsnorm( this->state_.x, this->state_.x, this->base_weights_.rms_final_weight, this->config_.dim );

  // classifier into logits
  ops::matmul(
    this->state_.logits, this->state_.x, this->base_weights_.wcls, this->config_.dim, this->config_.vocab_size );
}

template<typename DType>
uint32_t extract_token( const RunState<DType>& state, const Config& config, const float temp )
{
  uint32_t next_token;

  if ( temp == 0.0f ) {
    // greedy argmax sampling
    next_token = ops::argmax( state.logits, config.vocab_size );
  } else {
    throw runtime_error( "not implemented" );
  }

  return next_token;
}

template<typename DType>
InferenceState Llama2<DType>::forward( InferenceState&& state, ContextType& context )
{
  CHECK_EQ( state.next_layer(), this->config_.start_layer_num ) << "next_layer must be the start layer";

  if ( state.next_layer() == 0 ) {
    pass_begin( state.token() );
  } else {
    // load the activations
    CHECK_CUDA( cudaMemcpy(
      this->state_.x, state.activations().ptr.get(), this->config_.dim * sizeof( DType ), cudaMemcpyHostToDevice ) );
  }

  for ( int layer_num = this->config_.start_layer_num; layer_num <= this->config_.end_layer_num; layer_num++ ) {
    transformer_layer( layer_num, state.token_pos(), context );
  }

  if ( this->config_.end_layer_num == this->config_.n_layers - 1 ) {
    pass_end();

    InferenceState result { move( state ) };
    result.set_token( extract_token( this->state_, this->config_, result.temperature() ) );
    result.set_token_pos( result.token_pos() + 1 );
    result.set_next_layer( 0 );
    result.set_activations( {} );
    return result;
  }

  DataBuffer activations { is_same_v<DType, float> ? glinthawk::models::DataType::Type::Float32
                                                   : glinthawk::models::DataType::Type::Float16,
                           make_unique<uint8_t[]>( this->config_.dim * sizeof( DType ) ),
                           this->config_.dim };

  CHECK_CUDA(
    cudaMemcpy( activations.ptr.get(), this->state_.x, this->config_.dim * sizeof( DType ), cudaMemcpyDeviceToHost ) );

  InferenceState result { move( state ) };
  result.set_next_layer( static_cast<uint32_t>( this->config_.end_layer_num ) + 1 );
  result.set_activations( move( activations ) );
  return result;
}

template<typename DType>
uint32_t Llama2<DType>::forward( const uint32_t token )
{
  throw runtime_error( "not implemented" );
}

template class Context<__half>;
template class Llama2<__half>;

} // namespace glinthawk::models::llama2::cuda
