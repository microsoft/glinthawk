#include "model.cuh"

#include <cstdint>
#include <fcntl.h>
#include <memory>
#include <optional>
#include <set>
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
  : storage_( [&] {
    DType* ptr;
    CUDA_CHECK( cudaMalloc( &ptr, InferenceContext<DType>::context_size( config ) ) );
    return { ptr, cuda_deleter };
  }() )
  , glinthawk::models::llama2::InferenceContext<DType>::buffer_( storage_.get() )
{
}

template<typename DType>
Llama2<DType>::~Llama2()
{
  ops::destroy();
}

template<typename DType>
unique_ptr<Llama2<DType>> Llama2<DType>::load( const filesystem::path& model_path,
                                               const int32_t start_layer,
                                               const int32_t end_layer_raw,
                                               const uint64_t kv_prompt_limit,
                                               const uint64_t concurrency_limit )
{
  ops::init( concurrency_limit );

  const string filename_suffix = "_" + dtype_str<DType>();
  const auto config_path = model_path / "CONFIG";
  const auto base_path = model_path / ( "BASEWEIGHTS" + filename_suffix );

  llama2::Config config { config_path, start_layer, end_layer, kv_prompt_limit, concurrency_limit };

  CHECK_GT( 1025, config.n_heads ) << "Attention softmax has n_heads threads, and this cannot surpass 1024.";
  CHECK_GT( 1 << 16, config.n_heads ) << "RoPE has n_heads blocks, and this cannot surpass 2^16.";
  CHECK_GT( 1025, config.dim / config.n_heads / 2 ) << "RoPE has head_size / 2 threads, and this cannot surpass 1024.";
  CHECK_GT( 1 << 16, config.seq_len ) << "Attention softmax has seq_len blocks, and this cannot surpass 2^16.";
  const size_t tpb = ops::TPB;
  CHECK_GT( 1025, tpb ) << "Threads per block cannot surpass 1024.";
  CHECK_GT( 1 << 16, ( config.dim + tpb - 1 ) / tpb ) << "RMS Norm blocks cannot surpass 2^16.";
  CHECK_GT( 1 << 16, ( config.dim * config.concurrency_limit + tpb - 1 ) / tpb ) << "Accum blocks cannot surpass 2^16.";
  CHECK_GT( 1 << 16, ( config.hidden_dim * config.concurrency_limit + tpb - 1 ) / tpb )
    << "Silu blocks cannot surpass 2^16.";

  const int32_t layer_count = end_layer - start_layer + 1;

  const auto run_state_size = RunState<DType>::state_size( config );
  const auto base_size = BaseWeights<DType>::base_size( config );
  const auto layer_size = LayerWeights<DType>::layer_size( config );

  DType* base_raw_ptr;
  DType* layers_raw_ptr;
  DType* run_state_raw_ptr;

  // Allocate memory for the base weights
  ops::CHECK_CUDA( cudaMalloc( &base_raw_ptr, base_size ) );
  unique_ptr<DType, void ( * )( DType* )> base { base_raw_ptr, cuda_deleter };

  // Allocate memory for the layers
  ops::CHECK_CUDA( cudaMalloc( &layers_raw_ptr, layer_size * layer_count ) );
  unique_ptr<DType, void ( * )( DType* )> layers { layers_raw_ptr, cuda_deleter };

  // Allocate memory for the run state
  ops::CHECK_CUDA( cudaMalloc( &run_state_raw_ptr, run_state_size ) );
  unique_ptr<DType, void ( * )( DType* )> run_state { run_state_raw_ptr, cuda_deleter };

  // Load the model

  // (1) loading the base weights
  {
    CHECK_EQ( filesystem::file_size( base_path ), base_size ) << "Base weights are not the expected size.";

    FileDescriptor base_fd { CHECK_SYSCALL( "open", open( base_path.c_str(), O_RDONLY ) ) };
    MMap_Region base_mmap { nullptr, base_size, PROT_READ, MAP_PRIVATE, base_fd.fd_num(), 0 };
    ops::CHECK_CUDA( cudaMemcpy( base.get(), base_mmap.addr(), base_size, cudaMemcpyHostToDevice ) );

    LOG( INFO ) << "Loaded base weights (" << base_size << " bytes).";
  }

  // (2) load the layers
  for ( auto i = start_layer; i <= end_layer; i++ ) {
    const auto layer_path = model_path / ( "LAYER" + to_string( i ) + filename_suffix );

    CHECK_EQ( filesystem::file_size( layer_path ), layer_size ) << "Layer " << i << " is not the expected size.";

    FileDescriptor layer_fd { CHECK_SYSCALL( "open", open( layer_path.c_str(), O_RDONLY ) ) };
    MMap_Region layer_mmap { nullptr, layer_size, PROT_READ, MAP_PRIVATE, layer_fd.fd_num(), 0 };

    ops::CHECK_CUDA( cudaMemcpy( reinterpret_cast<uint8_t*>( layers.get() ) + ( i - start_layer ) * layer_size,
                                 layer_mmap.addr(),
                                 layer_size,
                                 cudaMemcpyHostToDevice ) );

    LOG( INFO ) << "Loaded layer " << i << " (" << layer_size << " bytes).";
  }

  auto model = unique_ptr<Llama2<DType>>( new Llama2<DType> {
    config, move( base ), move( layers ), move( run_state ) } );

  return model;
}

template<typename DType>
void Llama2<DType>::pass_begin( const std::vector<uint32_t>& token )
{
  // copy the token embedding into the state
  for ( size_t i = 0; i < token.size(); i++ ) {
    CHECK_LT( token[i], this->config_.vocab_size ) << "token index must not surpass vocab size";
    const DType* content_row = this->base_weights_.token_embedding_table + token[i] * this->config_.dim;
    ops::CHECK_CUDA( cudaMemcpyAsync( this->state_.x + i * this->config_.dim,
                                      content_row,
                                      this->config_.dim * sizeof( DType ),
                                      cudaMemcpyDeviceToDevice ) );
  }
}

template<typename DType>
void Llama2<DType>::transformer_layer( const int32_t layer_num ) // NEEDS CONTEXT
{
  const uint64_t dim = this->config_.dim;
  const uint64_t kv_dim = this->config_.kv_dim;
  const uint64_t gqa_size = this->config_.gqa_size;
  const uint64_t hidden_dim = this->config_.hidden_dim;
  const uint64_t head_size = dim / this->config_.n_heads;
  const uint64_t n_heads = this->config_.n_heads;
  const uint64_t n_kv_heads = this->config_.n_kv_heads;
  const uint64_t seq_len = this->config_.seq_len;
  const uint64_t n_layers_loaded = this->kv_cache_.n_layers_;
  const uint64_t kv_prompt_limit = this->config_.kv_prompt_limit;
  const uint64_t curr_conc_lvl = this->curr_concurrency_size;

  const auto& layer_weights = this->layer_weights_[layer_num];

  // attention rmsnorm
  ops::rmsnorm( this->state_.xb, this->state_.x, layer_weights.rms_att_weight, dim, curr_conc_lvl );

  // qkv matmuls for this position
  ops::matmul( this->state_.q, this->state_.xb, layer_weights.wq, curr_conc_lvl, dim, dim );
  ops::matmul( this->state_.k, this->state_.xb, layer_weights.wk, curr_conc_lvl, dim, kv_dim );
  ops::matmul( this->state_.v, this->state_.xb, layer_weights.wv, curr_conc_lvl, dim, kv_dim );

  ops::apply_rope( head_size,
                   n_kv_heads,
                   gqa_size,
                   curr_conc_lvl,
                   this->token_pos_,
                   this->base_weights_.freq_cis_real,
                   this->base_weights_.freq_cis_imag,
                   this->state_.q,
                   this->state_.k );

  // save key,value at each time step (pos) to our kv cache
  ops::copy_kv_cache( this->state_.k,
                      this->state_.v,
                      this->kv_cache_.key( layer_num, 0, 0 ),
                      this->kv_cache_.value( layer_num, 0, 0 ),
                      kv_dim,
                      n_layers_loaded,
                      curr_conc_lvl,
                      kv_prompt_limit,
                      this->id_allocation_,
                      this->token_pos_ );

  // multihead attention. for each head and for each token up to and including the current one
  ops::attention_0_gemm( this->state_.q,
                         this->kv_cache_.key( layer_num, 0, 0, 0 ),
                         this->state_.att,
                         n_layers_loaded,
                         seq_len,
                         head_size,
                         n_kv_heads,
                         gqa_size,
                         curr_conc_lvl,
                         kv_prompt_limit,
                         this->id_allocation_,
                         this->token_pos_ );

  // softmax
  ops::attention_softmax( this->state_.att,
                          this->token_pos_,
                          seq_len,
                          n_heads,
                          this->state_.temp_softmax,
                          curr_conc_lvl );

  ops::attention_2_gemm( this->state_.att,
                         this->kv_cache_.value( layer_num, 0, 0, 0 ),
                         this->state_.xb,
                         n_layers_loaded,
                         seq_len,
                         head_size,
                         n_kv_heads,
                         gqa_size,
                         curr_conc_lvl,
                         kv_prompt_limit,
                         this->id_allocation_,
                         this->token_pos_ );
  // end of multihead attention

  // final matmul to get the output of the attention
  ops::matmul( this->state_.xb2, this->state_.xb, layer_weights.wo, curr_conc_lvl, dim, dim );

  // residual connection back into x
  ops::accum( this->state_.x, this->state_.xb2, dim, curr_conc_lvl );

  // ffn rmsnorm
  ops::rmsnorm( this->state_.xb, this->state_.x, layer_weights.rms_ffn_weight, dim, curr_conc_lvl );

  // now for ffn in we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
  // first calculate self.w1(x) and self.w3(x)
  ops::matmul( this->state_.hb, this->state_.xb, layer_weights.w1, curr_conc_lvl, dim, hidden_dim );
  ops::matmul( this->state_.hb2, this->state_.xb, layer_weights.w3, curr_conc_lvl, dim, hidden_dim );

  ops::silu( this->state_.hb, this->state_.hb2, hidden_dim, curr_conc_lvl );

  // final matmul to get the output of the ffn
  ops::matmul( this->state_.xb, this->state_.hb, layer_weights.w2, curr_conc_lvl, hidden_dim, dim );

  // residual connection
  ops::accum( this->state_.x, this->state_.xb, dim, curr_conc_lvl );
}

template<typename DType>
void Llama2<DType>::pass_end()
{
  // final rmsnorm
  ops::rmsnorm( this->state_.x,
                this->state_.x,
                this->base_weights_.rms_final_weight,
                this->config_.dim,
                this->curr_concurrency_size );

  // classifier into logits
  ops::matmul( this->state_.logits,
               this->state_.x,
               this->base_weights_.wcls,
               this->curr_concurrency_size,
               this->config_.dim,
               this->config_.vocab_size );
}

template<typename DType>
uint32_t extract_token( const RunState<DType>& state,
                        const Config& config,
                        const float temp,
                        const uint64_t batch_index = 0 )
{
  uint32_t next_token;

  if ( temp == 0.0f ) {
    // greedy argmax sampling
    next_token = ops::argmax( state.logits + batch_index * config.vocab_size, config.vocab_size );
  } else {
    // apply the temperature to the logits
    for ( auto q = batch_index * config.vocab_size; q < ( batch_index + 1 ) * config.vocab_size; q++ ) {
      state.logits[q] /= temp;
    }

    // apply softmax to the logits to get the probabilities for next token
    ops::softmax( state.logits + batch_index * config.vocab_size, config.vocab_size );

    // we now want to sample from this distribution to get the next token
    next_token = ops::sample( state.logits + batch_index * config.vocab_size, config.vocab_size );
  }

  return next_token;
}

template<typename DType>
std::vector<uint32_t> extract_batch_token( const RunState<DType>& state,
                                           const Config& config,
                                           const std::vector<float>& temp )
{
  std::vector<uint32_t> next_tokens;
  for ( size_t i = 0; i < temp.size(); i++ )
    next_tokens.push_back( extract_token( state, config, temp[i], i ) );
  return next_tokens;
}

template<typename DType>
std::vector<InferenceState> Llama2<DType>::forward(
  const std::vector<std::reference_wrapper<const InferenceState>>& inference_state_s,
  const std::vector<uint32_t>& prompt_id_s )
{
  CHECK_GT( inference_state_s.size(), 0 ) << "batch size must be at least 1";
  CHECK_EQ( inference_state_s.size(), prompt_id_s.size() ) << "token size must be the same as prompt_id size";
  CHECK_LE( inference_state_s.size(), this->config_.concurrency_limit )
    << "current batch cannot be larger than max concurrency size";
  for ( auto& item : inference_state_s ) {
    CHECK_EQ( item.get().next_layer(), this->start_layer_num_ ) << "next_layer must be the start layer";
  }

  for ( size_t i = 0; i < prompt_id_s.size(); i++ ) {
    this->id_allocation_[i] = prompt_id_s[i];
    this->token_pos_[i] = inference_state_s[i].get().token_pos();
    CHECK_LT( this->token_pos_[i], this->config_.seq_len ) << "token position cannot be larger than sequence length";
  }
  this->curr_concurrency_size = inference_state_s.size();

  if ( inference_state_s[0].get().next_layer() == 0 ) {
    std::vector<uint32_t> token_vector;
    for ( size_t i = 0; i < inference_state_s.size(); i++ )
      token_vector.push_back( inference_state_s[i].get().token() );
    pass_begin( token_vector );
  } else {
    for ( size_t i = 0; i < inference_state_s.size(); i++ )
      // load the activations
      ops::CHECK_CUDA( cudaMemcpyAsync( this->state_.x + i * this->config_.dim * sizeof( DType ),
                                        inference_state_s[i].get().activations().ptr.get(),
                                        this->config_.dim * sizeof( DType ),
                                        cudaMemcpyHostToDevice ) );
  }

  for ( int layer_num = this->config_.start_layer_num; layer_num <= this->config_.end_layer_num; layer_num++ ) {
    transformer_layer( layer_num );
  }

  std::vector<InferenceState> token_vector;

  if ( this->config_.end_layer_num == this->config_.n_layers - 1 ) {
    pass_end();

    std::vector<float> batch_temps;
    for ( size_t i = 0; i < inference_state_s.size(); i++ )
      batch_temps.push_back( inference_state_s[i].get().temperature() );

    std::vector<uint32_t> next_tokens = extract_batch_token( this->state_, this->config_, batch_temps );

    for ( size_t i = 0; i < inference_state_s.size(); i++ )
      token_vector.emplace_back( inference_state[i].prompt_id(),             // prompt_id
                                 inference_state[i].model_id(),              // model_id
                                 next_tokens[i],                             // token
                                 inference_state_s[i].get().token_pos() + 1, // token_pos
                                 0,                                          // next_layer
                                 inference_state_s[i].get().temperature(),   // temperature
                                 DataBuffer {}                               // activations
      );

    return token_vector;
  }

  for ( size_t i = 0; i < inference_state_s.size(); i++ ) {
    DataBuffer activations { is_same_v<DType, float> ? DataType::Type::Float32 : DataType::Type::Float16,
                             make_unique<uint8_t[]>( this->config_.dim * sizeof( DType ) ),
                             this->config_.dim };

    ops::CHECK_CUDA( cudaMemcpy( activations.ptr.get(),
                                 this->state_.x + i * this->config_.dim * sizeof( DType ),
                                 this->config_.dim * sizeof( DType ),
                                 cudaMemcpyDeviceToHost ) );

    token_vector.emplace_back( inference_state_s[i].prompt_id(),                         // prompt_id
                               inference_state_s[i].get().model_id(),                    // model_id
                               inference_state_s[i].get().token(),                       // token
                               inference_state_s[i].get().token_pos(),                   // token_pos
                               static_cast<uint32_t>( this->config_.end_layer_num ) + 1, // next_layer
                               inference_state_s[i].get().temperature(),                 // temperature
                               move( activations )                                       // activations
    );
  }

  return token_vector;
}

template<typename DType>
std::vector<uint32_t> Llama2<DType>::forward( const std::vector<uint32_t>& token_s,
                                              const std::vector<uint32_t>& prompt_id_s,
                                              const std::vector<uint32_t>& token_pos_s )
{
  CHECK_GT( token_s.size(), 0 ) << "batch size must be at least 1";
  CHECK_EQ( token_s.size(), prompt_id_s.size() ) << "token size must be the same as prompt_id size";
  CHECK_EQ( token_s.size(), token_pos_s.size() ) << "token size must be the same as token_pos size";
  CHECK_LE( token_s.size(), this->config_.concurrency_limit )
    << "current batch cannot be larger than max concurrency size";

  for ( size_t i = 0; i < prompt_id_s.size(); i++ ) {
    this->id_allocation_[i] = prompt_id_s[i];
    this->token_pos_[i] = token_pos_s[i];
    CHECK_LT( token_pos_s[i], this->config_.seq_len ) << "token position cannot be larger than sequence length";
  }
  this->curr_concurrency_size = token_s.size();

  pass_begin( token_s );

  for ( int layer_num = this->config_.start_layer_num; layer_num <= this->config_.end_layer_num; layer_num++ ) {
    transformer_layer( layer_num );
  }

  pass_end();

  return extract_batch_token( this->state_, this->config_, std::vector<float>( token_s.size(), temperature_ ) );
}

template<typename DType>
std::vector<InferenceState> Llama2<DType>::forward( const std::vector<InferenceState>& inference_state_s,
                                                           const std::vector<uint32_t>& prompt_id_s )
{
  std::vector<std::reference_wrapper<const InferenceState>> res;
  for ( auto& state : inference_state_s )
    res.push_back( std::ref( state ) );
  return forward( res, prompt_id_s );
}

template<typename DType>
InferenceState Llama2<DType>::forward( const InferenceState& inference_state, const uint32_t& prompt_id )
{
  std::vector<std::reference_wrapper<const InferenceState>> token_vector;
  token_vector.push_back( std::ref( inference_state ) );
  std::vector<uint32_t> prompt_id_vector = { prompt_id };
  return move( forward( token_vector, prompt_id_vector )[0] );
}

template<typename DType>
uint32_t Llama2<DType>::forward( const uint32_t& token, const uint32_t& prompt_id, const uint32_t& token_pos )
{
  std::vector<uint32_t> token_vector = { token };
  std::vector<uint32_t> prompt_id_vector = { prompt_id };
  std::vector<uint32_t> token_pos_vector = { token_pos };
  return forward( token_vector, prompt_id_vector, token_pos_vector )[0];
}

template class Llama2<__half>;

} // namespace glinthawk::models::llama2::cuda
