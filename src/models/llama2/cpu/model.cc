#include "model.hh"

#include <cstdint>
#include <fcntl.h>
#include <memory>
#include <optional>
#include <set>
#include <source_location>

#include "util/exception.hh"
#include "util/file_descriptor.hh"
#include "util/ring_buffer.hh"

#include "models/common/cpu/ops.hh"
#include "models/llama2/base.hh"

using namespace std;
using namespace glinthawk::models;
using namespace glinthawk::models::common::cpu;

namespace glinthawk::models::llama2::cpu {

template<typename DType>
void basic_deleter( DType* ptr )
{
  free( ptr );
}

template<typename DType>
string dtype_str()
{
  if constexpr ( is_same_v<DType, float> ) {
    return "FP32"s;
  } else if constexpr ( is_same_v<DType, _Float16> ) {
    return "FP16"s;
  } else {
    return "unknown"s;
  }
}

template<typename DType>
Context<DType>::Context( const Config& config )
  : storage_( reinterpret_cast<DType*>( malloc( InferenceContext<DType>::context_size( config ) ) ),
              basic_deleter<DType> )
{
  this->buffer_ = storage_.get();
}

template<typename DType>
Llama2<DType>::Llama2( const filesystem::path& model_path,
                       const uint32_t start_layer,
                       const uint32_t end_layer,
                       const uint64_t concurrency_limit )
{
  const string filename_suffix = "_"s + dtype_str<DType>();
  const auto config_path = model_path / "CONFIG";
  const auto base_path = model_path / ( "BASEWEIGHTS" + filename_suffix );

  llama2::Config config { config_path, start_layer, end_layer, concurrency_limit };

  const auto run_state_size = RunState<DType>::state_size( config );
  const auto base_size = BaseWeights<DType>::base_size( config );
  const auto layer_size = LayerWeights<DType>::layer_size( config );

  DType* run_state_buffer = reinterpret_cast<DType*>( malloc( run_state_size ) );
  DType* base_weights_buffer = reinterpret_cast<DType*>( malloc( base_size ) );
  DType* layers_buffer = reinterpret_cast<DType*>( malloc( layer_size * config.n_layers_loaded() ) );

  memset( run_state_buffer, 0, run_state_size );

  // Allocate memory for the base weights, layers and run state
  unique_ptr<DType, void ( * )( DType* )> base { base_weights_buffer, basic_deleter<DType> };
  unique_ptr<DType, void ( * )( DType* )> layers { layers_buffer, basic_deleter<DType> };
  unique_ptr<DType, void ( * )( DType* )> run_state { run_state_buffer, basic_deleter<DType> };

  // Load the model

  // (1) loading the base weights
  {
    CHECK_EQ( filesystem::file_size( base_path ), base_size ) << "Base weights are not the expected size.";

    FileDescriptor base_fd { CHECK_SYSCALL( "open", open( base_path.c_str(), O_RDONLY ) ) };
    MMap_Region base_mmap { nullptr, base_size, PROT_READ, MAP_PRIVATE, base_fd.fd_num(), 0 };
    memcpy( base.get(), base_mmap.addr(), base_size );

    LOG( INFO ) << "Loaded base weights (" << base_size << " bytes).";
  }

  // (2) load the layers
  for ( auto i = config.start_layer_num; i <= config.end_layer_num; i++ ) {
    const auto layer_path = model_path / ( "LAYER" + to_string( i ) + filename_suffix );

    CHECK_EQ( filesystem::file_size( layer_path ), layer_size ) << "Layer " << i << " is not the expected size.";

    FileDescriptor layer_fd { CHECK_SYSCALL( "open", open( layer_path.c_str(), O_RDONLY ) ) };
    MMap_Region layer_mmap { nullptr, layer_size, PROT_READ, MAP_PRIVATE, layer_fd.fd_num(), 0 };

    memcpy( reinterpret_cast<uint8_t*>( layers.get() ) + ( i - config.start_layer_num ) * layer_size,
            layer_mmap.addr(),
            layer_size );

    LOG( INFO ) << "Loaded layer " << i << " (" << layer_size << " bytes).";
  }

  this->init( config, move( base ), move( layers ), move( run_state ) );
}

template<typename DType>
void Llama2<DType>::pass_begin( const vector<uint32_t>& token )
{
  // copy the token embedding into the state
  for ( size_t i = 0; i < token.size(); i++ ) {
    CHECK_LT( token[i], this->config_.vocab_size ) << "token index must not surpass vocab size";
    const DType* content_row = this->base_weights_.token_embedding_table + token[i] * this->config_.dim;
    memcpy( this->state_.x + i * this->config_.dim, content_row, this->config_.dim * sizeof( DType ) );
  }
}

template<typename DType>
void Llama2<DType>::transformer_layer( const int32_t layer_num )
{
  const uint64_t dim = this->config_.dim;
  const uint64_t kv_dim = this->config_.kv_dim;
  const uint64_t gqa_size = this->config_.gqa_size;
  const uint64_t hidden_dim = this->config_.hidden_dim;
  const uint64_t head_size = dim / this->config_.n_heads;
  const uint64_t n_heads = this->config_.n_heads;
  const uint64_t n_kv_heads = this->config_.n_kv_heads;
  const uint64_t seq_len = this->config_.seq_len;
  const uint64_t curr_conc_lvl = this->state_.curr_concurrency_size;

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
                   this->state_.batch_token_positions,
                   this->base_weights_.freq_cis_real,
                   this->base_weights_.freq_cis_imag,
                   this->state_.q,
                   this->state_.k );

  // save key,value at each time step (pos) to our kv cache
  ops::copy_kv_cache( this->state_.batch_context_pointers,
                      this->state_.k,
                      this->state_.v,
                      kv_dim,
                      curr_conc_lvl,
                      this->state_.batch_token_positions );

  // <multihead attention> for each head and for each token up to and including the current one
  ops::attention_0_gemm_fast( this->state_.q,
                              this->state_.batch_context_pointers,
                              this->state_.att,
                              seq_len,
                              head_size,
                              n_kv_heads,
                              gqa_size,
                              curr_conc_lvl,
                              this->state_.batch_token_positions );

  // softmax
  ops::attention_softmax(
    this->state_.att, this->state_.batch_token_positions, seq_len, n_heads, this->state_.temp_softmax, curr_conc_lvl );

  ops::attention_2_gemm_fast( this->state_.att,
                              this->state_.batch_context_pointers,
                              this->state_.xb,
                              seq_len,
                              head_size,
                              n_kv_heads,
                              gqa_size,
                              curr_conc_lvl,
                              this->state_.batch_token_positions );
  // </multihead attention>

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
                this->state_.curr_concurrency_size );

  // classifier into logits
  ops::matmul( this->state_.logits,
               this->state_.x,
               this->base_weights_.wcls,
               this->state_.curr_concurrency_size,
               this->config_.dim,
               this->config_.vocab_size );
}

template<typename DType>
void extract_batch_token( RunState<DType>& state, const Config& config, const std::vector<float>& temp )
{
  ops::soft_sample( state.logits, temp, config.vocab_size, temp.size() );
  ops::argmax( state.argmax_pos, state.logits, config.vocab_size, temp.size() );
}

template<typename DType>
vector<InferenceState> Llama2<DType>::forward( vector<InferenceState>&& inference_states,
                                               const vector<shared_ptr<ContextType>>& contexts )
{
  // TODO(sadjad): refactor the checks into a separate function
  CHECK_GT( inference_states.size(), 0 ) << "batch size must be at least 1";
  CHECK_EQ( inference_states.size(), contexts.size() ) << "token size must be the same as context size";
  CHECK_LE( inference_states.size(), this->config_.concurrency_limit )
    << "current batch cannot be larger than max concurrency size";

  for ( auto& item : inference_states ) {
    CHECK_EQ( item.next_layer(), this->config_.start_layer_num ) << "next_layer must be the start layer";
  }

  for ( size_t i = 0; i < inference_states.size(); i++ ) {
    this->state_.batch_token_positions[i] = inference_states[i].token_pos();
    CHECK_LT( this->state_.batch_token_positions[i], this->config_.seq_len )
      << "token position cannot be larger than sequence length";
  }

  this->state_.curr_concurrency_size = inference_states.size();

  if ( inference_states[0].next_layer() == 0 ) {
    vector<uint32_t> token_vector;
    for ( size_t i = 0; i < inference_states.size(); i++ ) {
      token_vector.push_back( inference_states[i].token() );
    }
    pass_begin( token_vector );
  } else {
    for ( size_t i = 0; i < inference_states.size(); i++ )
      // load the activations
      memcpy( this->state_.x + i * this->config_.dim,
              inference_states[i].activations().ptr.get(),
              this->config_.dim * sizeof( DType ) );
  }

  for ( size_t layer_num = this->config_.start_layer_num; layer_num <= this->config_.end_layer_num; layer_num++ ) {
    for ( size_t i = 0; i < inference_states.size(); i++ ) {
      this->state_.batch_context_pointers[i] = contexts[i]->key( this->config_, layer_num, 0 );
    }
    transformer_layer( layer_num );
  }

  vector<InferenceState> output_states;

  if ( this->config_.end_layer_num == this->config_.n_layers - 1 ) {
    pass_end();

    vector<float> batch_temps;
    for ( size_t i = 0; i < inference_states.size(); i++ )
      batch_temps.push_back( inference_states[i].temperature() );

    extract_batch_token( this->state_, this->config_, batch_temps );

    for ( size_t i = 0; i < inference_states.size(); i++ ) {
      output_states.push_back( move( inference_states[i] ) );
      auto& item = output_states.back();
      item.set_token( this->state_.argmax_pos[i] );
      item.set_token_pos( item.token_pos() + 1 );
      item.set_next_layer( 0 );
      item.set_activations( {} );
    }

    return output_states;
  }

  for ( size_t i = 0; i < inference_states.size(); i++ ) {
    DataBuffer activations { is_same_v<DType, float> ? SerializedDataType::Type::Float32
                                                     : SerializedDataType::Type::Float16,
                             make_unique<uint8_t[]>( this->config_.dim * sizeof( DType ) ),
                             this->config_.dim };

    memcpy( activations.ptr.get(), this->state_.x + i * this->config_.dim, this->config_.dim * sizeof( DType ) );

    output_states.push_back( move( inference_states[i] ) );
    auto& item = output_states.back();
    item.set_next_layer( this->config_.end_layer_num + 1 );
    item.set_activations( move( activations ) );
  }

  return output_states;
}

template<typename DType>
InferenceState Llama2<DType>::forward( InferenceState&& inference_state, shared_ptr<ContextType> context )
{
  vector<InferenceState> token_vector;
  token_vector.push_back( move( inference_state ) );
  vector<shared_ptr<ContextType>> context_vector;
  context_vector.push_back( move( context ) );
  return move( forward( move( token_vector ), context_vector )[0] );
}

template class Context<_Float16>;
template class Llama2<_Float16>;

template class Context<float>;
template class Llama2<float>;

} // namespace glinthawk::models::llama2::cpu
