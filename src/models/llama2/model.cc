#include "model.hh"

#include <algorithm>
#include <execution>
#include <fcntl.h>
#include <filesystem>
#include <glog/logging.h>
#include <random>

#include "util/exception.hh"
#include "util/file_descriptor.hh"
#include "util/ring_buffer.hh"

namespace glinthawk::models::llama2 {

namespace {

template<typename DType>
std::string dtype_str()
{
  if constexpr ( std::is_same_v<DType, glinthawk::float32_t> ) {
    return { "FP32" };
  } else if constexpr ( std::is_same_v<DType, glinthawk::float16_t> ) {
    return { "FP16" };
  } else {
    LOG( FATAL ) << "invalid dtype";
  }
}

template<typename DType>
void CHECK_DTYPE( const DataType dtype )
{
  if constexpr ( std::is_same_v<DType, glinthawk::float32_t> ) {
    CHECK( dtype == DataType::Float32 );
  } else if constexpr ( std::is_same_v<DType, glinthawk::float16_t> ) {
    CHECK( dtype == DataType::Float16 );
  } else {
    LOG( FATAL ) << "invalid dtype";
  }
}

template<typename Config, typename DType, typename LlamaOperations, typename Context>
void extract_batch_token( LlamaOperations& ops,
                          RunState<Config, DType, Context>& state,
                          const std::vector<float>& temp )
{
  ops.soft_sample( state.logits, temp, temp.size() );
  ops.template argmax<Config::vocab_size>( state.argmax_pos, state.logits, state.x, temp.size() );
}

}

template<typename Config, typename DType, typename LlamaOperations, typename Context>
Llama2<Config, DType, LlamaOperations, Context>::Llama2( const std::filesystem::path& model_dir,
                                                         const uint32_t start_layer,
                                                         const uint32_t end_layer,
                                                         const uint64_t concurrency_limit,
                                                         const uint64_t max_context_count,
                                                         const bool randomize_parameters )
  : settings_( model_dir / "CONFIG",
               start_layer,
               end_layer,
               concurrency_limit,
               max_context_count,
               randomize_parameters )
  , base_weights_buffer_( ops_.device_allocate( BaseWeights<Config, DType>::base_size() ) )
  , layers_buffer_( ops_.device_allocate( LayerWeights<Config, DType>::layer_size() * settings_.n_layers_loaded() ) )
  , run_state_buffer_( ops_.device_allocate( RunState<Config, DType, Context>::state_size( settings_ ) ) )
  , base_weights_( base_weights_buffer_.get() )
  , layer_weights_( [&] {
    std::array<LayerWeights<Config, DType>, Config::n_layers> layers {};
    constexpr size_t layer_size = LayerWeights<Config, DType>::layer_size();
    auto ptr = layers_buffer_.get();
    for ( auto i = settings_.start_layer_num; i <= settings_.end_layer_num; i++ ) {
      layers[i] = LayerWeights<Config, DType> { reinterpret_cast<DType*>(
        reinterpret_cast<uint8_t*>( ptr ) + ( i - settings_.start_layer_num ) * layer_size ) };
    }

    return layers;
  }() )
  , state_( settings_, run_state_buffer_.get() )
{
  const std::string filename_suffix = "_" + dtype_str<DType>();
  const auto base_path = model_dir / ( "BASEWEIGHTS" + filename_suffix );

  constexpr auto base_size = BaseWeights<Config, DType>::base_size();
  constexpr auto layer_size = LayerWeights<Config, DType>::layer_size();

  if ( randomize_parameters ) {
    LOG( WARNING ) << "Randomizing weights and run state...";

    ops_.randomize_device_buffer( base_weights_buffer_.get(),
                                  base_size / sizeof( DType ),
                                  -10.0 / sqrt( Config::dim ),
                                  10.0 / sqrt( Config::dim ) );

    ops_.randomize_device_buffer( layers_buffer_.get(),
                                  layer_size * settings_.n_layers_loaded() / sizeof( DType ),
                                  -10.0 / sqrt( Config::dim ),
                                  10.0 / sqrt( Config::dim ) );

    ops_.randomize_device_buffer( run_state_buffer_.get(),
                                  RunState<Config, DType, Context>::state_size( settings_ ) / sizeof( DType ),
                                  -10.0 / sqrt( Config::dim ),
                                  10.0 / sqrt( Config::dim ) );

    LOG( WARNING ) << "Randomizing weights and run state... done.";
  }

  if ( not randomize_parameters ) {
    // Load BASEWEIGHTS
    CHECK_EQ( std::filesystem::file_size( base_path ), base_size ) << "Base weights are not the expected size.";
    FileDescriptor base_fd { CHECK_SYSCALL( "open", open( base_path.c_str(), O_RDONLY ) ) };
    MMap_Region base_mmap { nullptr, base_size, PROT_READ, MAP_PRIVATE, base_fd.fd_num(), 0 };

    ops_.copy(
      base_weights_buffer_.get(), reinterpret_cast<DType*>( base_mmap.addr() ), base_size, CopyType::HostToDevice );

    LOG( INFO ) << "Loaded base weights (" << base_size << " bytes).";

    // Load LAYERi
    for ( auto i = settings_.start_layer_num; i <= settings_.end_layer_num; i++ ) {
      const auto layer_path = model_dir / ( "LAYER" + std::to_string( i ) + filename_suffix );

      CHECK_EQ( std::filesystem::file_size( layer_path ), layer_size ) << "Layer " << i << " is not the expected size.";

      FileDescriptor layer_fd { CHECK_SYSCALL( "open", open( layer_path.c_str(), O_RDONLY ) ) };
      MMap_Region layer_mmap { nullptr, layer_size, PROT_READ, MAP_PRIVATE, layer_fd.fd_num(), 0 };

      ops_.copy( reinterpret_cast<DType*>( reinterpret_cast<uint8_t*>( layers_buffer_.get() )
                                           + ( i - settings_.start_layer_num ) * layer_size ),
                 reinterpret_cast<DType*>( layer_mmap.addr() ),
                 layer_size,
                 CopyType::HostToDevice );

      LOG( INFO ) << "Loaded layer " << i << " (" << layer_size << " bytes).";
    }
  }
}

template<typename Config, typename DType, typename LlamaOperations, typename Context>
void Llama2<Config, DType, LlamaOperations, Context>::dummy_forward( InferenceState& state )
{
  CHECK_GE( state.next_layer(), settings_.start_layer_num );
  CHECK_LE( state.next_layer(), settings_.end_layer_num );
  //  state.erase_from_workers( state.next_layer(), state.next_stage() );
  state.loop_till_next_worker( Config::n_layers );
}

template<typename Config, typename DType, typename LlamaOperations, typename Context>
bool Llama2<Config, DType, LlamaOperations, Context>::is_finished( const InferenceState& state )
{
  return ( state.next_layer() == 0 and state.next_stage() == InferenceState::Stage::PreAttention )
         and ( state.token() == TOKEN_EOS or state.token_pos() >= Config::seq_len ); // EOS or out of length
}

template<typename Config, typename DType, typename LlamaOperations, typename Context>
void Llama2<Config, DType, LlamaOperations, Context>::check_batch(
  const std::vector<InferenceState>& states,
  const std::vector<std::shared_ptr<Context>>& contexts,
  const InferenceState::Stage stage ) const
{
  CHECK_GT( states.size(), 0 );
  CHECK_LE( states.size(), settings_.concurrency_limit );

  if ( stage == InferenceState::Stage::PreAttention or stage == InferenceState::Stage::Attention ) {
    CHECK_EQ( states.size(), contexts.size() );
  }

  const uint32_t next_layer_batch = states[0].next_layer();

  CHECK_LE( settings_.start_layer_num, next_layer_batch );
  CHECK_LE( next_layer_batch, settings_.end_layer_num );

  for ( auto& item : states ) {
    CHECK( item.next_stage() == stage );
    if ( stage != InferenceState::Stage::Attention ) {
      CHECK_DTYPE<DType>( item.dtype() );
      CHECK_EQ( item.next_layer(), next_layer_batch );
    }
    CHECK_LT( item.token_pos(), Config::seq_len );
  }
}

template<typename Config, typename DType, typename LlamaOperations, typename Context>
void Llama2<Config, DType, LlamaOperations, Context>::check_batch(
  const BatchedState& states,
  const std::vector<std::shared_ptr<Context>>& contexts,
  const InferenceState::Stage stage ) const
{
  CHECK_GT( states.batch_size(), 0 );
  CHECK_LE( states.batch_size(), settings_.concurrency_limit );

  if ( stage == InferenceState::Stage::PreAttention or stage == InferenceState::Stage::Attention ) {
    CHECK_EQ( states.batch_size(), contexts.size() );
  }

  const uint32_t next_layer_batch = states.next_layer();

  CHECK_LE( settings_.start_layer_num, next_layer_batch );
  CHECK_LE( next_layer_batch, settings_.end_layer_num );

  CHECK( states.next_stage() == stage );

  if ( stage != InferenceState::Stage::Attention ) {
    CHECK_DTYPE<DType>( states.dtype() );
    CHECK_EQ( states.next_layer(), next_layer_batch );
  }
}

template<typename Config, typename DType, typename LlamaOperations, typename Context>
void Llama2<Config, DType, LlamaOperations, Context>::load_embedding( const std::vector<InferenceState>& states )
{
  for ( size_t i = 0; i < states.size(); i++ ) {
    const auto token = states[i].token();
    CHECK_LT( token, Config::vocab_size );

    const DType* content_row = this->base_weights_.token_embedding_table + token * Config::dim;
    ops_.copy( this->state_.x + i * Config::dim, content_row, Config::dim * sizeof( DType ), CopyType::HostToDevice );
  }
}

template<typename Config, typename DType, typename LlamaOperations, typename Context>
void Llama2<Config, DType, LlamaOperations, Context>::load_embedding( const BatchedState& states )
{
  for ( size_t i = 0; i < states.batch_size(); i++ ) {
    const auto token = states.token( i );
    CHECK_LT( token, Config::vocab_size );

    const DType* content_row = this->base_weights_.token_embedding_table + token * Config::dim;
    ops_.copy( this->state_.x + i * Config::dim, content_row, Config::dim * sizeof( DType ), CopyType::HostToDevice );
  }
}

template<typename Config, typename DType, typename LlamaOperations, typename Context>
void Llama2<Config, DType, LlamaOperations, Context>::pre_attention_ops( const int32_t layer_num )
{
  const auto& layer_weights = this->layer_weights_[layer_num];

  // attention rmsnorm
  ops_.template rmsnorm<Config::dim>( this->state_.xb,
                                      this->state_.x,
                                      this->state_.xb2,
                                      layer_weights.rms_att_weight,
                                      this->state_.curr_concurrency_size );

  // qkv matmuls for this position
  ops_.template matmul<Config::dim, Config::dim>(
    this->state_.q, this->state_.xb, layer_weights.wq, this->state_.curr_concurrency_size );
  ops_.template matmul<Config::dim, Config::kv_dim * 2>(
    this->state_.kv, this->state_.xb, layer_weights.wkv, this->state_.curr_concurrency_size );

  // save key, value at each time step (pos) to our kv cache, if the context resides on memory
  ops_.copy_kv_cache( this->state_.batch_token_contexts, this->state_.kv, this->state_.curr_concurrency_size );
}

template<typename Config, typename DType, typename LlamaOperations, typename Context>
void Llama2<Config, DType, LlamaOperations, Context>::attention_ops()
{
  // TODO: We should either make parallel tokens in one prompt work, or remove the feature altogether (and put
  //  protections in place).
  ops_.apply_rope( this->state_.curr_concurrency_size,
                   this->state_.batch_token_positions,
                   this->base_weights_.freq_cis_real,
                   this->base_weights_.freq_cis_imag,
                   this->state_.q,
                   this->state_.batch_token_contexts );

  // <multihead attention> for each head and for each token up to and including the current one

  ops_.attention_0_gemm( this->state_.q,
                         this->state_.batch_layer_contexts,
                         this->state_.att,
                         this->state_.curr_concurrency_size,
                         this->state_.batch_token_positions );

  // softmax
  ops_.attention_softmax( this->state_.att,
                          this->state_.batch_token_positions,
                          this->state_.temp_softmax,
                          this->state_.curr_concurrency_size );

  ops_.attention_2_gemm( this->state_.att,
                         this->state_.batch_layer_contexts,
                         this->state_.xb,
                         this->state_.curr_concurrency_size,
                         this->state_.batch_token_positions );

  // </multihead attention>
}

template<typename Config, typename DType, typename LlamaOperations, typename Context>
void Llama2<Config, DType, LlamaOperations, Context>::post_attention_ops( const int32_t layer_num )
{
  const auto& layer_weights = this->layer_weights_[layer_num];

  // final matmul to get the output of the attention
  ops_.template matmul<Config::dim, Config::dim>(
    this->state_.xb2, this->state_.xb, layer_weights.wo, this->state_.curr_concurrency_size );

  // residual connection back into x
  ops_.template accum<Config::dim>( this->state_.x, this->state_.xb2, this->state_.curr_concurrency_size );

  // ffn rmsnorm
  ops_.template rmsnorm<Config::dim>( this->state_.xb,
                                      this->state_.x,
                                      this->state_.xb2,
                                      layer_weights.rms_ffn_weight,
                                      this->state_.curr_concurrency_size );

  // now for ffn in we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
  // first calculate self.w1(x) and self.w3(x)
  ops_.template matmul<Config::dim, Config::hidden_dim>(
    this->state_.hb, this->state_.xb, layer_weights.w1, this->state_.curr_concurrency_size );
  ops_.template matmul<Config::dim, Config::hidden_dim>(
    this->state_.hb2, this->state_.xb, layer_weights.w3, this->state_.curr_concurrency_size );

  ops_.template silu<Config::hidden_dim>( this->state_.hb, this->state_.hb2, this->state_.curr_concurrency_size );

  // final matmul to get the output of the ffn
  ops_.template matmul<Config::hidden_dim, Config::dim>(
    this->state_.xb, this->state_.hb, layer_weights.w2, this->state_.curr_concurrency_size );

  // residual connection
  ops_.template accum<Config::dim>( this->state_.x, this->state_.xb, this->state_.curr_concurrency_size );
}

template<typename Config, typename DType, typename LlamaOperations, typename Context>
void Llama2<Config, DType, LlamaOperations, Context>::classify_ops()
{
  // final rmsnorm
  ops_.template rmsnorm<Config::dim>( this->state_.x,
                                      this->state_.x,
                                      this->state_.xb2,
                                      this->base_weights_.rms_final_weight,
                                      this->state_.curr_concurrency_size );

  // classifier into logits
  ops_.template matmul<Config::dim, Config::vocab_size>(
    this->state_.logits, this->state_.x, this->base_weights_.wcls, this->state_.curr_concurrency_size );
}

template<typename Config, typename DType, typename LlamaOperations, typename Context>
void Llama2<Config, DType, LlamaOperations, Context>::forward_prelude(
  std::vector<InferenceState>& states,
  const std::vector<std::shared_ptr<Context>>& contexts )
{
  this->check_batch( states, contexts, InferenceState::Stage::PreAttention );

  this->state_.curr_concurrency_size = states.size();
  const uint32_t next_layer_batch = states[0].next_layer();

  for ( size_t i = 0; i < contexts.size(); i++ ) {
    this->state_.batch_token_positions[i] = states[i].token_pos();
    this->state_.batch_layer_contexts[i] = contexts[i]->layer( next_layer_batch );
    this->state_.batch_token_contexts[i] = contexts[i]->layer( next_layer_batch ).token( states[i].token_pos() );
  }

  if ( next_layer_batch == 0 ) {
    /* THE FIRST LAYER, just read the tokens */
    load_embedding( states );
  } else {
    /* NOT THE FIRST LAYER, load the activations */
    for ( size_t i = 0; i < states.size(); i++ ) {
      ops_.copy( this->state_.x + i * Config::dim,
                 reinterpret_cast<DType*>( states[i].activations().data() ),
                 Config::dim * sizeof( DType ),
                 CopyType::HostToDevice,
                 true );
    }
  }
}

template<typename Config, typename DType, typename LlamaOperations, typename Context>
void Llama2<Config, DType, LlamaOperations, Context>::forward_prelude( BatchedState& states,
                                                                       const ContextVector& contexts )
{
  this->check_batch( states, contexts, InferenceState::Stage::PreAttention );

  this->state_.curr_concurrency_size = states.batch_size();
  const uint32_t next_layer_batch = states.next_layer();

  for ( size_t i = 0; i < contexts.size(); i++ ) {
    this->state_.batch_token_positions[i] = states.token_pos( i );
    this->state_.batch_layer_contexts[i] = contexts[i]->layer( next_layer_batch );
    this->state_.batch_token_contexts[i] = contexts[i]->layer( next_layer_batch ).token( states.token_pos( i ) );
  }

  if ( next_layer_batch == 0 ) {
    /* THE FIRST LAYER, just read the tokens */
    load_embedding( states );
  } else {
    /* NOT THE FIRST LAYER, load the activations */
    ops_.copy( this->state_.x,
               reinterpret_cast<DType*>( states.activations().data() ),
               states.activations().len(),
               CopyType::HostToDevice,
               true );
  }
}

template<typename Config, typename DType, typename LlamaOperations, typename Context>
std::vector<InferenceState> Llama2<Config, DType, LlamaOperations, Context>::forward_postlude(
  std::vector<InferenceState>&& states,
  const int32_t most_recent_layer_num,
  const bool classified )
{
  std::vector<InferenceState> output_states;
  output_states.reserve( states.size() );

  if ( classified ) {
    CHECK_EQ( most_recent_layer_num, Config::n_layers - 1 );

    std::vector<float> batch_temps;
    for ( size_t i = 0; i < states.size(); i++ )
      batch_temps.push_back( states[i].temperature() );

    extract_batch_token( this->ops_, this->state_, batch_temps );

    for ( size_t i = 0; i < states.size(); i++ ) {
      states[i].set_token( this->state_.argmax_pos[i] );
      states[i].set_token_pos( states[i].token_pos() + 1 );
      states[i].set_next_stage( InferenceState::Stage::PreAttention );
      states[i].set_next_layer( 0 );
      states[i].set_activations( {} );
      output_states.push_back( std::move( states[i] ) );
    }
  } else {
    for ( size_t i = 0; i < states.size(); i++ ) {
      DataBuffer activations { Config::dim * sizeof( DType ) };

      ops_.copy( reinterpret_cast<DType*>( activations.data() ),
                 this->state_.x + i * Config::dim,
                 Config::dim * sizeof( DType ),
                 CopyType::DeviceToHost );

      if ( most_recent_layer_num == Config::n_layers - 1 ) {
        states[i].set_next_stage( InferenceState::Stage::Classification );
      } else {
        states[i].set_next_stage( InferenceState::Stage::PreAttention );
        states[i].set_next_layer( most_recent_layer_num + 1 );
      }
      states[i].set_activations( std::move( activations ) );
      output_states.push_back( std::move( states[i] ) );
    }
  }

  return output_states;
}

template<typename Config, typename DType, typename LlamaOperations, typename Context>
BatchedInferenceState<Config> Llama2<Config, DType, LlamaOperations, Context>::forward_postlude(
  BatchedState&& states,
  const int32_t most_recent_layer_num,
  const bool classification_done )
{
  if ( classification_done ) {
    CHECK_EQ( most_recent_layer_num, Config::n_layers - 1 );

    std::vector<float> batch_temps;
    for ( size_t i = 0; i < states.batch_size(); i++ ) {
      batch_temps.push_back( states.temperature( i ) );
    }

    extract_batch_token( this->ops_, this->state_, batch_temps );

    // we don't need to send the activations (or anything else) to the next worker, just the token
    states.deallocate_activations();
    states.deallocate_queries();
    states.deallocate_kvs();

    states.set_next_layer( 0 );
    states.set_next_stage( InferenceState::Stage::PreAttention );

    // XXX(sadjad): what do we do if the token is EOS?
    for ( size_t i = 0; i < states.batch_size(); i++ ) {
      states.set_token( i, this->state_.argmax_pos[i] );
      states.set_token_pos( i, states.token_pos( i ) + 1 );

      if ( states.token( i ) == TOKEN_EOS or states.token_pos( i ) >= Config::seq_len ) {
        states.set_finished( i, true );
      }
    }
  } else {
    // all we need to send is activations; the memory is already allocated
    states.deallocate_queries();
    states.deallocate_kvs();

    ops_.copy( reinterpret_cast<DType*>( states.activations().data() ),
               this->state_.x,
               states.batch_size() * Config::dim * sizeof( DType ),
               CopyType::DeviceToHost );

    if ( most_recent_layer_num == Config::n_layers - 1 ) {
      states.set_next_stage( InferenceState::Stage::Classification );
    } else {
      states.set_next_stage( InferenceState::Stage::PreAttention );
      states.set_next_layer( most_recent_layer_num + 1 );
    }
  }

  return std::move( states );
}

template<typename Config, typename DType, typename LlamaOperations, typename Context>
std::vector<InferenceState> Llama2<Config, DType, LlamaOperations, Context>::forward(
  std::vector<InferenceState>&& states,
  const ContextVector& contexts )
{
  forward_prelude( states, contexts );

  for ( size_t layer_num = states[0].next_layer(); layer_num <= this->settings_.end_layer_num; layer_num++ ) {
    for ( size_t i = 0; i < contexts.size(); i++ ) {
      this->state_.batch_layer_contexts[i] = contexts[i]->layer( layer_num );
      this->state_.batch_token_contexts[i] = contexts[i]->layer( layer_num ).token( states[i].token_pos() );
    }

    pre_attention_ops( layer_num );
    attention_ops();
    post_attention_ops( layer_num );
  }

  if ( this->settings_.end_layer_num == Config::n_layers - 1 ) {
    classify_ops();
    return forward_postlude( std::move( states ), this->settings_.end_layer_num, true );
  } else {
    return forward_postlude( std::move( states ), this->settings_.end_layer_num, false );
  }
}

template<typename Config, typename DType, typename LlamaOperations, typename Context>
BatchedInferenceState<Config> Llama2<Config, DType, LlamaOperations, Context>::forward( BatchedState&& states,
                                                                                        const ContextVector& contexts )
{
  forward_prelude( states, contexts );

  for ( size_t layer_num = states.next_layer(); layer_num <= this->settings_.end_layer_num; layer_num++ ) {
    for ( size_t i = 0; i < contexts.size(); i++ ) {
      this->state_.batch_layer_contexts[i] = contexts[i]->layer( layer_num );
      this->state_.batch_token_contexts[i] = contexts[i]->layer( layer_num ).token( states.token_pos( i ) );
    }

    pre_attention_ops( layer_num );
    attention_ops();
    post_attention_ops( layer_num );
  }

  if ( this->settings_.end_layer_num == Config::n_layers - 1 ) {
    classify_ops();
    return forward_postlude( std::move( states ), this->settings_.end_layer_num, true );
  } else {
    return forward_postlude( std::move( states ), this->settings_.end_layer_num, false );
  }
}

template<typename Config, typename DType, typename LlamaOperations, typename Context>
std::vector<InferenceState> Llama2<Config, DType, LlamaOperations, Context>::pre_attention_forward(
  std::vector<InferenceState>&& states,
  const ContextVector& contexts )
{
  forward_prelude( states, contexts );
  pre_attention_ops( states[0].next_layer() );

  std::vector<InferenceState> output_states;
  output_states.reserve( states.size() );
  for ( size_t i = 0; i < states.size(); i++ ) {
    size_t len_activation = 2 * Config::dim;
    if ( contexts[i]->empty() ) {
      len_activation += 2 * Config::kv_dim;
    }

    DataBuffer activations { len_activation * sizeof( DType ) };

    ops_.copy( reinterpret_cast<DType*>( activations.data() ),
               this->state_.x + i * Config::dim,
               Config::dim * sizeof( DType ),
               CopyType::DeviceToHost );

    ops_.copy( reinterpret_cast<DType*>( activations.data() ) + Config::dim,
               this->state_.q + i * Config::dim,
               Config::dim * sizeof( DType ),
               CopyType::DeviceToHost );

    if ( contexts[i]->empty() ) {
      ops_.copy( reinterpret_cast<DType*>( activations.data() ) + 2 * Config::dim,
                 this->state_.kv + i * 2 * Config::kv_dim,
                 2 * Config::kv_dim * sizeof( DType ),
                 CopyType::DeviceToHost );
    }

    states[i].set_next_stage( InferenceState::Stage::Attention );
    states[i].set_activations( std::move( activations ) );
    output_states.push_back( std::move( states[i] ) );
  }

  return output_states;
}

template<typename Config, typename DType, typename LlamaOperations, typename Context>
BatchedInferenceState<Config> Llama2<Config, DType, LlamaOperations, Context>::pre_attention_forward(
  BatchedState&& states,
  const ContextVector& contexts )
{
  forward_prelude( states, contexts );
  pre_attention_ops( states.next_layer() );

  if ( not states.has_activations() ) {
    states.allocate_activations();
  }

  if ( not states.has_queries() ) {
    states.allocate_queries();
  }

  if ( not states.has_kvs() ) {
    states.allocate_kvs();
  }

  ops_.copy( reinterpret_cast<DType*>( states.activations().data() ),
             this->state_.x,
             states.activations().len(),
             CopyType::DeviceToHost );

  ops_.copy( reinterpret_cast<DType*>( states.queries().data() ),
             this->state_.q,
             states.queries().len(),
             CopyType::DeviceToHost );

  // XXX(sadjad): Copying KV is not always necessary (i.e., if context[i]->empty()), but for convenience we always do it
  ops_.copy(
    reinterpret_cast<DType*>( states.kvs().data() ), this->state_.kv, states.kvs().len(), CopyType::DeviceToHost );

  states.set_next_stage( InferenceState::Stage::Attention );
  return std::move( states );
}

template<typename Config, typename DType, typename LlamaOperations, typename Context>
std::vector<InferenceState> Llama2<Config, DType, LlamaOperations, Context>::attention_forward(
  std::vector<InferenceState>&& states,
  const ContextVector& contexts )
{
  this->check_batch( states, contexts, InferenceState::Stage::Attention );
  this->state_.curr_concurrency_size = states.size();

  for ( size_t i = 0; i < states.size(); i++ ) {
    auto& state = states[i];

    this->state_.batch_token_positions[i] = state.token_pos();
    this->state_.batch_layer_contexts[i] = contexts[i]->layer( state.next_layer() );
    this->state_.batch_token_contexts[i] = contexts[i]->layer( state.next_layer() ).token( state.token_pos() );

    // load the activations
    auto activation_data = state.activations().data();
    auto activation_len = state.activations().len();

    switch ( state.dtype() ) {
      case DataType::Float16:
        // Q should go to run state
        ops_.template convert_and_copy( this->state_.q + i * Config::dim,
                                        reinterpret_cast<LlamaOperations::Float16*>( activation_data ) + Config::dim,
                                        Config::dim,
                                        CopyType::HostToDevice );

        // if KV is not already in context, put it there
        if ( activation_len > 2 * Config::dim * DataTypeSize( state.dtype() ) ) {
          ops_.template convert_and_copy( contexts[i]->layer( state.next_layer() ).token( state.token_pos() ).key(),
                                          reinterpret_cast<LlamaOperations::Float16*>( activation_data )
                                            + 2 * Config::dim,
                                          Config::kv_dim * 2,
                                          CopyType::HostToDevice );
        }
        break;

      case DataType::Float32:
        // Q should go to run state
        ops_.template convert_and_copy( this->state_.q + i * Config::dim,
                                        reinterpret_cast<LlamaOperations::Float32*>( activation_data ) + Config::dim,
                                        Config::dim,
                                        CopyType::HostToDevice );

        // if KV is not already in context, put it there
        if ( activation_len > 2 * Config::dim * DataTypeSize( state.dtype() ) ) {
          ops_.template convert_and_copy( contexts[i]->layer( state.next_layer() ).token( state.token_pos() ).key(),
                                          reinterpret_cast<LlamaOperations::Float32*>( activation_data )
                                            + 2 * Config::dim,
                                          Config::kv_dim * 2,
                                          CopyType::HostToDevice );
        }
        break;

      default: LOG( FATAL ) << "invalid dtype";
    }
  }

  attention_ops();

  std::vector<InferenceState> output_states;
  output_states.reserve( states.size() );

  for ( size_t i = 0; i < states.size(); i++ ) {
    auto& state = states[i];

    DataBuffer activations { 2 * Config::dim * DataTypeSize( state.dtype() ) };

    ops_.copy( reinterpret_cast<DType*>( activations.data() ),
               reinterpret_cast<DType*>( state.activations().data() ),
               Config::dim * sizeof( DType ),
               CopyType::HostToHost );

    switch ( state.dtype() ) {
      case DataType::Float16:
        ops_.template convert_and_copy( reinterpret_cast<LlamaOperations::Float16*>( activations.data() ) + Config::dim,
                                        this->state_.xb + i * Config::dim,
                                        Config::dim,
                                        CopyType::DeviceToHost );
        break;

      case DataType::Float32:
        ops_.template convert_and_copy( reinterpret_cast<LlamaOperations::Float32*>( activations.data() ) + Config::dim,
                                        this->state_.xb + i * Config::dim,
                                        Config::dim,
                                        CopyType::DeviceToHost );
        break;

      default: throw std::runtime_error( "invalid dtype" );
    }

    state.set_next_stage( InferenceState::Stage::PostAttention );
    state.set_activations( std::move( activations ) );
    output_states.push_back( std::move( state ) );
  }

  return output_states;
}

template<typename Config, typename DType, typename LlamaOperations, typename Context>
BatchedInferenceState<Config> Llama2<Config, DType, LlamaOperations, Context>::attention_forward(
  BatchedState&& states,
  const ContextVector& contexts )
{
  this->check_batch( states, contexts, InferenceState::Stage::Attention );
  this->state_.curr_concurrency_size = states.batch_size();

  for ( size_t i = 0; i < states.batch_size(); i++ ) {
    this->state_.batch_token_positions[i] = states.token_pos( i );
    this->state_.batch_layer_contexts[i] = contexts[i]->layer( states.next_layer() );
    this->state_.batch_token_contexts[i] = contexts[i]->layer( states.next_layer() ).token( states.token_pos( i ) );
  }

  switch ( states.dtype() ) {
    case DataType::Float16:
      ops_.template convert_and_copy( this->state_.q,
                                      reinterpret_cast<LlamaOperations::Float16*>( states.queries().data() ),
                                      states.queries().len() / sizeof( typename LlamaOperations::Float16 ),
                                      CopyType::HostToDevice );
      break;

    case DataType::Float32:
      ops_.template convert_and_copy( this->state_.q,
                                      reinterpret_cast<LlamaOperations::Float32*>( states.queries().data() ),
                                      states.queries().len() / sizeof( typename LlamaOperations::Float32 ),
                                      CopyType::HostToDevice );
      break;

    default: LOG( FATAL ) << "invalid dtype";
  }

  if ( states.has_kvs() ) {
    for ( size_t i = 0; i < states.batch_size(); i++ ) {
      switch ( states.dtype() ) {
        case DataType::Float16:
          ops_.template convert_and_copy(
            contexts[i]->layer( states.next_layer() ).token( states.token_pos( i ) ).key(),
            reinterpret_cast<LlamaOperations::Float16*>( states.kv( i ).data() ),
            Config::kv_dim * 2,
            CopyType::HostToDevice );
          break;

        case DataType::Float32:
          ops_.template convert_and_copy(
            contexts[i]->layer( states.next_layer() ).token( states.token_pos( i ) ).key(),
            reinterpret_cast<LlamaOperations::Float32*>( states.kv( i ).data() ),
            Config::kv_dim * 2,
            CopyType::HostToDevice );
          break;

        default: LOG( FATAL ) << "invalid dtype";
      }
    }
  }

  attention_ops();

  if ( not states.has_queries() ) {
    states.allocate_queries();
  }

  // FIXME(sadjad): what we're copying here is not actually the query, but for the sake of simplicity we're reusing the
  // query buffer. I'm not proud of this.
  switch ( states.dtype() ) {
    case DataType::Float16:
      ops_.template convert_and_copy( reinterpret_cast<LlamaOperations::Float16*>( states.queries().data() ),
                                      this->state_.xb,
                                      states.queries().len() / sizeof( typename LlamaOperations::Float16 ),
                                      CopyType::DeviceToHost );
      break;

    case DataType::Float32:
      ops_.template convert_and_copy( reinterpret_cast<LlamaOperations::Float32*>( states.queries().data() ),
                                      this->state_.xb,
                                      states.queries().len() / sizeof( typename LlamaOperations::Float32 ),
                                      CopyType::DeviceToHost );
      break;

    default: throw std::runtime_error( "invalid dtype" );
  }

  states.set_next_stage( InferenceState::Stage::PostAttention );
  return std::move( states );
}

template<typename Config, typename DType, typename LlamaOperations, typename Context>
std::vector<InferenceState> Llama2<Config, DType, LlamaOperations, Context>::post_attention_forward(
  std::vector<InferenceState>&& states )
{
  this->check_batch( states, {}, InferenceState::Stage::PostAttention );
  this->state_.curr_concurrency_size = states.size();

  for ( size_t i = 0; i < states.size(); i++ ) {
    // load the activations
    ops_.copy( this->state_.x + i * Config::dim,
               reinterpret_cast<DType*>( states[i].activations().data() ),
               Config::dim * sizeof( DType ),
               CopyType::HostToDevice );

    ops_.copy( this->state_.xb + i * Config::dim,
               reinterpret_cast<DType*>( states[i].activations().data() ) + Config::dim,
               Config::dim * sizeof( DType ),
               CopyType::HostToDevice );
  }

  post_attention_ops( states[0].next_layer() );

  return forward_postlude( std::move( states ), states[0].next_layer(), false );
}

template<typename Config, typename DType, typename LlamaOperations, typename Context>
BatchedInferenceState<Config> Llama2<Config, DType, LlamaOperations, Context>::post_attention_forward(
  BatchedState&& states )
{
  this->check_batch( states, {}, InferenceState::Stage::PostAttention );
  this->state_.curr_concurrency_size = states.batch_size();

  ops_.copy( this->state_.x,
             reinterpret_cast<DType*>( states.activations().data() ),
             states.activations().len(),
             CopyType::HostToDevice );

  ops_.copy( this->state_.xb,
             reinterpret_cast<DType*>( states.queries().data() ),
             states.queries().len(),
             CopyType::HostToDevice );

  post_attention_ops( states.next_layer() );

  return forward_postlude( std::move( states ), states.next_layer(), false );
}

template<typename Config, typename DType, typename LlamaOperations, typename Context>
std::vector<InferenceState> Llama2<Config, DType, LlamaOperations, Context>::classify_forward(
  std::vector<InferenceState>&& states )
{
  this->check_batch( states, {}, InferenceState::Stage::Classification );
  this->state_.curr_concurrency_size = states.size();

  for ( size_t i = 0; i < states.size(); i++ ) {
    // load the activations
    ops_.copy( this->state_.x + i * Config::dim,
               reinterpret_cast<DType*>( states[i].activations().data() ),
               Config::dim * sizeof( DType ),
               CopyType::HostToDevice );
  }

  classify_ops();

  return forward_postlude( std::move( states ), states[0].next_layer(), true );
}

template<typename Config, typename DType, typename LlamaOperations, typename Context>
BatchedInferenceState<Config> Llama2<Config, DType, LlamaOperations, Context>::classify_forward( BatchedState&& states )
{
  this->check_batch( states, {}, InferenceState::Stage::Classification );
  this->state_.curr_concurrency_size = states.batch_size();

  // load the activations
  ops_.copy( this->state_.x,
             reinterpret_cast<DType*>( states.activations().data() ),
             states.activations().len(),
             CopyType::HostToDevice );

  classify_ops();

  return forward_postlude( std::move( states ), states.next_layer(), true );
}

#define INSTANTIATE_MODEL( PLATFORM, MODEL, DTYPE )                                                                    \
  template class Llama2<configs::MODEL,                                                                                \
                        DTYPE,                                                                                         \
                        PLATFORM::LlamaOperations<configs::MODEL, DTYPE>,                                              \
                        PLATFORM::Context<configs::MODEL, DTYPE>>;

#if defined( TARGET_PLATFORM_AMD64 ) || defined( TARGET_PLATFORM_CUDA )
INSTANTIATE_MODEL( amd64, Stories_110M, glinthawk::float32_t );
INSTANTIATE_MODEL( amd64, Llama2_7B_Chat, glinthawk::float32_t );
INSTANTIATE_MODEL( amd64, Llama2_13B_Chat, glinthawk::float32_t );
INSTANTIATE_MODEL( amd64, Llama2_70B_Chat, glinthawk::float32_t );
#endif

// _Float16 is not supported by `nvcc`
#if defined( TARGET_PLATFORM_AMD64 )
INSTANTIATE_MODEL( amd64, Stories_110M, glinthawk::float16_t );
INSTANTIATE_MODEL( amd64, Llama2_7B_Chat, glinthawk::float16_t );
INSTANTIATE_MODEL( amd64, Llama2_13B_Chat, glinthawk::float16_t );
INSTANTIATE_MODEL( amd64, Llama2_70B_Chat, glinthawk::float16_t );
#endif

#if defined( TARGET_PLATFORM_CUDA )
INSTANTIATE_MODEL( cuda, Stories_110M, glinthawk::float32_t );
INSTANTIATE_MODEL( cuda, Llama2_7B_Chat, glinthawk::float32_t );
INSTANTIATE_MODEL( cuda, Llama2_13B_Chat, glinthawk::float32_t );
INSTANTIATE_MODEL( cuda, Llama2_70B_Chat, glinthawk::float32_t );
INSTANTIATE_MODEL( cuda, Stories_110M, glinthawk::float16_t );
INSTANTIATE_MODEL( cuda, Llama2_7B_Chat, glinthawk::float16_t );
INSTANTIATE_MODEL( cuda, Llama2_13B_Chat, glinthawk::float16_t );
INSTANTIATE_MODEL( cuda, Llama2_70B_Chat, glinthawk::float16_t );
#endif

} // namespace glinthawk::models::llama2
