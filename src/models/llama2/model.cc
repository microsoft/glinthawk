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
  if constexpr ( std::is_same_v<DType, float> ) {
    return { "FP32" };
  }
#if defined( TARGET_PLATFORM_AMD64 )
  else if constexpr ( std::is_same_v<DType, _Float16> ) {
    return { "FP16" };
  }
#elif defined( TARGET_PLATFORM_CUDA )
  else if constexpr ( std::is_same_v<DType, __half> ) {
    return { "FP16" };
  }
#endif
  else {
    LOG( FATAL ) << "invalid dtype";
  }
}

template<typename DType>
void CHECK_DTYPE( const DataType dtype )
{
  if constexpr ( std::is_same_v<DType, float> ) {
    CHECK( dtype == DataType::Float32 );
  }
#if defined( TARGET_PLATFORM_AMD64 )
  else if constexpr ( std::is_same_v<DType, _Float16> ) {
    CHECK( dtype == DataType::Float16 );
  }
#elif defined( TARGET_PLATFORM_CUDA )
  else if constexpr ( std::is_same_v<DType, __half> ) {
    CHECK( dtype == DataType::Float16 );
  }
#endif
}

template<typename DType>
void randomize_buffer( DType* buffer, size_t len, const float min, const float max )
{
  static thread_local std::mt19937 generator { std::random_device {}() };
  std::uniform_real_distribution<float> distribution( min, max );

  size_t i;
#pragma omp parallel for schedule( static ) private( i )
  for ( i = 0; i < len; i++ ) {
    if constexpr ( std::is_same_v<DType, float> ) {
      buffer[i] = distribution( generator );
    } else {
      buffer[i] = static_cast<DType>( distribution( generator ) );
    }
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

  // Load the model
  // (1) loading the base weights
  //  if ( not randomize_parameters ) {
  // XXX(pouya): avoid randomizing weights right now so CPU doesn't overload
  CHECK_EQ( std::filesystem::file_size( base_path ), base_size ) << "Base weights are not the expected size.";
  FileDescriptor base_fd { CHECK_SYSCALL( "open", open( base_path.c_str(), O_RDONLY ) ) };
  MMap_Region base_mmap { nullptr, base_size, PROT_READ, MAP_PRIVATE, base_fd.fd_num(), 0 };

  ops_.copy(
    base_weights_buffer_.get(), reinterpret_cast<DType*>( base_mmap.addr() ), base_size, CopyType::HostToDevice );

  LOG( INFO ) << "Loaded base weights (" << base_size << " bytes).";
  //  } else {
  //    LOG( WARNING ) << "Randomizing BASEWEIGHTS...";
  //    std::unique_ptr<DType[]> base { new DType[base_size / sizeof( DType )] };
  //    randomize_buffer(
  //      base.get(), base_size / sizeof( DType ), -10.0 / sqrt( Config::dim ), 10.0 / sqrt( Config::dim ) );
  //
  //    ops_.copy( base_weights_buffer_.get(), base.get(), base_size, CopyType::HostToDevice );
  //  }

  // (2) load the layers
  if ( not randomize_parameters ) {
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
  } else {
    LOG( WARNING ) << "Randomizing LAYERS...";
    std::unique_ptr<DType[]> layer_host { new DType[layer_size * settings_.n_layers_loaded() / sizeof( DType )] };
    randomize_buffer( layer_host.get(),
                      layer_size * settings_.n_layers_loaded() / sizeof( DType ),
                      -10.0 / sqrt( Config::dim ),
                      10.0 / sqrt( Config::dim ) );
    ops_.copy(
      layers_buffer_.get(), layer_host.get(), layer_size * settings_.n_layers_loaded(), CopyType::HostToDevice );
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
         and ( state.token() == 2 or state.token_pos() >= Config::seq_len ); // EOS or out of length
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
InferenceState Llama2<Config, DType, LlamaOperations, Context>::forward( InferenceState&& state,
                                                                         std::shared_ptr<Context> context )
{
  std::vector<InferenceState> states;
  states.push_back( std::move( state ) );
  return std::move( forward( std::move( states ), std::vector { context } ).at( 0 ) );
}

template<typename Config, typename DType, typename LlamaOperations, typename Context>
InferenceState Llama2<Config, DType, LlamaOperations, Context>::pre_attention_forward(
  InferenceState&& state,
  std::shared_ptr<Context> context )
{
  std::vector<InferenceState> states;
  states.push_back( std::move( state ) );
  return std::move( pre_attention_forward( std::move( states ), std::vector { context } ).at( 0 ) );
}

template<typename Config, typename DType, typename LlamaOperations, typename Context>
InferenceState Llama2<Config, DType, LlamaOperations, Context>::attention_forward( InferenceState&& state,
                                                                                   std::shared_ptr<Context> context )
{
  std::vector<InferenceState> states;
  states.push_back( std::move( state ) );
  return std::move( attention_forward( std::move( states ), std::vector { context } ).at( 0 ) );
}

template<typename Config, typename DType, typename LlamaOperations, typename Context>
InferenceState Llama2<Config, DType, LlamaOperations, Context>::post_attention_forward( InferenceState&& state )
{
  std::vector<InferenceState> states;
  states.push_back( std::move( state ) );
  return std::move( post_attention_forward( std::move( states ) ).at( 0 ) );
}

template<typename Config, typename DType, typename LlamaOperations, typename Context>
InferenceState Llama2<Config, DType, LlamaOperations, Context>::classify_forward( InferenceState&& state )
{
  std::vector<InferenceState> states;
  states.push_back( std::move( state ) );
  return std::move( classify_forward( std::move( states ) ).at( 0 ) );
}

#define INSTANTIATE_MODEL( PLATFORM, MODEL, DTYPE )                                                                    \
  template class Llama2<configs::MODEL,                                                                                \
                        DTYPE,                                                                                         \
                        PLATFORM::LlamaOperations<configs::MODEL, DTYPE>,                                              \
                        PLATFORM::Context<configs::MODEL, DTYPE>>;

#if defined( TARGET_PLATFORM_AMD64 ) || defined( TARGET_PLATFORM_CUDA )
INSTANTIATE_MODEL( amd64, Stories_110M, float );
INSTANTIATE_MODEL( amd64, Llama2_7B_Chat, float );
INSTANTIATE_MODEL( amd64, Llama2_13B_Chat, float );
INSTANTIATE_MODEL( amd64, Llama2_70B_Chat, float );
#endif

// not supported with nvcc
#if defined( TARGET_PLATFORM_AMD64 )
INSTANTIATE_MODEL( amd64, Stories_110M, _Float16 );
INSTANTIATE_MODEL( amd64, Llama2_7B_Chat, _Float16 );
INSTANTIATE_MODEL( amd64, Llama2_13B_Chat, _Float16 );
INSTANTIATE_MODEL( amd64, Llama2_70B_Chat, _Float16 );
#endif

#if defined( TARGET_PLATFORM_CUDA )
INSTANTIATE_MODEL( cuda, Stories_110M, float );
INSTANTIATE_MODEL( cuda, Llama2_7B_Chat, float );
INSTANTIATE_MODEL( cuda, Llama2_13B_Chat, float );
INSTANTIATE_MODEL( cuda, Llama2_70B_Chat, float );
INSTANTIATE_MODEL( cuda, Stories_110M, __half );
INSTANTIATE_MODEL( cuda, Llama2_7B_Chat, __half );
INSTANTIATE_MODEL( cuda, Llama2_13B_Chat, __half );
INSTANTIATE_MODEL( cuda, Llama2_70B_Chat, __half );
#endif

} // namespace glinthawk::models::llama2
