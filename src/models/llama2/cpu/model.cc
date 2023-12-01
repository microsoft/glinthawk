#include "model.hh"

#include <random>

namespace glinthawk::models::llama2::cpu {

namespace {

template<typename DType>
std::string dtype_str()
{
  if constexpr ( std::is_same_v<DType, float> ) {
    return { "FP32" };
  } else if constexpr ( std::is_same_v<DType, _Float16> ) {
    return { "FP16" };
  } else {
    LOG( FATAL ) << "invalid dtype";
  }
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

} // namespace

template<typename Config, typename DType>
Context<Config, DType>::Context( const Settings<Config>& settings )
  : storage_( reinterpret_cast<DType*>( new uint8_t[InferenceContext<Config, DType>::context_size( settings )] ) )
{
  this->buffer_ = storage_.get();
  if ( settings.randomize_parameters ) {
    LOG( WARNING ) << "Randomizing context...";
    randomize_buffer( storage_.get(),
                      InferenceContext<Config, DType>::context_size( settings ) / sizeof( DType ),
                      -10.0 / sqrt( Config::dim ),
                      10.0 / sqrt( Config::dim ) );
  }
}

template<typename Config, typename DType>
Llama2<Config, DType>::Llama2( const std::filesystem::path& model_path,
                               const uint32_t start_layer,
                               const uint32_t end_layer,
                               const uint64_t concurrency_limit,
                               const bool randomize_parameters )
{
  const std::string filename_suffix = "_" + dtype_str<DType>();
  const auto config_path = model_path / "CONFIG";
  const auto base_path = model_path / ( "BASEWEIGHTS" + filename_suffix );

  llama2::Settings<Config> settings { config_path, start_layer, end_layer, concurrency_limit, randomize_parameters };

  constexpr auto base_size = BaseWeights<Config, DType>::base_size();
  constexpr auto layer_size = LayerWeights<Config, DType>::layer_size();
  const auto run_state_size = RunState<Config, DType>::state_size( settings );

  // Allocate memory for the base weights, layers and run state
  std::unique_ptr<DType> base { reinterpret_cast<DType*>( new uint8_t[base_size] ) };
  std::unique_ptr<DType> layers { reinterpret_cast<DType*>( new uint8_t[layer_size * settings.n_layers_loaded()] ) };
  std::unique_ptr<DType> run_state { reinterpret_cast<DType*>( new uint8_t[run_state_size] ) };

  memset( run_state.get(), 0, run_state_size );

  // Load the model

  // (1) loading the base weights
  if ( not randomize_parameters ) {
    CHECK_EQ( std::filesystem::file_size( base_path ), base_size ) << "Base weights are not the expected size.";

    FileDescriptor base_fd { CHECK_SYSCALL( "open", open( base_path.c_str(), O_RDONLY ) ) };
    MMap_Region base_mmap { nullptr, base_size, PROT_READ, MAP_PRIVATE, base_fd.fd_num(), 0 };
    ops_.copy( base.get(), reinterpret_cast<DType*>( base_mmap.addr() ), base_size );
    LOG( INFO ) << "Loaded base weights (" << base_size << " bytes).";
  } else {
    LOG( WARNING ) << "Randomizing BASEWEIGHTS...";
    randomize_buffer(
      base.get(), base_size / sizeof( DType ), -10.0 / sqrt( Config::dim ), 10.0 / sqrt( Config::dim ) );
  }

  // (2) load the layers
  if ( not randomize_parameters ) {
    for ( auto i = settings.start_layer_num; i <= settings.end_layer_num; i++ ) {
      const auto layer_path = model_path / ( "LAYER" + std::to_string( i ) + filename_suffix );

      CHECK_EQ( std::filesystem::file_size( layer_path ), layer_size ) << "Layer " << i << " is not the expected size.";

      FileDescriptor layer_fd { CHECK_SYSCALL( "open", open( layer_path.c_str(), O_RDONLY ) ) };
      MMap_Region layer_mmap { nullptr, layer_size, PROT_READ, MAP_PRIVATE, layer_fd.fd_num(), 0 };

      ops_.copy( reinterpret_cast<DType*>( reinterpret_cast<uint8_t*>( layers.get() )
                                           + ( i - settings.start_layer_num ) * layer_size ),
                 reinterpret_cast<DType*>( layer_mmap.addr() ),
                 layer_size );

      LOG( INFO ) << "Loaded layer " << i << " (" << layer_size << " bytes).";
    }
  } else {
    LOG( WARNING ) << "Randomizing LAYERS...";
    randomize_buffer( layers.get(),
                      layer_size * settings.n_layers_loaded() / sizeof( DType ),
                      -10.0 / sqrt( Config::dim ),
                      10.0 / sqrt( Config::dim ) );
  }

  this->init( settings, std::move( base ), std::move( layers ), std::move( run_state ) );
}

template<typename Config, typename DType>
void Llama2<Config, DType>::pass_begin( const std::vector<uint32_t>& token )
{
  // copy the token embedding into the state
  for ( size_t i = 0; i < token.size(); i++ ) {
    CHECK_LT( token[i], Config::vocab_size ) << "token index must not surpass vocab size";

    const DType* content_row = this->base_weights_.token_embedding_table + token[i] * Config::dim;
    ops_.copy( this->state_.x + i * Config::dim, content_row, Config::dim * sizeof( DType ) );
  }
}

template<typename Config, typename DType>
void Llama2<Config, DType>::pre_attention_ops( const int32_t layer_num )
{
  const uint64_t curr_conc_lvl = this->state_.curr_concurrency_size;
  const auto& layer_weights = this->layer_weights_[layer_num];

  // attention rmsnorm
  ops_.template rmsnorm<Config::dim>(
    this->state_.xb, this->state_.x, this->state_.xb2, layer_weights.rms_att_weight, curr_conc_lvl );

  // qkv matmuls for this position
  ops_.template matmul<Config::dim, Config::dim>( this->state_.q, this->state_.xb, layer_weights.wq, curr_conc_lvl );
  ops_.template matmul<Config::dim, Config::kv_dim>( this->state_.k, this->state_.xb, layer_weights.wk, curr_conc_lvl );
  ops_.template matmul<Config::dim, Config::kv_dim>( this->state_.v, this->state_.xb, layer_weights.wv, curr_conc_lvl );

  // save key, value at each time step (pos) to our kv cache, if the context resides on memory
  ops_.copy_kv_cache( this->state_.batch_context_pointers,
                      this->state_.k,
                      this->state_.v,
                      curr_conc_lvl,
                      this->state_.batch_token_positions );
}

template<typename Config, typename DType>
void Llama2<Config, DType>::attention_ops()
{
  const uint64_t curr_conc_lvl = this->state_.curr_concurrency_size;

  ops_.apply_rope( curr_conc_lvl,
                   this->state_.batch_token_positions,
                   this->base_weights_.freq_cis_real,
                   this->base_weights_.freq_cis_imag,
                   this->state_.q,
                   this->state_.batch_context_pointers );

  // <multihead attention> for each head and for each token up to and including the current one

  ops_.attention_0_gemm( this->state_.q,
                         this->state_.batch_context_pointers,
                         this->state_.att,
                         curr_conc_lvl,
                         this->state_.batch_token_positions );

  // softmax
  ops_.attention_softmax(
    this->state_.att, this->state_.batch_token_positions, this->state_.temp_softmax, curr_conc_lvl );

  ops_.attention_2_gemm( this->state_.att,
                         this->state_.batch_context_pointers,
                         this->state_.xb,
                         curr_conc_lvl,
                         this->state_.batch_token_positions );

  // </multihead attention>
}

template<typename Config, typename DType>
void Llama2<Config, DType>::post_attention_ops( const int32_t layer_num )
{
  const uint64_t curr_conc_lvl = this->state_.curr_concurrency_size;
  const auto& layer_weights = this->layer_weights_[layer_num];

  // final matmul to get the output of the attention
  ops_.template matmul<Config::dim, Config::dim>( this->state_.xb2, this->state_.xb, layer_weights.wo, curr_conc_lvl );

  // residual connection back into x
  ops_.template accum<Config::dim>( this->state_.x, this->state_.xb2, curr_conc_lvl );

  // ffn rmsnorm
  ops_.template rmsnorm<Config::dim>(
    this->state_.xb, this->state_.x, this->state_.xb2, layer_weights.rms_ffn_weight, curr_conc_lvl );

  // now for ffn in we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
  // first calculate self.w1(x) and self.w3(x)
  ops_.template matmul<Config::dim, Config::hidden_dim>(
    this->state_.hb, this->state_.xb, layer_weights.w1, curr_conc_lvl );
  ops_.template matmul<Config::dim, Config::hidden_dim>(
    this->state_.hb2, this->state_.xb, layer_weights.w3, curr_conc_lvl );

  ops_.template silu<Config::hidden_dim>( this->state_.hb, this->state_.hb2, curr_conc_lvl );

  // final matmul to get the output of the ffn
  ops_.template matmul<Config::hidden_dim, Config::dim>(
    this->state_.xb, this->state_.hb, layer_weights.w2, curr_conc_lvl );

  // residual connection
  ops_.template accum<Config::dim>( this->state_.x, this->state_.xb, curr_conc_lvl );
}

template<typename Config, typename DType>
void Llama2<Config, DType>::pass_end()
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

template<typename Config, typename DType>
void extract_batch_token( LlamaOperations<Config, DType>& ops,
                          RunState<Config, DType>& state,
                          const std::vector<float>& temp )
{
  ops.template soft_sample<Config::vocab_size>( state.logits, temp, temp.size() );
  ops.template argmax<Config::vocab_size>( state.argmax_pos, state.logits, state.x, temp.size() );
}

template<typename Config, typename DType>
std::vector<InferenceState> Llama2<Config, DType>::forward( std::vector<InferenceState>&& inference_states,
                                                            const std::vector<std::shared_ptr<ContextType>>& contexts )
{
  this->assert_safe_forward( inference_states, contexts );

  for ( size_t i = 0; i < inference_states.size(); i++ ) {
    this->state_.batch_token_positions[i] = inference_states[i].token_pos();
  }

  this->state_.curr_concurrency_size = inference_states.size();
  const uint32_t next_layer_batch = inference_states[0].next_layer();

  if ( next_layer_batch == 0 ) {
    std::vector<uint32_t> token_vector;
    token_vector.reserve( inference_states.size() );
    for ( auto& state : inference_states ) {
      token_vector.push_back( state.token() );
    }
    pass_begin( token_vector );
  } else {
    for ( size_t i = 0; i < inference_states.size(); i++ )
      // load the activations
      ops_.copy( this->state_.x + i * Config::dim,
                 reinterpret_cast<DType*>( inference_states[i].activations().data() ),
                 Config::dim * sizeof( DType ) );
  }

  for ( size_t layer_num = next_layer_batch; layer_num <= this->settings_.end_layer_num; layer_num++ ) {
    for ( size_t i = 0; i < inference_states.size(); i++ ) {
      this->state_.batch_context_pointers[i] = contexts[i]->key( this->settings_, layer_num, 0 );
    }

    pre_attention_ops( layer_num );
    attention_ops();
    post_attention_ops( layer_num );
  }

  std::vector<InferenceState> output_states;

  if ( this->settings_.end_layer_num == Config::n_layers - 1 ) {
    pass_end();

    std::vector<float> batch_temps;
    batch_temps.reserve( inference_states.size() );

    for ( size_t i = 0; i < inference_states.size(); i++ )
      batch_temps.push_back( inference_states[i].temperature() );

    extract_batch_token( this->ops_, this->state_, batch_temps );

    for ( size_t i = 0; i < inference_states.size(); i++ ) {
      inference_states[i].set_token( this->state_.argmax_pos[i] );
      inference_states[i].set_token_pos( inference_states[i].token_pos() + 1 );
      inference_states[i].set_next_layer( 0 );
      inference_states[i].set_activations( {} );
      output_states.push_back( std::move( inference_states[i] ) );
    }

    return output_states;
  }

  for ( size_t i = 0; i < inference_states.size(); i++ ) {
    DataBuffer activations { Config::dim * sizeof( DType ), this->state_.x + i * Config::dim };
    inference_states[i].set_next_layer( this->settings_.end_layer_num + 1 );
    inference_states[i].set_activations( std::move( activations ) );
    output_states.push_back( std::move( inference_states[i] ) );
  }

  return output_states;
}

template<typename Config, typename DType>
std::vector<InferenceState> Llama2<Config, DType>::pre_attention_forward(
  std::vector<InferenceState>&& inference_states,
  const std::vector<std::shared_ptr<ContextType>>& contexts )
{
  this->assert_safe_pre_attention( inference_states, contexts );
  const uint32_t next_layer_batch = inference_states[0].next_layer();

  this->state_.curr_concurrency_size = inference_states.size();

  if ( inference_states[0].next_layer() == 0 ) {
    std::vector<uint32_t> token_vector;
    for ( size_t i = 0; i < inference_states.size(); i++ ) {
      token_vector.push_back( inference_states[i].token() );
    }
    pass_begin( token_vector );
  } else {
    for ( size_t i = 0; i < inference_states.size(); i++ )
      // load the activations
      ops_.copy( this->state_.x + i * Config::dim,
                 reinterpret_cast<DType*>( inference_states[i].activations().data() ),
                 Config::dim * sizeof( DType ) );
  }

  for ( size_t i = 0; i < inference_states.size(); i++ ) {
    this->state_.batch_token_positions[i] = inference_states[i].token_pos();
    this->state_.batch_context_pointers[i] = contexts[i]->key( this->settings_, inference_states[i].next_layer(), 0 );
  }

  pre_attention_ops( next_layer_batch );

  std::vector<InferenceState> output_states;

  for ( size_t i = 0; i < inference_states.size(); i++ ) {
    DataBuffer activations { ( Config::dim + 2 * Config::kv_dim ) * sizeof( DType ) };
    ops_.copy(
      reinterpret_cast<DType*>( activations.data() ), this->state_.q + i * Config::dim, Config::dim * sizeof( DType ) );

    if ( contexts[i]->empty() ) {
      ops_.copy( reinterpret_cast<DType*>( activations.data() + Config::dim * sizeof( DType ) ),
                 this->state_.k + i * Config::kv_dim,
                 Config::kv_dim * sizeof( DType ) );

      ops_.copy( reinterpret_cast<DType*>( activations.data() + ( Config::dim + Config::kv_dim ) * sizeof( DType ) ),
                 this->state_.v + i * Config::kv_dim,
                 Config::kv_dim * sizeof( DType ) );
    }

    inference_states[i].set_next_stage( InferenceState::Stage::Attention );
    inference_states[i].set_next_layer( this->settings_.end_layer_num + 1 );
    inference_states[i].set_activations( std::move( activations ) );
    output_states.push_back( std::move( inference_states[i] ) );
  }

  return output_states;
}

template<typename Config, typename DType>
std::vector<InferenceState> Llama2<Config, DType>::attention_forward(
  std::vector<InferenceState>&& inference_states,
  const std::vector<std::shared_ptr<ContextType>>& contexts )
{
  this->assert_safe_attention( inference_states, contexts );

  this->state_.curr_concurrency_size = inference_states.size();

  for ( size_t i = 0; i < inference_states.size(); i++ ) {
    this->state_.batch_token_positions[i] = inference_states[i].token_pos();
    this->state_.batch_context_pointers[i] = contexts[i]->key( this->settings_, inference_states[i].next_layer(), 0 );
    // load the activations

    auto activation_data = inference_states[i].activations().data();
    auto activation_len = inference_states[i].activations().len();

    switch ( inference_states[i].dtype() ) {
      case DataType::Float16:
        // Q should go to run state
        ops_.template convert_and_copy( this->state_.q + i * Config::dim,
                                        reinterpret_cast<_Float16*>( activation_data ),
                                        Config::dim,
                                        CopyType::HostToDevice );

        // if KV is not already in context, put it there
        if ( activation_len > Config::dim * DataTypeSize( inference_states[i].dtype() ) ) {
          ops_.template convert_and_copy(
            contexts[i]->key( this->settings_, inference_states[i].next_layer(), inference_states[i].token_pos() ),
            reinterpret_cast<_Float16*>( activation_data + Config::dim * sizeof( DType ) ),
            Config::kv_dim * 2,
            CopyType::HostToDevice );
        }
        break;
      case DataType::Float32:
        // Q should go to run state
        ops_.template convert_and_copy( this->state_.q + i * Config::dim,
                                        reinterpret_cast<float*>( activation_data ),
                                        Config::dim,
                                        CopyType::HostToDevice );

        // if KV is not already in context, put it there
        if ( activation_len > Config::dim * DataTypeSize( inference_states[i].dtype() ) ) {
          ops_.template convert_and_copy(
            contexts[i]->key( this->settings_, inference_states[i].next_layer(), inference_states[i].token_pos() ),
            reinterpret_cast<float*>( activation_data + Config::dim * sizeof( DType ) ),
            Config::kv_dim * 2,
            CopyType::HostToDevice );
        }
        break;
      default: throw std::runtime_error( "invalid dtype" );
    }
  }

  attention_ops();

  std::vector<InferenceState> output_states;

  for ( size_t i = 0; i < inference_states.size(); i++ ) {
    DataBuffer activations { Config::dim * DataTypeSize( inference_states[i].dtype() ) };

    switch ( inference_states[i].dtype() ) {
      case DataType::Float16:
        ops_.template convert_and_copy( reinterpret_cast<_Float16*>( activations.data() ),
                                        this->state_.xb + i * Config::dim,
                                        Config::dim,
                                        CopyType::DeviceToHost );
        break;
      case DataType::Float32:
        ops_.template convert_and_copy( reinterpret_cast<float*>( activations.data() ),
                                        this->state_.xb + i * Config::dim,
                                        Config::dim,
                                        CopyType::DeviceToHost );
        break;
      default: throw std::runtime_error( "invalid dtype" );
    }

    inference_states[i].set_next_stage( InferenceState::Stage::PostAttention );
    inference_states[i].set_next_layer( this->settings_.end_layer_num + 1 );
    inference_states[i].set_activations( std::move( activations ) );
    output_states.push_back( std::move( inference_states[i] ) );
  }

  return output_states;
}

template<typename Config, typename DType>
std::vector<InferenceState> Llama2<Config, DType>::post_attention_forward(
  std::vector<InferenceState>&& inference_states )
{
  this->assert_safe_post_attention( inference_states );
  const uint32_t next_layer_batch = inference_states[0].next_layer();

  this->state_.curr_concurrency_size = inference_states.size();

  for ( size_t i = 0; i < inference_states.size(); i++ )
    // load the activations
    ops_.copy( this->state_.xb + i * Config::dim,
               reinterpret_cast<DType*>( inference_states[i].activations().data() ),
               Config::dim * sizeof( DType ) );

  post_attention_ops( next_layer_batch );

  std::vector<InferenceState> output_states;

  if ( next_layer_batch + 1 == Config::n_layers - 1 ) {
    pass_end();

    std::vector<float> batch_temps;
    for ( size_t i = 0; i < inference_states.size(); i++ )
      batch_temps.push_back( inference_states[i].temperature() );

    extract_batch_token( this->ops_, this->state_, batch_temps );

    for ( size_t i = 0; i < inference_states.size(); i++ ) {
      inference_states[i].set_token( this->state_.argmax_pos[i] );
      inference_states[i].set_token_pos( inference_states[i].token_pos() + 1 );
      inference_states[i].set_next_stage( InferenceState::Stage::PreAttention );
      inference_states[i].set_next_layer( 0 );
      inference_states[i].set_activations( {} );
      output_states.push_back( std::move( inference_states[i] ) );
    }

    return output_states;
  }

  for ( size_t i = 0; i < inference_states.size(); i++ ) {
    DataBuffer activations { Config::dim * sizeof( DType ), this->state_.x + i * Config::dim };
    inference_states[i].set_next_stage( InferenceState::Stage::PreAttention );
    inference_states[i].set_next_layer( this->settings_.end_layer_num + 1 );
    inference_states[i].set_activations( std::move( activations ) );
    output_states.push_back( std::move( inference_states[i] ) );
  }

  return output_states;
}

template<typename Config, typename DType>
InferenceState Llama2<Config, DType>::forward( InferenceState&& inference_state, std::shared_ptr<ContextType> context )
{
  std::vector<InferenceState> token_vector;
  token_vector.push_back( std::move( inference_state ) );
  std::vector<std::shared_ptr<ContextType>> context_vector;
  context_vector.push_back( std::move( context ) );
  return std::move( forward( std::move( token_vector ), context_vector )[0] );
}

template<typename Config, typename DType>
InferenceState Llama2<Config, DType>::pre_attention_forward( InferenceState&& inference_state,
                                                             std::shared_ptr<ContextType> context )
{
  std::vector<InferenceState> token_vector;
  token_vector.push_back( std::move( inference_state ) );
  std::vector<std::shared_ptr<ContextType>> context_vector;
  context_vector.push_back( std::move( context ) );
  return std::move( pre_attention_forward( std::move( token_vector ), context_vector )[0] );
}

template<typename Config, typename DType>
InferenceState Llama2<Config, DType>::attention_forward( InferenceState&& inference_state,
                                                         std::shared_ptr<ContextType> context )
{
  std::vector<InferenceState> token_vector;
  token_vector.push_back( std::move( inference_state ) );
  std::vector<std::shared_ptr<ContextType>> context_vector;
  context_vector.push_back( std::move( context ) );
  return std::move( attention_forward( std::move( token_vector ), context_vector )[0] );
}

template<typename Config, typename DType>
InferenceState Llama2<Config, DType>::post_attention_forward( InferenceState&& inference_state )
{
  std::vector<InferenceState> token_vector;
  token_vector.push_back( std::move( inference_state ) );
  return std::move( post_attention_forward( std::move( token_vector ) )[0] );
}

#define INSTANTIATE_FOR_MODEL( X )                                                                                     \
  template class Context<X, float>;                                                                                    \
  template class Context<X, _Float16>;                                                                                 \
  template class Llama2<X, float>;                                                                                     \
  template class Llama2<X, _Float16>;

INSTANTIATE_FOR_MODEL( configs::Llama2_7B_Chat )
INSTANTIATE_FOR_MODEL( configs::Llama2_13B_Chat )
INSTANTIATE_FOR_MODEL( configs::Llama2_70B_Chat )
INSTANTIATE_FOR_MODEL( configs::Stories_110M )

} // namespace glinthawk::models::llama2::cpu
