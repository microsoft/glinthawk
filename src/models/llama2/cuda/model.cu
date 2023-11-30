#include "model.cuh"

using namespace glinthawk::models::common::cuda;

namespace glinthawk::models::llama2::cuda {

namespace {

template<typename DType>
std::string dtype_str()
{
  if constexpr ( std::is_same_v<DType, float> ) {
    return "FP32";
  } else if constexpr ( std::is_same_v<DType, __half> ) {
    return "FP16";
  } else {
    throw std::runtime_error( "invalid dtype" );
  }
}

}

template<typename Config, typename DType>
Context<Config, DType>::Context( const Settings<Config>& settings )
  : storage_( [&]() -> decltype( storage_ ) {
    DType* ptr;
    const cudaError_t err = cudaMalloc( &ptr, InferenceContext<Config, DType>::context_size( settings ) );
    if ( err == cudaSuccess ) {
      return std::unique_ptr<DType, CUDADeleter<DType>> { ptr };
    } else {
      return std::unique_ptr<DType, CUDADeleter<DType>> { nullptr };
    }
  }() )
{
  this->buffer_ = storage_.get();
}

template<typename Config, typename DType>
Llama2<Config, DType>::~Llama2()
{
  ops::destroy();
}

template<typename Config, typename DType>
Llama2<Config, DType>::Llama2( const std::filesystem::path& model_path,
                               const uint32_t start_layer,
                               const uint32_t end_layer,
                               const uint64_t concurrency_limit )
{
  ops::init( concurrency_limit );

  const std::string filename_suffix = "_" + dtype_str<DType>();
  const auto config_path = model_path / "CONFIG";
  const auto base_path = model_path / ( "BASEWEIGHTS" + filename_suffix );

  llama2::Settings<Config> settings { config_path, start_layer, end_layer, concurrency_limit };

  CHECK_GE( 1024, Config::n_heads ) << "Attention softmax has n_heads threads, and this cannot surpass 1024.";
  CHECK_GE( 1024, Config::dim / Config::n_heads / 2 ) << "RoPE has head_size / 2 threads and cannot surpass 1024.";
  CHECK_GE( ( 1 << 16 ) - 1, Config::n_heads ) << "RoPE has n_heads blocks, and this cannot surpass 2^16.";
  CHECK_GE( ( 1 << 16 ) - 1, Config::seq_len ) << "Attention softmax has seq_len blocks, and this cannot surpass 2^16.";

  CHECK_GE( 1024, ops::TPB ) << "Threads per block cannot surpass 1024.";
  CHECK_GE( ( 1 << 16 ) - 1, ops::div_ceil( Config::dim * settings.concurrency_limit, ops::TPB ) )
    << "Accum blocks cannot surpass 2^16.";
  CHECK_GE( ( 1 << 16 ) - 1, ops::div_ceil( Config::hidden_dim * settings.concurrency_limit, ops::TPB ) )
    << "Silu blocks cannot surpass 2^16.";
  CHECK_GE( ( 1 << 16 ) - 1, ops::div_ceil( Config::vocab_size, ops::TPB ) ) << "CuRAND blocks cannot surpass 2^16.";
  CHECK_GE( ( 1 << 16 ) - 1, ops::div_ceil( Config::dim, ops::NRBS ) ) << "RMS Norm blocks cannot surpass 2^16.";
  CHECK_GE( sizeof( DType ) * Config::dim,
            sizeof( float )
              * ( ops::div_ceil( Config::dim, 2 * ops::NRBS )
                  + ops::div_ceil( ops::div_ceil( Config::dim, 2 * ops::NRBS ), 2 * ops::NRBS ) + 1 ) )
    << "RMS Norm scratch pad does not have enough space.";
  CHECK_GE( sizeof( DType ) * ( 4 * Config::dim + 2 * Config::hidden_dim ),
            sizeof( uint32_t )
                * ( ops::div_ceil( Config::vocab_size, 2 * ops::AMRBS )
                    + ops::div_ceil( ops::div_ceil( Config::vocab_size, 2 * ops::AMRBS ), 2 * ops::AMRBS ) + 1 )
              + sizeof( DType )
                  * ( ops::div_ceil( Config::vocab_size, 2 * ops::AMRBS )
                      + ops::div_ceil( ops::div_ceil( Config::vocab_size, 2 * ops::AMRBS ), 2 * ops::AMRBS ) ) )
    << "Argmax scratch pad does not have enough space.";

  const int32_t layer_count = settings.n_layers_loaded();

  const auto base_size = BaseWeights<Config, DType>::base_size();
  const auto layer_size = LayerWeights<Config, DType>::layer_size();
  const auto run_state_size = RunState<Config, DType>::state_size( settings );

  DType* base_raw_ptr;
  DType* layers_raw_ptr;
  DType* run_state_raw_ptr;

  // Allocate memory for the base weights
  ops::CHECK_CUDA( cudaMalloc( &base_raw_ptr, base_size ) );
  std::unique_ptr<DType, CUDADeleter<DType>> base { base_raw_ptr };

  // Allocate memory for the layers
  ops::CHECK_CUDA( cudaMalloc( &layers_raw_ptr, layer_size * layer_count ) );
  std::unique_ptr<DType, CUDADeleter<DType>> layers { layers_raw_ptr };

  // Allocate memory for the run state
  ops::CHECK_CUDA( cudaMalloc( &run_state_raw_ptr, run_state_size ) );
  ops::CHECK_CUDA( cudaMemset( run_state_raw_ptr, 0, run_state_size ) );
  std::unique_ptr<DType, CUDADeleter<DType>> run_state { run_state_raw_ptr };

  // Load the model

  // (1) loading the base weights
  {
    CHECK_EQ( std::filesystem::file_size( base_path ), base_size ) << "Base weights are not the expected size.";

    FileDescriptor base_fd { CHECK_SYSCALL( "open", open( base_path.c_str(), O_RDONLY ) ) };
    MMap_Region base_mmap { nullptr, base_size, PROT_READ, MAP_PRIVATE, base_fd.fd_num(), 0 };
    ops::CHECK_CUDA( cudaMemcpy( base.get(), base_mmap.addr(), base_size, cudaMemcpyHostToDevice ) );

    LOG( INFO ) << "Loaded base weights (" << base_size << " bytes).";
  }

  // (2) load the layers
  for ( auto i = settings.start_layer_num; i <= settings.end_layer_num; i++ ) {
    const auto layer_path = model_path / ( "LAYER" + std::to_string( i ) + filename_suffix );

    CHECK_EQ( std::filesystem::file_size( layer_path ), layer_size ) << "Layer " << i << " is not the expected size.";

    FileDescriptor layer_fd { CHECK_SYSCALL( "open", open( layer_path.c_str(), O_RDONLY ) ) };
    MMap_Region layer_mmap { nullptr, layer_size, PROT_READ, MAP_PRIVATE, layer_fd.fd_num(), 0 };

    ops::CHECK_CUDA(
      cudaMemcpy( reinterpret_cast<uint8_t*>( layers.get() ) + ( i - settings.start_layer_num ) * layer_size,
                  layer_mmap.addr(),
                  layer_size,
                  cudaMemcpyHostToDevice ) );

    LOG( INFO ) << "Loaded layer " << i << " (" << layer_size << " bytes).";
  }

  this->init( settings, std::move( base ), std::move( layers ), std::move( run_state ) );

  ops::setup_rng( this->state_.rng_state, 1234, Config::vocab_size, settings.concurrency_limit );
  cudaDeviceSynchronize();
}

template<typename Config, typename DType>
void Llama2<Config, DType>::pass_begin( const std::vector<uint32_t>& token )
{
  // copy the token embedding into the state
  for ( size_t i = 0; i < token.size(); i++ ) {
    CHECK_LT( token[i], Config::vocab_size ) << "token index must not surpass vocab size";
    const DType* content_row = this->base_weights_.token_embedding_table + token[i] * Config::dim;
    ops::CHECK_CUDA( cudaMemcpyAsync(
      this->state_.x + i * Config::dim, content_row, Config::dim * sizeof( DType ), cudaMemcpyDeviceToDevice ) );
  }
}

template<typename Config, typename DType>
void Llama2<Config, DType>::pre_attention_ops( const int32_t layer_num )
{
  const uint64_t curr_conc_lvl = this->state_.curr_concurrency_size;
  const auto& layer_weights = this->layer_weights_[layer_num];

  // attention rmsnorm
  ops::rmsnorm(
    this->state_.xb, this->state_.x, this->state_.xb2, layer_weights.rms_att_weight, Config::dim, curr_conc_lvl );

  // qkv matmuls for this position
  ops::matmul( this->state_.q, this->state_.xb, layer_weights.wq, curr_conc_lvl, Config::dim, Config::dim );
  ops::matmul( this->state_.k, this->state_.xb, layer_weights.wk, curr_conc_lvl, Config::dim, Config::kv_dim );
  ops::matmul( this->state_.v, this->state_.xb, layer_weights.wv, curr_conc_lvl, Config::dim, Config::kv_dim );

  // save key,value at each time step (pos) to our kv cache
  ops::copy_kv_cache( this->state_.batch_context_pointers,
                      this->state_.k,
                      this->state_.v,
                      Config::kv_dim,
                      curr_conc_lvl,
                      this->state_.batch_token_positions );
}

template<typename Config, typename DType>
void Llama2<Config, DType>::attention_ops()
{
  const uint64_t curr_conc_lvl = this->state_.curr_concurrency_size;

  ops::apply_rope( Config::head_size,
                   Config::n_kv_heads,
                   Config::gqa_size,
                   curr_conc_lvl,
                   this->state_.batch_token_positions,
                   this->base_weights_.freq_cis_real,
                   this->base_weights_.freq_cis_imag,
                   this->state_.q,
                   this->state_.batch_context_pointers );

  // <multihead attention> for each head and for each token up to and including the current one
  ops::attention_0_gemm( this->state_.q,
                         this->state_.batch_context_pointers,
                         this->state_.att,
                         Config::seq_len,
                         Config::head_size,
                         Config::n_kv_heads,
                         Config::gqa_size,
                         curr_conc_lvl,
                         this->state_.batch_token_positions );

  // softmax
  ops::attention_softmax( this->state_.att,
                          this->state_.batch_token_positions,
                          Config::seq_len,
                          Config::n_heads,
                          this->state_.temp_softmax,
                          curr_conc_lvl );

  ops::attention_2_gemm( this->state_.att,
                         this->state_.batch_context_pointers,
                         this->state_.xb,
                         Config::seq_len,
                         Config::head_size,
                         Config::n_kv_heads,
                         Config::gqa_size,
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
  ops::matmul( this->state_.xb2, this->state_.xb, layer_weights.wo, curr_conc_lvl, Config::dim, Config::dim );

  // residual connection back into x
  ops::accum( this->state_.x, this->state_.xb2, Config::dim, curr_conc_lvl );

  // ffn rmsnorm
  ops::rmsnorm(
    this->state_.xb, this->state_.x, this->state_.xb2, layer_weights.rms_ffn_weight, Config::dim, curr_conc_lvl );

  // now for ffn in we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
  // first calculate self.w1(x) and self.w3(x)
  ops::matmul( this->state_.hb, this->state_.xb, layer_weights.w1, curr_conc_lvl, Config::dim, Config::hidden_dim );
  ops::matmul( this->state_.hb2, this->state_.xb, layer_weights.w3, curr_conc_lvl, Config::dim, Config::hidden_dim );

  ops::silu( this->state_.hb, this->state_.hb2, Config::hidden_dim, curr_conc_lvl );

  // final matmul to get the output of the ffn
  ops::matmul( this->state_.xb, this->state_.hb, layer_weights.w2, curr_conc_lvl, Config::hidden_dim, Config::dim );

  // residual connection
  ops::accum( this->state_.x, this->state_.xb, Config::dim, curr_conc_lvl );
}

template<typename Config, typename DType>
void Llama2<Config, DType>::pass_end()
{
  // final rmsnorm
  ops::rmsnorm( this->state_.x,
                this->state_.x,
                this->state_.xb2,
                this->base_weights_.rms_final_weight,
                Config::dim,
                this->state_.curr_concurrency_size );

  // classifier into logits
  ops::matmul( this->state_.logits,
               this->state_.x,
               this->base_weights_.wcls,
               this->state_.curr_concurrency_size,
               Config::dim,
               Config::vocab_size );
}

template<typename Config, typename DType>
void extract_batch_token( RunState<Config, DType>& state, const std::vector<float>& temp )
{
  ops::soft_sample( state.logits, temp, state.rng_state, Config::vocab_size, temp.size() );
  ops::argmax( &( state.argmax_pos[0] ), state.logits, state.x, Config::vocab_size, temp.size() );
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
    for ( size_t i = 0; i < inference_states.size(); i++ ) {
      token_vector.push_back( inference_states[i].token() );
    }
    pass_begin( token_vector );
  } else {
    for ( size_t i = 0; i < inference_states.size(); i++ )
      // load the activations
      ops::CHECK_CUDA( cudaMemcpyAsync( this->state_.x + i * Config::dim,
                                        inference_states[i].activations().data(),
                                        Config::dim * sizeof( DType ),
                                        cudaMemcpyHostToDevice ) );
  }

  for ( auto layer_num = next_layer_batch; layer_num <= this->settings_.end_layer_num; layer_num++ ) {
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
    for ( size_t i = 0; i < inference_states.size(); i++ )
      batch_temps.push_back( inference_states[i].temperature() );

    extract_batch_token( this->state_, batch_temps );

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
    DataBuffer activations { Config::dim * sizeof( DType ) };

    ops::CHECK_CUDA( cudaMemcpy(
      activations.data(), this->state_.x + i * Config::dim, Config::dim * sizeof( DType ), cudaMemcpyDeviceToHost ) );

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
      ops::CHECK_CUDA( cudaMemcpyAsync( this->state_.x + i * Config::dim,
                                        inference_states[i].activations().data(),
                                        Config::dim * sizeof( DType ),
                                        cudaMemcpyHostToDevice ) );
  }

  for ( size_t i = 0; i < inference_states.size(); i++ ) {
    this->state_.batch_token_positions[i] = inference_states[i].token_pos();
    this->state_.batch_context_pointers[i] = contexts[i]->key( this->settings_, inference_states[i].next_layer(), 0 );
  }

  pre_attention_ops( next_layer_batch );

  std::vector<InferenceState> output_states;

  for ( size_t i = 0; i < inference_states.size(); i++ ) {
    DataBuffer activations { ( Config::dim + 2 * Config::kv_dim ) * sizeof( DType ) };

    ops::CHECK_CUDA( cudaMemcpy(
      activations.data(), this->state_.q + i * Config::dim, Config::dim * sizeof( DType ), cudaMemcpyDeviceToHost ) );

    if ( contexts[i]->empty() ) {
      ops::CHECK_CUDA( cudaMemcpy( activations.data() + Config::dim * sizeof( DType ),
                                   this->state_.k + i * Config::kv_dim,
                                   Config::kv_dim * sizeof( DType ),
                                   cudaMemcpyDeviceToHost ) );

      ops::CHECK_CUDA( cudaMemcpy( activations.data() + ( Config::dim + Config::kv_dim ) * sizeof( DType ),
                                   this->state_.v + i * Config::kv_dim,
                                   Config::kv_dim * sizeof( DType ),
                                   cudaMemcpyDeviceToHost ) );
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
        ops::cvt_and_copy_to_cuda(
          this->state_.q + i * Config::dim, reinterpret_cast<__half*>( activation_data ), Config::dim );
        // if KV is not already in context, put it there
        if ( activation_len > Config::dim * DataTypeSize( inference_states[i].dtype() ) ) {
          ops::cvt_and_copy_to_cuda(
            contexts[i]->key( this->settings_, inference_states[i].next_layer(), inference_states[i].token_pos() ),
            reinterpret_cast<__half*>( activation_data + Config::dim * sizeof( DType ) ),
            Config::kv_dim * 2 );
        }
        break;
      case DataType::Float32:
        // Q should go to run state
        ops::cvt_and_copy_to_cuda(
          this->state_.q + i * Config::dim, reinterpret_cast<float*>( activation_data ), Config::dim );
        // if KV is not already in context, put it there
        if ( activation_len > Config::dim * DataTypeSize( inference_states[i].dtype() ) ) {
          ops::cvt_and_copy_to_cuda(
            contexts[i]->key( this->settings_, inference_states[i].next_layer(), inference_states[i].token_pos() ),
            reinterpret_cast<float*>( activation_data + Config::dim * sizeof( DType ) ),
            Config::kv_dim * 2 );
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
        ops::cvt_and_copy_from_cuda(
          reinterpret_cast<__half*>( activations.data() ), this->state_.xb + i * Config::dim, Config::dim );
        break;
      case DataType::Float32:
        ops::cvt_and_copy_from_cuda(
          reinterpret_cast<float*>( activations.data() ), this->state_.xb + i * Config::dim, Config::dim );
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
    ops::CHECK_CUDA( cudaMemcpy( this->state_.xb + i * Config::dim,
                                 inference_states[i].activations().data(),
                                 Config::dim * sizeof( DType ),
                                 cudaMemcpyHostToDevice ) );

  post_attention_ops( next_layer_batch );

  std::vector<InferenceState> output_states;

  if ( next_layer_batch + 1 == Config::n_layers - 1 ) {
    pass_end();

    std::vector<float> batch_temps;
    for ( size_t i = 0; i < inference_states.size(); i++ )
      batch_temps.push_back( inference_states[i].temperature() );

    extract_batch_token( this->state_, batch_temps );

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
    DataBuffer activations { Config::dim * sizeof( DType ) };

    ops::CHECK_CUDA( cudaMemcpy(
      activations.data(), this->state_.x + i * Config::dim, Config::dim * sizeof( DType ), cudaMemcpyDeviceToHost ) );

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
  template class Context<X, __half>;                                                                                   \
  template class Llama2<X, float>;                                                                                     \
  template class Llama2<X, __half>;

INSTANTIATE_FOR_MODEL( configs::Llama2_7B_Chat );
INSTANTIATE_FOR_MODEL( configs::Llama2_13B_Chat );
INSTANTIATE_FOR_MODEL( configs::Llama2_70B_Chat );
INSTANTIATE_FOR_MODEL( configs::Stories_110M );

} // namespace glinthawk::models::llama2::cuda
