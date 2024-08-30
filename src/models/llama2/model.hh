#pragma once

#include <algorithm>
#include <execution>
#include <fcntl.h>
#include <filesystem>
#include <glog/logging.h>
#include <random>

#include "base.hh"
#include "variants.hh"

#include "models/common/state.hh"
#include "ops/concept.hh"
#include "util/exception.hh"
#include "util/file_descriptor.hh"
#include "util/ring_buffer.hh"

#if defined( TARGET_PLATFORM_AMD64 ) || defined( TARGET_PLATFORM_CUDA )
#include "arch/amd64/llama2/ops.hh"
#endif

#if defined( TARGET_PLATFORM_CUDA )
#include "arch/cuda/llama2/ops.cuh"
#endif

namespace glinthawk::models::llama2 {

template<typename Config, typename DType, typename LlamaOperations, typename Context>
requires ModelConfig<Config> && LlamaOperationsConcept<LlamaOperations, DType, ConfigRuntime<Config>>
         && ContextConcept<Context, DType>
class Llama2
{
public:
  using ContextPtr = std::shared_ptr<Context>;
  using Operations = LlamaOperations;
  using ContextVector = std::vector<ContextPtr>;

  using ModelDataType = DType;
  using ContextType = Context;
  using ConfigType = Config;
  using SettingsType = ConfigRuntime<Config>;

public:
  Llama2( const std::filesystem::path& model_dir,
          const uint32_t start_layer = 0,
          const uint32_t end_layer = std::numeric_limits<uint32_t>::max(),
          const uint64_t concurrency_limit = 1,
          const uint64_t max_context_count = 1,
          const bool randomize_parameters = false );

  /// \note forward_*() functions mutate the input state object.
  // (input token|activations) -> {forward: [(pre -> att -> post) x n_layers] -> (?classify -> output token)}

  template<StateConcept StateType>
  void forward( StateType& state, const ContextVector& ctxs );

  template<StateConcept StateType>
  void forward_pre_attention( StateType& state );

  template<StateConcept StateType>
  void forward_attention( StateType& state, const ContextVector& ctxs );

  template<StateConcept StateType>
  void forward_post_attention( StateType& state );

  template<StateConcept StateType>
  void forward_classify( StateType& state );

  ConfigRuntime<Config> settings() const { return instance_config_; }
  Operations& ops() { return ops_; }

private:
  static constexpr uint32_t TOKEN_BOS = 1; // Beginning-of-sequence token
  static constexpr uint32_t TOKEN_EOS = 2; // End-of-sequence token

protected:
  const ConfigRuntime<Config> instance_config_;
  Operations ops_ { instance_config_ };

  typename Operations::DeviceUniquePtr base_weights_buffer_;
  typename Operations::DeviceUniquePtr layers_buffer_;
  typename Operations::DeviceUniquePtr scratchpad_buffer_;

  BaseWeights<Config, DType> base_weights_;
  std::array<LayerWeights<Config, DType>, Config::n_layers> layer_weights_;
  ScratchPad<Config, DType, Context> scratchpad_;

  // Checking if the inference states are safe to pass to the model
  template<StateConcept StateType>
  void check_batch( const StateType& inference_states,
                    const ContextVector& contexts,
                    const InferenceStage stage ) const;

  template<StateConcept StateType>
  void load_embedding( const StateType& inference_state );

  template<StateConcept StateType>
  void forward_prelude( StateType& inference_state, const ContextVector& contexts );

  template<StateConcept StateType>
  void forward_postlude( StateType& inference_state, const int32_t most_recent_layer_num, const bool classified );

  void pre_attention_ops( const int32_t layer_num, const bool update_kv_cache = false );
  void attention_ops();
  void post_attention_ops( const int32_t layer_num );
  void classify_ops();
};

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
                          ScratchPad<Config, DType, Context>& state,
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
  : instance_config_( model_dir / "CONFIG",
                      start_layer,
                      end_layer,
                      concurrency_limit,
                      max_context_count,
                      randomize_parameters )
  , base_weights_buffer_( ops_.device_allocate( BaseWeights<Config, DType>::base_size() ) )
  , layers_buffer_(
      ops_.device_allocate( LayerWeights<Config, DType>::layer_size() * instance_config_.n_layers_loaded() ) )
  , scratchpad_buffer_(
      ops_.device_allocate( ScratchPad<Config, DType, Context>::scratchpad_size( instance_config_ ) ) )
  , base_weights_( base_weights_buffer_.get() )
  , layer_weights_( [&] {
    std::array<LayerWeights<Config, DType>, Config::n_layers> layers {};
    constexpr size_t layer_size = LayerWeights<Config, DType>::layer_size();
    auto ptr = layers_buffer_.get();
    for ( auto i = instance_config_.start_layer_num; i <= instance_config_.end_layer_num; i++ ) {
      layers[i] = LayerWeights<Config, DType> { reinterpret_cast<DType*>(
        reinterpret_cast<uint8_t*>( ptr ) + ( i - instance_config_.start_layer_num ) * layer_size ) };
    }

    return layers;
  }() )
  , scratchpad_( instance_config_, scratchpad_buffer_.get() )
{
  auto copy_file_to_buffer = [this]( const std::filesystem::path& path, DType* buffer, const size_t size ) {
    CHECK_EQ( std::filesystem::file_size( path ), size ) << "File " << path << " is not the expected size.";
    FileDescriptor fd { CHECK_SYSCALL( "open", open( path.c_str(), O_RDONLY ) ) };
    MMap_Region mmap { nullptr, size, PROT_READ, MAP_PRIVATE, fd.fd_num(), 0 };
    this->ops_.copy( buffer, reinterpret_cast<DType*>( mmap.addr() ), size, CopyType::HostToDevice );
  };

  // Ugly
  const std::string filename_suffix = "_" + dtype_str<DType>();

  const auto base_size = BaseWeights<Config, DType>::base_size();
  const auto layer_size = LayerWeights<Config, DType>::layer_size();

  if ( randomize_parameters ) {
    LOG( WARNING ) << "Randomizing weights and scratchpad...";

    ops_.randomize_device_buffer( base_weights_buffer_.get(),
                                  base_size / sizeof( DType ),
                                  -10.0 / sqrt( Config::dim ),
                                  10.0 / sqrt( Config::dim ) );

    ops_.randomize_device_buffer( layers_buffer_.get(),
                                  layer_size * instance_config_.n_layers_loaded() / sizeof( DType ),
                                  -10.0 / sqrt( Config::dim ),
                                  10.0 / sqrt( Config::dim ) );

    ops_.randomize_device_buffer( scratchpad_buffer_.get(),
                                  ScratchPad<Config, DType, Context>::scratchpad_size( instance_config_ )
                                    / sizeof( DType ),
                                  -10.0 / sqrt( Config::dim ),
                                  10.0 / sqrt( Config::dim ) );

    LOG( WARNING ) << "Randomizing weights and run state... done.";
  } else { // not randomize_parameters
    copy_file_to_buffer( model_dir / ( "BASEWEIGHTS" + filename_suffix ), base_weights_buffer_.get(), base_size );

    LOG( INFO ) << "Loaded base weights (" << base_size << " bytes).";

    // Load LAYER(i)
    for ( auto i = instance_config_.start_layer_num; i <= instance_config_.end_layer_num; i++ ) {
      const auto filename = model_dir / ( "LAYER" + std::to_string( i ) + filename_suffix );
      DType* ptr = reinterpret_cast<DType*>( reinterpret_cast<uint8_t*>( layers_buffer_.get() )
                                             + ( i - instance_config_.start_layer_num ) * layer_size );

      copy_file_to_buffer( filename, ptr, layer_size );
    }

    LOG( INFO ) << "Loaded layer weights (" << instance_config_.start_layer_num << " to "
                << instance_config_.end_layer_num << ", " << ( layer_size * instance_config_.n_layers_loaded() )
                << " bytes).";
  }

  LOG( INFO ) << "Model " << typeid( decltype( this ) ).name() << " instantiated.";
}
template<typename Config, typename DType, typename LlamaOperations, typename Context>
template<StateConcept StateType>
void Llama2<Config, DType, LlamaOperations, Context>::check_batch(
  const StateType& states,
  const std::vector<std::shared_ptr<Context>>& contexts,
  const InferenceStage stage ) const
{
  CHECK_GT( states.batch_size(), 0 );
  CHECK_LE( states.batch_size(), instance_config_.concurrency_limit );

  if ( stage == InferenceStage::Attention ) {
    CHECK_EQ( states.batch_size(), contexts.size() );
  }

  const uint32_t next_layer_batch = states.next_layer();

  CHECK_LE( instance_config_.start_layer_num, next_layer_batch );
  CHECK_LE( next_layer_batch, instance_config_.end_layer_num );

  CHECK( states.next_stage() == stage );

  if ( stage != InferenceStage::Attention ) {
    CHECK_DTYPE<DType>( states.dtype() );
    CHECK_EQ( states.next_layer(), next_layer_batch );
  }
}

template<typename Config, typename DType, typename LlamaOperations, typename Context>
template<StateConcept StateType>
void Llama2<Config, DType, LlamaOperations, Context>::load_embedding( const StateType& states )
{
  for ( size_t i = 0; i < states.batch_size(); i++ ) {
    if ( states.active( i ) ) {
      const auto token = states.token( i );
      CHECK_LT( token, Config::vocab_size );

      const DType* content_row = this->base_weights_.token_embedding_table + token * Config::dim;
      ops_.copy(
        this->scratchpad_.x + i * Config::dim, content_row, Config::dim * sizeof( DType ), CopyType::HostToDevice );
    }
  }
}

template<typename Config, typename DType, typename LlamaOperations, typename Context>
void Llama2<Config, DType, LlamaOperations, Context>::pre_attention_ops( const int32_t layer_num,
                                                                         const bool update_kv_cache )
{
  const auto& layer_weights = this->layer_weights_[layer_num];

  // attention rmsnorm
  ops_.template rmsnorm<Config::dim>( this->scratchpad_.xb,
                                      this->scratchpad_.x,
                                      this->scratchpad_.xb2,
                                      layer_weights.rms_att_weight,
                                      this->scratchpad_.curr_concurrency_size );

  // qkv matmuls for this position
  ops_.template matmul<Config::dim, Config::dim>(
    this->scratchpad_.q, this->scratchpad_.xb, layer_weights.wq, this->scratchpad_.curr_concurrency_size );

  ops_.template matmul<Config::dim, Config::kv_dim * 2>(
    this->scratchpad_.kv, this->scratchpad_.xb, layer_weights.wkv, this->scratchpad_.curr_concurrency_size );

  if ( update_kv_cache ) {
    // save key, value at each time step (pos) to our kv cache.
    // only necessary for the end-to-end forward() function.
    ops_.copy_kv_cache(
      this->scratchpad_.batch_token_contexts, this->scratchpad_.kv, this->scratchpad_.curr_concurrency_size );
  }
}

template<typename Config, typename DType, typename LlamaOperations, typename Context>
void Llama2<Config, DType, LlamaOperations, Context>::attention_ops()
{
  // TODO: We should either make parallel tokens in one prompt work, or remove the feature altogether (and put
  // protections in place).
  // XXX(sadjad): With HybridKernel, this will run on the CPU; maybe should be moved to pre-attention?
  ops_.apply_rope( this->scratchpad_.curr_concurrency_size,
                   this->scratchpad_.batch_token_positions,
                   this->base_weights_.freq_cis_real,
                   this->base_weights_.freq_cis_imag,
                   this->scratchpad_.q,
                   this->scratchpad_.batch_token_contexts );

  // <multihead attention> for each head and for each token up to and including the current one

  ops_.attention_0_gemm( this->scratchpad_.q,
                         this->scratchpad_.batch_layer_contexts,
                         this->scratchpad_.att,
                         this->scratchpad_.curr_concurrency_size,
                         this->scratchpad_.batch_token_positions );

  // softmax
  ops_.attention_softmax( this->scratchpad_.att,
                          this->scratchpad_.batch_token_positions,
                          this->scratchpad_.temp_softmax,
                          this->scratchpad_.curr_concurrency_size );

  ops_.attention_2_gemm( this->scratchpad_.att,
                         this->scratchpad_.batch_layer_contexts,
                         this->scratchpad_.xb,
                         this->scratchpad_.curr_concurrency_size,
                         this->scratchpad_.batch_token_positions );

  // </multihead attention>
}

template<typename Config, typename DType, typename LlamaOperations, typename Context>
void Llama2<Config, DType, LlamaOperations, Context>::post_attention_ops( const int32_t layer_num )
{
  const auto& layer_weights = this->layer_weights_[layer_num];

  // final matmul to get the output of the attention
  ops_.template matmul<Config::dim, Config::dim>(
    this->scratchpad_.xb2, this->scratchpad_.xb, layer_weights.wo, this->scratchpad_.curr_concurrency_size );

  // residual connection back into x
  ops_.template accum<Config::dim>(
    this->scratchpad_.x, this->scratchpad_.xb2, this->scratchpad_.curr_concurrency_size );

  // ffn rmsnorm
  ops_.template rmsnorm<Config::dim>( this->scratchpad_.xb,
                                      this->scratchpad_.x,
                                      this->scratchpad_.xb2,
                                      layer_weights.rms_ffn_weight,
                                      this->scratchpad_.curr_concurrency_size );

  // now for ffn in we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
  // first calculate self.w1(x) and self.w3(x)
  ops_.template matmul<Config::dim, Config::hidden_dim>(
    this->scratchpad_.hb, this->scratchpad_.xb, layer_weights.w1, this->scratchpad_.curr_concurrency_size );

  ops_.template matmul<Config::dim, Config::hidden_dim>(
    this->scratchpad_.hb2, this->scratchpad_.xb, layer_weights.w3, this->scratchpad_.curr_concurrency_size );

  ops_.template silu<Config::hidden_dim>(
    this->scratchpad_.hb, this->scratchpad_.hb2, this->scratchpad_.curr_concurrency_size );

  // final matmul to get the output of the ffn
  ops_.template matmul<Config::hidden_dim, Config::dim>(
    this->scratchpad_.xb, this->scratchpad_.hb, layer_weights.w2, this->scratchpad_.curr_concurrency_size );

  // residual connection
  ops_.template accum<Config::dim>(
    this->scratchpad_.x, this->scratchpad_.xb, this->scratchpad_.curr_concurrency_size );
}

template<typename Config, typename DType, typename LlamaOperations, typename Context>
void Llama2<Config, DType, LlamaOperations, Context>::classify_ops()
{
  // final rmsnorm
  ops_.template rmsnorm<Config::dim>( this->scratchpad_.x,
                                      this->scratchpad_.x,
                                      this->scratchpad_.xb2,
                                      this->base_weights_.rms_final_weight,
                                      this->scratchpad_.curr_concurrency_size );

  // classifier into logits
  ops_.template matmul<Config::dim, Config::vocab_size>(
    this->scratchpad_.logits, this->scratchpad_.x, this->base_weights_.wcls, this->scratchpad_.curr_concurrency_size );
}

template<typename Config, typename DType, typename LlamaOperations, typename Context>
template<StateConcept StateType>
void Llama2<Config, DType, LlamaOperations, Context>::forward_prelude( StateType& states,
                                                                       const ContextVector& contexts )
{
  this->check_batch( states, contexts, InferenceStage::PreAttention );

  this->scratchpad_.curr_concurrency_size = states.batch_size();
  const uint32_t next_layer_batch = states.next_layer();

  for ( size_t i = 0; i < contexts.size(); i++ ) {
    this->scratchpad_.batch_token_positions[i] = states.token_pos( i );
    this->scratchpad_.batch_layer_contexts[i] = contexts[i]->layer( next_layer_batch );
    this->scratchpad_.batch_token_contexts[i] = contexts[i]->layer( next_layer_batch ).token( states.token_pos( i ) );
  }

  if ( next_layer_batch == 0 ) {
    /* THE FIRST LAYER, just read the tokens */
    load_embedding( states );
  } else {
    /* NOT THE FIRST LAYER, load the activations */
    ops_.copy( this->scratchpad_.x,
               reinterpret_cast<DType*>( states.activations().data() ),
               states.activations().len(),
               CopyType::HostToDevice,
               true );
  }
}

template<typename Config, typename DType, typename LlamaOperations, typename Context>
template<StateConcept StateType>
void Llama2<Config, DType, LlamaOperations, Context>::forward_postlude( StateType& states,
                                                                        const int32_t most_recent_layer_num,
                                                                        const bool classification_done )
{
  if ( classification_done ) {
    CHECK_EQ( most_recent_layer_num, Config::n_layers - 1 );

    std::vector<float> batch_temps;
    for ( size_t i = 0; i < states.batch_size(); i++ ) {
      batch_temps.push_back( states.temperature( i ) );
    }

    extract_batch_token( this->ops_, this->scratchpad_, batch_temps );

    // we don't need to send the activations (or anything else) to the next worker, just the token
    states.deallocate_activations();
    states.deallocate_queries();
    states.deallocate_kvs();

    states.set_next_layer( 0 );
    states.set_next_stage( InferenceStage::PreAttention );

    for ( size_t i = 0; i < states.batch_size(); i++ ) {
      if ( states.active( i ) ) {
        states.set_token( i, this->scratchpad_.argmax_pos[i] );
        states.set_token_pos( i, states.token_pos( i ) + 1 );

        if ( states.token( i ) == TOKEN_EOS or states.token_pos( i ) >= Config::seq_len ) {
          // Discarding the prompt entry is left to the caller, we just set the finished flag here
          states.set_finished( i );
        }
      }
    }
  } else {
    // all we need to send is activations; the memory is already allocated
    states.allocate_activations();
    states.deallocate_queries();
    states.deallocate_kvs();

    ops_.copy( reinterpret_cast<DType*>( states.activations().data() ),
               this->scratchpad_.x,
               states.batch_size() * Config::dim * sizeof( DType ),
               CopyType::DeviceToHost );

    if ( most_recent_layer_num == Config::n_layers - 1 ) {
      states.set_next_stage( InferenceStage::Classification );
    } else {
      states.set_next_stage( InferenceStage::PreAttention );
      states.set_next_layer( most_recent_layer_num + 1 );
    }
  }
}

template<typename Config, typename DType, typename LlamaOperations, typename Context>
template<StateConcept StateType>
void Llama2<Config, DType, LlamaOperations, Context>::forward( StateType& states, const ContextVector& contexts )
{
  forward_prelude( states, contexts );

  for ( size_t layer_num = states.next_layer(); layer_num <= this->instance_config_.end_layer_num; layer_num++ ) {
    for ( size_t i = 0; i < contexts.size(); i++ ) {
      this->scratchpad_.batch_layer_contexts[i] = contexts[i]->layer( layer_num );
      this->scratchpad_.batch_token_contexts[i] = contexts[i]->layer( layer_num ).token( states.token_pos( i ) );
    }

    pre_attention_ops( layer_num, true );
    attention_ops();
    post_attention_ops( layer_num );
  }

  if ( this->instance_config_.end_layer_num == Config::n_layers - 1 ) {
    classify_ops();
    return forward_postlude( states, this->instance_config_.end_layer_num, true );
  } else {
    return forward_postlude( states, this->instance_config_.end_layer_num, false );
  }
}

template<typename Config, typename DType, typename LlamaOperations, typename Context>
template<StateConcept StateType>
void Llama2<Config, DType, LlamaOperations, Context>::forward_pre_attention( StateType& states )
{
  forward_prelude( states, {} );
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
             this->scratchpad_.x,
             states.activations().len(),
             CopyType::DeviceToHost );

  ops_.copy( reinterpret_cast<DType*>( states.queries().data() ),
             this->scratchpad_.q,
             states.queries().len(),
             CopyType::DeviceToHost );

  // XXX(sadjad): Copying KV is not always necessary (i.e., if context[i]->empty()), but for convenience we always do it
  ops_.copy(
    reinterpret_cast<DType*>( states.kvs().data() ), this->scratchpad_.kv, states.kvs().len(), CopyType::DeviceToHost );

  states.set_next_stage( InferenceStage::Attention );
}

template<typename Config, typename DType, typename LlamaOperations, typename Context>
template<StateConcept StateType>
void Llama2<Config, DType, LlamaOperations, Context>::forward_attention( StateType& states,
                                                                         const ContextVector& contexts )
{
  this->check_batch( states, contexts, InferenceStage::Attention );
  this->scratchpad_.curr_concurrency_size = states.batch_size();

  for ( size_t i = 0; i < states.batch_size(); i++ ) {
    this->scratchpad_.batch_token_positions[i] = states.token_pos( i );
    this->scratchpad_.batch_layer_contexts[i] = contexts[i]->layer( states.next_layer() );
    this->scratchpad_.batch_token_contexts[i]
      = contexts[i]->layer( states.next_layer() ).token( states.token_pos( i ) );
  }

  // NOTE: We allow mixing FP16 in pre-/post-attention with FP32 during attention. Hence, the conversions.

  switch ( states.dtype() ) {
    case DataType::Float16:
      ops_.template convert_and_copy( this->scratchpad_.q,
                                      reinterpret_cast<LlamaOperations::Float16*>( states.queries().data() ),
                                      states.queries().len() / sizeof( typename LlamaOperations::Float16 ),
                                      CopyType::HostToDevice );
      break;

    case DataType::Float32:
      ops_.template convert_and_copy( this->scratchpad_.q,
                                      reinterpret_cast<LlamaOperations::Float32*>( states.queries().data() ),
                                      states.queries().len() / sizeof( typename LlamaOperations::Float32 ),
                                      CopyType::HostToDevice );
      break;

    default: LOG( FATAL ) << "invalid dtype";
  }

  if ( states.has_kvs() ) {
    for ( size_t i = 0; i < states.batch_size(); i++ ) {
      if ( states.active( i ) ) {
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
                                      this->scratchpad_.xb,
                                      states.queries().len() / sizeof( typename LlamaOperations::Float16 ),
                                      CopyType::DeviceToHost );
      break;

    case DataType::Float32:
      ops_.template convert_and_copy( reinterpret_cast<LlamaOperations::Float32*>( states.queries().data() ),
                                      this->scratchpad_.xb,
                                      states.queries().len() / sizeof( typename LlamaOperations::Float32 ),
                                      CopyType::DeviceToHost );
      break;

    default: throw std::runtime_error( "invalid dtype" );
  }

  states.set_next_stage( InferenceStage::PostAttention );
}

template<typename Config, typename DType, typename LlamaOperations, typename Context>
template<StateConcept StateType>
void Llama2<Config, DType, LlamaOperations, Context>::forward_post_attention( StateType& states )
{
  this->check_batch( states, {}, InferenceStage::PostAttention );
  this->scratchpad_.curr_concurrency_size = states.batch_size();

  ops_.copy( this->scratchpad_.x,
             reinterpret_cast<DType*>( states.activations().data() ),
             states.activations().len(),
             CopyType::HostToDevice );

  ops_.copy( this->scratchpad_.xb,
             reinterpret_cast<DType*>( states.queries().data() ),
             states.queries().len(),
             CopyType::HostToDevice );

  post_attention_ops( states.next_layer() );
  forward_postlude( states, states.next_layer(), /* classification done? */ false );
}

template<typename Config, typename DType, typename LlamaOperations, typename Context>
template<StateConcept StateType>
void Llama2<Config, DType, LlamaOperations, Context>::forward_classify( StateType& states )
{
  this->check_batch( states, {}, InferenceStage::Classification );
  this->scratchpad_.curr_concurrency_size = states.batch_size();

  // load the activations
  ops_.copy( this->scratchpad_.x,
             reinterpret_cast<DType*>( states.activations().data() ),
             states.activations().len(),
             CopyType::HostToDevice );

  classify_ops();
  forward_postlude( states, states.next_layer(), /* classification done? */ true );
}

#define DECLARE_MODEL( PLATFORM, MODEL_NAME )                                                                          \
  template<typename DType>                                                                                             \
  using MODEL_NAME                                                                                                     \
    = Llama2<configs::MODEL_NAME,                                                                                      \
             DType,                                                                                                    \
             PLATFORM::LlamaOperations<configs::MODEL_NAME, DType, PLATFORM::Context<configs::MODEL_NAME, DType>>,     \
             PLATFORM::Context<configs::MODEL_NAME, DType>>

#if defined( TARGET_PLATFORM_AMD64 ) || defined( TARGET_PLATFORM_CUDA )
namespace amd64 {
DECLARE_MODEL( amd64, Llama2_7B_Chat );
DECLARE_MODEL( amd64, Llama2_13B_Chat );
DECLARE_MODEL( amd64, Llama2_70B_Chat );
DECLARE_MODEL( amd64, Stories_110M );
}
#endif

#if defined( TARGET_PLATFORM_CUDA )
namespace cuda {
DECLARE_MODEL( cuda, Llama2_7B_Chat );
DECLARE_MODEL( cuda, Llama2_13B_Chat );
DECLARE_MODEL( cuda, Llama2_70B_Chat );
DECLARE_MODEL( cuda, Stories_110M );
}
#endif

} // namespace glinthawk::models::llama2
