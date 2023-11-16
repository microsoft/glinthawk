#include "base.hh"

#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>

#ifdef GLINTHAWK_CUDA_ENABLED
#include <cuda_fp16.h>
#endif

#include <glog/logging.h>

using namespace std;
using namespace glinthawk::models::llama2;

/* MODEL CONFIG */

namespace {

template<typename DType>
DType* _advance_pointer( DType*& ptr, const size_t size )
{
  auto old = ptr;
  ptr += size;
  return old;
}

}

Config::Config( const filesystem::path& config_file,
                const uint32_t start_layer,
                const uint32_t end_layer,
                uint64_t concurrency_limit_ )
  : concurrency_limit( concurrency_limit_ )
{
  ifstream fin { config_file, ios::binary };
  CHECK( fin ) << "Failed to open config file: " << config_file;

  string raw_config;
  raw_config.resize( config_size() );

  fin.read( raw_config.data(), config_size() );
  if ( fin.gcount() != static_cast<streamsize>( config_size() ) ) {
    LOG( FATAL ) << "Failed to read config file: " << config_file;
  }

  auto ptr = raw_config.data();

  this->dim = *reinterpret_cast<const int32_t*>( _advance_pointer( ptr, sizeof( int32_t ) ) );
  this->hidden_dim = *reinterpret_cast<const int32_t*>( _advance_pointer( ptr, sizeof( int32_t ) ) );
  this->n_layers = *reinterpret_cast<const int32_t*>( _advance_pointer( ptr, sizeof( int32_t ) ) );
  this->n_heads = *reinterpret_cast<const int32_t*>( _advance_pointer( ptr, sizeof( int32_t ) ) );
  this->n_kv_heads = *reinterpret_cast<const int32_t*>( _advance_pointer( ptr, sizeof( int32_t ) ) );

  // if vocab size is negative, that means that wcls is present
  const auto original_vocab_size = *reinterpret_cast<const int32_t*>( _advance_pointer( ptr, sizeof( int32_t ) ) );

  this->vocab_size = abs( original_vocab_size );
  this->seq_len = *reinterpret_cast<const int32_t*>( _advance_pointer( ptr, sizeof( int32_t ) ) );

  if ( original_vocab_size < 0 ) {
    wcls_present = true;
  }

  this->gqa_size = this->n_heads / this->n_kv_heads;
  this->kv_dim = this->dim / this->gqa_size;
  this->head_size = this->dim / this->n_heads;

  CHECK_GT( dim, 0 ) << "Transformer dimension must be positive.";
  CHECK_GT( kv_dim, 0 ) << "key/value dimension must be positive.";
  CHECK_GT( hidden_dim, 0 ) << "FFN hidden dimension must be positive.";
  CHECK_GT( n_layers, 0 ) << "Number of layers must be positive.";
  CHECK_GT( head_size, 0 ) << "Head dimension must be positive.";
  CHECK_GT( n_heads, 0 ) << "Number of query heads must be positive.";
  CHECK_GT( n_kv_heads, 0 ) << "Number of key/value heads must be positive.";
  CHECK_GT( gqa_size, 0 ) << "GQA sharing rate must be positive.";
  CHECK_GT( vocab_size, 0 ) << "Vocabulary size must be positive.";
  CHECK_GT( seq_len, 0 ) << "Sequence length must be positive.";
  CHECK_GT( concurrency_limit, 0 ) << "Max concurrent inference size must be positive.";

  this->start_layer_num = start_layer;
  this->end_layer_num = ( end_layer == numeric_limits<uint32_t>::max() ) ? ( n_layers - 1 ) : end_layer;

  CHECK_GE( start_layer_num, 0 ) << "Start layer must be non-negative.";
  CHECK_LT( end_layer_num, n_layers ) << "End layer must be less than the number of layers.";
  CHECK_LE( start_layer_num, end_layer_num ) << "Start layer must be less than or equal to end layer.";

  LOG( INFO ) << "Loaded config: " << to_string();
}

string Config::to_string() const
{
  ostringstream oss;
  oss << "{ ";
  oss << "dim: " << dim << ", ";
  oss << "kv_dim: " << kv_dim << ", ";
  oss << "hidden_dim: " << hidden_dim << ", ";
  oss << "n_layers: " << n_layers << ", ";
  oss << "head_size: " << head_size << ", ";
  oss << "n_heads: " << n_heads << ", ";
  oss << "n_kv_heads: " << n_kv_heads << ", ";
  oss << "gqa_size: " << gqa_size << ", ";
  oss << "vocab_size: " << vocab_size << ", ";
  oss << "seq_len: " << seq_len << ", ";
  oss << "concurrency_limit: " << concurrency_limit << ", ";
  oss << "wcls_present: " << wcls_present << ", ";
  oss << "start_layer_num: " << start_layer_num << ", ";
  oss << "end_layer_num: " << end_layer_num;
  oss << " }";
  return oss.str();
}

/* VOCABULARY */

Vocabulary::Vocabulary( const filesystem::path& vocabulary_path )
{
  ifstream fin { vocabulary_path, ios::binary };
  int len = 0;

  CHECK( fin.good() ) << "Failed to open vocabulary file: " << vocabulary_path;

  int i;
  for ( i = 0;; i++ ) {
    if ( not fin.read( reinterpret_cast<char*>( &len ), sizeof( int ) ) ) {
      break;
    }

    CHECK_GT( len, 0 ) << "Vocabulary entry length must be positive.";

    string val;
    val.resize( len );
    CHECK( fin.read( val.data(), val.length() ) ) << "Failed to read vocabulary entry.";

    token_to_word_.push_back( val );
    word_to_token_.emplace( val, i );
  }

  LOG( INFO ) << "Loaded vocabulary of size " << i << " from " << vocabulary_path;
}

string Vocabulary::get_word( int token ) const
{
  CHECK_GE( token, 0 ) << "Token index must be non-negative.";
  CHECK_LT( token, token_to_word_.size() ) << "Token index out of bounds.";
  return token_to_word_[token];
}

int Vocabulary::get_token( const string& word ) const
{
  auto it = word_to_token_.find( word );
  CHECK( it != word_to_token_.end() ) << "Unknown word: " << word;
  return it->second;
}

/* BASE WEIGHTS */

template<typename DType>
BaseWeights<DType>::BaseWeights( const Config& config, const DType* model )
{
  const int head_size = config.dim / config.n_heads;

  auto ptr = model;
  token_embedding_table = _advance_pointer( ptr, config.vocab_size * config.dim );
  rms_final_weight = _advance_pointer( ptr, config.dim );
  freq_cis_real = _advance_pointer( ptr, config.seq_len * head_size / 2 );
  freq_cis_imag = _advance_pointer( ptr, config.seq_len * head_size / 2 );
  wcls = config.wcls_present ? ptr : token_embedding_table;
}

template<typename DType>
size_t BaseWeights<DType>::base_size( const Config& config )
{
  return sizeof( DType )
         * ( config.vocab_size * config.dim + config.dim + config.seq_len * config.dim / config.n_heads
             + ( config.wcls_present ? ( config.vocab_size * config.dim ) : 0 ) );
}

/* LAYER WEIGHTS */

template<typename DType>
LayerWeights<DType>::LayerWeights( const Config& config, const DType* model )
{
  auto ptr = model;

  // base pointers
  rms_att_weight = _advance_pointer( ptr, config.dim );
  this->wq = _advance_pointer( ptr, config.dim * config.dim );
  this->wk = _advance_pointer( ptr, config.dim * config.kv_dim );
  this->wv = _advance_pointer( ptr, config.dim * config.kv_dim );
  this->wo = _advance_pointer( ptr, config.dim * config.dim );
  this->rms_ffn_weight = _advance_pointer( ptr, config.dim );
  this->w1 = _advance_pointer( ptr, config.dim * config.hidden_dim );
  this->w2 = _advance_pointer( ptr, config.dim * config.hidden_dim );
  this->w3 = _advance_pointer( ptr, config.dim * config.hidden_dim );
}

template<typename DType>
size_t LayerWeights<DType>::layer_size( const Config& config )
{
  return sizeof( DType )
         * ( 2 * config.dim + 2 * config.dim * config.dim + 2 * config.dim * config.kv_dim
             + 3 * config.dim * config.hidden_dim );
}

/* RUN STATE */

// TODO: optimize run state memory usage
template<typename DType>
RunState<DType>::RunState( const Config& config, DType* buffer )
  : buffer_( buffer )
  , x( buffer_ )
  , xb( buffer_ + config.dim * config.concurrency_limit )
  , xb2( xb + config.dim * config.concurrency_limit )
  , q( xb2 + config.dim * config.concurrency_limit )
  , k( q + config.dim * config.concurrency_limit )
  , v( k + config.kv_dim * config.concurrency_limit )
  , hb( v + config.kv_dim * config.concurrency_limit )
  , hb2( hb + config.hidden_dim * config.concurrency_limit )
  , att( hb2 + config.hidden_dim * config.concurrency_limit )
  , logits( att + config.n_heads * config.seq_len * config.concurrency_limit )
  , temp_softmax( logits + config.vocab_size * config.concurrency_limit )
#ifdef GLINTHAWK_CUDA_ENABLED
  , rng_state( reinterpret_cast<curandState*>( temp_softmax + config.n_heads * config.concurrency_limit ) )
#endif
{
}

template<typename DType>
size_t RunState<DType>::state_size( const Config& config )
{
  size_t rng_size = 0;
#ifdef GLINTHAWK_CUDA_ENABLED
  rng_size += sizeof( curandState ) * config.concurrency_limit * config.vocab_size;
#endif
  return sizeof( DType ) * config.concurrency_limit
           * ( config.dim * 4 + config.kv_dim * 2 + config.hidden_dim * 2 + config.n_heads * config.seq_len
               + config.vocab_size + config.n_heads )
         + rng_size;
}

/* InferenceContext */

template<typename DType>
size_t InferenceContext<DType>::context_size( const Config& config )
{
  return sizeof( DType ) * config.seq_len * config.kv_dim * 2 * config.n_layers_loaded();
}

template<typename DType>
DType* InferenceContext<DType>::key( const Config& config, int layer_num, const int token_num, const int head_num )
{
  if ( empty() )
    return nullptr;
  return buffer_ + ( layer_num - config.start_layer_num ) * ( config.seq_len * config.kv_dim * 2 )
         + token_num * ( config.kv_dim * 2 ) + head_num * ( config.dim / config.n_heads );
}

template<typename DType>
DType* InferenceContext<DType>::value( const Config& config, int layer_num, const int token_num, const int head_num )
{
  if ( empty() )
    return nullptr;
  return key( config, layer_num, token_num, head_num ) + config.kv_dim;
}

template<typename DType>
bool InferenceContext<DType>::empty()
{
  return buffer_ == nullptr;
}

/* BaseLlama2 */

template<typename DType, typename Context>
void BaseLlama2<DType, Context>::init( const Config& config,
                                       unique_ptr<DType, void ( * )( DType* )>&& base_weights,
                                       unique_ptr<DType, void ( * )( DType* )>&& layers_weights,
                                       unique_ptr<DType, void ( * )( DType* )>&& run_state )
{
  this->config_ = config;
  this->base_weights_buffer_ = move( base_weights );
  this->layers_buffer_ = move( layers_weights );
  this->run_state_buffer_ = move( run_state );

  this->state_ = RunState<DType> { config_, run_state_buffer_.get() };
  this->base_weights_ = BaseWeights<DType> { config_, base_weights_buffer_.get() };

  this->layer_weights_ = [&] {
    vector<LayerWeights<DType>> layers {};
    layers.resize( config_.n_layers );

    const size_t layer_size = LayerWeights<DType>::layer_size( config_ );
    auto ptr = layers_buffer_.get();

    for ( auto i = config_.start_layer_num; i <= config_.end_layer_num; i++ ) {
      layers[i] = LayerWeights { config_,
                                 reinterpret_cast<DType*>( reinterpret_cast<uint8_t*>( ptr )
                                                           + ( i - config_.start_layer_num ) * layer_size ) };
    }

    return layers;
  }();
}

template<typename DType, typename Context>
void BaseLlama2<DType, Context>::dummy_forward( InferenceState& inference_state )
{
  CHECK_EQ( inference_state.next_layer(), this->config_.start_layer_num );
  inference_state.erase_from_workers( this->config().start_layer_num );
  if ( this->config_.end_layer_num == this->config_.n_layers - 1 ) {
    inference_state.set_next_layer( 0 );
  } else {
    inference_state.set_next_layer( this->config().end_layer_num + 1 );
  }
}

template<typename DType, typename Context>
bool BaseLlama2<DType, Context>::is_finished( const InferenceState& inference_state )
{
  CHECK_EQ( inference_state.next_layer(), this->config_.start_layer_num );
  return ( inference_state.next_layer() == 0 )
         and ( inference_state.token() == 2
               or inference_state.token_pos() >= this->config_.seq_len ); // EOS or out of length
}

template<typename DType, typename Context>
void BaseLlama2<DType, Context>::assert_safe_forward( const vector<InferenceState>& inference_states,
                                                      const std::vector<std::shared_ptr<ContextType>>& contexts )
{
  CHECK_GT( inference_states.size(), 0 ) << "batch size must be at least 1";
  CHECK_LE( inference_states.size(), this->config_.concurrency_limit )
    << "current batch cannot be larger than max concurrency size";

  const uint32_t next_layer_batch = inference_states[0].next_layer();

  CHECK_LE( this->config_.start_layer_num, next_layer_batch ) << "next layer in batch can not be before starting layer";
  CHECK_LE( next_layer_batch, this->config_.end_layer_num ) << "next layer in batch can not be after end layer";

  for ( auto& item : inference_states ) {
    CHECK_EQ( item.next_stage(), glinthawk::models::InferenceState::Stage::PreAttention ) << "next_stage must be pre attention";
    CHECK_EQ( item.next_layer(), next_layer_batch ) << "next_layer must be the same across batch";
    if ( constexpr( DType == _Float16 ) )
      CHECK_EQ( item.activations().dtype.dtype, glinthawk::models::SerializedDataType::Type::Float16 )
        << "Inference State data type does not match model data type (float16)";
#ifdef GLINTHAWK_CUDA_ENABLED
    if ( constexpr( DType == __half ) )
      CHECK_EQ( item.activations().dtype.dtype, glinthawk::models::SerializedDataType::Type::Float16 )
        << "Inference State data type does not match model data type (float16)";
#endif
    if ( constexpr( DType == float ) )
      CHECK_EQ( item.activations().dtype.dtype, glinthawk::models::SerializedDataType::Type::Float32 )
        << "Inference State data type does not match model data type (float16)";
    CHECK_LT( item.token_pos(), this->config_.seq_len ) << "token position cannot be larger than sequence length";
  }
  CHECK_EQ( inference_states.size(), contexts.size() ) << "token size must be the same as context size";
  for ( auto& item : contexts ) {
    CHECK_EQ( item.empty(), false ) << "context cannot be empty";
  }
}

template<typename DType, typename Context>
void BaseLlama2<DType, Context>::assert_safe_pre_attention( const vector<InferenceState>& inference_states,
                                                            const std::vector<std::shared_ptr<ContextType>>& contexts )
{
  CHECK_GT( inference_states.size(), 0 ) << "batch size must be at least 1";
  CHECK_LE( inference_states.size(), this->config_.concurrency_limit )
    << "current batch cannot be larger than max concurrency size";

  const uint32_t next_layer_batch = inference_states[0].next_layer();

  CHECK_LE( this->config_.start_layer_num, next_layer_batch ) << "next layer in batch can not be before starting layer";
  CHECK_LE( next_layer_batch, this->config_.end_layer_num ) << "next layer in batch can not be after end layer";

  for ( auto& item : inference_states ) {
    CHECK_EQ( item.next_stage(), glinthawk::models::InferenceState::Stage::PreAttention ) << "next_stage must be pre attention";
    CHECK_EQ( item.next_layer(), next_layer_batch ) << "next_layer must be the same across batch";
    if ( constexpr( DType == _Float16 ) )
      CHECK_EQ( item.activations().dtype.dtype, glinthawk::models::SerializedDataType::Type::Float16 )
        << "Inference State data type does not match model data type (float16)";
#ifdef GLINTHAWK_CUDA_ENABLED
    if ( constexpr( DType == __half ) )
      CHECK_EQ( item.activations().dtype.dtype, glinthawk::models::SerializedDataType::Type::Float16 )
        << "Inference State data type does not match model data type (float16)";
#endif
    if ( constexpr( DType == float ) )
      CHECK_EQ( item.activations().dtype.dtype, glinthawk::models::SerializedDataType::Type::Float32 )
        << "Inference State data type does not match model data type (float16)";
    CHECK_LT( item.token_pos(), this->config_.seq_len ) << "token position cannot be larger than sequence length";
  }
  CHECK_EQ( inference_states.size(), contexts.size() ) << "token size must be the same as context size";
}

template<typename DType, typename Context>
void BaseLlama2<DType, Context>::assert_safe_attention( const vector<InferenceState>& inference_states,
                                                        const std::vector<std::shared_ptr<ContextType>>& contexts )
{
  CHECK_GT( inference_states.size(), 0 ) << "batch size must be at least 1";
  CHECK_LE( inference_states.size(), this->config_.concurrency_limit )
    << "current batch cannot be larger than max concurrency size";

  const uint32_t next_layer_batch = inference_states[0].next_layer();

  CHECK_LE( this->config_.start_layer_num, next_layer_batch ) << "next layer in batch can not be before starting layer";
  CHECK_LE( next_layer_batch, this->config_.end_layer_num ) << "next layer in batch can not be after end layer";

  for ( auto& item : inference_states ) {
    CHECK_EQ( item.next_stage(), glinthawk::models::InferenceState::Stage::Attention ) << "next_stage must be attention";
    CHECK_LT( item.token_pos(), this->config_.seq_len ) << "token position cannot be larger than sequence length";
  }
  CHECK_EQ( inference_states.size(), contexts.size() ) << "token size must be the same as context size";
  for ( auto& item : contexts ) {
    CHECK_EQ( item.empty(), false ) << "context cannot be empty";
  }
}

template<typename DType, typename Context>
void BaseLlama2<DType, Context>::assert_safe_post_attention( const vector<InferenceState>& inference_states )
{
  CHECK_GT( inference_states.size(), 0 ) << "batch size must be at least 1";
  CHECK_LE( inference_states.size(), this->config_.concurrency_limit )
    << "current batch cannot be larger than max concurrency size";

  const uint32_t next_layer_batch = inference_states[0].next_layer();

  CHECK_LE( this->config_.start_layer_num, next_layer_batch ) << "next layer in batch can not be before starting layer";
  CHECK_LE( next_layer_batch, this->config_.end_layer_num ) << "next layer in batch can not be after end layer";

  for ( auto& item : inference_states ) {
    CHECK_EQ( item.next_stage(), glinthawk::models::InferenceState::Stage::PostAttention ) << "next_stage must be post attention";
    CHECK_EQ( item.next_layer(), next_layer_batch ) << "next_layer must be the same across batch";
    if ( constexpr( DType == _Float16 ) )
      CHECK_EQ( item.activations().dtype.dtype, glinthawk::models::SerializedDataType::Type::Float16 )
        << "Inference State data type does not match model data type (float16)";
#ifdef GLINTHAWK_CUDA_ENABLED
    if ( constexpr( DType == __half ) )
      CHECK_EQ( item.activations().dtype.dtype, glinthawk::models::SerializedDataType::Type::Float16 )
        << "Inference State data type does not match model data type (float16)";
#endif
    if ( constexpr( DType == float ) )
      CHECK_EQ( item.activations().dtype.dtype, glinthawk::models::SerializedDataType::Type::Float32 )
        << "Inference State data type does not match model data type (float16)";
    CHECK_LT( item.token_pos(), this->config_.seq_len ) << "token position cannot be larger than sequence length";
  }
}

namespace glinthawk::models::llama2 {
template class RunState<float>;
template class BaseWeights<float>;
template class LayerWeights<float>;
template class InferenceContext<float>;

template class RunState<_Float16>;
template class BaseWeights<_Float16>;
template class LayerWeights<_Float16>;
template class InferenceContext<_Float16>;

namespace cpu {
template<typename DType>
class Context;
}

template class BaseLlama2<float, cpu::Context<float>>;
template class BaseLlama2<_Float16, cpu::Context<_Float16>>;

#ifdef GLINTHAWK_CUDA_ENABLED
template class RunState<__half>;
template class BaseWeights<__half>;
template class LayerWeights<__half>;
template class InferenceContext<__half>;

// forward declare Context
namespace cuda {
template<typename DType>
class Context;
}

template class BaseLlama2<__half, cuda::Context<__half>>;
#endif
}
