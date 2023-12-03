#include "base.hh"

#include <glog/logging.h>

#ifdef GLINTHAWK_CUDA_ENABLED
#include "models/common/cuda/ops.cuh"
#endif

namespace glinthawk::models::llama2 {

namespace {

template<typename DType>
DType* _advance_pointer( DType*& ptr, const size_t size )
{
  auto old = ptr;
  ptr += size;
  return old;
}

}

template<typename T>
Settings<T>::Settings( const std::filesystem::path& config_file,
                       const uint32_t start_layer,
                       const uint32_t end_layer,
                       const uint64_t concurrency_limit_,
                       const bool randomize_parameters_ )
  : concurrency_limit( concurrency_limit_ )
  , randomize_parameters( randomize_parameters_ )
{
  std::ifstream fin { config_file, std::ios::binary };
  CHECK( fin ) << "Failed to open config file: " << config_file;

  std::string raw_config;
  raw_config.resize( config_size() );

  fin.read( raw_config.data(), config_size() );
  if ( fin.gcount() != static_cast<std::streamsize>( config_size() ) ) {
    LOG( FATAL ) << "Failed to read config file: " << config_file;
  }

  auto ptr = raw_config.data();

  const int32_t dim = *reinterpret_cast<const int32_t*>( _advance_pointer( ptr, sizeof( int32_t ) ) );
  const int32_t hidden_dim = *reinterpret_cast<const int32_t*>( _advance_pointer( ptr, sizeof( int32_t ) ) );
  const int32_t n_layers = *reinterpret_cast<const int32_t*>( _advance_pointer( ptr, sizeof( int32_t ) ) );
  const int32_t n_heads = *reinterpret_cast<const int32_t*>( _advance_pointer( ptr, sizeof( int32_t ) ) );
  const int32_t n_kv_heads = *reinterpret_cast<const int32_t*>( _advance_pointer( ptr, sizeof( int32_t ) ) );
  const int32_t gqa_size = n_heads / n_kv_heads;
  const int32_t kv_dim = dim / gqa_size;
  const int32_t head_size = dim / n_heads;

  // if vocab size is negative, that means that wcls is present
  const auto original_vocab_size = *reinterpret_cast<const int32_t*>( _advance_pointer( ptr, sizeof( int32_t ) ) );
  const int32_t vocab_size = abs( original_vocab_size );
  const int32_t seq_len = *reinterpret_cast<const int32_t*>( _advance_pointer( ptr, sizeof( int32_t ) ) );
  const bool wcls_present = ( original_vocab_size < 0 );

  // make sure that the data read from config file matches the ModelConfig (T)
  CHECK_EQ( dim, T::dim ) << "dim does not match config file";
  CHECK_EQ( kv_dim, T::kv_dim ) << "kv_dim does not match config file";
  CHECK_EQ( hidden_dim, T::hidden_dim ) << "hidden_dim does not match config file";
  CHECK_EQ( n_layers, T::n_layers ) << "n_layers does not match config file";
  CHECK_EQ( head_size, T::head_size ) << "head_size does not match config file";
  CHECK_EQ( n_heads, T::n_heads ) << "n_heads does not match config file";
  CHECK_EQ( n_kv_heads, T::n_kv_heads ) << "n_kv_heads does not match config file";
  CHECK_EQ( gqa_size, T::gqa_size ) << "gqa_size does not match config file";
  CHECK_EQ( vocab_size, T::vocab_size ) << "vocab_size does not match config file";
  CHECK_EQ( seq_len, T::seq_len ) << "seq_len does not match config file";
  CHECK_EQ( wcls_present, T::wcls_present ) << "wcls_present does not match config file";

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
  this->end_layer_num = ( end_layer == std::numeric_limits<uint32_t>::max() ) ? ( T::n_layers - 1 ) : end_layer;

  CHECK_GE( start_layer_num, 0 ) << "Start layer must be non-negative.";
  CHECK_LT( end_layer_num, T::n_layers ) << "End layer must be less than the number of layers.";
  CHECK_LE( start_layer_num, end_layer_num ) << "Start layer must be less than or equal to end layer.";

  LOG( INFO ) << "Loaded settings and validated " << to_string();
}

template<typename T>
std::string Settings<T>::to_string() const
{
  std::ostringstream oss;
  oss << "{ ";
  oss << "dim: " << T::dim << ", ";
  oss << "kv_dim: " << T::kv_dim << ", ";
  oss << "hidden_dim: " << T::hidden_dim << ", ";
  oss << "n_layers: " << T::n_layers << ", ";
  oss << "head_size: " << T::head_size << ", ";
  oss << "n_heads: " << T::n_heads << ", ";
  oss << "n_kv_heads: " << T::n_kv_heads << ", ";
  oss << "gqa_size: " << T::gqa_size << ", ";
  oss << "vocab_size: " << T::vocab_size << ", ";
  oss << "seq_len: " << T::seq_len << ", ";
  oss << "wcls_present: " << T::wcls_present << ", ";
  oss << "concurrency_limit: " << concurrency_limit << ", ";
  oss << "start_layer_num: " << start_layer_num << ", ";
  oss << "end_layer_num: " << end_layer_num;
  oss << " }";
  return oss.str();
}

/* VOCABULARY */

Vocabulary::Vocabulary( const std::filesystem::path& vocabulary_path )
{
  std::ifstream fin { vocabulary_path, std::ios::binary };
  int len = 0;

  CHECK( fin.good() ) << "Failed to open vocabulary file: " << vocabulary_path;

  int i;
  for ( i = 0;; i++ ) {
    if ( not fin.read( reinterpret_cast<char*>( &len ), sizeof( int ) ) ) {
      break;
    }

    CHECK_GT( len, 0 ) << "Vocabulary entry length must be positive.";

    std::string val;
    val.resize( len );
    CHECK( fin.read( val.data(), val.length() ) ) << "Failed to read vocabulary entry.";

    token_to_word_.push_back( val );
    word_to_token_.emplace( val, i );
  }

  LOG( INFO ) << "Loaded vocabulary of size " << i << " from " << vocabulary_path;
}

std::string Vocabulary::get_word( int token ) const
{
  CHECK_GE( token, 0 ) << "Token index must be non-negative.";
  CHECK_LT( token, token_to_word_.size() ) << "Token index out of bounds.";
  return token_to_word_[token];
}

int Vocabulary::get_token( const std::string& word ) const
{
  auto it = word_to_token_.find( word );
  CHECK( it != word_to_token_.end() ) << "Unknown word: " << word;
  return it->second;
}

/* BASE WEIGHTS */

template<typename Config, typename DType>
BaseWeights<Config, DType>::BaseWeights( const DType* model )
{
  const int head_size = Config::dim / Config::n_heads;

  auto ptr = model;
  token_embedding_table = _advance_pointer( ptr, Config::vocab_size * Config::dim );
  rms_final_weight = _advance_pointer( ptr, Config::dim );
  freq_cis_real = _advance_pointer( ptr, Config::seq_len * head_size / 2 );
  freq_cis_imag = _advance_pointer( ptr, Config::seq_len * head_size / 2 );
  wcls = Config::wcls_present ? ptr : token_embedding_table;
}

/* LAYER WEIGHTS */

template<typename Config, typename DType>
LayerWeights<Config, DType>::LayerWeights( const DType* model )
{
  auto ptr = model;

  // base pointers
  this->rms_att_weight = _advance_pointer( ptr, Config::dim );
  this->wq = _advance_pointer( ptr, Config::dim * Config::dim );
  this->wk = _advance_pointer( ptr, Config::dim * Config::kv_dim );
  this->wv = _advance_pointer( ptr, Config::dim * Config::kv_dim );
  this->wo = _advance_pointer( ptr, Config::dim * Config::dim );
  this->rms_ffn_weight = _advance_pointer( ptr, Config::dim );
  this->w1 = _advance_pointer( ptr, Config::dim * Config::hidden_dim );
  this->w2 = _advance_pointer( ptr, Config::dim * Config::hidden_dim );
  this->w3 = _advance_pointer( ptr, Config::dim * Config::hidden_dim );
}

/* RUN STATE */

// run-state memory usage is unoptimized, but too small to matter
template<typename Config, typename DType>
RunState<Config, DType>::RunState( const Settings<Config>& settings, DType* buffer )
  : buffer_( buffer )
  , x( buffer_ )
  , xb( buffer_ + Config::dim * settings.concurrency_limit )
  , xb2( xb + Config::dim * settings.concurrency_limit )
  , q( xb2 + Config::dim * settings.concurrency_limit )
  , k( q + Config::dim * settings.concurrency_limit )
  , v( k + Config::kv_dim * settings.concurrency_limit )
  , hb( v + Config::kv_dim * settings.concurrency_limit )
  , hb2( hb + Config::hidden_dim * settings.concurrency_limit )
  , att( hb2 + Config::hidden_dim * settings.concurrency_limit )
  , logits( att + Config::n_heads * Config::seq_len * settings.concurrency_limit )
  , temp_softmax( logits + Config::vocab_size * settings.concurrency_limit )
#ifdef GLINTHAWK_CUDA_ENABLED
  , rng_state( reinterpret_cast<curandState*>( temp_softmax + Config::n_heads * settings.concurrency_limit ) )
#endif
{
}

template<typename Config, typename DType>
size_t RunState<Config, DType>::state_size( const Settings<Config>& settings )
{
  size_t rng_size = 0;

#ifdef GLINTHAWK_CUDA_ENABLED
  rng_size += sizeof( curandState ) * settings.concurrency_limit * Config::vocab_size;
#endif

  return sizeof( DType ) * settings.concurrency_limit
           * ( Config::dim * 4 + Config::kv_dim * 2 + Config::hidden_dim * 2 + Config::n_heads * Config::seq_len
               + Config::vocab_size + Config::n_heads )
         + rng_size;
}

/* InferenceContext */

template<typename Config, typename DType>
size_t InferenceContext<Config, DType>::context_size( const Settings<Config>& settings )
{
  return sizeof( DType ) * Config::seq_len * Config::kv_dim * 2 * settings.n_layers_loaded();
}

template<typename Config, typename DType>
DType* InferenceContext<Config, DType>::key( const Settings<Config>& settings,
                                             int layer_num,
                                             const int token_num,
                                             const int head_num )
{
  if ( empty() )
    return nullptr;
  return buffer_ + ( layer_num - settings.start_layer_num ) * ( Config::seq_len * Config::kv_dim * 2 )
         + token_num * ( Config::kv_dim * 2 ) + head_num * ( Config::dim / Config::n_heads );
}

template<typename Config, typename DType>
DType* InferenceContext<Config, DType>::value( const Settings<Config>& settings,
                                               int layer_num,
                                               const int token_num,
                                               const int head_num )
{
  if ( empty() )
    return nullptr;
  return key( settings, layer_num, token_num, head_num ) + Config::kv_dim;
}

template<typename Config, typename DType>
bool InferenceContext<Config, DType>::empty()
{
  return buffer_ == nullptr;
}

/* BaseLlama2 */

template<typename Config, typename DType, typename Context, typename StorageDeleter>
void BaseLlama2<Config, DType, Context, StorageDeleter>::init( const Settings<Config>& settings,
                                                               std::unique_ptr<DType, StorageDeleter>&& base_weights,
                                                               std::unique_ptr<DType, StorageDeleter>&& layers_weights,
                                                               std::unique_ptr<DType, StorageDeleter>&& run_state )
{
  this->settings_ = settings;
  this->base_weights_buffer_ = move( base_weights );
  this->layers_buffer_ = move( layers_weights );
  this->run_state_buffer_ = move( run_state );

  this->state_ = RunState<Config, DType> { settings_, run_state_buffer_.get() };
  this->base_weights_ = BaseWeights<Config, DType> { base_weights_buffer_.get() };

  this->layer_weights_ = [&] {
    // XXX std::array
    std::vector<LayerWeights<Config, DType>> layers {};
    layers.resize( Config::n_layers );

    constexpr size_t layer_size = LayerWeights<Config, DType>::layer_size();
    auto ptr = layers_buffer_.get();

    for ( auto i = settings_.start_layer_num; i <= settings_.end_layer_num; i++ ) {
      layers[i] = LayerWeights<Config, DType> { reinterpret_cast<DType*>(
        reinterpret_cast<uint8_t*>( ptr ) + ( i - settings_.start_layer_num ) * layer_size ) };
    }

    return layers;
  }();
}

template<typename Config, typename DType, typename Context, typename StorageDeleter>
void BaseLlama2<Config, DType, Context, StorageDeleter>::dummy_forward( InferenceState& inference_state )
{
  // TODO: rewrite this for split pipes
  CHECK_EQ( inference_state.next_layer(), settings_.start_layer_num );

  inference_state.erase_from_workers( settings_.start_layer_num );
  if ( settings_.end_layer_num == Config::n_layers - 1 ) {
    inference_state.set_next_layer( 0 );
  } else {
    inference_state.set_next_layer( settings_.end_layer_num + 1 );
  }
}

template<typename Config, typename DType, typename Context, typename StorageDeleter>
bool BaseLlama2<Config, DType, Context, StorageDeleter>::is_finished( const InferenceState& inference_state )
{
  // TODO: rewrite this for split pipes
  CHECK_EQ( inference_state.next_layer(), settings_.start_layer_num );
  return ( inference_state.next_layer() == 0 )
         and ( inference_state.token() == 2 or inference_state.token_pos() >= Config::seq_len ); // EOS or out of length
}

template<typename Config, typename DType, typename Context, typename StorageDeleter>
void BaseLlama2<Config, DType, Context, StorageDeleter>::assert_safe_forward(
  const InferenceStateVector& inference_states,
  const ContextVector& contexts ) const
{
  CHECK_GT( inference_states.size(), 0 ) << "batch size must be at least 1";
  CHECK_LE( inference_states.size(), settings_.concurrency_limit )
    << "current batch cannot be larger than max concurrency size";

  const uint32_t next_layer_batch = inference_states[0].next_layer();

  CHECK_LE( settings_.start_layer_num, next_layer_batch ) << "next layer in batch can not be before starting layer";
  CHECK_LE( next_layer_batch, settings_.end_layer_num ) << "next layer in batch can not be after end layer";

  for ( auto& item : inference_states ) {
    CHECK( item.next_stage() == InferenceState::Stage::PreAttention ) << "next_stage must be pre attention";
    CHECK_EQ( item.next_layer(), next_layer_batch ) << "next_layer must be the same across batch";
    if constexpr ( std::is_same_v<DType, _Float16> )
      CHECK_EQ( item.dtype(), DataType::Float16 ) << "Inference State data type does not match model data type";
#ifdef GLINTHAWK_CUDA_ENABLED
    else if constexpr ( std::is_same_v<DType, __half> )
      CHECK_EQ( item.dtype(), DataType::Float16 ) << "Inference State data type does not match model data type";
#endif
    else if constexpr ( std::is_same_v<DType, float> )
      CHECK_EQ( item.dtype(), DataType::Float32 ) << "Inference State data type does not match model data type";
    CHECK_LT( item.token_pos(), Config::seq_len ) << "token position cannot be larger than sequence length";
  }
  CHECK_EQ( inference_states.size(), contexts.size() ) << "token size must be the same as context size";
}

template<typename Config, typename DType, typename Context, typename StorageDeleter>
void BaseLlama2<Config, DType, Context, StorageDeleter>::assert_safe_pre_attention(
  const InferenceStateVector& inference_states,
  const ContextVector& contexts ) const
{
  CHECK_GT( inference_states.size(), 0 ) << "batch size must be at least 1";
  CHECK_LE( inference_states.size(), settings_.concurrency_limit )
    << "current batch cannot be larger than max concurrency size";

  const uint32_t next_layer_batch = inference_states[0].next_layer();

  CHECK_LE( settings_.start_layer_num, next_layer_batch ) << "next layer in batch can not be before starting layer";
  CHECK_LE( next_layer_batch, settings_.end_layer_num ) << "next layer in batch can not be after end layer";

  for ( auto& item : inference_states ) {
    CHECK( item.next_stage() == InferenceState::Stage::PreAttention ) << "next_stage must be pre attention";
    CHECK_EQ( item.next_layer(), next_layer_batch ) << "next_layer must be the same across batch";
    if constexpr ( std::is_same_v<DType, _Float16> )
      CHECK_EQ( item.dtype(), DataType::Float16 ) << "Inference State data type does not match model data type";
#ifdef GLINTHAWK_CUDA_ENABLED
    else if constexpr ( std::is_same_v<DType, __half> )
      CHECK_EQ( item.dtype(), DataType::Float16 ) << "Inference State data type does not match model data type";
#endif
    else if constexpr ( std::is_same_v<DType, float> )
      CHECK_EQ( item.dtype(), DataType::Float32 ) << "Inference State data type does not match model data type";
    CHECK_LT( item.token_pos(), Config::seq_len ) << "token position cannot be larger than sequence length";
  }
  CHECK_EQ( inference_states.size(), contexts.size() ) << "token size must be the same as context size";
}

template<typename Config, typename DType, typename Context, typename StorageDeleter>
void BaseLlama2<Config, DType, Context, StorageDeleter>::assert_safe_attention(
  const InferenceStateVector& inference_states,
  const ContextVector& contexts ) const
{
  CHECK_GT( inference_states.size(), 0 ) << "batch size must be at least 1";
  CHECK_LE( inference_states.size(), settings_.concurrency_limit )
    << "current batch cannot be larger than max concurrency size";

  const uint32_t next_layer_batch = inference_states[0].next_layer();

  CHECK_LE( settings_.start_layer_num, next_layer_batch ) << "next layer in batch can not be before starting layer";
  CHECK_LE( next_layer_batch, settings_.end_layer_num ) << "next layer in batch can not be after end layer";

  for ( auto& item : inference_states ) {
    CHECK( item.next_stage() == InferenceState::Stage::Attention ) << "next_stage must be attention";
    CHECK_LT( item.token_pos(), Config::seq_len ) << "token position cannot be larger than sequence length";
  }
  CHECK_EQ( inference_states.size(), contexts.size() ) << "token size must be the same as context size";
}

template<typename Config, typename DType, typename Context, typename StorageDeleter>
void BaseLlama2<Config, DType, Context, StorageDeleter>::assert_safe_post_attention(
  const InferenceStateVector& inference_states ) const
{
  CHECK_GT( inference_states.size(), 0 ) << "batch size must be at least 1";
  CHECK_LE( inference_states.size(), settings_.concurrency_limit )
    << "current batch cannot be larger than max concurrency size";

  const uint32_t next_layer_batch = inference_states[0].next_layer();

  CHECK_LE( settings_.start_layer_num, next_layer_batch ) << "next layer in batch can not be before starting layer";
  CHECK_LE( next_layer_batch, settings_.end_layer_num ) << "next layer in batch can not be after end layer";

  for ( auto& item : inference_states ) {
    CHECK( item.next_stage() == InferenceState::Stage::PostAttention ) << "next_stage must be post attention";
    CHECK_EQ( item.next_layer(), next_layer_batch ) << "next_layer must be the same across batch";
    if constexpr ( std::is_same_v<DType, _Float16> )
      CHECK_EQ( item.dtype(), DataType::Float16 ) << "Inference State data type does not match model data type";
#ifdef GLINTHAWK_CUDA_ENABLED
    else if constexpr ( std::is_same_v<DType, __half> )
      CHECK_EQ( item.dtype(), DataType::Float16 ) << "Inference State data type does not match model data type";
#endif
    else if constexpr ( std::is_same_v<DType, float> )
      CHECK_EQ( item.dtype(), DataType::Float32 ) << "Inference State data type does not match model data type";
    CHECK_LT( item.token_pos(), Config::seq_len ) << "token position cannot be larger than sequence length";
  }
}

namespace cpu {
template<typename Config, typename DType>
requires ModelConfig<Config>
class Context;
}

namespace cuda {
template<typename Config, typename DType>
requires ModelConfig<Config>
class Context;
}

#define INSTANTIATE_FOR_MODEL( X, DTYPE )                                                                              \
  template class RunState<X, DTYPE>;                                                                                   \
  template class BaseWeights<X, DTYPE>;                                                                                \
  template class LayerWeights<X, DTYPE>;                                                                               \
  template class InferenceContext<X, DTYPE>;                                                                           \
  template class BaseLlama2<X, DTYPE, cpu::Context<X, DTYPE>, std::default_delete<DTYPE>>;

#define INSTANTIATE_FOR_MODEL_CPU( X )                                                                                 \
  template class Settings<X>;                                                                                          \
  INSTANTIATE_FOR_MODEL( X, _Float16 )                                                                                 \
  INSTANTIATE_FOR_MODEL( X, float )

INSTANTIATE_FOR_MODEL_CPU( configs::Llama2_7B_Chat )
INSTANTIATE_FOR_MODEL_CPU( configs::Llama2_13B_Chat )
INSTANTIATE_FOR_MODEL_CPU( configs::Llama2_70B_Chat )
INSTANTIATE_FOR_MODEL_CPU( configs::Stories_110M )

#ifdef GLINTHAWK_CUDA_ENABLED
#define INSTANTIATE_FOR_MODEL_CUDA( X )                                                                                \
  INSTANTIATE_FOR_MODEL( X, __half )                                                                                   \
  template class BaseLlama2<X, __half, cuda::Context<X, __half>, common::cuda::ops::CUDADeleter<__half>>;              \
  template class BaseLlama2<X, float, cuda::Context<X, float>, common::cuda::ops::CUDADeleter<float>>;

INSTANTIATE_FOR_MODEL_CUDA( configs::Llama2_7B_Chat )
INSTANTIATE_FOR_MODEL_CUDA( configs::Llama2_13B_Chat )
INSTANTIATE_FOR_MODEL_CUDA( configs::Llama2_70B_Chat )
INSTANTIATE_FOR_MODEL_CUDA( configs::Stories_110M )
#endif

} // namespace glinthawk::models::llama2
