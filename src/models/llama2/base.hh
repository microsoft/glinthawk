#pragma once

#include <array>
#include <filesystem>
#include <fstream>
#include <glog/logging.h>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include "context.hh"
#include "models/llama2/ops/concept.hh"
#include "models/types.hh"
#include "variants.hh"

namespace glinthawk::models::llama2 {

constexpr size_t MAX_BATCH_SIZE = 1024;

template<typename Config>
requires ModelConfig<Config>
struct Settings
{
  Settings() {}

  Settings( const std::filesystem::path& config_file,
            const uint32_t start_layer,
            const uint32_t end_layer,
            const uint64_t pre_att_concurrency_limit,
            const uint64_t att_concurrency_limit,
            const uint64_t post_att_concurrency_limit,
            const uint64_t cls_concurrency_limit,
            const uint64_t max_context_count,
            const bool randomize_parameters );

  std::string to_string() const;

  /// @brief Size of the config stored on disk (in bytes)
  static size_t config_size() { return sizeof( int32_t ) * 7; }
  uint64_t n_layers_loaded() const { return end_layer_num - start_layer_num + 1; }

  uint64_t start_layer_num {};
  uint64_t end_layer_num {};
  uint64_t concurrency_limit { 1 };          // max concurrent inference size
  uint64_t pre_att_concurrency_limit { 1 };  // max concurrent inference size
  uint64_t att_concurrency_limit { 1 };      // max concurrent inference size
  uint64_t post_att_concurrency_limit { 1 }; // max concurrent inference size
  uint64_t cls_concurrency_limit { 1 };      // max concurrent inference size
  uint64_t max_context_count { 1 };          // max number of contexts
  bool randomize_parameters { false };
};

class Vocabulary
{
private:
  std::vector<std::string> token_to_word_ {};
  std::unordered_multimap<std::string, int> word_to_token_ {};

public:
  Vocabulary( const std::filesystem::path& vocabulary_path );

  size_t size() const { return token_to_word_.size(); }
  int get_token( const std::string& word ) const;
  std::string get_word( const int token ) const;
};

template<typename Config, typename DType>
requires ModelConfig<Config>
struct BaseWeights
{
  BaseWeights() = default;
  BaseWeights( const DType* base_weights, const Settings<Config>& settings );

  BaseWeights( const BaseWeights& ) = delete;
  BaseWeights operator=( const BaseWeights& ) = delete;
  BaseWeights( BaseWeights&& ) = default;
  BaseWeights& operator=( BaseWeights&& ) = default;

  static consteval size_t read_size()
  {
    return sizeof( DType )
           * ( Config::vocab_size * Config::dim + Config::dim + Config::seq_len * Config::head_size
               + ( Config::wcls_present ? ( Config::vocab_size * Config::dim ) : 0 ) );
  }

  static size_t base_size( const Settings<Config>& settings )
  {
    const bool model_hosts_embedding = ( settings.pre_att_concurrency_limit > 0 and settings.start_layer_num == 0 )
                                       or ( settings.cls_concurrency_limit > 0 and not Config::wcls_present );
    const bool model_hosts_att = settings.att_concurrency_limit > 0;
    const bool model_hosts_cls = settings.cls_concurrency_limit > 0;
    size_t size = 0;
    if ( model_hosts_embedding )
      size += Config::vocab_size * Config::dim;
    if ( model_hosts_att )
      size += Config::seq_len * Config::head_size;
    if ( model_hosts_cls and Config::wcls_present )
      size += Config::dim + Config::vocab_size * Config::dim;
    if ( model_hosts_cls and not Config::wcls_present )
      size += Config::dim;
    return sizeof( DType ) * size;
  }

  static std::vector<uint64_t> weight_offset( const Settings<Config>& settings )
  {
    const bool model_hosts_embedding = ( settings.pre_att_concurrency_limit > 0 and settings.start_layer_num == 0 )
                                       or ( settings.cls_concurrency_limit > 0 and not Config::wcls_present );
    const bool model_hosts_att = settings.att_concurrency_limit > 0;
    const bool model_hosts_cls = settings.cls_concurrency_limit > 0;
    std::vector<uint64_t> offsets {};
    offsets.push_back( 0 );
    offsets.push_back( offsets.back() + ( model_hosts_embedding ? Config::vocab_size * Config::dim : 0 ) );
    offsets.push_back( offsets.back() + ( model_hosts_cls ? Config::dim : 0 ) );
    if ( Config::wcls_present and model_hosts_cls )
      offsets.push_back( offsets.back() + ( model_hosts_att ? Config::seq_len * Config::head_size : 0 ) );
    return offsets;
  }

  static std::vector<uint64_t> read_offset()
  {
    std::vector<uint64_t> offsets {};
    offsets.push_back( 0 );
    offsets.push_back( offsets.back() + Config::vocab_size * Config::dim );
    offsets.push_back( offsets.back() + Config::dim );
    if ( Config::wcls_present )
      offsets.push_back( offsets.back() + Config::seq_len * Config::head_size );
    return offsets;
  }

  static std::vector<uint64_t> weight_size( const Settings<Config>& settings )
  {
    const bool model_hosts_embedding = ( settings.pre_att_concurrency_limit > 0 and settings.start_layer_num == 0 )
                                       or ( settings.cls_concurrency_limit > 0 and not Config::wcls_present );
    const bool model_hosts_att = settings.att_concurrency_limit > 0;
    const bool model_hosts_cls = settings.cls_concurrency_limit > 0;
    std::vector<uint64_t> offsets {};
    offsets.push_back( sizeof( DType ) * ( model_hosts_embedding ? Config::vocab_size * Config::dim : 0 ) );
    offsets.push_back( sizeof( DType ) * ( model_hosts_cls ? Config::dim : 0 ) );
    offsets.push_back( sizeof( DType ) * ( model_hosts_att ? Config::seq_len * Config::head_size : 0 ) );
    if ( Config::wcls_present and model_hosts_cls )
      offsets.push_back( sizeof( DType ) * Config::vocab_size * Config::dim );
    return offsets;
  }

  const DType* token_embedding_table {}; // (vocab_size, dim)
  const DType* rms_final_weight {};      // (dim,)

  // freq_cis for RoPE relatively positional embeddings
  const DType* freq_cis_real {}; // (seq_len, dim/2)
  const DType* freq_cis_imag {}; // (seq_len, dim/2)

  // classifier weights for the logits, on the last layer
  const DType* wcls {};
};

template<typename Config, typename DType>
requires ModelConfig<Config>
struct LayerWeights
{
  LayerWeights() = default;
  LayerWeights( const DType* model, const Settings<Config>& settings );

  LayerWeights( const LayerWeights& ) = delete;
  LayerWeights operator=( const LayerWeights& ) = delete;
  LayerWeights( LayerWeights&& ) = default;
  LayerWeights& operator=( LayerWeights&& ) = default;

  static consteval size_t read_size()
  {
    return sizeof( DType )
           * ( 2 * Config::dim + 2 * Config::dim * Config::dim + 2 * Config::dim * Config::kv_dim
               + 3 * Config::dim * Config::hidden_dim );
  }

  static size_t layer_size( const Settings<Config>& settings )
  {
    const bool model_hosts_pre_att = settings.pre_att_concurrency_limit > 0;
    const bool model_hosts_post_att = settings.post_att_concurrency_limit > 0;
    size_t size = 0;
    if ( model_hosts_pre_att )
      size += Config::dim * Config::dim + 2 * Config::dim * Config::kv_dim + Config::dim;
    if ( model_hosts_post_att )
      size += 3 * Config::dim * Config::hidden_dim + Config::dim * Config::dim + Config::dim;
    return sizeof( DType ) * size;
  }

  static std::vector<uint64_t> weight_offset( const Settings<Config>& settings )
  {
    const bool model_hosts_pre_att = settings.pre_att_concurrency_limit > 0;
    std::vector<uint64_t> offsets {};
    offsets.push_back( 0 );
    offsets.push_back(
      offsets.back()
      + ( model_hosts_pre_att ? Config::dim + Config::dim * Config::dim + 2 * Config::dim * Config::kv_dim : 0 ) );
    return offsets;
  }

  static std::vector<uint64_t> read_offset()
  {
    std::vector<uint64_t> offsets {};
    offsets.push_back( 0 );
    offsets.push_back( offsets.back() + Config::dim + Config::dim * Config::dim + 2 * Config::dim * Config::kv_dim );
    return offsets;
  }

  static std::vector<uint64_t> weight_size( const Settings<Config>& settings )
  {
    const bool model_hosts_pre_att = settings.pre_att_concurrency_limit > 0;
    const bool model_hosts_post_att = settings.post_att_concurrency_limit > 0;
    std::vector<uint64_t> offsets {};
    offsets.push_back(
      sizeof( DType )
      * ( model_hosts_pre_att ? Config::dim + Config::dim * Config::dim + 2 * Config::dim * Config::kv_dim : 0 ) );
    offsets.push_back(
      sizeof( DType )
      * ( model_hosts_post_att ? Config::dim + Config::dim * Config::dim + 3 * Config::dim * Config::hidden_dim : 0 ) );
    return offsets;
  }

  // PreAttention
  // weights for rmsnorms
  const DType* rms_att_weight { nullptr }; // (dim) rmsnorm weights

  // weights for matmuls
  const DType* wq { nullptr };  // (dim, dim)
  const DType* wkv { nullptr }; // (dim, 2*kv_dim)

  // PostAttention
  // weights for rmsnorms
  const DType* rms_ffn_weight { nullptr }; // (dim)

  // weights for matmuls
  const DType* wo { nullptr }; // (dim, dim)

  // weights for ffn
  const DType* w1 { nullptr }; // (hidden_dim, dim)
  const DType* w2 { nullptr }; // (dim, hidden_dim)
  const DType* w3 { nullptr }; // (hidden_dim, dim)
};

/// @brief This class acts as the scratchpad for the computations.
/// None of this data needs to be saved between calls to `forward*()` functions.
template<typename Config, typename DType, typename ContextType>
requires ModelConfig<Config>
struct ScratchPad
{
  ScratchPad() = default;
  ScratchPad( const Settings<Config>& settings, DType* buffer );

  ScratchPad( const ScratchPad& ) = delete;
  ScratchPad operator=( const ScratchPad& ) = delete;
  ScratchPad( ScratchPad&& ) = default;
  ScratchPad& operator=( ScratchPad&& ) = default;

  static size_t scratchpad_size( const Settings<Config>& settings );

  DType* buffer_ {};      // we use this buffer for everything, including activations
  DType* x {};            // activation at current time stamp (B, dim)
  DType* xb {};           // same, but inside a residual branch (B, dim)
  DType* xb2 {};          // an additional buffer just for convenience (B, dim)
  DType* q {};            // query (B, dim)
  DType* kv {};           // key and value (B, kv_dim, 2)
  DType* hb {};           // buffer for hidden dimension in the ffn (B, hidden_dim)
  DType* hb2 {};          // buffer for hidden dimension in the ffn (B, hidden_dim)
  DType* att {};          // buffer for scores/attention values (B, n_heads, seq_len)
  DType* logits {};       // output logits (B, vocab_size)
  DType* temp_softmax {}; // temporary buffer for computing softmax (B, n_heads)

  // This memory is on the host
  uint32_t argmax_pos[MAX_BATCH_SIZE] {}; // argmax results (B, )

  // information about the current batch
  uint64_t curr_concurrency_size { 1 };
  uint32_t batch_token_positions[MAX_BATCH_SIZE] {};
  typename ContextType::LayerContextType batch_layer_contexts[MAX_BATCH_SIZE] {};
  typename ContextType::TokenContextType batch_token_contexts[MAX_BATCH_SIZE] {};
};

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
                       const uint64_t pre_att_concurrency_limit_,
                       const uint64_t att_concurrency_limit_,
                       const uint64_t post_att_concurrency_limit_,
                       const uint64_t cls_concurrency_limit_,
                       const uint64_t max_context_count_,
                       const bool randomize_parameters_ )
  : concurrency_limit( std::max(
    { pre_att_concurrency_limit_, att_concurrency_limit_, post_att_concurrency_limit_, cls_concurrency_limit_ } ) )
  , pre_att_concurrency_limit( pre_att_concurrency_limit_ )
  , att_concurrency_limit( att_concurrency_limit_ )
  , post_att_concurrency_limit( post_att_concurrency_limit_ )
  , cls_concurrency_limit( cls_concurrency_limit_ )
  , max_context_count( max_context_count_ )
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

  LOG( INFO ) << "Instantiated settings for " << typeid( T ).name() << ": " << to_string();
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
  oss << "pre_att_concurrency_limit: " << pre_att_concurrency_limit << ", ";
  oss << "att_concurrency_limit: " << att_concurrency_limit << ", ";
  oss << "post_att_concurrency_limit: " << post_att_concurrency_limit << ", ";
  oss << "cls_concurrency_limit: " << cls_concurrency_limit << ", ";
  oss << "max_context_count: " << max_context_count << ", ";
  oss << "start_layer_num: " << start_layer_num << ", ";
  oss << "end_layer_num: " << end_layer_num;
  oss << " }";
  return oss.str();
}

/* BASE WEIGHTS */

template<typename Config, typename DType>
BaseWeights<Config, DType>::BaseWeights( const DType* model, const Settings<Config>& settings )
{
  auto ptr = model;

  const bool model_hosts_embedding = ( settings.pre_att_concurrency_limit > 0 and settings.start_layer_num == 0 )
                                     or ( settings.cls_concurrency_limit > 0 and not Config::wcls_present );
  const bool model_hosts_att = settings.att_concurrency_limit > 0;
  const bool model_hosts_cls = settings.cls_concurrency_limit > 0;

  token_embedding_table = _advance_pointer( ptr, model_hosts_embedding ? Config::vocab_size * Config::dim : 0 );
  rms_final_weight = _advance_pointer( ptr, model_hosts_cls ? Config::dim : 0 );
  freq_cis_real = _advance_pointer( ptr, model_hosts_att ? Config::seq_len * Config::head_size / 2 : 0 );
  freq_cis_imag = _advance_pointer( ptr, model_hosts_att ? Config::seq_len * Config::head_size / 2 : 0 );
  wcls = Config::wcls_present ? ptr : token_embedding_table;
}

/* LAYER WEIGHTS */

template<typename Config, typename DType>
LayerWeights<Config, DType>::LayerWeights( const DType* model, const Settings<Config>& settings )
{
  auto ptr = model;

  const bool model_hosts_pre_att = settings.pre_att_concurrency_limit > 0;
  const bool model_hosts_post_att = settings.post_att_concurrency_limit > 0;

  // base pointers
  this->rms_att_weight = _advance_pointer( ptr, model_hosts_pre_att ? Config::dim : 0 );
  this->wq = _advance_pointer( ptr, model_hosts_pre_att ? Config::dim * Config::dim : 0 );
  this->wkv = _advance_pointer( ptr, model_hosts_pre_att ? Config::dim * Config::kv_dim * 2 : 0 );
  this->wo = _advance_pointer( ptr, model_hosts_post_att ? Config::dim * Config::dim : 0 );
  this->rms_ffn_weight = _advance_pointer( ptr, model_hosts_post_att ? Config::dim : 0 );
  this->w1 = _advance_pointer( ptr, model_hosts_post_att ? Config::dim * Config::hidden_dim : 0 );
  this->w2 = _advance_pointer( ptr, model_hosts_post_att ? Config::dim * Config::hidden_dim : 0 );
  this->w3 = _advance_pointer( ptr, model_hosts_post_att ? Config::dim * Config::hidden_dim : 0 );
}

/* RUN STATE */

template<typename Config, typename DType, typename ContextType>
ScratchPad<Config, DType, ContextType>::ScratchPad( const Settings<Config>& settings, DType* buffer )
  : buffer_( buffer )
  , x( buffer_ )
  , xb( buffer_ + Config::dim * settings.concurrency_limit )
  , xb2( xb + Config::dim * settings.concurrency_limit )
  , q( xb2 + Config::dim * settings.concurrency_limit )
  , kv( q + Config::dim * settings.concurrency_limit )
  , hb( kv + Config::kv_dim * 2 * settings.concurrency_limit )
  , hb2( hb + Config::hidden_dim * settings.concurrency_limit )
  , att( hb2 + Config::hidden_dim * settings.concurrency_limit )
  , logits( att + Config::n_heads * Config::seq_len * settings.concurrency_limit )
  , temp_softmax( logits + Config::vocab_size * settings.concurrency_limit )
{
}

template<typename Config, typename DType, typename ContextType>
size_t ScratchPad<Config, DType, ContextType>::scratchpad_size( const Settings<Config>& settings )
{
  return sizeof( DType ) * settings.concurrency_limit
         * ( Config::dim * 4 + Config::kv_dim * 2 + Config::hidden_dim * 2 + Config::n_heads * Config::seq_len
             + Config::vocab_size + Config::n_heads );
}

} // namespace glinthawk::models::llama2
