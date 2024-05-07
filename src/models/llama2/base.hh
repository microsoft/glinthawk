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
struct ConfigRuntime
{
  ConfigRuntime() {}

  ConfigRuntime( const std::filesystem::path& config_file,
                 const uint32_t start_layer,
                 const uint32_t end_layer,
                 const uint64_t concurrency_limit,
                 const uint64_t max_context_count,
                 const bool randomize_parameters );

  std::string to_string() const;

  /// @brief Size of the config stored on disk (in bytes)
  static size_t config_size() { return sizeof( int32_t ) * 7; }
  uint64_t n_layers_loaded() const { return end_layer_num - start_layer_num + 1; }

  uint64_t start_layer_num {};
  uint64_t end_layer_num {};
  uint64_t concurrency_limit { 1 }; // max concurrent inference size
  uint64_t max_context_count { 1 }; // max number of contexts
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
  // TODO: We should have partial model loading
  BaseWeights() = default;
  BaseWeights( const DType* base_weights );

  BaseWeights( const BaseWeights& ) = delete;
  BaseWeights operator=( const BaseWeights& ) = delete;
  BaseWeights( BaseWeights&& ) = default;
  BaseWeights& operator=( BaseWeights&& ) = default;

  static consteval size_t base_size()
  {
    return sizeof( DType )
           * ( Config::vocab_size * Config::dim + Config::dim + Config::seq_len * Config::dim / Config::n_heads
               + ( Config::wcls_present ? ( Config::vocab_size * Config::dim ) : 0 ) );
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
  LayerWeights( const DType* model );

  LayerWeights( const LayerWeights& ) = delete;
  LayerWeights operator=( const LayerWeights& ) = delete;
  LayerWeights( LayerWeights&& ) = default;
  LayerWeights& operator=( LayerWeights&& ) = default;

  static consteval size_t layer_size()
  {
    return sizeof( DType )
           * ( 2 * Config::dim + 2 * Config::dim * Config::dim + 2 * Config::dim * Config::kv_dim
               + 3 * Config::dim * Config::hidden_dim );
  }

  // weights for rmsnorms
  const DType* rms_att_weight { nullptr }; // (dim) rmsnorm weights
  const DType* rms_ffn_weight { nullptr }; // (dim)

  // weights for matmuls
  const DType* wq { nullptr };  // (dim, dim)
  const DType* wkv { nullptr }; // (dim, 2*kv_dim)
  const DType* wo { nullptr };  // (dim, dim)

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
  ScratchPad( const ConfigRuntime<Config>& settings, DType* buffer );

  ScratchPad( const ScratchPad& ) = delete;
  ScratchPad operator=( const ScratchPad& ) = delete;
  ScratchPad( ScratchPad&& ) = default;
  ScratchPad& operator=( ScratchPad&& ) = default;

  static size_t scratchpad_size( const ConfigRuntime<Config>& settings );

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
ConfigRuntime<T>::ConfigRuntime( const std::filesystem::path& config_file,
                                 const uint32_t start_layer,
                                 const uint32_t end_layer,
                                 const uint64_t concurrency_limit_,
                                 const uint64_t max_context_count_,
                                 const bool randomize_parameters_ )
  : concurrency_limit( concurrency_limit_ )
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
std::string ConfigRuntime<T>::to_string() const
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
  oss << "max_context_count: " << max_context_count << ", ";
  oss << "start_layer_num: " << start_layer_num << ", ";
  oss << "end_layer_num: " << end_layer_num;
  oss << " }";
  return oss.str();
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
  this->wkv = _advance_pointer( ptr, Config::dim * Config::kv_dim * 2 );
  this->wo = _advance_pointer( ptr, Config::dim * Config::dim );
  this->rms_ffn_weight = _advance_pointer( ptr, Config::dim );
  this->w1 = _advance_pointer( ptr, Config::dim * Config::hidden_dim );
  this->w2 = _advance_pointer( ptr, Config::dim * Config::hidden_dim );
  this->w3 = _advance_pointer( ptr, Config::dim * Config::hidden_dim );
}

/* RUN STATE */

template<typename Config, typename DType, typename ContextType>
ScratchPad<Config, DType, ContextType>::ScratchPad( const ConfigRuntime<Config>& settings, DType* buffer )
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
size_t ScratchPad<Config, DType, ContextType>::scratchpad_size( const ConfigRuntime<Config>& settings )
{
  return sizeof( DType ) * settings.concurrency_limit
         * ( Config::dim * 4 + Config::kv_dim * 2 + Config::hidden_dim * 2 + Config::n_heads * Config::seq_len
             + Config::vocab_size + Config::n_heads );
}

} // namespace glinthawk::models::llama2
