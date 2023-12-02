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
#include <unordered_map>
#include <vector>

#include "models/common/model.hh"
#include "models/llama2/ops/concept.hh"
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
            const uint64_t concurrency_limit,
            const bool randomize_parameters );

  std::string to_string() const;

  /// @brief Size of the config stored on disk (in bytes)
  static size_t config_size() { return sizeof( int32_t ) * 7; }
  uint64_t n_layers_loaded() const { return end_layer_num - start_layer_num + 1; }

  uint64_t start_layer_num {};
  uint64_t end_layer_num {};
  uint64_t concurrency_limit { 1 }; // max concurrent inference size
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
  const DType* wq { nullptr }; // (dim, dim)
  const DType* wk { nullptr }; // (dim, dim)
  const DType* wv { nullptr }; // (dim, dim)
  const DType* wo { nullptr }; // (dim, dim)

  // weights for ffn
  const DType* w1 { nullptr }; // (hidden_dim, dim)
  const DType* w2 { nullptr }; // (dim, hidden_dim)
  const DType* w3 { nullptr }; // (hidden_dim, dim)
};

/// @brief This class acts as the scratchpad for the computations
template<typename Config, typename DType>
requires ModelConfig<Config>
struct RunState
{
  RunState() = default;
  RunState( const Settings<Config>& settings, DType* buffer );

  RunState( const RunState& ) = delete;
  RunState operator=( const RunState& ) = delete;
  RunState( RunState&& ) = default;
  RunState& operator=( RunState&& ) = default;

  static size_t state_size( const Settings<Config>& settings );

  DType* buffer_ {};      // we use this buffer for everything, including activations
  DType* x {};            // activation at current time stamp (B, dim)
  DType* xb {};           // same, but inside a residual branch (B, dim)
  DType* xb2 {};          // an additional buffer just for convenience (B, dim)
  DType* q {};            // query (B, dim)
  DType* k {};            // key (B, kv_dim)
  DType* v {};            // value (B, kv_dim)
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
  DType* batch_context_pointers[MAX_BATCH_SIZE] {};
};

/// @brief InferenceContext for Llama2 model is the KV-cache
template<typename Config, typename DType>
requires ModelConfig<Config>
struct InferenceContext
{
  static size_t context_size( const Settings<Config>& settings );
  DType* key( const Settings<Config>& settings, int layer_num, const int token_pos, const int head = 0 );
  DType* value( const Settings<Config>& settings, int layer_num, const int token_pos, const int head = 0 );
  bool empty() const { return buffer_ == nullptr; }

  DType* buffer_ { nullptr };
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

// TODO: optimize run state memory usage
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
{
}

template<typename Config, typename DType>
size_t RunState<Config, DType>::state_size( const Settings<Config>& settings )
{
  return sizeof( DType ) * settings.concurrency_limit
         * ( Config::dim * 4 + Config::kv_dim * 2 + Config::hidden_dim * 2 + Config::n_heads * Config::seq_len
             + Config::vocab_size + Config::n_heads );
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

} // namespace glinthawk::models::llama2
