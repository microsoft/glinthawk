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
#include "util/demangle.hh"
#include "util/util.hh"
#include "variants.hh"

namespace glinthawk::models::llama2 {

constexpr size_t MAX_BATCH_SIZE = 1024;

template<typename Config>
requires ModelConfig<Config>
struct ConfigRuntime
{
  ConfigRuntime() {}

  ConfigRuntime( const std::filesystem::path& config_file,
                 std::array<std::array<bool, util::to_underlying( models::InferenceStage::__COUNT__ )>,
                            Config::n_layers> hosting_table,
                 const uint64_t concurrency_limit,
                 const uint64_t max_context_count,
                 const bool randomize_parameters );

  [[nodiscard]] std::string to_string() const;

  [[nodiscard]] bool hosts( size_t layer, models::InferenceStage stage ) const
  {
    return hosting_table_[layer][util::to_underlying( stage )];
  }
  [[nodiscard]] bool hosts_in_any_layer( models::InferenceStage stage ) const
  {
    return hosting_stage_table_[util::to_underlying( stage )];
  }
  [[nodiscard]] size_t num_attention_layers_hosted() const { return num_attention_layers_hosted_; };

  /// @brief Size of the config stored on disk (in bytes)
  static size_t config_size() { return sizeof( int32_t ) * 7; }

  std::array<std::array<bool, util::to_underlying( models::InferenceStage::__COUNT__ )>, Config::n_layers>
    hosting_table_;
  std::array<bool, util::to_underlying( models::InferenceStage::__COUNT__ )> hosting_stage_table_;
  uint64_t concurrency_limit { 1 };          // max concurrent inference size
  uint64_t max_context_count { 1 };          // max number of contexts
  size_t num_attention_layers_hosted_ { 0 }; // how many attention layers are supported
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
  using ParameterArray = typename std::array<uint64_t, 5>;
  BaseWeights() = default;
  BaseWeights( const DType* ptr, const ConfigRuntime<Config>& settings );

  BaseWeights( const BaseWeights& ) = delete;
  BaseWeights operator=( const BaseWeights& ) = delete;
  BaseWeights( BaseWeights&& ) = default;
  BaseWeights& operator=( BaseWeights&& ) = default;

  static consteval ParameterArray on_disk_element_size();
  static consteval size_t on_disk_total_byte_size();
  static consteval ParameterArray on_disk_offset();

  static ParameterArray in_memory_element_size( const ConfigRuntime<Config>& settings );
  static size_t in_memory_total_byte_size( const ConfigRuntime<Config>& settings );
  static ParameterArray in_memory_offset( const ConfigRuntime<Config>& settings );

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
  using ParameterArray = typename std::array<uint64_t, 8>;

  LayerWeights() = default;
  LayerWeights( const DType* ptr, const ConfigRuntime<Config>& settings );

  LayerWeights( const LayerWeights& ) = delete;
  LayerWeights operator=( const LayerWeights& ) = delete;
  LayerWeights( LayerWeights&& ) = default;
  LayerWeights& operator=( LayerWeights&& ) = default;

  static consteval ParameterArray on_disk_element_size();
  static consteval size_t on_disk_total_byte_size();
  static consteval ParameterArray on_disk_offset();

  static ParameterArray in_memory_element_size( const ConfigRuntime<Config>& settings, const size_t layer_num );
  static size_t in_memory_total_byte_size( const ConfigRuntime<Config>& settings, const size_t layer_num );
  static ParameterArray in_memory_offset( const ConfigRuntime<Config>& settings, const size_t layer_num );

  static size_t in_memory_all_layers_total_byte_size( const ConfigRuntime<Config>& settings );

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
  ScratchPad( const ConfigRuntime<Config>& settings, DType* buffer );

  ScratchPad( const ScratchPad& ) = delete;
  ScratchPad operator=( const ScratchPad& ) = delete;
  ScratchPad( ScratchPad&& ) = default;
  ScratchPad& operator=( ScratchPad&& ) = default;

  static size_t scratchpad_size( const ConfigRuntime<Config>& settings );
  static size_t temp_buffer_size( const ConfigRuntime<Config>& settings );

  /* The following reside on the device */
  DType* buffer_ {}; // we use this buffer for everything, including activations
  DType* x {};       // activation at current time stamp (B, dim)
  DType* xb {};      // same, but inside a residual branch (B, dim)
  DType* xb2 {};     // an additional buffer just for convenience (B, dim)
  DType* q {};       // query (B, dim)
  DType* kv {};      // key and value (B, kv_dim, 2)
  DType* hb {};      // buffer for hidden dimension in the ffn (B, hidden_dim)
  DType* hb2 {};     // buffer for hidden dimension in the ffn (B, hidden_dim)
  DType* att {};     // buffer for scores/attention values (B, n_heads, seq_len)
  DType* logits {};  // output logits (B, vocab_size)

  // An auxiliary buffer of size (B * max(n_heads, dim) * max(sizeof(DType), sizeof(float32_t))) used for several
  // operations, including softmax and rmsnorm. Since some operations require space for float32_t, regardless of the
  // DType, we need to allocate enough space for the largest type. See `temp_buffer_size()` for more details.
  DType* temp {};

  /* The following reside on the host */
  uint64_t curr_concurrency_size { 1 };              // current batch size
  uint32_t argmax_pos[MAX_BATCH_SIZE] {};            // argmax results (B, )
  uint32_t batch_token_positions[MAX_BATCH_SIZE] {}; // token positions for the current batch
  typename ContextType::LayerContextType batch_layer_contexts[MAX_BATCH_SIZE] {}; // layer KV-cache addresses
  typename ContextType::TokenContextType batch_token_contexts[MAX_BATCH_SIZE] {}; // token KV-cache addresses
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

template<typename Config>
requires ModelConfig<Config>
ConfigRuntime<Config>::ConfigRuntime(
  const std::filesystem::path& config_file,
  std::array<std::array<bool, util::to_underlying( models::InferenceStage::__COUNT__ )>, Config::n_layers>
    hosting_table,
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
  CHECK_EQ( dim, Config::dim ) << "dim does not match config file";
  CHECK_EQ( kv_dim, Config::kv_dim ) << "kv_dim does not match config file";
  CHECK_EQ( hidden_dim, Config::hidden_dim ) << "hidden_dim does not match config file";
  CHECK_EQ( n_layers, Config::n_layers ) << "n_layers does not match config file";
  CHECK_EQ( head_size, Config::head_size ) << "head_size does not match config file";
  CHECK_EQ( n_heads, Config::n_heads ) << "n_heads does not match config file";
  CHECK_EQ( n_kv_heads, Config::n_kv_heads ) << "n_kv_heads does not match config file";
  CHECK_EQ( gqa_size, Config::gqa_size ) << "gqa_size does not match config file";
  CHECK_EQ( vocab_size, Config::vocab_size ) << "vocab_size does not match config file";
  CHECK_EQ( seq_len, Config::seq_len ) << "seq_len does not match config file";
  CHECK_EQ( wcls_present, Config::wcls_present ) << "wcls_present does not match config file";

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

  this->hosting_table_ = hosting_table;
  bool hosts_anything = false;

  for ( size_t i = 0; i < hosting_stage_table_.size(); i++ ) {
    hosting_stage_table_[i] = false;
    for ( size_t j = 0; j < Config::n_layers; j++ ) {
      hosting_stage_table_[i] = hosting_stage_table_[i] or hosting_table_[j][i];
    }
    hosts_anything = hosts_anything or hosting_stage_table_[i];
  }

  CHECK( hosts_anything ) << "This hosting table does not host any part of the model";

  num_attention_layers_hosted_ = 0;
  for ( size_t j = 0; j < Config::n_layers; j++ ) {
    if ( hosting_table_[j][util::to_underlying( models::InferenceStage::Attention )] ) {
      num_attention_layers_hosted_++;
    }
  }

  LOG( INFO ) << "Instantiated settings for " << util::demangle( typeid( Config ).name() ) << ": " << to_string();
}

template<typename Config>
requires ModelConfig<Config>
std::string ConfigRuntime<Config>::to_string() const
{
  // TODO: update to_string with something to represent what the node is hosting
  std::ostringstream oss;
  oss << "{ ";
  oss << "dim: " << Config::dim << ", ";
  oss << "kv_dim: " << Config::kv_dim << ", ";
  oss << "hidden_dim: " << Config::hidden_dim << ", ";
  oss << "n_layers: " << Config::n_layers << ", ";
  oss << "head_size: " << Config::head_size << ", ";
  oss << "n_heads: " << Config::n_heads << ", ";
  oss << "n_kv_heads: " << Config::n_kv_heads << ", ";
  oss << "gqa_size: " << Config::gqa_size << ", ";
  oss << "vocab_size: " << Config::vocab_size << ", ";
  oss << "seq_len: " << Config::seq_len << ", ";
  oss << "wcls_present: " << Config::wcls_present << ", ";
  oss << "concurrency_limit: " << concurrency_limit << ", ";
  oss << "max_context_count: " << max_context_count << ", ";
  oss << " }";
  return oss.str();
}

/* BASE WEIGHTS */

template<typename Config, typename DType>
requires ModelConfig<Config>
BaseWeights<Config, DType>::BaseWeights( const DType* ptr, const ConfigRuntime<Config>& settings )
{
  ParameterArray offsets = in_memory_offset( settings );
  this->token_embedding_table = ptr + offsets[0];
  this->rms_final_weight = ptr + offsets[1];
  this->freq_cis_real = ptr + offsets[2];
  this->freq_cis_imag = ptr + offsets[3];
  this->wcls = ptr + offsets[4];
}

template<typename Config, typename DType>
requires ModelConfig<Config>
consteval BaseWeights<Config, DType>::ParameterArray BaseWeights<Config, DType>::on_disk_element_size()
{
  ParameterArray sizes {};

  // Embedding table (token_embedding_table)
  sizes[0] = Config::vocab_size * Config::dim;
  // Classification (rms_final_weight)
  sizes[1] = Config::dim;
  // RoPE (freq_cis_real)
  sizes[2] = Config::seq_len * Config::head_size / 2;
  // RoPE (freq_cis_imag)
  sizes[3] = Config::seq_len * Config::head_size / 2;
  // Classification (wcls)
  sizes[4] = Config::wcls_present ? Config::vocab_size * Config::dim : 0;

  return sizes;
}

template<typename Config, typename DType>
requires ModelConfig<Config>
consteval size_t BaseWeights<Config, DType>::on_disk_total_byte_size()
{
  ParameterArray sizes = on_disk_element_size();
  return sizeof( DType ) * ( std::accumulate( sizes.begin(), sizes.end(), 0 ) );
}

template<typename Config, typename DType>
requires ModelConfig<Config>
consteval BaseWeights<Config, DType>::ParameterArray BaseWeights<Config, DType>::on_disk_offset()
{
  ParameterArray sizes;
  ParameterArray offsets = on_disk_element_size();

  offsets[0] = 0;
  for ( size_t i = 1; i < sizes.size(); ++i ) {
    offsets[i] = offsets[i - 1] + sizes[i - 1];
  }

  // Classification (wcls)
  offsets[4] = Config::wcls_present ? offsets[4] : 0;

  return offsets;
}

template<typename Config, typename DType>
requires ModelConfig<Config>
BaseWeights<Config, DType>::ParameterArray BaseWeights<Config, DType>::in_memory_element_size(
  const ConfigRuntime<Config>& settings )
{

  const bool model_hosts_embedding
    = settings.hosts( 0, models::InferenceStage::PreAttention )
      or ( settings.hosts_in_any_layer( models::InferenceStage::Classification ) and not Config::wcls_present );
  const bool model_hosts_att = settings.hosts_in_any_layer( models::InferenceStage::Attention );
  const bool model_hosts_cls = settings.hosts_in_any_layer( models::InferenceStage::Classification );

  ParameterArray sizes {};

  // Embedding table (token_embedding_table)
  sizes[0] = model_hosts_embedding ? Config::vocab_size * Config::dim : 0;
  // Classification (rms_final_weight)
  sizes[1] = model_hosts_cls ? Config::dim : 0;
  // RoPE (freq_cis_real)
  sizes[2] = model_hosts_att ? Config::seq_len * Config::head_size / 2 : 0;
  // RoPE (freq_cis_imag)
  sizes[3] = model_hosts_att ? Config::seq_len * Config::head_size / 2 : 0;
  // Classification (wcls)
  sizes[4] = Config::wcls_present and model_hosts_cls ? Config::vocab_size * Config::dim : 0;

  return sizes;
}

template<typename Config, typename DType>
requires ModelConfig<Config>
size_t BaseWeights<Config, DType>::in_memory_total_byte_size( const ConfigRuntime<Config>& settings )
{
  ParameterArray sizes = in_memory_element_size( settings );
  return sizeof( DType ) * ( std::accumulate( sizes.begin(), sizes.end(), 0 ) );
}

template<typename Config, typename DType>
requires ModelConfig<Config>
BaseWeights<Config, DType>::ParameterArray BaseWeights<Config, DType>::in_memory_offset(
  const ConfigRuntime<Config>& settings )
{
  ParameterArray sizes;
  ParameterArray offsets = in_memory_element_size( settings );

  offsets[0] = 0;
  for ( size_t i = 1; i < sizes.size(); ++i ) {
    offsets[i] = offsets[i - 1] + sizes[i - 1];
  }

  // Classification (wcls)
  offsets[4] = Config::wcls_present ? offsets[4] : 0;

  return offsets;
}

/* LAYER WEIGHTS */

template<typename Config, typename DType>
requires ModelConfig<Config>
LayerWeights<Config, DType>::LayerWeights( const DType* ptr, const ConfigRuntime<Config>& settings )
{
  ParameterArray offsets = in_memory_offset( settings );
  this->rms_att_weight = ptr + offsets[0];
  this->wq = ptr + offsets[1];
  this->wkv = ptr + offsets[2];
  this->wo = ptr + offsets[3];
  this->rms_ffn_weight = ptr + offsets[4];
  this->w1 = ptr + offsets[5];
  this->w2 = ptr + offsets[6];
  this->w3 = ptr + offsets[7];
}

template<typename Config, typename DType>
requires ModelConfig<Config>
consteval LayerWeights<Config, DType>::ParameterArray LayerWeights<Config, DType>::on_disk_element_size()
{
  ParameterArray sizes {};

  // PreAttention (rms_att_weight)
  sizes[0] = Config::vocab_size * Config::dim;
  // PreAttention (wq)
  sizes[1] = Config::vocab_size * Config::dim * Config::dim;
  // PreAttention (wkv)
  sizes[2] = Config::vocab_size * Config::dim * Config::kv_dim * 2;
  // PostAttention (wo)
  sizes[3] = Config::vocab_size * Config::dim * Config::dim;
  // PostAttention (rms_ffn_weight)
  sizes[4] = Config::vocab_size * Config::dim;
  // PostAttention (w1)
  sizes[5] = Config::vocab_size * Config::dim * Config::hidden_dim;
  // PostAttention (w2)
  sizes[6] = Config::vocab_size * Config::dim * Config::hidden_dim;
  // PostAttention (w3)
  sizes[7] = Config::vocab_size * Config::dim * Config::hidden_dim;

  return sizes;
}

template<typename Config, typename DType>
requires ModelConfig<Config>
consteval size_t LayerWeights<Config, DType>::on_disk_total_byte_size()
{
  ParameterArray sizes = on_disk_element_size();
  return sizeof( DType ) * ( std::accumulate( sizes.begin(), sizes.end(), 0 ) );
}

template<typename Config, typename DType>
requires ModelConfig<Config>
consteval LayerWeights<Config, DType>::ParameterArray LayerWeights<Config, DType>::on_disk_offset()
{
  ParameterArray sizes;
  ParameterArray offsets = on_disk_element_size();

  offsets[0] = 0;
  for ( size_t i = 1; i < sizes.size(); ++i ) {
    offsets[i] = offsets[i - 1] + sizes[i - 1];
  }

  return offsets;
}

template<typename Config, typename DType>
requires ModelConfig<Config>
LayerWeights<Config, DType>::ParameterArray LayerWeights<Config, DType>::in_memory_element_size(
  const ConfigRuntime<Config>& settings,
  const size_t layer_num )
{
  const bool model_hosts_pre_att = settings.hosts( layer_num, models::InferenceStage::PreAttention );
  const bool model_hosts_post_att = settings.hosts( layer_num, models::InferenceStage::PostAttention );

  ParameterArray sizes {};

  // PreAttention (rms_att_weight)
  sizes[0] = model_hosts_pre_att ? Config::vocab_size * Config::dim : 0;
  // PreAttention (wq)
  sizes[1] = model_hosts_pre_att ? Config::vocab_size * Config::dim * Config::dim : 0;
  // PreAttention (wkv)
  sizes[2] = model_hosts_pre_att ? Config::vocab_size * Config::dim * Config::kv_dim * 2 : 0;
  // PostAttention (wo)
  sizes[3] = model_hosts_post_att ? Config::vocab_size * Config::dim * Config::dim : 0;
  // PostAttention (rms_ffn_weight)
  sizes[4] = model_hosts_post_att ? Config::vocab_size * Config::dim : 0;
  // PostAttention (w1)
  sizes[5] = model_hosts_post_att ? Config::vocab_size * Config::dim * Config::hidden_dim : 0;
  // PostAttention (w2)
  sizes[6] = model_hosts_post_att ? Config::vocab_size * Config::dim * Config::hidden_dim : 0;
  // PostAttention (w3)
  sizes[7] = model_hosts_post_att ? Config::vocab_size * Config::dim * Config::hidden_dim : 0;

  return sizes;
}

template<typename Config, typename DType>
requires ModelConfig<Config>
size_t LayerWeights<Config, DType>::in_memory_total_byte_size( const ConfigRuntime<Config>& settings,
                                                               const size_t layer_num )
{
  ParameterArray sizes = in_memory_element_size( settings, layer_num );
  return sizeof( DType ) * ( std::accumulate( sizes.begin(), sizes.end(), 0 ) );
}

template<typename Config, typename DType>
requires ModelConfig<Config>
LayerWeights<Config, DType>::ParameterArray LayerWeights<Config, DType>::in_memory_offset(
  const ConfigRuntime<Config>& settings,
  const size_t layer_num )
{
  ParameterArray sizes;
  ParameterArray offsets = in_memory_element_size( settings, layer_num );

  offsets[0] = 0;
  for ( size_t i = 1; i < sizes.size(); ++i ) {
    offsets[i] = offsets[i - 1] + sizes[i - 1];
  }

  return offsets;
}

template<typename Config, typename DType>
requires ModelConfig<Config>
size_t LayerWeights<Config, DType>::in_memory_all_layers_total_byte_size( const ConfigRuntime<Config>& settings )
{
  size_t total_size = 0;
  for ( size_t i = 0; i < Config::n_layers; ++i ) {
    total_size += in_memory_total_byte_size( settings, i );
  }
  return total_size;
}

/* RUN STATE */

template<typename Config, typename DType, typename ContextType>
requires ModelConfig<Config>
ScratchPad<Config, DType, ContextType>::ScratchPad( const ConfigRuntime<Config>& settings, DType* buffer )
  : buffer_( buffer )
{
  auto ptr = buffer_;

  x = _advance_pointer( ptr, Config::dim * settings.concurrency_limit );
  xb = _advance_pointer( ptr, Config::dim * settings.concurrency_limit );
  xb2 = _advance_pointer( ptr, Config::dim * settings.concurrency_limit );
  q = _advance_pointer( ptr, Config::dim * settings.concurrency_limit );
  kv = _advance_pointer( ptr, Config::kv_dim * 2 * settings.concurrency_limit );
  hb = _advance_pointer( ptr, Config::hidden_dim * settings.concurrency_limit );
  hb2 = _advance_pointer( ptr, Config::hidden_dim * settings.concurrency_limit );
  att = _advance_pointer( ptr, Config::n_heads * Config::seq_len * settings.concurrency_limit );
  logits = _advance_pointer( ptr, Config::vocab_size * settings.concurrency_limit );
  temp = _advance_pointer( ptr, temp_buffer_size( settings ) );
}

template<typename Config, typename DType, typename ContextType>
requires ModelConfig<Config>
size_t ScratchPad<Config, DType, ContextType>::scratchpad_size( const ConfigRuntime<Config>& settings )
{
  return sizeof( DType ) * settings.concurrency_limit
           * ( Config::dim * 4                     // x, xb, xb2, q
               + Config::kv_dim * 2                // kv
               + Config::hidden_dim * 2            // hb, hb2
               + Config::n_heads * Config::seq_len // att
               + Config::vocab_size )              // logits
         + temp_buffer_size( settings )            // temp
    ;
}

template<typename Config, typename DType, typename ContextType>
requires ModelConfig<Config>
size_t ScratchPad<Config, DType, ContextType>::temp_buffer_size( const ConfigRuntime<Config>& settings )
{
  return settings.concurrency_limit * std::max( sizeof( DType ), sizeof( glinthawk::float32_t ) )
         * std::max( Config::n_heads, Config::dim );
}

} // namespace glinthawk::models::llama2
