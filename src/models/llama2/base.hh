#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "models/common/model.hh"

namespace glinthawk::models::llama2 {

struct Config
{
  Config( const std::filesystem::path& config_file );

  std::string to_string() const;

  static size_t config_size() { return sizeof( int32_t ) * 7; }

  uint64_t dim {};        // transformer dimension
  uint64_t hidden_dim {}; // for ffn layers
  uint64_t n_layers {};   // number of layers
  uint64_t n_heads {};    // number of query heads
  uint64_t n_kv_heads {}; // number of key/value heads (can be < query heads because of multiquery)
  uint64_t vocab_size {}; // vocabulary size (byte-level)
  uint64_t seq_len {};    // max sequence length

  bool wcls_present { false };
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

template<typename DType>
struct BaseWeights
{
  BaseWeights( const Config& config, const DType* base_weights );
  BaseWeights( const BaseWeights& ) = delete;
  BaseWeights operator=( const BaseWeights& ) = delete;

  static size_t base_size( const Config& config );

  const DType* token_embedding_table {}; // (vocab_size, dim)
  const DType* rms_final_weight {};      // (dim,)

  // freq_cis for RoPE relatively positional embeddings
  const DType* freq_cis_real {}; // (seq_len, dim/2)
  const DType* freq_cis_imag {}; // (seq_len, dim/2)

  // classifier weights for the logits, on the last layer
  const DType* wcls {};
};

template<typename DType>
struct LayerWeights
{
  LayerWeights() = default;
  LayerWeights( const Config& config, const DType* model );

  static size_t layer_size( const Config& config );

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

template<typename DType>
struct RunState
{
  RunState( const Config& config, DType* buffer );
  RunState( const RunState& ) = delete;
  RunState operator=( const RunState& ) = delete;

  static size_t state_size( const Config& config );

  DType* buffer_;         // we use this buffer for everything, including activations
  DType* x {};            // activation at current time stamp (dim,)
  DType* xb {};           // same, but inside a residual branch (dim,)
  DType* xb2 {};          // an additional buffer just for convenience (dim,)
  DType* q {};            // query (dim,)
  DType* k {};            // key (dim,)
  DType* v {};            // value (dim,)
  DType* hb {};           // buffer for hidden dimension in the ffn (hidden_dim,)
  DType* hb2 {};          // buffer for hidden dimension in the ffn (hidden_dim,)
  DType* att {};          // buffer for scores/attention values (n_heads, seq_len)
  DType* logits {};       // output logits
  DType* temp_softmax {}; // temporary buffer for computing softmax (n_heads,)
};

template<typename DType>
struct KVCache
{
  KVCache( const Config& config, DType* buffer, const int32_t start_layer, const int32_t end_layer );

  static size_t cache_size( const Config& config, const int32_t start_layer, const int32_t end_layer );

  const int32_t start_layer_;
  const int32_t end_layer_;

  DType* buffer_;
  const int seq_len_;
  const int dim_;
  const int n_layers_;
  const int head_size_;

  DType* key( int layer, const int step, const int head = 0 );
  DType* value( int layer, const int step, const int head = 0 );
};

template<typename DType>
class BaseLlama2 : public virtual glinthawk::models::Model
{
protected:
  std::unique_ptr<DType, void ( * )( DType* )> base_weights_buffer_;
  std::unique_ptr<DType, void ( * )( DType* )> layers_buffer_;
  std::unique_ptr<DType, void ( * )( DType* )> run_state_buffer_;
  std::unique_ptr<DType, void ( * )( DType* )> kv_cache_buffer_;

  const Config config_;
  const int32_t start_layer_num_;
  const int32_t end_layer_num_;

  RunState<DType> state_;
  KVCache<DType> kv_cache_;
  const BaseWeights<DType> base_weights_;
  const std::vector<LayerWeights<DType>> layer_weights_;

protected:
  BaseLlama2( const Config& config,
              std::unique_ptr<DType, void ( * )( DType* )>&& base_weights,
              std::unique_ptr<DType, void ( * )( DType* )>&& layers_weights,
              std::unique_ptr<DType, void ( * )( DType* )>&& run_state,
              std::unique_ptr<DType, void ( * )( DType* )>&& kv_cache,
              const int32_t start_layer = 0,
              const int32_t end_layer = -1 );

public:
  ~BaseLlama2() override = default;
};

} // namespace glinthawk::models::llama2
