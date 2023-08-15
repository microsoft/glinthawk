#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace glinthawk::models::llama2 {

struct Config
{
  Config( const std::filesystem::path& weights_path );
  std::string to_string() const;

  int32_t dim {};        // transformer dimension
  int32_t hidden_dim {}; // for ffn layers
  int32_t n_layers {};   // number of layers
  int32_t n_heads {};    // number of query heads
  int32_t n_kv_heads {}; // number of key/value heads (can be < query heads because of multiquery)
  int32_t vocab_size {}; // vocabulary size (byte-level)
  int32_t seq_len {};    // max sequence length
};

class Vocabulary
{
private:
  std::vector<std::string> token_to_word_ {};
  std::unordered_multimap<std::string, int> word_to_token_ {};

public:
  Vocabulary( const Config& config, const std::filesystem::path& vocabulary_path );

  size_t size() const { return token_to_word_.size(); }
  int get_token( const std::string& word ) const;
  std::string get_word( const int token ) const;
};

struct BaseWeights
{
  BaseWeights( const Config& config, const float* model );
  BaseWeights( const BaseWeights& ) = delete;
  BaseWeights operator=( const BaseWeights& ) = delete;

  // token embedding table
  const float* token_embedding_table {}; // (vocab_size, dim)

  // final rmsnorm
  const float* rms_final_weight {}; // (dim,)

  // freq_cis for RoPE relatively positional embeddings
  const float* freq_cis_real {}; // (seq_len, dim/2)
  const float* freq_cis_imag {}; // (seq_len, dim/2)

  // classifier weights for the logits, on the last layer
  const float* wcls {};
};

struct LayerWeights
{
  LayerWeights() = default;
  LayerWeights( const Config& config, const float* model, const int32_t layer_num );

  // weights for rmsnorms
  const float* rms_att_weight { nullptr }; // (dim) rmsnorm weights
  const float* rms_ffn_weight { nullptr }; // (dim)

  // weights for matmuls
  const float* wq { nullptr }; // (dim, dim)
  const float* wk { nullptr }; // (dim, dim)
  const float* wv { nullptr }; // (dim, dim)
  const float* wo { nullptr }; // (dim, dim)

  // weights for ffn
  const float* w1 { nullptr }; // (hidden_dim, dim)
  const float* w2 { nullptr }; // (dim, hidden_dim)
  const float* w3 { nullptr }; // (hidden_dim, dim)
};

struct RunState
{
  RunState( const Config& config, const int32_t start_layer, const int32_t end_layer );
  RunState( const RunState& ) = delete;
  RunState operator=( const RunState& ) = delete;

  static size_t state_size( const Config& config )
  {
    return sizeof( float )
           * ( config.dim * 5 + config.hidden_dim * 2 + config.n_heads * config.seq_len + config.vocab_size
               + config.n_heads )
  }

  float* buffer_;         // we use this buffer for everything except for activations
  float* x {};            // activation at current time stamp (dim,)
  float* xb {};           // same, but inside a residual branch (dim,)
  float* xb2 {};          // an additional buffer just for convenience (dim,)
  float* q {};            // query (dim,)
  float* k {};            // key (dim,)
  float* v {};            // value (dim,)
  float* hb {};           // buffer for hidden dimension in the ffn (hidden_dim,)
  float* hb2 {};          // buffer for hidden dimension in the ffn (hidden_dim,)
  float* att {};          // buffer for scores/attention values (n_heads, seq_len)
  float* logits {};       // output logits
  float* temp_softmax {}; // temporary buffer for computing softmax (n_heads,)

  // k-v cache
  struct KVCache
  {
    const int32_t start_layer_;
    const int32_t end_layer_;

    float* buffer_;
    const int seq_len_;
    const int dim_;
    const int n_layers_;
    const int head_size_;

    inline float* key( const int layer, const int step, const int head = 0 );
    inline float* value( const int layer, const int step, const int head = 0 );

    void pop();

    KVCache( const Config& config, const int32_t start_layer, const int32_t end_layer );
  };

  KVCache kv_cache;
};

} // namespace glinthawk::models::llama2
