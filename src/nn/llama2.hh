#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "util/file_descriptor.hh"
#include "util/ring_buffer.hh"

namespace glinthawk {

class Llama2
{
private:
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
    std::unique_ptr<float[]> buffer_;

    // current wave of activations
    float* x {};      // activation at current time stamp (dim,)
    float* xb {};     // same, but inside a residual branch (dim,)
    float* xb2 {};    // an additional buffer just for convenience (dim,)
    float* hb {};     // buffer for hidden dimension in the ffn (hidden_dim,)
    float* hb2 {};    // buffer for hidden dimension in the ffn (hidden_dim,)
    float* q {};      // query (dim,)
    float* k {};      // key (dim,)
    float* v {};      // value (dim,)
    float* att {};    // buffer for scores/attention values (n_heads, seq_len)
    float* logits {}; // output logits

    // k-v cache
    struct KVCache
    {
      std::unique_ptr<float[]> buffer_;
      const int seq_len_;
      const int dim_;
      const int n_layers_;
      const int head_size_;

      inline float* key( const int layer, const int step, const int head = 0 );
      inline float* value( const int layer, const int step, const int head = 0 );

      void pop();

      KVCache( const Config& config );
    };

    KVCache kv_cache;

    // other information
    int current_token { 1 }; // BOS
    int current_pos { 0 };   // current position in the sequence

    RunState( const Config& config );
    RunState( const RunState& ) = delete;
    RunState operator=( const RunState& ) = delete;
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

private:
  MMap_Region model_mmap_region_;
  const float* model_ptr_;

  const Config config_;
  const BaseWeights base_weights_;
  const std::vector<LayerWeights> layer_weights_;
  const Vocabulary vocabulary_;

  RunState state_;
  float temperature_ { 0.0f };

  void transformer( const int token );

public:
  Llama2( const std::filesystem::path& tokenizer_path, const std::filesystem::path& model_path );

  std::string next_token();

  Llama2( const Llama2& ) = delete;
  Llama2& operator=( const Llama2& ) = delete;
  Llama2( Llama2&& ) = default;
  Llama2& operator=( Llama2&& ) = default;
};

}
