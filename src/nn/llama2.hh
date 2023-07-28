#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "util/ring_buffer.hh"

namespace glinthawk {

class Llama2
{
private:
  struct Config
  {
    int dim {};        // transformer dimension
    int hidden_dim {}; // for ffn layers
    int n_layers {};   // number of layers
    int n_heads {};    // number of query heads
    int n_kv_heads {}; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size {}; // vocabulary size (byte-level)
    int seq_len {};    // max sequence length

    Config( const std::filesystem::path& weights_path );
    std::string to_string() const;
  };

  struct TransformerWeights
  {
    struct LayerWeights
    {
      // weights for rmsnorms
      float* rms_att_weight; // (dim) rmsnorm weights
      float* rms_ffn_weight; // (dim)

      // weights for matmuls
      float* wq; // (dim, dim)
      float* wk; // (dim, dim)
      float* wv; // (dim, dim)
      float* wo; // (dim, dim)

      // weights for ffn
      float* w1; // (hidden_dim, dim)
      float* w2; // (dim, hidden_dim)
      float* w3; // (hidden_dim, dim)
    };

    std::unique_ptr<float[]> buffer_ {};

    // token embedding table
    float* token_embedding_table {}; // (vocab_size, dim)

    // transformer layers
    std::unique_ptr<LayerWeights[]> layers {}; // (n_layers,)

    // final rmsnorm
    float* rms_final_weight {}; // (dim,)

    // freq_cis for RoPE relatively positional embeddings
    float* freq_cis_real {}; // (seq_len, dim/2)
    float* freq_cis_imag {}; // (seq_len, dim/2)

    // classifier weights for the logits, on the last layer
    float* wcls {};

    TransformerWeights( const Config& config, const std::filesystem::path& weights_path );
    TransformerWeights( const TransformerWeights& ) = delete;
    TransformerWeights operator=( const TransformerWeights& ) = delete;
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
  Config config_;
  TransformerWeights weights_;
  Vocabulary vocabulary_;
  RunState state_;

  float temperature_ { 0.0f };
  int current_token_ { 1 }; // BOS
  int current_pos_ { 0 };   // current position in the sequence

  void transformer( const int token );

public:
  Llama2( const std::filesystem::path& tokenizer_path, const std::filesystem::path& weights_path );

  std::string next_token();

  Llama2( const Llama2& ) = delete;
  Llama2& operator=( const Llama2& ) = delete;
  Llama2( Llama2&& ) = default;
  Llama2& operator=( Llama2&& ) = default;
};

}
