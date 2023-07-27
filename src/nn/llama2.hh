#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace glinthawk {

class Llama2
{
private:
  struct Config
  {
    int dim;        // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers;   // number of layers
    int n_heads;    // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len;    // max sequence length

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
    float* token_embedding_table; // (vocab_size, dim)

    // transformer layers
    std::unique_ptr<LayerWeights[]> layers {}; // (n_layers,)

    // final rmsnorm
    float* rms_final_weight; // (dim,)

    // freq_cis for RoPE relatively positional embeddings
    float* freq_cis_real; // (seq_len, dim/2)
    float* freq_cis_imag; // (seq_len, dim/2)

    // (optional) classifier weights for the logits, on the last layer
    float* wcls;
  };

  struct RunState
  {
    std::unique_ptr<float[]> buffer_ {};

    // current wave of activations
    float* x;      // activation at current time stamp (dim,)
    float* xb;     // same, but inside a residual branch (dim,)
    float* xb2;    // an additional buffer just for convenience (dim,)
    float* hb;     // buffer for hidden dimension in the ffn (hidden_dim,)
    float* hb2;    // buffer for hidden dimension in the ffn (hidden_dim,)
    float* q;      // query (dim,)
    float* k;      // key (dim,)
    float* v;      // value (dim,)
    float* att;    // buffer for scores/attention values (n_heads, seq_len)
    float* logits; // output logits

    struct LayerKVCache
    {
      float* k; // (seq_len, dim)
      float* v; // (seq_len, dim)
    };

    std::unique_ptr<LayerKVCache[]> kv_caches; // (n_layers,)
  };

private:
  std::vector<std::string> vocabulary_ {};

  Config config_ {};
  TransformerWeights weights_ {};
  RunState state_ {};

  float temperature_ { 0.0f };
  int max_steps_ { 128 };
  int current_token_ { 1 }; // BOS
  int current_pos_ { 0 };   // current position in the sequence

  void init_weights( const std::filesystem::path& weights_path );
  void init_vocabulary( const std::filesystem::path& vocabulary_path );
  void init_state();

  void transformer( const int token, const int pos );

public:
  Llama2( const std::filesystem::path& tokenizer_path, const std::filesystem::path& weights_path );

  Llama2( const Llama2& ) = delete;
  Llama2& operator=( const Llama2& ) = delete;

  Llama2( Llama2&& ) = default;
  Llama2& operator=( Llama2&& ) = default;

  std::string next_token();
};

}
