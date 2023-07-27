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
    // token embedding table
    float* token_embedding_table; // (vocab_size, dim)

    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)

    // weights for matmuls
    float* wq; // (layer, dim, dim)
    float* wk; // (layer, dim, dim)
    float* wv; // (layer, dim, dim)
    float* wo; // (layer, dim, dim)

    // weights for ffn
    float* w1; // (layer, hidden_dim, dim)
    float* w2; // (layer, dim, hidden_dim)
    float* w3; // (layer, hidden_dim, dim)

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
    // current wave of activations
    std::unique_ptr<float[]> x;      // activation at current time stamp (dim,)
    std::unique_ptr<float[]> xb;     // same, but inside a residual branch (dim,)
    std::unique_ptr<float[]> xb2;    // an additional buffer just for convenience (dim,)
    std::unique_ptr<float[]> hb;     // buffer for hidden dimension in the ffn (hidden_dim,)
    std::unique_ptr<float[]> hb2;    // buffer for hidden dimension in the ffn (hidden_dim,)
    std::unique_ptr<float[]> q;      // query (dim,)
    std::unique_ptr<float[]> k;      // key (dim,)
    std::unique_ptr<float[]> v;      // value (dim,)
    std::unique_ptr<float[]> att;    // buffer for scores/attention values (n_heads, seq_len)
    std::unique_ptr<float[]> logits; // output logits

    // kv cache
    std::unique_ptr<float[]> key_cache;   // (layer, seq_len, dim)
    std::unique_ptr<float[]> value_cache; // (layer, seq_len, dim)
  };

private:
  std::unique_ptr<float[]> weights_buffer_ {};
  std::vector<std::string> vocabulary_ {};

  Config config_ {};
  TransformerWeights weights_ {};
  RunState state_ {};

  void init_weights( const std::filesystem::path& weights_path );
  void init_vocabulary( const std::filesystem::path& vocabulary_path );
  void init_state();

  void transformer( const int token, const int pos );

public:
  Llama2( const std::filesystem::path& tokenizer_path, const std::filesystem::path& weights_path );
};

}
