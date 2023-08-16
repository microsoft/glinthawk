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
  Config( const std::filesystem::path& weights_path );
  Config( const void* model_buffer );

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

template<typename DType>
struct BaseWeights
{
  BaseWeights( const Config& config, const DType* model );
  BaseWeights( const BaseWeights& ) = delete;
  BaseWeights operator=( const BaseWeights& ) = delete;

  // token embedding table
  const DType* token_embedding_table {}; // (vocab_size, dim)

  // final rmsnorm
  const DType* rms_final_weight {}; // (dim,)

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
  LayerWeights( const Config& config, const DType* model, const int32_t layer_num );

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
  RunState( const Config& config, DType* buffer, const int32_t start_layer, const int32_t end_layer );
  RunState( const RunState& ) = delete;
  RunState operator=( const RunState& ) = delete;

  static size_t state_size( const Config& config );

  DType* buffer_;         // we use this buffer for everything except for activations
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

  // k-v cache
  struct KVCache
  {
    const int32_t start_layer_;
    const int32_t end_layer_;

    DType* buffer_;
    const int seq_len_;
    const int dim_;
    const int n_layers_;
    const int head_size_;

    inline DType* key( const int layer, const int step, const int head = 0 );
    inline DType* value( const int layer, const int step, const int head = 0 );

    void pop();

    KVCache( const Config& config, DType* buffer, const int32_t start_layer, const int32_t end_layer );
  };

  KVCache kv_cache;
};

template<typename DType>
class Llama2 : public glinthawk::models::Model<DType>
{
protected:
  std::unique_ptr<DType, void ( * )( DType* )> model_buffer_;
  std::unique_ptr<DType, void ( * )( DType* )> run_state_buffer_;

  const Config config_;
  const Vocabulary vocabulary_;
  const BaseWeights<DType> base_weights_;
  const std::vector<LayerWeights<DType>> layer_weights_;

public:
  Llama2( std::unique_ptr<DType, void ( * )( DType* )> model,
          std::unique_ptr<DType, void ( * )( DType* )> run_state,
          const Config& config,
          const Vocabulary& vocabulary,
          const BaseWeights<DType>& base_weights,
          const std::vector<LayerWeights<DType>>& layer_weights );

  InferenceState<DType> forward( const InferenceState<DType>& inference_state ) override;
};

} // namespace glinthawk::models::llama2
