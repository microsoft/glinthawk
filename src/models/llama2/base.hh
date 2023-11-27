#pragma once

#include "models/common/model.hh"
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef GLINTHAWK_CUDA_ENABLED
#include "cuda_runtime.h"
#include <curand.h>
#include <curand_kernel.h>
#endif

namespace glinthawk::models::llama2 {

constexpr size_t MAX_BATCH_SIZE = 1024;

struct Config
{
  Config() {}

  Config( const std::filesystem::path& config_file,
          const uint32_t start_layer,
          const uint32_t end_layer,
          uint64_t concurrency_limit_ );

  std::string to_string() const;

  /// @brief Size of the config stored on disk (in bytes)
  static size_t config_size() { return sizeof( int32_t ) * 7; }

  uint64_t dim {};                  // transformer dimension
  uint64_t kv_dim {};               // key/value dimension
  uint64_t hidden_dim {};           // for ffn layers
  uint64_t n_layers {};             // total number of layers
  uint64_t head_size {};            // dimension of each head
  uint64_t n_heads {};              // number of query heads
  uint64_t n_kv_heads {};           // number of key/value heads (can be < query heads because of multiquery)
  uint64_t gqa_size {};             // GQA sharing rate
  uint64_t vocab_size {};           // vocabulary size (byte-level)
  uint64_t seq_len {};              // max sequence length
  uint64_t concurrency_limit { 1 }; // max concurrent inference size // XXX(sadjad): I don't like this!

  // which layers to serve
  uint64_t start_layer_num {};
  uint64_t end_layer_num {};

  bool wcls_present { false };

  uint64_t n_layers_loaded() const { return end_layer_num - start_layer_num + 1; }
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
  BaseWeights() = default;
  BaseWeights( const Config& config, const DType* base_weights );

  BaseWeights( const BaseWeights& ) = delete;
  BaseWeights operator=( const BaseWeights& ) = delete;
  BaseWeights( BaseWeights&& ) = default;
  BaseWeights& operator=( BaseWeights&& ) = default;

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

  LayerWeights( const LayerWeights& ) = delete;
  LayerWeights operator=( const LayerWeights& ) = delete;
  LayerWeights( LayerWeights&& ) = default;
  LayerWeights& operator=( LayerWeights&& ) = default;

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

/// @brief This class acts as the scratchpad for the computations
template<typename DType>
struct RunState
{
  RunState() = default;
  RunState( const Config& config, DType* buffer );

  RunState( const RunState& ) = delete;
  RunState operator=( const RunState& ) = delete;
  RunState( RunState&& ) = default;
  RunState& operator=( RunState&& ) = default;

  static size_t state_size( const Config& config );

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

#ifdef GLINTHAWK_CUDA_ENABLED
  curandState* rng_state {}; // CURAND state (B, vocab_size)
#endif

  // This memory is on CPU
  uint32_t argmax_pos[MAX_BATCH_SIZE] {}; // argmax results (B, )

  // information about the current batch
  uint64_t curr_concurrency_size { 1 };
  uint32_t batch_token_positions[MAX_BATCH_SIZE] {};
  DType* batch_context_pointers[MAX_BATCH_SIZE] {};
};

/// @brief InferenceContext for Llama2 model is the KV-cache
template<typename DType>
struct InferenceContext
{
  // TODO: add context state here
  static size_t context_size( const Config& config );

  DType* buffer_ { nullptr };
  DType* key( const Config& config, int layer_num, const int token_pos, const int head = 0 );
  DType* value( const Config& config, int layer_num, const int token_pos, const int head = 0 );

  bool empty();
};

template<typename DType, typename Context>
class BaseLlama2 : public glinthawk::models::Model<Context>
{
protected:
  Config config_ {};

  std::unique_ptr<DType, void ( * )( DType* )> base_weights_buffer_ { nullptr, nullptr };
  std::unique_ptr<DType, void ( * )( DType* )> layers_buffer_ { nullptr, nullptr };
  std::unique_ptr<DType, void ( * )( DType* )> run_state_buffer_ { nullptr, nullptr };

  RunState<DType> state_ {};
  BaseWeights<DType> base_weights_ {};
  std::vector<LayerWeights<DType>> layer_weights_ {};

  void init( const Config& config,
             std::unique_ptr<DType, void ( * )( DType* )>&& base_weights,
             std::unique_ptr<DType, void ( * )( DType* )>&& layers_weights,
             std::unique_ptr<DType, void ( * )( DType* )>&& run_state );

  BaseLlama2() = default;

  void assert_safe_forward( const std::vector<InferenceState>& inference_states,
                            const std::vector<std::shared_ptr<Context>>& contexts );

  void assert_safe_pre_attention( const std::vector<InferenceState>& inference_states,
                                  const std::vector<std::shared_ptr<Context>>& contexts );

  void assert_safe_attention( const std::vector<InferenceState>& inference_states,
                              const std::vector<std::shared_ptr<Context>>& contexts );

  void assert_safe_post_attention( const std::vector<InferenceState>& inference_states );

public:
  virtual ~BaseLlama2() = default;

  BaseLlama2( BaseLlama2&& ) = default;
  BaseLlama2& operator=( BaseLlama2&& ) = default;
  BaseLlama2( const BaseLlama2& ) = delete;
  BaseLlama2& operator=( const BaseLlama2& ) = delete;

  using ConfigType = Config;
  using ContextType = Context;
  using DataType = DType;
  using TokenizerType = Vocabulary;

  void dummy_forward( InferenceState& inference_state ) override;

  bool is_finished( const InferenceState& inference_state ) override;

  Config config() const { return config_; }
};

} // namespace glinthawk::models::llama2
