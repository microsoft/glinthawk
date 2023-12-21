#pragma once

#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "models/common/model.hh"
#include "variants.hh"

#ifdef GLINTHAWK_CUDA_ENABLED
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#endif

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
            const uint64_t concurrency_limit_,
            const uint64_t max_context_,
            const bool randomize_parameters_ );

  std::string to_string() const;

  /// @brief Size of the config stored on disk (in bytes)
  static size_t config_size() { return sizeof( int32_t ) * 7; }
  uint64_t n_layers_loaded() const { return end_layer_num - start_layer_num + 1; }

  uint64_t start_layer_num {};
  uint64_t end_layer_num {};
  uint64_t concurrency_limit { 1 }; // max concurrent inference size
  uint64_t max_context { 1 };       // max context size, only valid with a pre-allocated context manager
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
  const DType* wq { nullptr };   // (dim, dim)
  const DType* wkv { nullptr };  // (dim, 2*kv_dim)
  const DType* wo { nullptr };   // (dim, dim)

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
  DType* kv {};           // key-value (B, kv_dim, 2)
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
template<typename Config, typename DType>
requires ModelConfig<Config>
struct InferenceContext
{
  static size_t context_size( const Settings<Config>& settings );

  DType* buffer_ { nullptr };
  DType* key( const Settings<Config>& settings, int layer_num, const int token_pos, const int head = 0 );
  DType* value( const Settings<Config>& settings, int layer_num, const int token_pos, const int head = 0 );

  bool empty();
};

template<typename Config, typename DType, typename Context, typename StorageDeleter = std::default_delete<DType>>
requires ModelConfig<Config>
class BaseLlama2 : public glinthawk::models::Model<Context>
{
public:
  using InferenceStateVector = std::vector<InferenceState>;
  using ContextVector = std::vector<std::shared_ptr<Context>>;

protected:
  Settings<Config> settings_ {};

  std::unique_ptr<DType, StorageDeleter> base_weights_buffer_ { nullptr };
  std::unique_ptr<DType, StorageDeleter> layers_buffer_ { nullptr };
  std::unique_ptr<DType, StorageDeleter> run_state_buffer_ { nullptr };

  RunState<Config, DType> state_ {};
  BaseWeights<Config, DType> base_weights_ {};
  std::vector<LayerWeights<Config, DType>> layer_weights_ {};

  void init( const Settings<Config>& settings,
             std::unique_ptr<DType, StorageDeleter>&& base_weights,
             std::unique_ptr<DType, StorageDeleter>&& layers_weights,
             std::unique_ptr<DType, StorageDeleter>&& run_state );

  BaseLlama2() = default;

  void assert_safe_forward( const InferenceStateVector& inference_states, const ContextVector& contexts ) const;
  void assert_safe_pre_attention( const InferenceStateVector& inference_states, const ContextVector& contexts ) const;
  void assert_safe_attention( const InferenceStateVector& inference_states, const ContextVector& contexts ) const;
  void assert_safe_post_attention( const InferenceStateVector& inference_states ) const;

public:
  ~BaseLlama2() override = default;

  BaseLlama2( BaseLlama2&& ) = default;
  BaseLlama2& operator=( BaseLlama2&& ) = default;
  BaseLlama2( const BaseLlama2& ) = delete;
  BaseLlama2& operator=( const BaseLlama2& ) = delete;

  using ConfigType = Config;
  using SettingsType = Settings<Config>;
  using ContextType = Context;
  using TokenizerType = Vocabulary;

  void dummy_forward( InferenceState& inference_state ) override;
  bool is_finished( const InferenceState& inference_state ) override;

  Settings<Config> settings() const { return settings_; }
};

} // namespace glinthawk::models::llama2
