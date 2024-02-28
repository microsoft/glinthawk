#pragma once

#include "base.hh"
#include "models/common/state.hh"
#include "ops/concept.hh"
#include "variants.hh"

#if defined( TARGET_PLATFORM_AMD64 ) || defined( TARGET_PLATFORM_CUDA )
#include "arch/amd64/llama2/ops.hh"
#endif

#if defined( TARGET_PLATFORM_CUDA )
#include "arch/cuda/llama2/ops.cuh"
#endif

namespace glinthawk::models::llama2 {

template<typename Config, typename DType, typename LlamaOperations, typename Context>
requires ModelConfig<Config> && LlamaOperationsConcept<LlamaOperations, DType, Settings<Config>>
         && ContextConcept<Context, DType>
class Llama2
{
public:
  using BatchedState = BatchedInferenceState<Config>;
  using ContextPtr = std::shared_ptr<Context>;
  using ContextVector = std::vector<ContextPtr>;
  using Operations = LlamaOperations;

  using ModelDataType = DType;
  using ContextType = Context;
  using ConfigType = Config;
  using SettingsType = Settings<Config>;

public:
  Llama2( const std::filesystem::path& model_dir,
          const uint32_t start_layer = 0,
          const uint32_t end_layer = std::numeric_limits<uint32_t>::max(),
          const uint64_t concurrency_limit = 1,
          const uint64_t max_context_count = 1,
          const bool randomize_parameters = false );

  [[nodiscard]] BatchedState forward( BatchedState&& state, const ContextVector& ctxs );

  // input token -> [(pre -> att -> post) x n_layers] -> classify -> output token
  [[nodiscard]] BatchedState pre_attention_forward( BatchedState&& state );
  [[nodiscard]] BatchedState attention_forward( BatchedState&& state, const ContextVector& ctxs );
  [[nodiscard]] BatchedState post_attention_forward( BatchedState&& state );
  [[nodiscard]] BatchedState classify_forward( BatchedState&& state );

  Settings<Config> settings() const { return settings_; }
  Operations& ops() { return ops_; }

private:
  static constexpr uint32_t TOKEN_BOS = 1; // Beginning-of-sequence token
  static constexpr uint32_t TOKEN_EOS = 2; // End-of-sequence token

protected:
  const Settings<Config> settings_;
  Operations ops_ { settings_ };

  typename Operations::DeviceUniquePtr base_weights_buffer_;
  typename Operations::DeviceUniquePtr layers_buffer_;
  typename Operations::DeviceUniquePtr run_state_buffer_;

  BaseWeights<Config, DType> base_weights_;
  std::array<LayerWeights<Config, DType>, Config::n_layers> layer_weights_;
  RunState<Config, DType, Context> state_;

  // Checking if the inference states are safe to pass to the model
  void check_batch( const BatchedState& inference_states,
                    const ContextVector& contexts,
                    const InferenceStage stage ) const;

  void load_embedding( const BatchedState& inference_state );

  void forward_prelude( BatchedState& inference_state, const ContextVector& contexts );
  [[nodiscard]] BatchedState forward_postlude( BatchedState&& inference_state,
                                               const int32_t most_recent_layer_num,
                                               const bool classified );

  void pre_attention_ops( const int32_t layer_num, const bool update_kv_cache = false );
  void attention_ops();
  void post_attention_ops( const int32_t layer_num );
  void classify_ops();
};

#define DECLARE_MODEL( PLATFORM, MODEL_NAME )                                                                          \
  template<typename DType>                                                                                             \
  using MODEL_NAME                                                                                                     \
    = Llama2<configs::MODEL_NAME,                                                                                      \
             DType,                                                                                                    \
             PLATFORM::LlamaOperations<configs::MODEL_NAME, DType, PLATFORM::Context<configs::MODEL_NAME, DType>>,     \
             PLATFORM::Context<configs::MODEL_NAME, DType>>

#if defined( TARGET_PLATFORM_AMD64 ) || defined( TARGET_PLATFORM_CUDA )
namespace amd64 {
DECLARE_MODEL( amd64, Llama2_7B_Chat );
DECLARE_MODEL( amd64, Llama2_13B_Chat );
DECLARE_MODEL( amd64, Llama2_70B_Chat );
DECLARE_MODEL( amd64, Stories_110M );
}
#endif

#if defined( TARGET_PLATFORM_CUDA )
namespace cuda {
DECLARE_MODEL( cuda, Llama2_7B_Chat );
DECLARE_MODEL( cuda, Llama2_13B_Chat );
DECLARE_MODEL( cuda, Llama2_70B_Chat );
DECLARE_MODEL( cuda, Stories_110M );
}
#endif

} // namespace glinthawk::models::llama2
