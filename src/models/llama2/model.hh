#pragma once

#include "base.hh"
#include "models/common/state.hh"
#include "ops/concept.hh"
#include "variants.hh"

#if defined( TARGET_PLATFORM_AMD64 )
#include "arch/amd64/llama2/ops.hh"
#elif defined( TARGET_PLATFORM_CUDA )
#include "arch/cuda/llama2/ops.cuh"
#endif

namespace glinthawk::models::llama2 {

template<typename Config, typename DType, typename LlamaOperations, typename Context>
requires ModelConfig<Config> && LlamaOperationsConcept<LlamaOperations, DType, Settings<Config>>
         && ContextConcept<Context, DType>
class Llama2
{
public:
  using StateVector = std::vector<InferenceState>;
  using ContextPtr = std::shared_ptr<Context>;
  using ContextVector = std::vector<ContextPtr>;
  using Operations = LlamaOperations;

  using ModelDataType = DType;
  using ContextType = Context;
  using ConfigType = Config;
  using SettingsType = Settings<Config>;

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
  void check_batch( const StateVector& inference_states,
                    const ContextVector& contexts,
                    const InferenceState::Stage stage ) const;

  void check_batch( const BatchedInferenceState& inference_states,
                    const ContextVector& contexts,
                    const InferenceState::Stage stage ) const;

  void load_embedding( const StateVector& inference_state );
  void load_embedding( const BatchedInferenceState& inference_state );

  void forward_prelude( StateVector& inference_state, const ContextVector& contexts );
  void forward_prelude( BatchedInferenceState& inference_state, const ContextVector& contexts );

  void pre_attention_ops( const int32_t layer_num );
  void attention_ops();
  void post_attention_ops( const int32_t layer_num );
  void classify_ops();

  [[nodiscard]] StateVector forward_postlude( StateVector&& inference_state,
                                              const int32_t most_recent_layer_num,
                                              const bool classified );

  [[nodiscard]] BatchedInferenceState forward_postlude( BatchedInferenceState&& inference_state,
                                                        const int32_t most_recent_layer_num,
                                                        const bool classified );

public:
  Llama2( const std::filesystem::path& model_dir,
          const uint32_t start_layer = 0,
          const uint32_t end_layer = std::numeric_limits<uint32_t>::max(),
          const uint64_t pre_att_concurrency_limit = 1,
          const uint64_t att_concurrency_limit = 1,
          const uint64_t post_att_concurrency_limit = 1,
          const uint64_t cls_concurrency_limit = 1,
          const uint64_t max_context_count = 1,
          const bool randomize_parameters = false );

  [[nodiscard]] InferenceState forward( InferenceState&& state, ContextPtr ctx );
  [[nodiscard]] InferenceState pre_attention_forward( InferenceState&& state, ContextPtr ctx );
  [[nodiscard]] InferenceState attention_forward( InferenceState&& state, ContextPtr ctx );
  [[nodiscard]] InferenceState post_attention_forward( InferenceState&& state );
  [[nodiscard]] InferenceState classify_forward( InferenceState&& state );

  [[nodiscard]] StateVector forward( StateVector&& states, const ContextVector& ctxs );
  [[nodiscard]] StateVector pre_attention_forward( StateVector&& states, const ContextVector& ctxs );
  [[nodiscard]] StateVector attention_forward( StateVector&& states, const ContextVector& ctxs );
  [[nodiscard]] StateVector post_attention_forward( StateVector&& states );
  [[nodiscard]] StateVector classify_forward( StateVector&& states );

  [[nodiscard]] BatchedInferenceState forward( BatchedInferenceState&& state, const ContextVector& ctxs );
  [[nodiscard]] BatchedInferenceState pre_attention_forward( BatchedInferenceState&& state, const ContextVector& ctxs );
  [[nodiscard]] BatchedInferenceState attention_forward( BatchedInferenceState&& state, const ContextVector& ctxs );
  [[nodiscard]] BatchedInferenceState post_attention_forward( BatchedInferenceState&& state );
  [[nodiscard]] BatchedInferenceState classify_forward( BatchedInferenceState&& state );

  void dummy_forward( InferenceState& inference_state );
  bool is_finished( const InferenceState& inference_state );

  Settings<Config> settings() const { return settings_; }
};

#define DECLARE_MODEL( PLATFORM, MODEL_NAME )                                                                          \
  template<typename DType>                                                                                             \
  using MODEL_NAME                                                                                                     \
    = Llama2<configs::MODEL_NAME,                                                                                      \
             DType,                                                                                                    \
             PLATFORM::LlamaOperations<configs::MODEL_NAME, DType, PLATFORM::Context<configs::MODEL_NAME, DType>>,     \
             PLATFORM::Context<configs::MODEL_NAME, DType>>

#if defined( TARGET_PLATFORM_AMD64 )
namespace amd64 {
DECLARE_MODEL( amd64, Llama2_7B_Chat );
DECLARE_MODEL( amd64, Llama2_13B_Chat );
DECLARE_MODEL( amd64, Llama2_70B_Chat );
DECLARE_MODEL( amd64, Stories_110M );
}
#elif defined( TARGET_PLATFORM_CUDA )
namespace cuda {
DECLARE_MODEL( cuda, Llama2_7B_Chat );
DECLARE_MODEL( cuda, Llama2_13B_Chat );
DECLARE_MODEL( cuda, Llama2_70B_Chat );
DECLARE_MODEL( cuda, Stories_110M );
}
#endif

} // namespace glinthawk::models::llama2
