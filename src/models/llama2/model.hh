#pragma once

#include "base.hh"
#include "ops/concept.hh"
#include "variants.hh"

#if defined( TARGET_PLATFORM_CPU )
#include "arch/cpu/llama2/ops.hh"
#elif defined( TARGET_PLATFORM_CUDA )
#include "arch/cuda/llama2/ops.cuh"
#endif

namespace glinthawk::models::llama2 {

template<typename Config, typename DType, typename LlamaOperations, typename Context>
requires ModelConfig<Config> && LlamaOperationsConcept<LlamaOperations, DType, Settings<Config>>
class Llama2 : public glinthawk::models::Model<Context>
{
public:
  using StateVector = std::vector<InferenceState>;
  using ContextPtr = std::shared_ptr<Context>;
  using ContextVector = std::vector<ContextPtr>;
  using Operations = LlamaOperations;

  using ContextType = Context;
  using ConfigType = Config;
  using SettingsType = Settings<Config>;

protected:
  Settings<Config> settings_;
  Operations ops_ { settings_ };

  typename Operations::DeviceUniquePtr base_weights_buffer_;
  typename Operations::DeviceUniquePtr layers_buffer_;
  typename Operations::DeviceUniquePtr run_state_buffer_;

  BaseWeights<Config, DType> base_weights_;
  std::array<LayerWeights<Config, DType>, Config::n_layers> layer_weights_;
  RunState<Config, DType> state_;

  // Checking if the inference states are safe to pass to the model
  void check_batch( const StateVector& inference_states,
                    const ContextVector& contexts,
                    const InferenceState::Stage stage ) const;

  void pass_begin( const std::vector<uint32_t>& token );
  void pass_end();

  void forward_prelude( StateVector& inference_state, const ContextVector& contexts );
  void pre_attention_ops( const int32_t layer_num );
  void attention_ops();
  void post_attention_ops( const int32_t layer_num );
  [[nodiscard]] StateVector forward_postlude( StateVector&& inference_state );

public:
  Llama2( const std::filesystem::path& model_dir,
          const uint32_t start_layer = 0,
          const uint32_t end_layer = std::numeric_limits<uint32_t>::max(),
          const uint64_t concurrency_limit = 1,
          const bool randomize_parameters = false );

  [[nodiscard]] InferenceState forward( InferenceState&& inference_state, ContextPtr context ) override;
  [[nodiscard]] InferenceState pre_attention_forward( InferenceState&& inference_state, ContextPtr context ) override;
  [[nodiscard]] InferenceState attention_forward( InferenceState&& inference_state, ContextPtr context ) override;
  [[nodiscard]] InferenceState post_attention_forward( InferenceState&& inference_state ) override;

  [[nodiscard]] StateVector forward( StateVector&& inference_states, const ContextVector& contexts ) override;
  [[nodiscard]] StateVector pre_attention_forward( StateVector&& inference_states,
                                                   const ContextVector& contexts ) override;
  [[nodiscard]] StateVector attention_forward( StateVector&& inference_states, const ContextVector& contexts ) override;
  [[nodiscard]] StateVector post_attention_forward( StateVector&& inference_states ) override;

  void dummy_forward( InferenceState& inference_state ) override;
  bool is_finished( const InferenceState& inference_state ) override;

  Settings<Config> settings() const { return settings_; }
};

#define DECLARE_MODEL( PLATFORM, MODEL_NAME )                                                                          \
  template<typename DType>                                                                                             \
  using MODEL_NAME = Llama2<configs::MODEL_NAME,                                                                       \
                            DType,                                                                                     \
                            PLATFORM::LlamaOperations<configs::MODEL_NAME, DType>,                                     \
                            PLATFORM::Context<configs::MODEL_NAME, DType>>

#if defined( TARGET_PLATFORM_CPU )
namespace cpu {
DECLARE_MODEL( cpu, Llama2_7B_Chat );
DECLARE_MODEL( cpu, Llama2_13B_Chat );
DECLARE_MODEL( cpu, Llama2_70B_Chat );
DECLARE_MODEL( cpu, Stories_110M );
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
