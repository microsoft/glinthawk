#pragma once

#include "models/llama2/base.hh"

#include <limits>

namespace glinthawk::models::llama2::cpu {

template<typename DType>
struct Context : public glinthawk::models::llama2::InferenceContext<DType>
{
private:
  std::unique_ptr<DType, void ( * )( DType* )> storage_;

public:
  Context( const Config& config );
};

template<typename DType>
class Llama2 : public glinthawk::models::llama2::BaseLlama2<DType, Context<DType>>
{
public:
  using ContextType = glinthawk::models::llama2::BaseLlama2<DType, Context<DType>>::ContextType;
  using ConfigType = glinthawk::models::llama2::BaseLlama2<DType, Context<DType>>::ConfigType;
  using DataType = glinthawk::models::llama2::BaseLlama2<DType, Context<DType>>::DataType;
  using TokenizerType = glinthawk::models::llama2::BaseLlama2<DType, Context<DType>>::TokenizerType;

private:
  void pass_begin( const std::vector<uint32_t>& token );
  void transformer_layer( const int32_t layer_num );
  void pass_end();

  std::vector<InferenceState> forward(
    const std::vector<std::reference_wrapper<const InferenceState>>& inference_states,
    const std::vector<std::shared_ptr<ContextType>>& contexts );

public:
  using BaseLlama2<DType, Context<DType>>::BaseLlama2;

  Llama2( const std::filesystem::path& model_dir,
          const uint32_t start_layer = 0,
          const uint32_t end_layer = std::numeric_limits<uint32_t>::max(),
          const uint64_t concurrency_limit = 1 );

  Llama2( const Llama2& ) = delete;
  Llama2& operator=( const Llama2& ) = delete;
  Llama2( Llama2&& ) = default;
  Llama2& operator=( Llama2&& ) = default;

  virtual ~Llama2() {}

  InferenceState forward( const InferenceState& inference_state, std::shared_ptr<ContextType>& context ) override;

  std::vector<InferenceState> forward( const std::vector<InferenceState>& inference_states,
                                       const std::vector<std::shared_ptr<ContextType>>& contexts ) override;
};

} // namespace glinthawk::models::llama2::cpu
