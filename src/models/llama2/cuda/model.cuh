#pragma once

#include "models/llama2/base.hh"

namespace glinthawk::models::llama2::cuda {

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

private:
  void pass_begin( const std::vector<uint32_t>& token );
  void transformer_layer( const int32_t layer_num );
  void pass_end();

  float temperature_ { 0.0f };
  std::vector<DType*> pointer_scratchpad { };

  std::vector<InferenceState<DType>> forward( const std::vector<std::reference_wrapper<const InferenceState<DType>>>& inference_state_s, const std::vector<uint32_t>& prompt_id_s );

protected:
  using BaseLlama2<DType, Context<DType>>::BaseLlama2;

public:
  static std::unique_ptr<Llama2<DType>> load( const std::filesystem::path& model_dir,
                      const int32_t start_layer = 0,
                      const int32_t end_layer = -1,
                      const uint64_t kv_prompt_limit = 1,
                      const uint64_t concurrency_limit = 1 );


  Llama2( const Llama2& ) = delete;
  Llama2& operator=( const Llama2& ) = delete;
  Llama2( Llama2&& ) = default;
  Llama2& operator=( Llama2&& ) = default;

  ~Llama2();

<<<<<<< HEAD
  InferenceState forward( const InferenceState& inference_state, const uint32_t& prompt_id ) override;
||||||| parent of d22c38f (model.hh: Pass an R-value reference to the state in forward().)
  InferenceState forward( const InferenceState& inference_state, ContextType& context ) override;
=======
  InferenceState forward( InferenceState&& inference_state, ContextType& context ) override;
>>>>>>> d22c38f (model.hh: Pass an R-value reference to the state in forward().)

  std::vector<InferenceState> forward( const std::vector<InferenceState>& inference_state_s, const std::vector<uint32_t>& prompt_id_s ) override;

  uint32_t forward( const uint32_t& token, const uint32_t& prompt_id, const uint32_t& token_pos );

  std::vector<uint32_t> forward( const std::vector<uint32_t>& token_s, const std::vector<uint32_t>& prompt_id_s, const std::vector<uint32_t>& token_pos_s );
};

} // namespace glinthawk::models::llama2::cuda
