#pragma once

#include "models/llama2/base.hh"

namespace glinthawk::models::llama2::cuda {

template<typename DType>
class Llama2 : public glinthawk::models::llama2::BaseLlama2<DType>
{
private:
  void pass_begin( const std::vector<uint32_t>& token );
  void transformer_layer( const int32_t layer_num );
  void pass_end();

  std::vector<InferenceState<DType>> forward( const std::vector<std::reference_wrapper<const InferenceState<DType>>>& inference_state_s, const std::vector<uint32_t>& prompt_id_s );

protected:
  using BaseLlama2<DType>::BaseLlama2;

public:
  static std::unique_ptr<Llama2<DType>> load( const std::filesystem::path& model_dir,
                      const int32_t start_layer = 0,
                      const int32_t end_layer = -1,
                      const uint64_t kv_prompt_limit = 1,
                      const uint64_t concurrency_limit = 1 );

  ~Llama2();

  InferenceState<DType> forward( const InferenceState<DType>& inference_state, const uint32_t& prompt_id ) override;

  std::vector<InferenceState<DType>> forward( const std::vector<InferenceState<DType>>& inference_state_s, const std::vector<uint32_t>& prompt_id_s ) override;

  uint32_t forward( const uint32_t& token, const uint32_t& prompt_id, const uint32_t& token_pos, const float& temperature );

  std::vector<uint32_t> forward( const std::vector<uint32_t>& token_s, const std::vector<uint32_t>& prompt_id_s, const std::vector<uint32_t>& token_pos_s, const std::vector<float>& temperature_s );
};

} // namespace glinthawk::models::llama2::cuda
