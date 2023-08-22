#pragma once

#include "models/llama2/base.hh"

namespace glinthawk::models::llama2::cuda {

template<typename DType>
class Llama2 : public glinthawk::models::llama2::BaseLlama2<DType>
{
private:
  void pass_begin( const std::vector<uint32_t>& token );
  void transformer_layer( const int32_t layer_num, const uint64_t token_pos );
  void pass_end();

  uint64_t token_pos_ { 0 };
  float temperature_ { 0.0f };
  uint64_t curr_batch_size { 1 };

  std::vector<InferenceState<DType>> forward( const std::vector<std::reference_wrapper<const InferenceState<DType>>>& inference_state_s );

protected:
  using BaseLlama2<DType>::BaseLlama2;

public:
  static Llama2 load( const std::filesystem::path& model_dir,
                      const int32_t start_layer = 0,
                      const int32_t end_layer = -1,
                      const uint64_t batch_size = 1 );

  ~Llama2();

  InferenceState<DType> forward( const InferenceState<DType>& inference_state ) override;

  std::vector<InferenceState<DType>> forward( const std::vector<InferenceState<DType>>& inference_state_s ) override;

  uint32_t forward( const uint32_t& token );

  std::vector<uint32_t> forward( const std::vector<uint32_t>& token_s );
};

} // namespace glinthawk::models::llama2::cuda
