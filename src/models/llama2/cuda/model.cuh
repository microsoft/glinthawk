#pragma once

#include "models/llama2/base.hh"

namespace glinthawk::models::llama2::cuda {

template<typename DType>
class Llama2 : public glinthawk::models::llama2::BaseLlama2<DType>
{
private:
  void pass_begin( const int token );
  void transformer_layer( const int layer_num, const int token_pos );
  void pass_end();

public:
  Llama2( const std::filesystem::path& model_config,
          const std::filesystem::path& model_weights,
          const int32_t start_layer = 0,
          const int32_t end_layer = -1 );
};

} // namespace glinthawk::models::llama2::cuda
