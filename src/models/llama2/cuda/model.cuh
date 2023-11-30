#pragma once

#include "models/llama2/base.hh"

#include <cstdint>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <source_location>

#include <cuda_fp16.h>

#include "util/exception.hh"
#include "util/file_descriptor.hh"
#include "util/ring_buffer.hh"

#include "models/common/cuda/ops.cuh"
#include "models/llama2/base.hh"

#include "models/llama2/variants.hh"

namespace glinthawk::models::llama2::cuda {

template<typename DType>
using CUDADeleter = models::common::cuda::ops::CUDADeleter<DType>;

template<typename Config, typename DType>
requires ModelConfig<Config>
struct Context : public InferenceContext<Config, DType>
{
private:
  std::unique_ptr<DType, CUDADeleter<DType>> storage_;

public:
  Context( const Settings<Config>& settings );
};

template<typename Config, typename DType>
class Llama2 : public BaseLlama2<Config, DType, Context<Config, DType>, CUDADeleter<DType>>
{
public:
  using BaseModel = BaseLlama2<Config, DType, Context<Config, DType>, CUDADeleter<DType>>;

  using ContextType = BaseModel::ContextType;
  using SettingsType = BaseModel::SettingsType;
  using ConfigType = BaseModel::ConfigType;
  using TokenizerType = BaseModel::TokenizerType;

private:
  void pass_begin( const std::vector<uint32_t>& token );
  void pre_attention_ops( const int32_t layer_num );
  void attention_ops();
  void post_attention_ops( const int32_t layer_num );
  void pass_end();

public:
  Llama2( const std::filesystem::path& model_dir,
          const uint32_t start_layer = 0,
          const uint32_t end_layer = std::numeric_limits<uint32_t>::max(),
          const uint64_t concurrency_limit = 1 );

  Llama2( const Llama2& ) = delete;
  Llama2& operator=( const Llama2& ) = delete;
  Llama2( Llama2&& ) = default;
  Llama2& operator=( Llama2&& ) = default;

  ~Llama2();

  InferenceState forward( InferenceState&& inference_state, std::shared_ptr<ContextType> context );
  InferenceState pre_attention_forward( InferenceState&& inference_state, std::shared_ptr<ContextType> context );
  InferenceState attention_forward( InferenceState&& inference_state, std::shared_ptr<ContextType> context );
  InferenceState post_attention_forward( InferenceState&& inference_state );

  std::vector<InferenceState> forward( std::vector<InferenceState>&& inference_states,
                                       const std::vector<std::shared_ptr<ContextType>>& contexts );

  std::vector<InferenceState> pre_attention_forward( std::vector<InferenceState>&& inference_states,
                                                     const std::vector<std::shared_ptr<ContextType>>& contexts );

  std::vector<InferenceState> attention_forward( std::vector<InferenceState>&& inference_states,
                                                 const std::vector<std::shared_ptr<ContextType>>& contexts );

  std::vector<InferenceState> post_attention_forward( std::vector<InferenceState>&& inference_states );
};

template<typename T>
using Llama2_70B_Chat = Llama2<configs::Llama2_70B_Chat, T>;

template<typename T>
using Llama2_13B_Chat = Llama2<configs::Llama2_13B_Chat, T>;

template<typename T>
using Llama2_7B_Chat = Llama2<configs::Llama2_7B_Chat, T>;

template<typename T>
using Stories_110M = Llama2<configs::Stories_110M, T>;

} // namespace glinthawk::models::llama2::cuda
