#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "kernel.hh"
#include "models/llama2/cpu/model.hh"

#ifdef GLINTHAWK_CUDA_ENABLED
#include "models/llama2/cuda/model.cuh"
#endif

namespace glinthawk::compute {

namespace {

template<class Model>
class SimpleComputeKernelBase
{
private:
  Model model_;
  compute::ContextManager<Model> context_manager_ { model_.settings() };

public:
  using ModelType = Model;

  SimpleComputeKernelBase() = delete;
  SimpleComputeKernelBase( const SimpleComputeKernelBase& ) = delete;
  SimpleComputeKernelBase& operator=( const SimpleComputeKernelBase& ) = delete;
  SimpleComputeKernelBase( SimpleComputeKernelBase&& ) = default;
  SimpleComputeKernelBase& operator=( SimpleComputeKernelBase&& ) = default;

  SimpleComputeKernelBase( const std::filesystem::path& model_path,
                           uint32_t start_slice,
                           uint32_t end_slice,
                           uint64_t batch_size )
    : model_( model_path, start_slice, end_slice, batch_size )
  {
  }

  std::vector<models::InferenceState> forward( std::vector<models::InferenceState>&& inference_states )
  {
    std::vector<models::InferenceState> output_states;
    output_states.reserve( inference_states.size() );

    std::vector<std::shared_ptr<typename Model::ContextType>> contexts;
    contexts.reserve( inference_states.size() );

    for ( auto& state : inference_states ) {
      auto context = context_manager_.get_context( state.prompt_id() );
      CHECK( context != nullptr ) << "Could not get context";
      contexts.emplace_back( move( context ) );
    }

    return model_.forward( move( inference_states ), contexts );
  }
};

template<Platform platform, DataType data_type>
struct SimpleComputeKernelTraits;

template<>
struct SimpleComputeKernelTraits<Platform::CPU, DataType::Float32>
{
  using DType = float;

  using Stories_110M = models::llama2::cpu::Stories_110M<DType>;
  using Llama2_7B_Chat = models::llama2::cpu::Llama2_7B_Chat<DType>;
  using Llama2_13B_Chat = models::llama2::cpu::Llama2_13B_Chat<DType>;
  using Llama2_70B_Chat = models::llama2::cpu::Llama2_70B_Chat<DType>;

  using KernelVariantT = std::variant<SimpleComputeKernelBase<Stories_110M>,
                                      SimpleComputeKernelBase<Llama2_7B_Chat>,
                                      SimpleComputeKernelBase<Llama2_13B_Chat>,
                                      SimpleComputeKernelBase<Llama2_70B_Chat>>;
};

template<>
struct SimpleComputeKernelTraits<Platform::CPU, DataType::Float16>
{
  using DType = _Float16;

  using Stories_110M = models::llama2::cpu::Stories_110M<DType>;
  using Llama2_7B_Chat = models::llama2::cpu::Llama2_7B_Chat<DType>;
  using Llama2_13B_Chat = models::llama2::cpu::Llama2_13B_Chat<DType>;
  using Llama2_70B_Chat = models::llama2::cpu::Llama2_70B_Chat<DType>;

  using KernelVariantT = std::variant<SimpleComputeKernelBase<Stories_110M>,
                                      SimpleComputeKernelBase<Llama2_7B_Chat>,
                                      SimpleComputeKernelBase<Llama2_13B_Chat>,
                                      SimpleComputeKernelBase<Llama2_70B_Chat>>;
};

#ifdef GLINTHAWK_CUDA_ENABLED
template<>
struct SimpleComputeKernelTraits<Platform::CUDA, DataType::Float32>
{
  using DType = float;

  using Stories_110M = models::llama2::cuda::Stories_110M<DType>;
  using Llama2_7B_Chat = models::llama2::cuda::Llama2_7B_Chat<DType>;
  using Llama2_13B_Chat = models::llama2::cuda::Llama2_13B_Chat<DType>;
  using Llama2_70B_Chat = models::llama2::cuda::Llama2_70B_Chat<DType>;

  using KernelVariantT = std::variant<SimpleComputeKernelBase<Stories_110M>,
                                      SimpleComputeKernelBase<Llama2_7B_Chat>,
                                      SimpleComputeKernelBase<Llama2_13B_Chat>,
                                      SimpleComputeKernelBase<Llama2_70B_Chat>>;
};

template<>
struct SimpleComputeKernelTraits<Platform::CUDA, DataType::Float16>
{
  using DType = __half;

  using Stories_110M = models::llama2::cuda::Stories_110M<DType>;
  using Llama2_7B_Chat = models::llama2::cuda::Llama2_7B_Chat<DType>;
  using Llama2_13B_Chat = models::llama2::cuda::Llama2_13B_Chat<DType>;
  using Llama2_70B_Chat = models::llama2::cuda::Llama2_70B_Chat<DType>;

  using KernelVariantT = std::variant<SimpleComputeKernelBase<Stories_110M>,
                                      SimpleComputeKernelBase<Llama2_7B_Chat>,
                                      SimpleComputeKernelBase<Llama2_13B_Chat>,
                                      SimpleComputeKernelBase<Llama2_70B_Chat>>;
};
#endif

} // namespace

template<Platform platform, DataType data_type>
class SimpleComputeKernel
{
private:
  using Traits = SimpleComputeKernelTraits<platform, data_type>;
  using DType = typename Traits::DType;
  using KernelVariantT = typename Traits::KernelVariantT;

private:
  KernelVariantT kernel_;

public:
  SimpleComputeKernel( const std::filesystem::path model_path,
                       const std::string model_name,
                       uint32_t start_slice,
                       uint32_t end_slice,
                       uint64_t batch_size )
    : kernel_( [&]() -> KernelVariantT {
      if ( model_name == "llama2-7b-chat" ) {
        return KernelVariantT { std::in_place_type<SimpleComputeKernelBase<typename Traits::Llama2_7B_Chat>>,
                                model_path,
                                start_slice,
                                end_slice,
                                batch_size };
      } else if ( model_name == "llama2-13b-chat" ) {
        return KernelVariantT { std::in_place_type<SimpleComputeKernelBase<typename Traits::Llama2_13B_Chat>>,
                                model_path,
                                start_slice,
                                end_slice,
                                batch_size };
      } else if ( model_name == "llama2-70b-chat" ) {
        return KernelVariantT { std::in_place_type<SimpleComputeKernelBase<typename Traits::Llama2_70B_Chat>>,
                                model_path,
                                start_slice,
                                end_slice,
                                batch_size };
      } else if ( model_name == "stories-110m" ) {
        return KernelVariantT { std::in_place_type<SimpleComputeKernelBase<typename Traits::Stories_110M>>,
                                model_path,
                                start_slice,
                                end_slice,
                                batch_size };
      } else {
        throw std::runtime_error( "Unknown model name" );
      }
    }() )
  {
  }

  std::vector<models::InferenceState> forward( std::vector<models::InferenceState>&& inference_states )
  {
    return std::visit(
      [&]( auto& kernel ) -> std::vector<models::InferenceState> {
        return kernel.forward( std::move( inference_states ) );
      },
      kernel_ );
  }

  size_t max_seq_len() const
  {
    return std::visit( []<typename T>( const T& ) -> size_t { return T::ModelType::ConfigType::seq_len; }, kernel_ );
  }

  size_t dim() const
  {
    return std::visit( []<typename T>( const T& ) -> size_t { return T::ModelType::ConfigType::dim; }, kernel_ );
  }
};

} // namespace glinthawk::compute
