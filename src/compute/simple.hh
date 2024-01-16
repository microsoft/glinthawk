#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "kernel.hh"
#include "models/llama2/model.hh"

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
                           const uint32_t start_slice,
                           const uint32_t end_slice,
                           const uint64_t batch_size,
                           const bool randomize_parameters )
    : model_( model_path,
              start_slice,
              end_slice,
              batch_size,
              batch_size,
              batch_size,
              batch_size,
              batch_size,
              randomize_parameters )
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

#if defined( TARGET_PLATFORM_AMD64 )

template<>
struct SimpleComputeKernelTraits<Platform::AMD64, DataType::Float32>
{
  using DType = glinthawk::float32_t;

  using Stories_110M = models::llama2::amd64::Stories_110M<DType>;
  using Llama2_7B_Chat = models::llama2::amd64::Llama2_7B_Chat<DType>;
  using Llama2_13B_Chat = models::llama2::amd64::Llama2_13B_Chat<DType>;
  using Llama2_70B_Chat = models::llama2::amd64::Llama2_70B_Chat<DType>;

  using KernelVariantT = std::variant<SimpleComputeKernelBase<Stories_110M>,
                                      SimpleComputeKernelBase<Llama2_7B_Chat>,
                                      SimpleComputeKernelBase<Llama2_13B_Chat>,
                                      SimpleComputeKernelBase<Llama2_70B_Chat>>;
};

template<>
struct SimpleComputeKernelTraits<Platform::AMD64, DataType::Float16>
{
  using DType = glinthawk::float16_t;

  using Stories_110M = models::llama2::amd64::Stories_110M<DType>;
  using Llama2_7B_Chat = models::llama2::amd64::Llama2_7B_Chat<DType>;
  using Llama2_13B_Chat = models::llama2::amd64::Llama2_13B_Chat<DType>;
  using Llama2_70B_Chat = models::llama2::amd64::Llama2_70B_Chat<DType>;

  using KernelVariantT = std::variant<SimpleComputeKernelBase<Stories_110M>,
                                      SimpleComputeKernelBase<Llama2_7B_Chat>,
                                      SimpleComputeKernelBase<Llama2_13B_Chat>,
                                      SimpleComputeKernelBase<Llama2_70B_Chat>>;
};

#elif defined( TARGET_PLATFORM_CUDA )

template<>
struct SimpleComputeKernelTraits<Platform::CUDA, DataType::Float32>
{
  using DType = glinthawk::float32_t;

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
  using DType = glinthawk::float16_t;

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
                       const uint32_t start_slice,
                       const uint32_t end_slice,
                       const uint64_t batch_size,
                       const bool randomize_parameters = false )
    : kernel_( [&]() -> KernelVariantT {

#define CREATE_MODEL( MODEL_NAME, CLASS_NAME )                                                                         \
  if ( model_name == MODEL_NAME )                                                                                      \
    return KernelVariantT { std::in_place_type<SimpleComputeKernelBase<typename Traits::CLASS_NAME>>,                  \
                            model_path,                                                                                \
                            start_slice,                                                                               \
                            end_slice,                                                                                 \
                            batch_size,                                                                                \
                            randomize_parameters };                                                                    \
  // clang-format off
    CREATE_MODEL( "llama2-7b-chat", Llama2_7B_Chat )
    else CREATE_MODEL( "llama2-13b-chat", Llama2_13B_Chat )
    else CREATE_MODEL( "llama2-70b-chat", Llama2_70B_Chat )
    else CREATE_MODEL( "stories-110m", Stories_110M )
    else throw std::runtime_error( "Unknown model name" );
    // clang-format on
#undef CREATE_MODEL
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
