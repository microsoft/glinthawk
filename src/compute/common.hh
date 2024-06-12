#pragma once

#include <concepts>
#include <type_traits>
#include <utility>

namespace glinthawk::compute {

enum class KernelType
{
  Batched,     /* kernel.hh */
  Hybrid,      /* kernel_hybrid.hh */
  SimpleHybrid /* kernel_hybrid_simple.hh */
};

template<typename Kernel, typename StateType>
concept KernelConcept = requires( Kernel k, StateType s ) {
  std::is_same_v<typename std::decay<decltype( Kernel::Type )>::type, KernelType>;
  { k.push( std::move( s ) ) } -> std::same_as<void>;
  { k.pop( s ) } -> std::same_as<bool>;
};

class Concurrency
{
private:
  std::array<size_t, util::to_underlying( models::InferenceStage::__COUNT__ )> v_;

public:
  Concurrency( const size_t pre, const size_t att, const size_t post, const size_t classify )
    : v_ { pre, att, post, classify }
  {
    CHECK_GT( pre + att + post + classify, 0 ) << "At least one stage must be enabled";
  }

  void set( const models::InferenceStage stage, const size_t value ) { v_[util::to_underlying( stage )] = value; }
  size_t get( const models::InferenceStage stage ) const { return v_[util::to_underlying( stage )]; }
};

} // namespace glinthawk::compute
