#pragma once

#include <concepts>
#include <type_traits>
#include <utility>

namespace glinthawk::compute {

enum class KernelType
{
  Batched, /* kernel.hh */
  Hybrid,  /* kernel_hybrid.hh */
};

template<typename Kernel, typename StateType>
concept KernelConcept = requires( Kernel k, StateType s ) {
  std::is_same_v<typename std::decay<decltype( Kernel::Type )>::type, KernelType>;
  { k.push( std::move( s ) ) } -> std::same_as<void>;
  { k.pop( s ) } -> std::same_as<bool>;
};

} // namespace glinthawk::compute
