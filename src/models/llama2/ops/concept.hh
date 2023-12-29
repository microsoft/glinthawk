#pragma once

#include <concepts>
#include <cstdint>
#include <cstring>
#include <vector>

#include "models/common/ops/concept.hh"

namespace glinthawk::models::llama2 {

namespace {
constexpr uint64_t UI64 = 1;

template<typename T, typename DType, typename Settings>
concept AdditionalLlamaOperationsConcept = requires( T t,
                                                     void* ptr_void,
                                                     DType* ptr,
                                                     DType* arr[],
                                                     const DType* cptr,
                                                     const DType* carr[],
                                                     const uint64_t size,
                                                     const uint32_t* int_arr,
                                                     const CopyType cpt,
                                                     const Settings& s,
                                                     const std::vector<float>& vec,
                                                     typename T::ContextType::LayerContextType lc[],
                                                     typename T::ContextType::TokenContextType tc[] ) {
  { T( s ) };
  { t.template attention_0_gemm( cptr, lc, ptr, size, int_arr ) } -> std::same_as<void>;
  { t.template attention_2_gemm( cptr, lc, ptr, size, int_arr ) } -> std::same_as<void>;
  { t.template attention_softmax( ptr, int_arr, ptr, size ) } -> std::same_as<void>;
  { t.template apply_rope( size, int_arr, cptr, cptr, ptr, tc ) } -> std::same_as<void>;
  { t.template soft_sample( ptr, vec, size ) } -> std::same_as<void>;
  { t.template copy_kv_cache( tc, cptr, size ) } -> std::same_as<void>;
  { t.template convert_and_copy<void, void>( ptr_void, ptr_void, size, cpt ) } -> std::same_as<void>;
};

}

template<typename T, typename DType, typename Settings>
concept LlamaOperationsConcept
  = AdditionalLlamaOperationsConcept<T, DType, Settings> && common::OperationsConcept<T, DType>;

} // namespace glinthawk::models::common
