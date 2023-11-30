#pragma once

#include <concepts>
#include <cstdint>
#include <cstring>
#include <vector>

#include "models/common/ops/concept.hh"

namespace glinthawk::models::llama2 {

namespace {
constexpr uint64_t UI64 = 1;

template<typename T, typename DType, typename T1, typename T2>
concept AdditionalLlamaOperationsConcept = requires( T t,
                                                     T1* t1_ptr,
                                                     T2* t2_ptr,
                                                     DType* ptr,
                                                     DType* ptr_arr[],
                                                     const DType* cptr,
                                                     const DType* cptr_arr[],
                                                     const uint64_t size,
                                                     const uint32_t* int_arr,
                                                     const CopyType cpt ) {
  { t.template attention_0_gemm<UI64, UI64, UI64, UI64>( cptr, cptr_arr, size, int_arr ) } -> std::same_as<void>;
  { t.template attention_2_gemm<UI64, UI64, UI64, UI64>( cptr ctpr_arr, ptr, size, int_arr ) } -> std::same_as<void>;
  { t.template attention_softmax<UI64, UI64>( ptr, int_arr, ptr, size ) } -> std::same_as<void>;
  { t.template apply_rope<UI64, UI64, UI64>( size, int_arr, cptr, cptr, ptr, ptr_arr ) } -> std::same_as<void>;
  { t.template copy_kv_cache<UI64>( ptr_arr, cptr, cptr, size, int_arr ) } -> std::same_as<void>;
  { t.template convert_and_copy<T1, T2>( t1_ptr, t2_ptr, size, cpt ) } -> std::same_as<void>;
};

}

template<typename T, typename DType, typename T1, typename T2>
concept LlamaOperationsConcept
  = AdditionalLlamaOperationsConcept<T, DType, T1, T2> && CommonOperationsConcept<T, DType, T1, T2>;

} // namespace glinthawk::models::common
