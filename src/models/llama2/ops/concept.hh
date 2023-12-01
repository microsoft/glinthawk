#pragma once

#include <concepts>
#include <cstdint>
#include <cstring>
#include <vector>

#include "models/common/ops/concept.hh"

namespace glinthawk::models::llama2 {

namespace {
constexpr uint64_t UI64 = 1;

template<typename T, typename DType, typename T1, typename T2, typename Settings>
concept AdditionalLlamaOperationsConcept = requires( T t,
                                                     T1* t1_ptr,
                                                     T2* t2_ptr,
                                                     DType* ptr,
                                                     DType* arr[],
                                                     const DType* cptr,
                                                     const DType* carr[],
                                                     const uint64_t size,
                                                     const uint32_t* int_arr,
                                                     const CopyType cpt,
                                                     const Settings& s,
                                                     const std::vector<float>& vec ) {
  { T( s ) };
  { t.template attention_0_gemm( cptr, carr, ptr, size, int_arr ) } -> std::same_as<void>;
  { t.template attention_2_gemm( cptr, carr, ptr, size, int_arr ) } -> std::same_as<void>;
  { t.template attention_softmax( ptr, int_arr, ptr, size ) } -> std::same_as<void>;
  { t.template apply_rope( size, int_arr, cptr, cptr, ptr, arr ) } -> std::same_as<void>;
  { t.template soft_sample( ptr, vec, size ) } -> std::same_as<void>;
  { t.template copy_kv_cache( arr, cptr, cptr, size, int_arr ) } -> std::same_as<void>;
  { t.template convert_and_copy<T1, T2>( t1_ptr, t2_ptr, size, cpt ) } -> std::same_as<void>;
};

}

template<typename T, typename DType, typename T1, typename T2, typename Settings>
concept LlamaOperationsConcept
  = AdditionalLlamaOperationsConcept<T, DType, T1, T2, Settings> && common::OperationsConcept<T, DType>;

} // namespace glinthawk::models::common
