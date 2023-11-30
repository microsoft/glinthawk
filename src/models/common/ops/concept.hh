#pragma once

#include <concepts>
#include <cstdint>
#include <cstring>
#include <vector>

#include "models/types.hh"

namespace glinthawk::models::common {

namespace {
constexpr uint64_t UI64 = 1;
}

template<typename T, typename DType>
concept OperationsConcept = requires( T t,
                                      DType* ptr1,
                                      const DType* ptr2,
                                      const std::vector<float>& vec,
                                      uint32_t* ptr_uint32,
                                      const uint64_t size,
                                      const float val_f,
                                      const bool flag,
                                      const CopyType cpt ) {
  { t.template accum<UI64>( ptr1, ptr2, size ) } -> std::same_as<void>;
  { t.template rmsnorm<UI64>( ptr1, ptr2, ptr2, size ) } -> std::same_as<void>;
  { t.template argmax<UI64>( ptr_uint32, ptr2, size ) } -> std::same_as<void>;
  { t.template silu<UI64>( ptr1, ptr1, size ) } -> std::same_as<void>;
  { t.template matmul<UI64, UI64>( ptr1, ptr2, ptr2, size ) } -> std::same_as<void>;
  { t.template soft_sample<UI64>( ptr1, vec, size ) } -> std::same_as<void>;
  { t.copy( ptr1, ptr2, size, flag, cpt ) } -> std::same_as<void>;
  { t.randomize_buffer( ptr1, size, val_f, val_f ) } -> std::same_as<void>;
};

} // namespace glinthawk::models::common
