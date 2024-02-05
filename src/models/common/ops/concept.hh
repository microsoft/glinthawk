#pragma once

#include <concepts>
#include <cstdint>
#include <cstring>

#include "models/types.hh"

namespace glinthawk::models::common {

namespace {
constexpr uint64_t UI64 = 1;
}

template<typename T, typename DType>
concept OperationsConcept = requires( T t,
                                      DType* ptr1,
                                      const DType* ptr2,
                                      uint32_t* ptr_uint32,
                                      const uint64_t size,
                                      const std::vector<uint64_t> size_table,
                                      const float val_f,
                                      const bool flag,
                                      const CopyType cpt ) {
  typename T::Float16;
  typename T::Float32;
  typename T::DeviceUniquePtr;
  { t.template accum<UI64>( ptr1, ptr2, size ) } -> std::same_as<void>;
  { t.template rmsnorm<UI64>( ptr1, ptr2, ptr1, ptr2, size ) } -> std::same_as<void>;
  { t.template argmax<UI64>( ptr_uint32, ptr2, ptr1, size ) } -> std::same_as<void>;
  { t.template silu<UI64>( ptr1, ptr1, size ) } -> std::same_as<void>;
  { t.template matmul<UI64, UI64>( ptr1, ptr2, ptr2, size ) } -> std::same_as<void>;
  { t.copy( ptr1, ptr2, size, cpt, flag ) } -> std::same_as<void>;
  { t.copy_table( ptr1, ptr2, size_table, size_table, size_table, cpt, flag ) } -> std::same_as<void>;
  { t.device_allocate( size ) } -> std::same_as<typename T::DeviceUniquePtr>;
};

} // namespace glinthawk::models::common
