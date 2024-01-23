#pragma once

#include <random>
#include <cstdint>

namespace glinthawk::util {

template<typename DType>
void randomize_buffer( DType* buffer, size_t len, const float min, const float max )
{
  static thread_local std::mt19937 generator { std::random_device {}() };
  std::uniform_real_distribution<float> distribution( min, max );

  size_t i;
#pragma omp parallel for schedule( static ) private( i )
  for ( i = 0; i < len; i++ ) {
    if constexpr ( std::is_same_v<DType, float> ) {
      buffer[i] = distribution( generator );
    } else {
      buffer[i] = static_cast<DType>( distribution( generator ) );
    }
  }
}

}
