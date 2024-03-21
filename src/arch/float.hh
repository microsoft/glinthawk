#pragma once

#if defined( TARGET_PLATFORM_CUDA )
#include <cuda_fp16.h>
#endif

namespace glinthawk {

#if defined( TARGET_PLATFORM_AMD64 )
using float16_t = _Float16;
using float32_t = float;
#elif defined( TARGET_PLATFORM_CUDA )
using float16_t = __half;
using float32_t = float;
#endif

}
