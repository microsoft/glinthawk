#pragma once

#if defined( TARGET_PLATFORM_CUDA )
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#endif

namespace glinthawk {

#if defined( TARGET_PLATFORM_AMD64 )
using float16_t = _Float16;
using float32_t = float;
using bfloat16_t = _Float16; // NOTE: BF16 support on AMD64 is not implemented yet and is broken
#elif defined( TARGET_PLATFORM_CUDA )
using float16_t = __half;
using float32_t = float;
using bfloat16_t = __nv_bfloat16;
#endif

}
