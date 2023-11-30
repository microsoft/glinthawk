#pragma once

#if defined( TARGET_PLATFORM_CPU )
#define _GLINTHAWK_ARCH_NS_ cpu
#define _GLINTHAWK_PLATFORM_NAME_ CPU
#endif

#if defined( TARGET_PLATFORM_CUDA )
#define _GLINTHAWK_ARCH_NS_ cuda
#define _GLINTHAWK_PLATFORM_NAME_ CUDA
#endif

#if defined( TARGET_DTYPE_FLOAT16 )
#define _GLINTHAWK_DTYPE_NAME_ Float16
#if defined( TARGET_PLATFORM_CPU )
#define _GLINTHAWK_DTYPE_ _Float16
#else
#define _GLINTHAWK_DTYPE_ __half
#endif
#endif

#if defined( TARGET_DTYPE_FLOAT32 )
#define _GLINTHAWK_DTYPE_NAME_ Float32
#define _GLINTHAWK_DTYPE_ float
#endif

#if !( defined( _GLINTHAWK_ARCH_NS_ ) && defined( _GLINTHAWK_PLATFORM_NAME_ ) && defined( _GLINTHAWK_DTYPE_ ) )
#error "TARGET_PLATFORM and TARGET_DTYPE must be defined"
#endif
